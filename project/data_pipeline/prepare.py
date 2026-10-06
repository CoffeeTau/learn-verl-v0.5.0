"""Freeze a modest 2Wiki task set and a shared, sentence-aligned corpus."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import traceback
import unicodedata

from project.common import (code_info, config_from, resource_path, save_json,
                            save_jsonl, sha256, show_summary, signature)
from project.scripts.smoke import find_split, validate_row

TYPES = ("comparison", "inference", "compositional", "bridge_comparison")


def question_key(text):
    return re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def paragraph_chunks(title, sentences, tokenizer, max_tokens):
    """Preserve sentence boundaries; reject an oversized sentence instead of losing evidence."""
    chunks, current = [], []
    def count(entries):
        text = "passage: " + title + "\n" + " ".join(s for _, s in entries)
        return len(tokenizer.encode(text, add_special_tokens=True))
    for index, sentence in enumerate(sentences):
        if count([(index, sentence)]) > max_tokens:
            raise ValueError("sentence_too_long")
        if current and count(current + [(index, sentence)]) > max_tokens:
            chunks.append(current)
            current = []
        current.append((index, sentence))
    if current:
        chunks.append(current)
    paragraph_id = signature([title, sentences])[:24]
    return [{"id": signature([title, entries])[:24], "paragraph_id": paragraph_id,
             "title": title, "sentence_ids": [i for i, _ in entries],
             "sentences": [s for _, s in entries], "text": " ".join(s for _, s in entries)}
            for entries in chunks]


def prepare_task(row, tokenizer, chunk_tokens):
    validate_row(row)
    if row.get("type") not in TYPES:
        raise ValueError("unsupported_question_type")
    if len({fact[0] for fact in row["supporting_facts"]}) < 2:
        raise ValueError("fewer_than_two_support_titles")
    chunks, lookup = {}, {}
    for title, sentences in row["context"]:
        for chunk in paragraph_chunks(title, sentences, tokenizer, chunk_tokens):
            chunks[chunk["id"]] = chunk
            for index in chunk["sentence_ids"]:
                lookup[title, index] = chunk["id"]
    support = [{"title": title, "sentence_id": index, "passage_id": lookup[title, index]}
               for title, index in row["supporting_facts"]]
    task = {"id": row["_id"], "question": row["question"].strip(), "type": row["type"],
            "answer": row["answer"].strip(), "supporting_facts": support,
            "evidences": row["evidences"], "answer_id": row.get("answer_id")}
    return task, list(chunks.values())


def select(rows, count, tokenizer, config, seen_ids, seen_questions, rejected, source):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row.get("type", "unknown")].append(row)
    rng = random.Random(config["seed"])
    for bucket in buckets.values():
        rng.shuffle(bucket)
    tasks, chunks = [], {}
    while len(tasks) < count:
        added = False
        for kind in TYPES:
            if len(tasks) >= count:
                break
            while buckets[kind]:
                row = buckets[kind].pop()
                try:
                    key = question_key(row["question"])
                    if row["_id"] in seen_ids or key in seen_questions:
                        raise ValueError("duplicate_id_or_question")
                    task, passages = prepare_task(row, tokenizer, config["chunk_tokens"])
                except (ValueError, KeyError, TypeError) as exc:
                    rejected.append({"source": source, "id": row.get("_id"), "reason": str(exc)})
                    continue
                tasks.append(task)
                chunks.update({p["id"]: p for p in passages})
                seen_ids.add(row["_id"])
                seen_questions.add(key)
                added = True
                break
        if not added:
            raise ValueError(f"Insufficient eligible {source} tasks: {len(tasks)}/{count}")
    return tasks, chunks


def build(config, raw, output, corpus_dir, tokenizer):
    # Freeze once: changes require a new dataset version/directory, not silent rebuilding.
    if (output / "manifest.json").exists() or (corpus_dir / "manifest.json").exists():
        raise ValueError("Dataset version already exists; use the existing artifacts or a new version")
    if not 1 <= config["chunk_tokens"] <= 512:
        raise ValueError("chunk_tokens must be in [1,512] for E5")
    for name in ("train_size", "dev_size", "test_size"):
        if config[name] <= 0:
            raise ValueError(f"{name} must be positive")
    output.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    paths = {split: find_split(raw, split) for split in ("train", "dev")}
    rejected = []
    print("Preparing frozen test tasks from official dev; reserving all dev IDs/questions.", flush=True)
    dev = json.loads(paths["dev"].read_text(encoding="utf-8"))
    test, corpus = select(dev, config["test_size"], tokenizer, config, set(), set(), rejected, "official_dev")
    # Reserve ALL official dev questions/IDs, including those not sampled for test.
    seen_ids = {row["_id"] for row in dev}
    seen_questions = {question_key(row["question"]) for row in dev}
    del dev
    print("Preparing train/development tasks and shared corpus.", flush=True)
    rows = json.loads(paths["train"].read_text(encoding="utf-8"))
    train_dev, passages = select(rows, config["train_size"] + config["dev_size"], tokenizer,
                                 config, seen_ids, seen_questions, rejected, "official_train")
    corpus.update(passages)
    # Selection interleaves types; prefix is a balanced development set.
    dev = train_dev[:config["dev_size"]]
    train = train_dev[config["dev_size"]:]
    chosen_ids = {row["id"] for row in train_dev}
    rng = random.Random(config["seed"] + 1)
    order = list(range(len(rows)))
    rng.shuffle(order)
    existing_paragraphs = {p["paragraph_id"] for p in corpus.values()}
    background_count = 0
    for index in order:
        if background_count >= config["background_paragraphs"]:
            break
        row = rows[index]
        if row["_id"] in chosen_ids:
            continue
        for title, sentences in row["context"]:
            if background_count >= config["background_paragraphs"]:
                break
            try:
                chunks = paragraph_chunks(title, sentences, tokenizer, config["chunk_tokens"])
            except ValueError:
                continue
            if not chunks or chunks[0]["paragraph_id"] in existing_paragraphs:
                continue
            corpus.update({p["id"]: p for p in chunks})
            existing_paragraphs.add(chunks[0]["paragraph_id"])
            background_count += 1
    del rows
    if background_count < config["background_paragraphs"]:
        raise ValueError(f"Insufficient unique background paragraphs: {background_count}")
    splits = {"train": train, "dev": dev, "test": test}
    coverage = sum(len(row["supporting_facts"]) for split in splits.values() for row in split)
    for split in splits.values():
        for row in split:
            for fact in row["supporting_facts"]:
                assert fact["passage_id"] in corpus
                assert fact["sentence_id"] in corpus[fact["passage_id"]]["sentence_ids"]
    files = {}
    for split, tasks in splits.items():
        # Public question files exclude all answers/support labels, including test labels.
        public = [{key: row[key] for key in ("id", "question", "type")} for row in tasks]
        labels = [{key: value for key, value in row.items() if key not in ("question", "type")} for row in tasks]
        for name, data in ((f"{split}.jsonl", public), (f"{split}.labels.jsonl", labels)):
            save_jsonl(output / name, data)
            files[name] = sha256(output / name)
    save_jsonl(output / "rejected.jsonl", rejected)
    save_jsonl(corpus_dir / "corpus.jsonl", [corpus[key] for key in sorted(corpus)])
    title_versions = defaultdict(set)
    for passage in corpus.values():
        title_versions[passage["title"]].add(passage["paragraph_id"])
    manifest = {"version": config["dataset_version"], "config": config, "code": code_info(),
                "source_files": {key: {"path": str(p), "sha256": sha256(p)} for key, p in paths.items()},
                "files": files, "corpus_sha256": sha256(corpus_dir / "corpus.jsonl"),
                "sizes": {key: len(value) for key, value in splits.items()},
                "types": {key: dict(Counter(r["type"] for r in value)) for key, value in splits.items()},
                "corpus_chunks": len(corpus), "background_paragraphs": background_count,
                "support_sentences_mapped": coverage,
                "title_variant_count": sum(len(v) > 1 for v in title_versions.values()),
                "rejected": len(rejected), "rejection_reasons": dict(Counter(r["reason"] for r in rejected)),
                "alias_scoring": False,
                "limits": ["Official dev reserved from training; test is a sampled project split.",
                           "Evidence triples preserved, not automatically proven or semantically sentence-aligned.",
                           "Source mirror revision may differ from official April 7 release.",
                           "Scoring uses canonical answer only; alias-aware scoring not yet enabled."]}
    save_json(corpus_dir / "manifest.json", {"version": manifest["version"],
                                            "corpus_sha256": manifest["corpus_sha256"],
                                            "chunks": len(corpus)})
    save_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    args = parser.parse_args()
    report_dir = resource_path("AGENTIC_RUNS_DIR") / "prepare_v1"
    report_dir.mkdir(parents=True, exist_ok=True)
    try:
        from transformers import AutoTokenizer
        config = config_from(args.config)
        output = resource_path("AGENTIC_PROCESSED_DATA_DIR")
        corpus = resource_path("AGENTIC_CORPUS_DIR")
        if output.name != config["dataset_version"] or corpus.name != config["dataset_version"]:
            raise ValueError("Configured dataset_version must match output directory names")
        tokenizer = AutoTokenizer.from_pretrained(str(resource_path("AGENTIC_RETRIEVER_DIR")), local_files_only=True)
        manifest = build(config, resource_path("AGENTIC_RAW_DATA_DIR"), output, corpus, tokenizer)
        save_json(report_dir / "report.json", {"status": "passed", **manifest})
        show_summary(report_dir, "\n".join([
            "=== PREPARE | PASSED ===",
            f"Tasks: train={manifest['sizes']['train']} dev={manifest['sizes']['dev']} test={manifest['sizes']['test']}",
            f"Corpus: {manifest['corpus_chunks']} chunks | background={manifest['background_paragraphs']} paragraphs",
            f"Mapped support sentences: {manifest['support_sentences_mapped']} | missing=0",
            f"Rejected candidates: {manifest['rejected']} | title variants preserved={manifest['title_variant_count']}",
            "Split: deduplicated IDs/questions | official dev excluded from training",
            "Labels: separate files | canonical-answer scoring (aliases pending)",
            "Selection: four question types interleaved | test frozen",
            f"Corpus SHA256: {manifest['corpus_sha256'][:16]}",
            "Next: build E5 index; then V0 development evaluation.",
        ]))
        return 0
    except Exception as exc:
        save_json(report_dir / "report.json", {"status": "failed", "traceback": traceback.format_exc()})
        show_summary(report_dir, f"=== PREPARE | FAILED ===\n{type(exc).__name__}: {str(exc)[:240]}\nDetails: runs/prepare_v1/report.json")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
