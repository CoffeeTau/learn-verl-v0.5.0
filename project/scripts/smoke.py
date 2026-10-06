"""Resource smoke checks, not training or benchmark evaluation.

Search action protocol follows Search-R1/infer.py; generation uses vLLM and
Qwen3's non-thinking chat template. No gold labels are passed to generation.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time
import traceback

from project.scripts.summarize_smoke import placeholder_query, render


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def required(condition, message):
    if not condition:
        raise ValueError(message)


def find_split(root, split):
    matches = sorted(root.rglob(f"{split}.json"))
    required(len(matches) == 1,
             f"Expected one {split}.json under {root}; found {matches}. "
             "Keep only the repaired dataset here; store older archives separately.")
    return matches[0]


def validate_row(row):
    for key in ("_id", "question", "answer"):
        required(isinstance(row.get(key), str) and row[key].strip(), f"Invalid {key}")
    required(isinstance(row.get("context"), list) and row["context"], "Missing context")
    titles = {}
    for passage in row["context"]:
        required(isinstance(passage, list) and len(passage) == 2, "Invalid context entry")
        title, sentences = passage
        required(isinstance(title, str) and title.strip(), "Invalid title")
        required(isinstance(sentences, list) and sentences and
                 all(isinstance(s, str) for s in sentences), "Invalid sentences")
        if title in titles:
            required(titles[title] == sentences, f"Ambiguous title: {title}")
        titles[title] = sentences
    facts = row.get("supporting_facts")
    required(isinstance(facts, list) and facts, "Missing supporting_facts")
    for fact in facts:
        required(isinstance(fact, list) and len(fact) == 2, "Invalid supporting fact")
        title, index = fact
        required(title in titles, f"Support title absent from context: {title}")
        required(type(index) is int and 0 <= index < len(titles[title]),
                 f"Invalid supporting sentence index: {fact}")
        required(titles[title][index].strip(), f"Empty support sentence: {fact}")
    evidences = row.get("evidences")
    required(isinstance(evidences, list) and evidences, "Missing evidences")
    required(all(isinstance(e, list) and len(e) == 3 and
                 all(isinstance(x, str) for x in e) for e in evidences), "Invalid evidence triples")
    return titles


def check_data(args, report, out):
    root = Path(os.environ["AGENTIC_RAW_DATA_DIR"])
    report["raw_data_dir"] = str(root)
    report["scope"] = "Seeded sample checks only; not a full data audit or formal split."
    report["splits"] = {}
    report["errors"] = []
    corpus = {}
    train_questions = []
    sample_ids = {}
    for split in ("train", "dev"):
        report["phase"] = f"read_{split}"
        path = find_split(root, split)
        print(f"Reading {path}; JSON is loaded one split at a time.", flush=True)
        with path.open(encoding="utf-8") as handle:
            rows = json.load(handle)
        required(isinstance(rows, list) and rows, f"{path} must contain a nonempty JSON array")
        chosen = random.Random(42).sample(range(len(rows)), min(args.samples, len(rows)))
        report["splits"][split] = {
            "file": str(path), "bytes": path.stat().st_size, "total_rows": len(rows),
            "sampled_rows": len(chosen), "valid_rows": 0, "types": {},
        }
        types = Counter()
        sample_ids[split] = set()
        for index in chosen:
            row = rows[index]
            try:
                validate_row(row)
                required(row["_id"] not in sample_ids[split], "Duplicate sampled ID")
                sample_ids[split].add(row["_id"])
                report["splits"][split]["valid_rows"] += 1
                types[row.get("type", "unknown")] += 1
                if split == "train":
                    # corpus has text only; questions/answers/labels remain separate.
                    for title, sentences in row["context"]:
                        content = title + "\n" + " ".join(sentences)
                        pid = hashlib.sha256(content.encode()).hexdigest()[:20]
                        corpus[pid] = {"id": pid, "title": title, "text": " ".join(sentences)}
                    train_questions.append({"id": row["_id"], "question": row["question"]})
            except (ValueError, TypeError, KeyError) as exc:
                report["errors"].append({"split": split, "row_index": index,
                                         "error": str(exc)})
        report["splits"][split]["types"] = dict(types)
        del rows
    overlap = sample_ids["train"] & sample_ids["dev"]
    if overlap:
        report["errors"].append({"error": "Sampled train/dev ID overlap", "ids": sorted(overlap)})
    report["alias_files"] = [str(p) for p in root.rglob("id_aliases.json")]
    report["warnings"] = [] if report["alias_files"] else [
        "id_aliases.json not found; basic smoke can proceed, alias-aware evaluation needs it later."
    ]
    required(not report["errors"], f"{len(report['errors'])} data errors; see data_report.json")
    with (out / "sample_corpus.jsonl").open("w", encoding="utf-8") as handle:
        for passage in corpus.values():
            handle.write(json.dumps(passage, ensure_ascii=False) + "\n")
    write_json(out / "sample_questions.json", train_questions)
    report["corpus_passages"] = len(corpus)
    report["phase"] = "data_complete"


def extract_tag(text, tag):
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    required(len(matches) == 1 and matches[0].strip(), f"Expected exactly one nonempty <{tag}> action: {text!r}")
    return matches[0].strip()


def check_models(args, report, out):
    write_json(out / "model_trace.json", [])
    report["phase"] = "check_local_resources"
    data_report = json.loads((out / "data_report.json").read_text())
    required(data_report.get("status") == "passed", "Run data smoke successfully first")
    report["data_report_started_at"] = data_report["started_at"]
    for key in ("AGENTIC_MODEL_DIR", "AGENTIC_RETRIEVER_DIR"):
        path = Path(os.environ[key])
        required((path / "config.json").is_file(), f"Missing local model config: {path}")
        report[key] = str(path)
        manifest = path.parent / (path.name + ".revision.json")
        report[key + "_revision"] = json.loads(manifest.read_text()) if manifest.exists() else None
    import torch
    from transformers import AutoTokenizer
    from project.retrieval.e5 import E5Encoder

    required(torch.cuda.is_available(), "CUDA unavailable; activate the server training environment")
    report["gpu"] = {"visible_count": torch.cuda.device_count(), "name": torch.cuda.get_device_name(0),
                     "memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30}
    required(torch.cuda.is_bf16_supported(), "BF16 not supported by this CUDA device/environment")
    report["phase"] = "e5_encode"
    print("Loading E5 on CPU and encoding the sampled corpus.", flush=True)
    corpus = [json.loads(line) for line in (out / "sample_corpus.jsonl").read_text().splitlines()]
    questions = json.loads((out / "sample_questions.json").read_text())
    required(corpus and questions, "Empty smoke corpus/questions")
    encoder = E5Encoder(os.environ["AGENTIC_RETRIEVER_DIR"])
    vectors = encoder.encode([p["title"] + "\n" + p["text"] for p in corpus])
    report["e5"] = {"device": "cpu", "passages": len(corpus), "embedding_shape": list(vectors.shape)}
    report["phase"] = "qwen_template"
    tokenizer = AutoTokenizer.from_pretrained(os.environ["AGENTIC_MODEL_DIR"], local_files_only=True)

    def format_chat(messages):
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)

    template = format_chat([{"role": "user", "content": "Hello"}])
    report["template_suffix"] = template[-200:]
    required("<think>\n\n</think>" in template, "Qwen3 non-thinking template not applied")
    report["phase"] = "vllm_load"
    print("Loading Qwen3-4B via vLLM on one visible GPU.", flush=True)
    write_json(out / "models_report.json", report)
    # Spawn avoids CUDA fork errors after the hardware checks above.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams
    llm = LLM(model=os.environ["AGENTIC_MODEL_DIR"], tokenizer=os.environ["AGENTIC_MODEL_DIR"],
              dtype="bfloat16", tensor_parallel_size=1, max_model_len=8192,
              gpu_memory_utilization=0.65, max_num_seqs=1, enforce_eager=True, seed=42)
    trace = []

    def generate(messages, stop=None):
        prompt = format_chat(messages)
        required(len(tokenizer.encode(prompt)) + 256 <= 8192, "Smoke prompt exceeds context budget")
        params = SamplingParams(temperature=0, max_tokens=256, stop=stop,
                                include_stop_str_in_output=True)
        result = llm.generate([prompt], params, use_tqdm=False)[0].outputs[0]
        item = {"messages": messages.copy(), "output": result.text,
                "finish_reason": result.finish_reason, "tokens": len(result.token_ids)}
        trace.append(item)
        # Persist after each generation, even when subsequent parsing fails.
        write_json(out / "model_trace.json", trace)
        required(result.text.strip(), "Empty Qwen3 output")
        required(result.finish_reason != "length", "Generation hit 256-token smoke limit")
        return result.text

    report["phase"] = "qwen_short_generation"
    print("Testing short generation, then one real search roundtrip.", flush=True)
    short = generate([{"role": "user", "content": "Reply with exactly <answer>READY</answer>."}])
    report["short_generation"] = short
    extract_tag(short, "answer")
    report["phase"] = "search_action"
    question = questions[0]
    messages = [{"role": "system", "content": (
        "You are answering the user's question using a search tool. First formulate a concrete "
        "search query using relevant entity names and the missing fact. Put your actual search "
        "words between the opening tag <search> and closing tag </search>, with no other text. "
        "Do not output placeholder words such as 'query'. After receiving information, put "
        "a concise answer to the user's question between <answer> and </answer>. "
        "If evidence is insufficient, say insufficient evidence inside the answer tags."
    )}, {"role": "user", "content": question["question"]}]
    action = generate(messages, ["</search>"])
    query = extract_tag(action, "search")
    report["tool_roundtrip"] = {"question_id": question["id"], "question": question["question"], "query": query}
    required(not placeholder_query(query), "Model emitted a placeholder instead of a concrete search query")
    report["phase"] = "retrieve_and_continue"
    query_vector = encoder.encode([question["question"] + " [SEP] " + query], is_query=True)
    scores = (query_vector @ vectors.T)[0]
    top = torch.topk(scores, min(3, len(corpus)))
    hits = [{**corpus[i], "score": score} for i, score in zip(top.indices.tolist(), top.values.tolist())]
    observation = "\n".join(f"[{p['id']}] {p['title']}\n{p['text'][:1200]}" for p in hits)
    messages += [{"role": "assistant", "content": action},
                 {"role": "user", "content": "<information>\n" + observation + "\n</information>"}]
    answer_text = generate(messages)
    answer = extract_tag(answer_text, "answer")
    report["tool_roundtrip"] = {"question_id": question["id"], "question": question["question"],
                                "query": query, "hits": hits, "answer": answer}
    report["scope"] = (
        "One prompted search roundtrip on a sampled training corpus. Checks loading/encoding/"
        "generation/protocol only; no answer accuracy, training update, masking or weight-sync claim."
    )
    report["phase"] = "models_complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["data", "models"])
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    required(args.samples > 0, "--samples must be positive")
    out = Path(os.environ["AGENTIC_RUNS_DIR"]) / "resource_smoke"
    out.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "stage": args.stage,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "platform": platform.platform(), "packages": {}}
    start = time.monotonic()
    target = out / f"{args.stage}_report.json"
    write_json(target, report)  # replace stale success before doing any work
    try:
        for package in ("torch", "transformers", "vllm", "huggingface-hub", "verl"):
            try:
                report["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                report["packages"][package] = None
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        report["code_commit"] = commit.stdout.strip() or None
        (check_data if args.stage == "data" else check_models)(args, report, out)
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr, flush=True)
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - start, 3)
        write_json(target, report)
        summary = render(report)
        (out / f"{args.stage}_summary.txt").write_text(summary + "\n", encoding="utf-8")
        print(f"{args.stage.upper()}: {report['status'].upper()}\n{summary}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
