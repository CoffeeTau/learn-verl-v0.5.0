"""Create reward-side Parquet views of frozen train/dev, without re-splitting."""
import json
from pathlib import Path
from project.agent.search import SYSTEM_PROMPT
from project.common import config_from, load_jsonl, resource_path, save_json, sha256, show_summary
from project.training.protocol import continuation_ids


def prepare(version="v1", output_dir=None):
    prompt = SYSTEM_PROMPT
    if version in ("v2", "v4"):
        from project.agent.correction import SYSTEM_PROMPT as prompt
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    cfg = config_from()
    source = resource_path("AGENTIC_PROCESSED_DATA_DIR")
    manifest = json.loads((source / "manifest.json").read_text())
    out = Path(output_dir) if output_dir else source / (f"verl_{version}" if version in ("v2", "v4") else "verl")
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(resource_path("AGENTIC_MODEL_DIR")), local_files_only=True)
    # Verify the hard-coded continuation against this actual local Qwen3 template.
    base = [{"role": "system", "content": prompt.format(max_searches=cfg["max_searches"])},
            {"role": "user", "content": "Template check"}]
    raw = tokenizer.apply_chat_template(base, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    suffix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    if not raw.endswith(suffix):
        raise ValueError("Unsupported Qwen3 non-thinking template")
    expected = tokenizer.apply_chat_template(base + [{"role": "assistant", "content": "<search>Paris</search>"},
                                                     {"role": "user", "content": "observation"}],
                                              tokenize=False, add_generation_prompt=True, enable_thinking=False)
    continuation = tokenizer.decode(continuation_ids(tokenizer, "observation", []), skip_special_tokens=False)
    if not expected.endswith("<search>Paris</search>" + continuation):
        raise ValueError("Continuation does not match installed chat template")
    sizes, files = {}, {}
    for split in ("train", "dev"):
        for name in (f"{split}.jsonl", f"{split}.labels.jsonl"):
            if sha256(source / name) != manifest["files"][name]:
                raise ValueError(f"Frozen input changed: {name}")
        tasks = load_jsonl(source / f"{split}.jsonl")
        labels = {row["id"]: row for row in load_jsonl(source / f"{split}.labels.jsonl")}
        records = []
        for i, row in enumerate(tasks):
            messages = [{"role": "system", "content": base[0]["content"]},
                        {"role": "user", "content": row["question"]}]
            if len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=False)) > 512:
                raise ValueError(f"Prompt exceeds 512 tokens: {row['id']}; do not silently truncate")
            records.append({"data_source": "agentic_2wiki", "agent_name": ("agentic_correction_train" if split == "train" else "agentic_correction_eval") if version in ("v2", "v4") else "agentic_search", "prompt": messages,
                            "ability": "multi_hop_search", "reward_model": {"style": "rule", "ground_truth": labels[row['id']]['answer']},
                            "extra_info": {"split": split, "index": i, "task_id": row["id"]}})
        path = out / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(records), path)
        sizes[split], files[path.name] = len(records), sha256(path)
    report = {"sizes": sizes, "files": files, "source_manifest_sha256": sha256(source / "manifest.json"),
              "corpus_sha256": manifest["corpus_sha256"], "task": cfg, "enable_thinking": False, "version": version}
    save_json(out / "manifest.json", report)
    show_summary(resource_path("AGENTIC_RUNS_DIR") / f"{version}_prepare",
                 f"=== {version.upper()} PREPARE | PASSED ===\nTasks: train={sizes['train']} dev={sizes['dev']} | test untouched\n"
                 "Prompt: system + question | gold: reward side only\n"
                 "Qwen3 continuation: checked | no-thinking | prompt <=512\n"
                 f"Corpus SHA256: {manifest['corpus_sha256'][:16]}")
    return out, report


if __name__ == "__main__":
    prepare()
