"""Evaluate the untrained Qwen3 search agent on the project development split."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
import traceback
from uuid import uuid4

from project.agent.search import ContextBudgetError, SYSTEM_PROMPT, run_episode
from project.common import (code_info, config_from, load_jsonl, model_inventory, resource_path,
                            save_json, sha256, show_summary)
from project.evaluation.metrics import normalize_answer, score_answer


def aggregate(results, planned):
    # All attempted tasks, including invalid actions and budget exhaustion, stay in the denominator.
    n = len(results)
    return {"completed": n, "planned": planned,
            "em_percent": 100 * sum(row["em"] for row in results) / n if n else 0,
            "f1_percent": 100 * sum(row["f1"] for row in results) / n if n else 0,
            "mean_searches": sum(row["searches"] for row in results) / n if n else 0,
            "mean_tokens": sum(row["total_tokens"] for row in results) / n if n else 0,
            "mean_seconds": sum(row["seconds"] for row in results) / n if n else 0,
            "statuses": dict(Counter(row["status"] for row in results)),
            "insufficient_evidence": sum(normalize_answer(row["answer"]) == "insufficient evidence" for row in results),
            "invalid_citations": sum(bool(row.get("invalid_source_ids")) for row in results)}


def summary_text(report, results, relative_run):
    m = report["metrics"]
    lines = [f"=== V0 DEV | {report['status'].upper()} ===", f"Run: {relative_run}",
             f"Tasks: {m['completed']}/{m['planned']} | concurrency=1 | max_searches={report['config']['max_searches']}",
             f"Answer EM={m['em_percent']:.2f}% | F1={m['f1_percent']:.2f}% (canonical only)",
             f"Mean: searches={m['mean_searches']:.2f} | tokens={m['mean_tokens']:.0f} | seconds={m['mean_seconds']:.2f}",
             f"Status counts: {m['statuses']}",
             f"Insufficient evidence={m['insufficient_evidence']} | invalid citation IDs={m['invalid_citations']}",
             "Tokens include repeated prompt reads; episode time excludes model/index loading."]
    # Two representative cases, no long documents or absolute machine paths.
    samples = []
    for predicate in (lambda row: row["em"] == 0, lambda row: row["em"] == 1):
        match = next((row for row in results if predicate(row)), None)
        if match:
            samples.append(match)
    for number, row in enumerate(samples, 1):
        lines.append(f"Case {number}: {row['id']} | searches={row['searches']} | EM={row['em']:.0f}")
        lines.append("  Q: " + row["question"][:130])
        lines.append("  Queries: " + " -> ".join(step["query"] for step in row["steps"] if "query" in step)[:170])
        lines.append("  Pred: " + " ".join(row["answer"].split())[:100] + " | Gold: " + row["gold"][:70])
    if report.get("error"):
        lines.append("Error: " + report["error"][:200])
    if report["status"] != "passed":
        lines.append("Incomplete run: scores cover completed tasks only; do not use as final baseline.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--limit", type=int, default=None, help="Optional partial dev run; omitted = all 200")
    args = parser.parse_args()
    config = config_from(args.config)
    runs = resource_path("AGENTIC_RUNS_DIR") / "v0"
    run_id = "dev_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:6]
    out = runs / run_id
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "config": config, "metrics": aggregate([], 0)}
    results = []
    planned = 0
    try:
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        data_dir = resource_path("AGENTIC_PROCESSED_DATA_DIR")
        dataset = json.loads((data_dir / "manifest.json").read_text())
        for name in ("dev.jsonl", "dev.labels.jsonl"):
            if sha256(data_dir / name) != dataset["files"][name]:
                raise ValueError(f"Frozen file changed: {name}")
        questions = load_jsonl(data_dir / "dev.jsonl")
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be positive")
            questions = questions[:args.limit]
        planned = len(questions)
        if not planned:
            raise ValueError("Empty development split")
        labels = {row["id"]: row for row in load_jsonl(data_dir / "dev.labels.jsonl")}
        report["metrics"] = aggregate([], planned)
        report["full_dev_size"] = dataset["sizes"]["dev"]
        from project.retrieval.local import LocalRetriever
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        retriever = LocalRetriever(resource_path("AGENTIC_CORPUS_DIR"), resource_path("AGENTIC_INDEX_DIR"),
                                   resource_path("AGENTIC_RETRIEVER_DIR"))
        if retriever.manifest["corpus_sha256"] != dataset["corpus_sha256"]:
            raise ValueError("Dataset and retrieval corpus versions differ")
        model_path = resource_path("AGENTIC_MODEL_DIR")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        manifest = {"code": code_info(), "config": config, "question_split": "dev", "limit": args.limit,
                    "dataset_manifest": dataset, "index_manifest": retriever.manifest,
                    "policy_model": model_inventory(model_path), "system_prompt": SYSTEM_PROMPT,
                    "enable_thinking": False, "scoring": "canonical_answer_em_f1_no_aliases"}
        save_json(out / "manifest.json", manifest)
        save_json(out / "resolved_config.json", {**config, "model_path": str(model_path), "dataset_path": str(data_dir)})
        save_json(out / "report.json", report)
        llm = LLM(model=str(model_path), tokenizer=str(model_path), dtype="bfloat16",
                  tensor_parallel_size=1, max_model_len=config["max_context_tokens"],
                  gpu_memory_utilization=config["gpu_memory_utilization"], max_num_seqs=1,
                  enforce_eager=True, seed=config["seed"])
        params = SamplingParams(temperature=config["temperature"], max_tokens=config["max_new_tokens"],
                                stop=["</search>", "</answer>"], include_stop_str_in_output=True)

        def generate(messages):
            token_ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                                      add_generation_prompt=True, enable_thinking=False)
            if len(token_ids) + config["max_new_tokens"] > config["max_context_tokens"]:
                raise ContextBudgetError()
            response = llm.generate([{"prompt_token_ids": token_ids}], params, use_tqdm=False)[0]
            output = response.outputs[0]
            return {"text": output.text, "finish_reason": output.finish_reason,
                    "prompt_tokens": len(token_ids), "output_tokens": len(output.token_ids)}

        with (out / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
            for number, row in enumerate(questions, 1):
                # Only row['question'] crosses into the agent. Labels are read for scoring afterwards.
                result = run_episode(row["question"], generate, retriever.search, config)
                gold = labels[row["id"]]
                score = score_answer(result["answer"], gold["answer"]) if result["status"] == "answered" else {"em": 0.0, "f1": 0.0}
                retrieved = {hit["id"] for step in result["steps"] for hit in step.get("hits", [])}
                required_ids = {fact["passage_id"] for fact in gold["supporting_facts"]}
                result.update(id=row["id"], question=row["question"], gold=gold["answer"], **score,
                              support_chunks_found=len(retrieved & required_ids), support_chunks_total=len(required_ids))
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                results.append(result)
                if number % 10 == 0 or number == planned:
                    print(f"V0 dev: {number}/{planned}", flush=True)
                    report["metrics"] = aggregate(results, planned)
                    save_json(out / "report.json", report)
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        report["metrics"] = aggregate(results, planned)
        save_json(out / "report.json", report)
        text = summary_text(report, results, f"runs/v0/{run_id}")
        show_summary(out, text)
        (runs / "latest_summary.txt").write_text(text + "\n", encoding="utf-8")
        save_json(runs / "latest_run.json", {"run_id": run_id, "status": report["status"]})
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
