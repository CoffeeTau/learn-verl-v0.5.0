"""Evaluate the untrained Qwen3 search agent on the project development split."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
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
    lines = [f"=== {report.get('variant', 'V0')} DEV | {report['status'].upper()} ===", f"Run: {relative_run}",
             f"Tasks: {m['completed']}/{m['planned']} | concurrency=1 | max_searches={report['config']['max_searches']}",
             f"Answer EM={m['em_percent']:.2f}% | F1={m['f1_percent']:.2f}% (canonical only)",
             f"Mean: searches={m['mean_searches']:.2f} | tokens={m['mean_tokens']:.0f} | seconds={m['mean_seconds']:.2f}",
             f"Status counts: {m['statuses']}",
             f"Insufficient evidence={m['insufficient_evidence']} | invalid citation IDs={m['invalid_citations']}",
             "Tokens include repeated prompt reads; episode time excludes model/index loading."]
    if report.get("baseline_metrics") and report["status"] == "passed":
        baseline = report["baseline_metrics"]
        lines.append(f"Vs V0: EM {m['em_percent'] - baseline['em_percent']:+.2f} pp | "
                     f"F1 {m['f1_percent'] - baseline['f1_percent']:+.2f} pp")
    if report.get("comparison_metrics") and report["status"] == "passed":
        reference = report["comparison_metrics"]
        lines.append(f"Vs {report.get('comparison_variant', 'V1')}: EM {m['em_percent'] - reference['em_percent']:+.2f} pp | "
                     f"F1 {m['f1_percent'] - reference['f1_percent']:+.2f} pp")
        lines.append("V3: adapted E5 + source-preserving state; frozen V2 policy." if report.get('variant') == 'V3'
                     else "V2 adds judge protocol; clean retrieval, same dev and resource budgets.")
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


def validate_baseline(manifest, report, dataset, planned):
    """Reject changing the questions, labels or action protocol between policies."""
    if manifest["system_prompt"] != SYSTEM_PROMPT or manifest["enable_thinking"] is not False:
        raise ValueError("Prompt/thinking mode differs from V0")
    if manifest["scoring"] != "canonical_answer_em_f1_no_aliases":
        raise ValueError("Scoring differs from V0")
    for name in ("dev.jsonl", "dev.labels.jsonl"):
        if dataset["files"][name] != manifest["dataset_manifest"]["files"][name]:
            raise ValueError(f"Development input differs from V0: {name}")
    if dataset["corpus_sha256"] != manifest["dataset_manifest"]["corpus_sha256"]:
        raise ValueError("Corpus differs from V0")
    if report["metrics"]["completed"] != planned or report["metrics"]["planned"] != planned:
        raise ValueError("Development task count differs from V0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--protocol", choices=("v1", "v2", "v3"), default="v1")
    parser.add_argument('--retriever-run', type=Path, help='Completed V3 E5 training run with index')
    parser.add_argument("--compare-run", type=Path, help="Optional completed V1 reference for V2")
    parser.add_argument("--limit", type=int, default=None, help="Optional partial dev run; omitted = all 200")
    parser.add_argument("--model-path", type=Path, help="Exported V1 policy; tokenizer stays identical to V0")
    parser.add_argument("--baseline-run", type=Path, help="Saved V0 run directory for matched V1 evaluation")
    args = parser.parse_args()
    if args.protocol in ("v2", "v3") and not args.model_path:
        parser.error("V2 evaluation requires an exported model")
    if (args.protocol == 'v3') != bool(args.retriever_run) or (args.protocol == 'v3' and not args.compare_run):
        parser.error('V3 requires retriever-run and V2 compare-run; other protocols cannot change retriever')
    if bool(args.model_path) != bool(args.baseline_run):
        parser.error("--model-path and --baseline-run must be supplied together")
    if args.baseline_run and (args.limit is not None or args.config):
        parser.error("Matched V1 evaluation uses the complete dev split and saved V0 config")
    config = config_from(args.config)
    baseline_manifest = None
    baseline_report = None
    if args.baseline_run:
        baseline_manifest = json.loads((args.baseline_run / "manifest.json").read_text())
        baseline_report = json.loads((args.baseline_run / "report.json").read_text())
        if baseline_report["status"] != "passed" or baseline_manifest.get("limit") is not None:
            raise ValueError("Baseline must be a completed full dev evaluation")
        config = baseline_manifest["config"]
    variant = args.protocol.upper() if args.protocol in ('v2', 'v3') else ("V1" if args.model_path else "V0")
    prompt_text, action_parser = SYSTEM_PROMPT, None
    history_builder = None
    if args.protocol in ('v2', 'v3'):
        from project.agent.correction import SYSTEM_PROMPT as prompt_text, parse_action as action_parser
    if args.protocol == 'v3':
        from project.agent.state import SYSTEM_PROMPT as prompt_text, state_messages as history_builder
    group = args.protocol + '_eval' if args.protocol in ('v2', 'v3') else ("v1_eval" if args.model_path else "v0")
    runs = resource_path("AGENTIC_RUNS_DIR") / group
    run_id = "dev_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:6]
    out = runs / run_id
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "variant": variant, "config": config, "metrics": aggregate([], 0)}
    if baseline_report:
        report.update(baseline_metrics=baseline_report["metrics"], baseline_run=str(args.baseline_run))
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
        if baseline_manifest:
            validate_baseline(baseline_manifest, baseline_report, dataset, planned)
        if args.compare_run:
            comparison = json.loads((args.compare_run / "report.json").read_text())
            comparison_manifest = json.loads((args.compare_run / "manifest.json").read_text())
            if comparison["status"] != "passed" or comparison_manifest.get("limit") is not None:
                raise ValueError("V1 comparison must be complete")
            if args.protocol == 'v3':
                from project.agent.correction import SYSTEM_PROMPT as v2_prompt
                if comparison_manifest.get('protocol') != 'v2' or comparison_manifest['system_prompt'] != v2_prompt:
                    raise ValueError('V3 comparison must use the saved V2 protocol')
                # All other frozen-data checks remain identical; only the declared prompt differs.
                validate_baseline({**comparison_manifest, 'system_prompt': SYSTEM_PROMPT}, comparison, dataset, planned)
                if comparison_manifest['policy_model'] != model_inventory(args.model_path):
                    raise ValueError('V3 must freeze the exact evaluated V2 policy')
                report['comparison_variant'] = 'V2'
            else:
                validate_baseline(comparison_manifest, comparison, dataset, planned)
            if comparison_manifest["config"] != config or comparison_manifest["index_manifest"] != baseline_manifest["index_manifest"]:
                raise ValueError("V1 comparison settings differ")
            report.update(comparison_metrics=comparison["metrics"], comparison_run=str(args.compare_run))
        labels = {row["id"]: row for row in load_jsonl(data_dir / "dev.labels.jsonl")}
        report["metrics"] = aggregate([], planned)
        report["full_dev_size"] = dataset["sizes"]["dev"]
        from project.retrieval.local import LocalRetriever
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        index_path, retriever_path = resource_path('AGENTIC_INDEX_DIR'), resource_path('AGENTIC_RETRIEVER_DIR')
        if args.retriever_run:
            adapted = json.loads((args.retriever_run / 'report.json').read_text())
            provenance = json.loads((args.retriever_run / 'manifest.json').read_text())
            retriever_path, index_path = args.retriever_run / 'model', args.retriever_run / 'index'
            if adapted['status'] != 'passed' or adapted['model_inventory'] != model_inventory(retriever_path):
                raise ValueError('V3 E5 training/export changed')
            if provenance['pairs_manifest']['source_manifest_sha256'] != sha256(data_dir / 'manifest.json'):
                raise ValueError('V3 E5 trained on a different data version')
            if provenance['pairs_manifest']['base_index'] != baseline_manifest['index_manifest']:
                raise ValueError('V3 E5 mining did not use the frozen baseline index')
        retriever = LocalRetriever(resource_path("AGENTIC_CORPUS_DIR"), index_path, retriever_path)
        if retriever.manifest["corpus_sha256"] != dataset["corpus_sha256"]:
            raise ValueError("Dataset and retrieval corpus versions differ")
        if baseline_manifest and args.protocol != 'v3' and retriever.manifest != baseline_manifest["index_manifest"]:
            raise ValueError("Retrieval index differs from V0")
        model_path = args.model_path or resource_path("AGENTIC_MODEL_DIR")
        tokenizer_path = resource_path("AGENTIC_MODEL_DIR")
        if baseline_manifest and model_inventory(tokenizer_path) != baseline_manifest["policy_model"]:
            raise ValueError("Original V0 model/tokenizer inventory changed")
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
        manifest = {"code": code_info(), "config": config, "question_split": "dev", "limit": args.limit,
                    "dataset_manifest": dataset, "index_manifest": retriever.manifest,
                    "policy_model": model_inventory(model_path), "system_prompt": prompt_text, "protocol": args.protocol,
                    "enable_thinking": False, "scoring": "canonical_answer_em_f1_no_aliases",
                    "tokenizer_path": str(tokenizer_path), "history_mode": 'source_preserving_state' if history_builder else "v0_rerender",
                    "retriever_training_run": str(args.retriever_run) if args.retriever_run else None,
                    "baseline_run": str(args.baseline_run) if args.baseline_run else None}
        save_json(out / "manifest.json", manifest)
        save_json(out / "resolved_config.json", {**config, "model_path": str(model_path), "dataset_path": str(data_dir)})
        save_json(out / "report.json", report)
        llm = LLM(model=str(model_path), tokenizer=str(tokenizer_path), dtype="bfloat16",
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
                result = run_episode(row["question"], generate, retriever.search, config,
                                     system_prompt=prompt_text, action_parser=action_parser, history_builder=history_builder)
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
                    print(f"{variant} dev: {number}/{planned}", flush=True)
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
        text = summary_text(report, results, f"runs/{group}/{run_id}")
        show_summary(out, text)
        (runs / "latest_summary.txt").write_text(text + "\n", encoding="utf-8")
        save_json(runs / "latest_run.json", {"run_id": run_id, "status": report["status"]})
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
