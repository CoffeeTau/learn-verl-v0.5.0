"""Compact, conservative summary of completed optimizer logs and reward audits."""
import json
import math
import re
from collections import Counter
from pathlib import Path
from project.common import load_jsonl, save_json, show_summary


def summarize(out, steps, exit_code):
    out = Path(out)
    log = (out / "train.log").read_text(errors="replace") if (out / "train.log").exists() else ""
    metrics = {}
    for line in log.splitlines():
        if "actor/grad_norm:" not in line:
            continue
        step = re.search(r"step:(\d+)", line)
        if step:
            metrics[int(step[1])] = {key: float(value) for key, value in
                                    re.findall(r"(actor/[a-zA-Z_]+):(?:np\.float(?:32|64)\()?([-+\deE.]+|nan|inf)", line)}
    metric_source = "console"
    if (out / "metrics.jsonl").exists():
        # Authoritative synchronous records override console output, which Ray
        # may not forward completely before shutdown. Never invent absent steps.
        metric_source = "metrics.jsonl"
        metrics = {int(row["step"]): row["metrics"] for row in load_jsonl(out / "metrics.jsonl")
                   if "actor/grad_norm" in row["metrics"]}
    batches = []
    for path in sorted((out / "reward_audit").glob("*.jsonl")):
        batches.extend(load_jsonl(path))
    records = [record for batch in batches for record in batch["records"]]
    statuses = dict(Counter(row["status"] for row in records))
    varied = sum(batch["varying_groups"] for batch in batches)
    groups = sum(batch["groups"] for batch in batches)
    masked = sum(row["tool_template_tokens"] for row in records)
    updates_ok = all(step in metrics and
                     math.isfinite(metrics[step].get("actor/grad_norm", float("nan"))) and
                     metrics[step].get("actor/grad_norm", 0) > 0 and
                     math.isfinite(metrics[step].get("actor/probe_parameter_delta_max", float("nan"))) and
                     metrics[step].get("actor/probe_parameter_delta_max", 0) > 0
                     for step in range(1, steps + 1))
    checkpoint = out / "checkpoints" / f"global_step_{steps}" / "actor"
    saved = checkpoint.is_dir() and bool(list(checkpoint.glob("model_world_size_*_rank_*.pt")))
    passed = exit_code == 0 and updates_ok and saved and varied > 0 and masked > 0
    status = "PASSED" if passed else ("FAILED" if exit_code else "NEEDS_REVIEW")
    report = {"status": status, "exit_code": exit_code, "steps": metrics, "expected_steps": steps,
              "metric_source": metric_source,
              "varying_groups": varied, "groups": groups, "statuses": statuses,
              "masked_tool_template_tokens": masked, "checkpoint_found": saved}
    save_json(out / "report.json", report)
    lines = [f"=== V1 | {status} ===", f"Run: {out.name}",
             f"Updates logged: {len(metrics)}/{steps} | finite nonzero grad + sampled parameter delta: {updates_ok}",
             f"Metric source: {metric_source}",
             f"Reward groups with variation: {varied}/{groups} | episodes={len(records)}",
             f"Masked tool/template tokens: {masked} | terminal reward on model tokens only",
             f"Statuses: {statuses}", f"Final checkpoint: {saved} | exit={exit_code}"]
    for step in sorted(metrics)[-2:]:
        row = metrics[step]
        lines.append(f"Step {step}: grad={row.get('actor/grad_norm', 0):.4g} delta={row.get('actor/probe_parameter_delta_max', 0):.4g}")
    lines.append("Scope: training integration; no claim of dev improvement.")
    if exit_code:
        errors = [line.strip() for line in log.splitlines() if re.search(r'(Error:|Exception:|OutOfMemory|AssertionError)', line)]
        lines.extend(line[:200] for line in errors[-3:])
    show_summary(out, "\n".join(lines))
    (out.parent / "latest_summary.txt").write_text("\n".join(lines) + "\n")
    save_json(out.parent / "latest_run.json", {"run_id": out.name, "path": str(out)})
    return passed
