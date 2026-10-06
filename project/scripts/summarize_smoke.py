"""Print a compact screenshot summary without loading models or running tests."""
import json
import os
from pathlib import Path
import textwrap


def placeholder_query(query):
    return query.strip().lower().strip(". ") in {"", "query", "search query", "your query", "..."}


def render(report):
    stage = report.get("stage", "unknown")
    lines = [f"=== {stage.upper()} | {report.get('status', 'unknown').upper()} ==="]
    packages = report.get("packages", {})
    lines.append("Env: " + " | ".join(f"{k}={packages.get(k, '?')}" for k in ("torch", "transformers", "vllm")))
    if stage == "data":
        for split, stats in report.get("splits", {}).items():
            lines.append(f"{split}: total={stats['total_rows']} | checked={stats['sampled_rows']} | valid={stats['valid_rows']}")
        lines.append(f"Corpus: {report.get('corpus_passages', '?')} passages | errors={len(report.get('errors', []))}")
        lines.append("Aliases: " + ("present" if report.get("alias_files") else "missing (not blocking smoke)"))
        lines.append("Scope: sampled structure/support offsets only; not full data validation.")
    else:
        gpu = report.get("gpu", {})
        e5 = report.get("e5", {})
        lines.append(f"GPU: {gpu.get('name', '?')} | visible={gpu.get('visible_count', '?')}")
        lines.append(f"E5: device={e5.get('device', '?')} | shape={e5.get('embedding_shape', '?')}")
        lines.append(f"Short generation: {report.get('short_generation', '(not reached)')}")
        trip = report.get("tool_roundtrip", {})
        for label, value in (("Question", trip.get("question")), ("Query", trip.get("query")),
                             ("Answer", trip.get("answer"))):
            if value:
                compact = " ".join(value.split())
                if len(compact) > 260:
                    compact = compact[:260] + " [truncated]"
                lines.extend(textwrap.wrap(f"{label}: {compact}", width=100))
        if "query" in trip:
            lines.append("Query check: " + ("INVALID PLACEHOLDER (legacy PASS is connectivity only)"
                         if placeholder_query(trip["query"]) else "no placeholder; relevance not scored"))
        if trip.get("hits"):
            titles = " | ".join(p["title"] for p in trip["hits"])
            lines.extend(textwrap.wrap("Top hits: " + titles[:230], width=100))
        lines.append("Scope: resource/protocol smoke; answer accuracy and RL not evaluated.")
    if report.get("traceback"):
        lines.append("Error: " + report["traceback"].strip().splitlines()[-1][:220])
    lines.append(f"Phase: {report.get('phase', '?')} | smoke elapsed: {report.get('elapsed_seconds', '?')}s")
    return "\n".join(lines)


def main():
    root = Path(os.environ["AGENTIC_RUNS_DIR"]) / "resource_smoke"
    for stage in ("data", "models"):
        path = root / f"{stage}_report.json"
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            summary = render(report)
            (root / f"{stage}_summary.txt").write_text(summary + "\n", encoding="utf-8")
            print(summary + "\n")
        else:
            print(f"{stage}: report not found\n")


if __name__ == "__main__":
    main()
