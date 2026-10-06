"""Screenshot-sized inspection of existing V0 trajectories; no model/data mutation."""
import argparse
from collections import Counter
import json

from project.common import load_jsonl, resource_path


def summarize(rows):
    counts = Counter(row["status"] for row in rows)
    wrong = [row for row in rows if row["em"] == 0]
    all_gold_found = [row for row in wrong if row.get("support_chunks_total", 0) > 0 and
                      row.get("support_chunks_found", 0) == row["support_chunks_total"]]
    partial_gold = [row for row in wrong if 0 < row.get("support_chunks_found", 0) < row.get("support_chunks_total", 0)]
    no_gold = [row for row in wrong if row.get("support_chunks_total", 0) > 0 and row.get("support_chunks_found", 0) == 0]
    repeated = 0
    for row in rows:
        queries = [" ".join(step["query"].casefold().split()) for step in row["steps"] if "query" in step]
        repeated += len(queries) != len(set(queries))
    lines = ["=== V0 SAVED-TRACE REVIEW (no inference) ===",
             f"Tasks={len(rows)} | exact_correct={sum(row['em'] == 1 for row in rows)} | nonexact={len(wrong)}",
             f"Statuses: {dict(counts)}",
             f"Search-count distribution: {dict(sorted(Counter(row['searches'] for row in rows).items()))}",
             f"Exact repeated-query episodes: {repeated}",
             f"Nonexact: all gold chunks seen={len(all_gold_found)} | some={len(partial_gold)} | none={len(no_gold)}",
             "Gold coverage is a diagnostic; other valid evidence and alias mismatches may exist."]
    invalid = [row for row in rows if row["status"] == "invalid_action_format"]
    for index, row in enumerate(invalid[:5], 1):
        output = row["steps"][-1]["output"] if row["steps"] else "(no output)"
        compact = " ".join(output.split())
        if len(compact) > 240:
            compact = compact[:110] + " ... [middle truncated] ... " + compact[-100:]
        lines.append(f"Format {index}: {compact}")
    if len(invalid) > 5:
        lines.append(f"Showing first 5 of {len(invalid)} format failures.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Defaults to latest V0 run; directory name only")
    args = parser.parse_args()
    runs = resource_path("AGENTIC_RUNS_DIR") / "v0"
    run_id = args.run_id or json.loads((runs / "latest_run.json").read_text())["run_id"]
    if "/" in run_id or "\\" in run_id or run_id in (".", ".."):
        raise ValueError("--run-id must be a directory name, not a path")
    folder = runs / run_id
    rows = load_jsonl(folder / "trajectories.jsonl")
    summary = summarize(rows)
    (folder / "review_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
