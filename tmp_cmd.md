bash project/scripts/run.sh python3 - <<'PY'
from project.common import load_jsonl, resource_path

p = (resource_path("AGENTIC_RUNS_DIR") / "v2_eval"
     / "dev_20261007T024032Z_0bed4f" / "trajectories.jsonl")
rows = [r for r in load_jsonl(p)
        if r["id"] == "2f761f100bb011ebab90acde48001122"]
if not rows:
    raise SystemExit("Case not found")

r = rows[0]
print("=== GRANDFATHER CASE | V2 ===")
print(f"Status={r['status']} | EM={r['em']} | searches={r['searches']}")
print("Pred:", r["answer"])
print("Gold:", r["gold"])
for i, step in enumerate(r["steps"], 1):
    print(f"Turn {i}:", " ".join(step["output"].split())[:450])
    if step.get("hits"):
        print("Retrieved:", " | ".join(h["title"] for h in step["hits"]))
PY