bash project/scripts/run.sh python3 - <<'PY'
import json, os
from pathlib import Path
from datetime import datetime

root = Path(os.environ["AGENTIC_RUNS_DIR"]) / "v1"
runs = sorted(root.glob("main_*"), key=lambda p: p.stat().st_mtime)
if not runs:
    raise SystemExit("未找到 main 训练目录")
run = runs[-1]
p = run / "metrics.jsonl"
print("Run:", run.name)
if not p.exists():
    raise SystemExit("尚无 metrics.jsonl；需要查看 train.log 确认进度")
rows = []
for line in p.read_text().splitlines():
    try:
        row = json.loads(line)
        if "actor/grad_norm" in row["metrics"]:
            rows.append(row)
    except json.JSONDecodeError:
        pass
if not rows:
    raise SystemExit("尚无已完成更新的指标")
last = rows[-1]
step = int(last["step"])
print(f"Completed: {step}/125")
print("Metrics updated:", datetime.fromtimestamp(p.stat().st_mtime))
times = [float(r["metrics"]["timing_s/step"]) for r in rows[-10:]
         if "timing_s/step" in r["metrics"]]
if times:
    avg = sum(times) / len(times)
    print(f"Recent mean: {avg:.1f} seconds/step")
    print(f"Estimated remaining: {max(0, 125-step)*avg/3600:.2f} hours")
print("Grad:", last["metrics"].get("actor/grad_norm"))
PY