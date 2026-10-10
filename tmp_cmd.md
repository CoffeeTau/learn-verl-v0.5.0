```
bash project/scripts/run.sh python3 - <<'PY'
import json
import textwrap
from pathlib import Path

root = Path("runtime/runs/v4_base_eval/dev_20261010T063219Z_9caa63")
lines = []
for group in ("natural", "hard"):
    rows = [json.loads(s) for s in (root / f"{group}.jsonl").read_text().splitlines()]
    matches = [r for r in rows if "some came running" in r["question"].lower()]
    if len(matches) != 1:
        raise SystemExit(f"{group}: 找到{len(matches)}题，请检查")
    r = matches[0]
    summary = f"{group}: EM={r['em']} searches={r['searches']} status={r['status']} pred={r['answer']}"
    print(summary)
    lines += [f"\n=== {group} ===", summary,
              f"Question: {r['question']}", f"Gold: {r['gold']}",
              "Perturbation: " + json.dumps(r.get("perturbation", {}), ensure_ascii=False)]
    for i, step in enumerate(r["steps"], 1):
        lines += [f"\n--- Turn {i} ---", step["output"]]
        for h in step.get("hits", []):
            lines += [f"\nEvidence [{h['id']}] {h['title']}", h["text"]]

out = Path("runtime/director_death_v4_base.txt")
out.write_text("\n".join(
    textwrap.fill(line, width=100, replace_whitespace=False)
    for block in lines for line in block.split("\n")
) + "\n")
print("完整轨迹已保存：", out)
PY
```

打开 `runtime/director_death_v4_base.txt`，发文件或截图即可。已加入自动换行，避免右侧证据被截掉。