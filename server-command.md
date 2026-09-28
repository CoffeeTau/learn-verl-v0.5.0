# veRL v0.5.0：定位 GSM8K 奖励全部为 0

当前结果：

- 单卡训练完成 `3/3` step
- 每一步生成 32 条 rollout
- `1.jsonl`、`2.jsonl`、`3.jsonl` 均正常生成
- 三步共 96 条 rollout，但所有 `score` 都为 0

这说明训练、vLLM、LoRA、反向传播和数据落盘链路已经跑通。当前只排查奖励为 0 的原因，暂时不重新训练。

veRL v0.5.0 的 GSM8K 默认奖励使用严格格式，只能从回答最后 300 个字符中匹配：

```text
#### 数字
```

例如：

```text
#### 42
```

## 1. 统计输出是否包含严格答案格式

在服务器项目根目录执行：

```bash
cd /home/h50061831/learn-verl-v0.5.0

python3 - <<'PY'
import glob
import json
import os
import re

directory = "/home/h50061831/learn-verl-v0.5.0/rollouts_3steps"
strict_pattern = re.compile(r"#### (\-?[0-9\.\,]+)")

for filename in sorted(glob.glob(f"{directory}/*.jsonl")):
    with open(filename, encoding="utf-8") as file:
        rows = [json.loads(line) for line in file]

    strict_answers = []
    contains_marker = 0
    empty_outputs = 0

    for row in rows:
        output = row.get("output", "")
        if not output.strip():
            empty_outputs += 1
        if "####" in output:
            contains_marker += 1
        matches = strict_pattern.findall(output[-300:])
        if matches:
            strict_answers.append(matches[-1])

    print(
        os.path.basename(filename),
        "rows=", len(rows),
        "empty=", empty_outputs,
        "contains_####=", contains_marker,
        "strict_matches=", len(strict_answers),
        "examples=", strict_answers[:5],
    )
PY
```

结果解释：

- `contains_####=0`：模型完全没有遵守答案格式。
- `contains_####>0` 但 `strict_matches=0`：模型输出了标记，但不是严格的 `#### 数字`。
- `strict_matches>0` 但奖励仍为 0：格式正确，但答案错误，或者数据中的 ground truth 有问题。
- `empty>0`：存在空生成，需要另外检查 tokenizer 或停止符。

## 2. 打印前 8 条完整回答

```bash
python3 - <<'PY'
import json

filename = "/home/h50061831/learn-verl-v0.5.0/rollouts_3steps/1.jsonl"

with open(filename, encoding="utf-8") as file:
    rows = [json.loads(line) for line in file]

for index, row in enumerate(rows[:8], start=1):
    print("=" * 80)
    print("INDEX:", index)
    print("SCORE:", row.get("score"))
    print("INPUT:")
    print(row.get("input", ""))
    print("OUTPUT:")
    print(row.get("output", ""))
PY
```

重点观察：

- prompt 是否明确要求最终答案放在 `####` 后面；
- output 是否出现 `####`；
- `####` 后是否紧跟一个空格和数字；
- 回答是否在推理中途突然结束；
- 是否出现大量重复文字或乱码。

## 3. 验证训练数据的奖励字段

```bash
python3 - <<'PY'
import os
import pandas as pd

train_file = os.environ["TRAIN_FILE"]
frame = pd.read_parquet(train_file)

print("rows:", len(frame))
print("columns:", frame.columns.tolist())

for index in range(min(3, len(frame))):
    row = frame.iloc[index]
    print("=" * 80)
    print("data_source:", row.get("data_source"))
    print("prompt:", row.get("prompt"))
    print("reward_model:", row.get("reward_model"))
PY
```

正常数据应满足：

```text
data_source: openai/gsm8k
reward_model.style: rule
reward_model.ground_truth: 一个数字字符串
prompt: 包含 output the final answer after "####"
```

## 4. 下一步判断规则

执行以上三项后，按照证据选择下一步：

1. 没有 `####`：增强 prompt 的格式约束或先做少量 SFT。
2. 有 `####` 但格式不匹配：调整 prompt 或奖励提取格式。
3. 格式匹配但答案全错：模型能力或采样问题，检查具体题目和答案。
4. ground truth 异常：重新生成 GSM8K Parquet。
5. 输出在中途结束：再检查长度、EOS和生成配置。

暂时不要直接修改奖励函数，也不要继续增加训练步数；先确定属于哪一种情况。
