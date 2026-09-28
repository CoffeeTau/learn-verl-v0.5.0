# veRL v0.5.0：修正 GSM8K 输出格式并做奖励预检

## 当前结论

三步训练共生成 96 条 rollout：

```text
包含 ####：18 条
严格匹配 #### 数字：2 条
非零奖励：0 条
```

模型多次算出了正确答案，例如 `72` 和 `10`，但常写成：

```text
#### Conclusion:
Final Answer: 10
\boxed{10}
#### Final Answer: **10**
```

veRL v0.5.0 的 GSM8K 默认奖励只接受回答最后 300 个字符中的：

```text
#### 数字
```

因此当前主要问题是格式遵循，而不是训练、CUDA、vLLM或响应长度。本轮先创建提示词更严格的数据集，再执行纯推理奖励预检；预检出现非零奖励后才继续训练。

## 1. 导出路径变量

进入项目目录，并将变量导出给 Python 子进程：

```bash
cd /home/h50061831/learn-verl-v0.5.0

export TRAIN_FILE=/home/h50061831/learn-verl-v0.5.0/data/gsm8k/train.parquet
export TEST_FILE=/home/h50061831/learn-verl-v0.5.0/data/gsm8k/test.parquet
export MODEL_PATH=/实际模型目录/Qwen2.5-0.5B-Instruct
```

`MODEL_PATH` 必须替换成服务器上的真实模型目录，不要照抄“实际模型目录”。

验证：

```bash
ls -lh "$TRAIN_FILE" "$TEST_FILE" "$MODEL_PATH/config.json"
```

## 2. 检查原始数据字段

```bash
python3 - <<'PY'
import os
import pandas as pd

frame = pd.read_parquet(os.environ["TRAIN_FILE"])

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
reward_model.ground_truth: 数字字符串
```

## 3. 创建格式要求更明确的数据集

不会覆盖原来的 Parquet，而是写入新目录：

```bash
python3 - <<'PY'
import os
from pathlib import Path
from datasets import load_dataset

train_file = os.environ["TRAIN_FILE"]
test_file = os.environ["TEST_FILE"]
output_dir = Path("/home/h50061831/learn-verl-v0.5.0/data/gsm8k_strict")
output_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset(
    "parquet",
    data_files={"train": train_file, "test": test_file},
)

format_instruction = (
    "\n\nIMPORTANT OUTPUT FORMAT:\n"
    "Your final line must contain only four hash signs, one space, and the numeric answer.\n"
    "Example final line: #### 42\n"
    "Do not use #### as a Markdown heading.\n"
    "Do not put words, labels, currency symbols, LaTeX, or bold markup after ####.\n"
    "Do not add a trailing period or any punctuation after the numeric answer.\n"
    "End your response immediately after that final line."
)

def strengthen_prompt(example):
    messages = [dict(message) for message in example["prompt"]]
    messages[-1]["content"] = messages[-1]["content"].rstrip() + format_instruction
    return {"prompt": messages}

for split in ("train", "test"):
    converted = dataset[split].map(strengthen_prompt)
    destination = output_dir / f"{split}.parquet"
    converted.to_parquet(destination)
    print(split, len(converted), destination)
PY
```

设置新数据路径：

```bash
export STRICT_TRAIN_FILE=/home/h50061831/learn-verl-v0.5.0/data/gsm8k_strict/train.parquet
export STRICT_TEST_FILE=/home/h50061831/learn-verl-v0.5.0/data/gsm8k_strict/test.parquet

ls -lh "$STRICT_TRAIN_FILE" "$STRICT_TEST_FILE"
```

## 4. 纯推理奖励预检

这个测试只加载模型并生成前8道题的4个候选答案，共32条，不做反向传播或参数更新。

```bash
ray stop --force

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

python3 - <<'PY'
import json
import os
import re
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from verl.utils.reward_score.gsm8k import compute_score

model_path = os.environ["MODEL_PATH"]
train_file = os.environ["STRICT_TRAIN_FILE"]
output_file = Path(
    "/home/h50061831/learn-verl-v0.5.0/gsm8k_strict_preflight.jsonl"
)

dataset = load_dataset(
    "parquet",
    data_files={"train": train_file},
    split="train",
).select(range(8))

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
)

prompts = [
    tokenizer.apply_chat_template(
        item["prompt"],
        tokenize=False,
        add_generation_prompt=True,
    )
    for item in dataset
]

llm = LLM(
    model=model_path,
    dtype="bfloat16",
    max_model_len=1024,
    gpu_memory_utilization=0.4,
    enforce_eager=True,
)

sampling_params = SamplingParams(
    n=4,
    temperature=1.0,
    top_p=1.0,
    max_tokens=512,
)

results = llm.generate(prompts, sampling_params)
strict_pattern = re.compile(r"#### (\-?[0-9\.\,]+)")
records = []

for item, result in zip(dataset, results, strict=True):
    ground_truth = str(item["reward_model"]["ground_truth"])
    for candidate in result.outputs:
        output = candidate.text
        score = compute_score(output, ground_truth)
        matches = strict_pattern.findall(output[-300:])
        records.append(
            {
                "ground_truth": ground_truth,
                "extracted": matches[-1] if matches else None,
                "score": score,
                "output": output,
            }
        )

with output_file.open("w", encoding="utf-8") as file:
    for record in records:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

print("total:", len(records))
print("strict_matches:", sum(record["extracted"] is not None for record in records))
print("nonzero_rewards:", sum(record["score"] != 0 for record in records))
print("output_file:", output_file)

for index, record in enumerate(records[:8], start=1):
    print("=" * 80)
    print("index:", index)
    print("ground_truth:", record["ground_truth"])
    print("extracted:", record["extracted"])
    print("score:", record["score"])
    print("output_tail:", record["output"][-500:])
PY
```

## 5. 判断标准

- `strict_matches` 明显大于原来的比例 `2/96`：强化提示有效。
- `nonzero_rewards > 0`：奖励闭环恢复，可以用新 Parquet 继续训练。
- `strict_matches > 0` 但 `nonzero_rewards = 0`：格式改善，但答案仍全错，需要检查模型能力或采样策略。
- `strict_matches = 0`：0.5B模型仍无法可靠遵循格式，应考虑少量SFT，暂时不要继续GRPO。

在奖励预检成功以前，不修改奖励函数，也不继续增加训练步数。
