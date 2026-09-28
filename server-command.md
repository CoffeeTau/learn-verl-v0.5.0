# veRL v0.5.0：修复模型路径末尾斜杠并重跑

## 报错原因

报错为：

```text
AssertionError: Make sure the last char in src is not /
Got /home/h50061831/data/models/Qwen2.5-0.5B-Instruct/
```

veRL v0.5.0 在复制或解析模型路径时，不允许路径以 `/` 结尾。模型文件本身没有损坏。

错误路径：

```text
/home/h50061831/data/models/Qwen2.5-0.5B-Instruct/
```

正确路径：

```text
/home/h50061831/data/models/Qwen2.5-0.5B-Instruct
```

## 1. 修复并检查路径

```bash
cd /home/h50061831/learn-verl-v0.5.0

export MODEL_PATH="${MODEL_PATH%/}"
export STRICT_TRAIN_FILE=/home/h50061831/learn-verl-v0.5.0/data/gsm8k_strict/train.parquet
export STRICT_TEST_FILE=/home/h50061831/learn-verl-v0.5.0/data/gsm8k_strict/test.parquet

printf 'MODEL_PATH=<%s>\n' "$MODEL_PATH"
ls -lh "$MODEL_PATH/config.json" "$STRICT_TRAIN_FILE" "$STRICT_TEST_FILE"
```

输出的 `MODEL_PATH=<...>` 中，右尖括号前不能出现 `/`。

如果变量为空，直接设置真实路径：

```bash
export MODEL_PATH=/home/h50061831/data/models/Qwen2.5-0.5B-Instruct
```

## 2. 重新运行单卡 3 步 GRPO

```bash
ray stop --force

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

set -o pipefail

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$STRICT_TRAIN_FILE" \
  data.val_files="$STRICT_TEST_FILE" \
  data.train_batch_size=8 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=False \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr=3e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.max_model_len=1024 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.logger=console \
  trainer.project_name=verl_smoke \
  trainer.experiment_name=qwen25_05b_grpo_strict_3steps \
  trainer.rollout_data_dir=/home/h50061831/learn-verl-v0.5.0/rollouts_strict_3steps \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_training_steps=3 \
  2>&1 | tee verl_strict_3steps.log

TRAIN_EXIT_CODE=$?
echo "TRAIN_EXIT_CODE=$TRAIN_EXIT_CODE"
```

## 3. 完成后检查

```bash
grep -E \
  'step:|Training Progress|critic/rewards|critic/advantages|actor/grad_norm|AssertionError|Traceback|OutOfMemory|CUDA error' \
  verl_strict_3steps.log \
  | tail -n 50
```

预期完成 `step:1`、`step:2`、`step:3`，并输出：

```text
TRAIN_EXIT_CODE=0
```

统计每一步的奖励组：

```bash
python3 - <<'PY'
import glob
import json
import os

directory = "/home/h50061831/learn-verl-v0.5.0/rollouts_strict_3steps"
group_size = 4

for filename in sorted(glob.glob(f"{directory}/*.jsonl")):
    with open(filename, encoding="utf-8") as file:
        rows = [json.loads(line) for line in file]

    scores = [float(row["score"]) for row in rows]
    groups = [scores[index:index + group_size] for index in range(0, len(scores), group_size)]
    mixed_groups = [group for group in groups if min(group) < max(group)]

    print("=" * 80)
    print("file:", os.path.basename(filename))
    print("nonzero:", sum(score != 0 for score in scores))
    print("mixed_groups:", len(mixed_groups), "/", len(groups))
    print("groups:", groups)
PY
```
