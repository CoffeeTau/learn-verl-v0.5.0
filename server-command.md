# veRL v0.5.0：单卡 3 步 GRPO 测试

本文只保留当前最新的服务器执行步骤。目标是在已经通过单卡单步冒烟测试的基础上，继续运行 3 个训练 step，并保存每一步的生成文本和奖励。

当前已确认：

- Python 3.12.9
- PyTorch 2.6.0+cu124
- vLLM 0.8.4
- FlashAttention 2.7.4.post1
- veRL v0.5.0
- 单卡、单 step GRPO 已完成
- 上一次命令正常退出，但所有 rollout 的奖励和 advantage 都为 0
- 上一次 `response_length/clip_ratio=0.5625`，因此本轮把最大回答长度从 256 提高到 512

## 1. 进入项目并确认路径变量

```bash
cd /home/h50061831/learn-verl-v0.5.0

echo "$TRAIN_FILE"
echo "$TEST_FILE"
echo "$MODEL_PATH"

ls -lh "$TRAIN_FILE" "$TEST_FILE" "$MODEL_PATH/config.json"
```

三个变量必须分别指向：

```text
TRAIN_FILE：GSM8K train.parquet
TEST_FILE：GSM8K test.parquet
MODEL_PATH：Qwen2.5-0.5B-Instruct 模型目录
```

如果变量为空，应先按照服务器上的实际路径重新设置，例如：

```bash
export TRAIN_FILE=/实际路径/gsm8k/train.parquet
export TEST_FILE=/实际路径/gsm8k/test.parquet
export MODEL_PATH=/实际路径/Qwen2.5-0.5B-Instruct
```

不要直接照抄“实际路径”三个字。

## 2. 运行单卡 3 步测试

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
  data.train_files="$TRAIN_FILE" \
  data.val_files="$TEST_FILE" \
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
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
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
  trainer.experiment_name=qwen25_05b_grpo_3steps \
  trainer.rollout_data_dir=/home/h50061831/learn-verl-v0.5.0/rollouts_3steps \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_training_steps=3 \
  2>&1 | tee verl_smoke_3steps.log

TRAIN_EXIT_CODE=$?
echo "TRAIN_EXIT_CODE=$TRAIN_EXIT_CODE"
```

`set -o pipefail` 用来保证最终退出码能反映训练进程是否失败，而不只是 `tee` 是否执行成功。

## 3. 检查训练是否完成

```bash
grep -E \
  'step:|Training Progress|Traceback|OutOfMemory|CUDA error' \
  verl_smoke_3steps.log \
  | tail -n 30
```

应当出现：

```text
step:1
step:2
step:3
Training Progress: 100%
TRAIN_EXIT_CODE=0
```

## 4. 检查保存的 rollout

```bash
ls -lh /home/h50061831/learn-verl-v0.5.0/rollouts_3steps
```

预期生成：

```text
1.jsonl
2.jsonl
3.jsonl
```

统计每一步的奖励：

```bash
python3 - <<'PY'
import glob
import json

directory = "/home/h50061831/learn-verl-v0.5.0/rollouts_3steps"

for filename in sorted(glob.glob(f"{directory}/*.jsonl")):
    with open(filename, encoding="utf-8") as file:
        rows = [json.loads(line) for line in file]
    scores = [row["score"] for row in rows]
    print(
        filename,
        "count=", len(scores),
        "min=", min(scores),
        "max=", max(scores),
        "nonzero=", sum(score != 0 for score in scores),
    )
PY
```

## 5. 本轮成功标准

- `TRAIN_EXIT_CODE=0`
- 日志中出现 `step:1`、`step:2`、`step:3`
- 生成 `1.jsonl`、`2.jsonl`、`3.jsonl`
- 至少一部分 `score` 非零
- `critic/advantages/min` 和 `critic/advantages/max` 不再同时为 0
- `response_length/clip_ratio` 明显低于上一轮的 0.5625

如果训练成功但奖励仍全部为 0，下一步检查 JSONL 中的 `output` 字段，确认模型是否按 GSM8K 奖励函数要求输出了 `#### 数字`。
