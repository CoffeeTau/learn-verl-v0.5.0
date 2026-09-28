# veRL v0.5.0：修正 mini batch 后重跑 3 步测试

本轮报错为：

```text
AssertionError
config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
```

原命令设置了：

```text
data.train_batch_size=8
actor_rollout_ref.actor.ppo_mini_batch_size=32
```

veRL v0.5.0 要求训练 batch 不小于 PPO mini batch。本轮修正为：

```text
data.train_batch_size=8
actor_rollout_ref.actor.ppo_mini_batch_size=8
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
```

其中 8 可以被 4 整除。

## 1. 确认路径变量

```bash
cd /home/h50061831/learn-verl-v0.5.0

echo "$TRAIN_FILE"
echo "$TEST_FILE"
echo "$MODEL_PATH"

ls -lh "$TRAIN_FILE" "$TEST_FILE" "$MODEL_PATH/config.json"
```

## 2. 重新运行单卡 3 步测试

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

## 3. 检查结果

```bash
grep -E \
  'step:|Training Progress|AssertionError|Traceback|OutOfMemory|CUDA error' \
  verl_smoke_3steps.log \
  | tail -n 40
```

预期看到：

```text
step:1
step:2
step:3
Training Progress: 100%
TRAIN_EXIT_CODE=0
```

检查 rollout 文件：

```bash
ls -lh /home/h50061831/learn-verl-v0.5.0/rollouts_3steps
```

统计奖励：

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

成功标准：命令退出码为 0、完成 3 个 step、生成三个 JSONL，并检查是否开始出现非零奖励。
