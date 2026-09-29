# veRL v0.5.0：8 卡 L40S 分布式冒烟测试

## 当前结论

单卡严格格式数据的三步奖励结果：

```text
step 1: nonzero=3, mixed_groups=3/8
step 2: nonzero=2, mixed_groups=2/8
step 3: nonzero=4, mixed_groups=3/8
```

同一道题的4个候选中已经同时出现0分和1分，因此GRPO可以计算非零组相对优势。单卡的数据、vLLM rollout、奖励、GRPO和actor更新链路已经通过。

本轮只验证8张L40S上的Ray、FSDP、NCCL、vLLM worker和参数更新能否协同运行。使用0.5B模型运行2步，不用于评估最终模型质量或吞吐性能。

## 1. 确认路径和8张GPU

```bash
cd /home/h50061831/learn-verl-v0.5.0

export MODEL_PATH="${MODEL_PATH%/}"
export STRICT_TRAIN_FILE=/home/h50061831/learn-verl-v0.5.0/data/gsm8k_strict/train.parquet
export STRICT_TEST_FILE=/home/h50061831/learn-verl-v0.5.0/data/gsm8k_strict/test.parquet

printf 'MODEL_PATH=<%s>\n' "$MODEL_PATH"
ls -lh "$MODEL_PATH/config.json" "$STRICT_TRAIN_FILE" "$STRICT_TEST_FILE"
nvidia-smi -L
```

必须看到8张GPU，且 `MODEL_PATH` 不能以 `/` 结尾。

## 2. 清理旧进程和错误网络变量

```bash
ray stop --force

unset NCCL_SOCKET_IFNAME
unset GLOO_SOCKET_IFNAME
unset NCCL_SOCKET_FAMILY

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
```

## 3. 运行8卡、2步GRPO

```bash
set -o pipefail

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$STRICT_TRAIN_FILE" \
  data.val_files="$STRICT_TEST_FILE" \
  data.train_batch_size=64 \
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
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
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
  trainer.experiment_name=qwen25_05b_grpo_strict_8gpu \
  trainer.rollout_data_dir=/home/h50061831/learn-verl-v0.5.0/rollouts_strict_8gpu \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_training_steps=2 \
  2>&1 | tee verl_strict_8gpu.log

TRAIN_EXIT_CODE=$?
echo "TRAIN_EXIT_CODE=$TRAIN_EXIT_CODE"
```

这里必须保持：

```text
data.train_batch_size=64
actor_rollout_ref.actor.ppo_mini_batch_size=64
```

因为 veRL v0.5.0 要求全局训练 batch 不小于 PPO mini batch。

## 4. 检查8卡训练结果

```bash
grep -E \
  'step:|Training Progress|critic/rewards|critic/advantages|actor/grad_norm|NCCL|Traceback|AssertionError|OutOfMemory|CUDA error' \
  verl_strict_8gpu.log \
  | tail -n 80
```

成功标准：

- `TRAIN_EXIT_CODE=0`
- 出现 `step:1` 和 `step:2`
- `Training Progress: 100%`
- 没有NCCL、CUDA、OOM或RayTaskError
- `critic/advantages/min` 与 `max` 不同时为0
- `actor/grad_norm` 为有限数值

## 5. 统计8卡 rollout 奖励组

```bash
python3 - <<'PY'
import glob
import json
import os

directory = "/home/h50061831/learn-verl-v0.5.0/rollouts_strict_8gpu"
group_size = 4

for filename in sorted(glob.glob(f"{directory}/*.jsonl")):
    with open(filename, encoding="utf-8") as file:
        rows = [json.loads(line) for line in file]

    scores = [float(row["score"]) for row in rows]
    groups = [scores[index:index + group_size] for index in range(0, len(scores), group_size)]
    mixed_groups = [group for group in groups if min(group) < max(group)]

    print("=" * 80)
    print("file:", os.path.basename(filename))
    print("total:", len(scores))
    print("nonzero:", sum(score != 0 for score in scores))
    print("mixed_groups:", len(mixed_groups), "/", len(groups))
PY
```

预期每一步生成：

```text
64个prompt × 每题4个候选 = 256条rollout
```

只要至少存在一个 mixed group，就说明8卡训练中也有真实GRPO信号。
