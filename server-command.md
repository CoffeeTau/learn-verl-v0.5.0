本文记录在 NVIDIA 8 卡 L40S 服务器上启动当前 `verl 0.5.0` 仓库的最短流程。第一阶段只使用 1 张 GPU 完成 1 个 GRPO 训练 step，确认数据、Ray、FSDP、vLLM、FlashAttention 和训练更新链路全部正常；之后再扩展到 8 卡。

# 1. 当前环境与完成状态

当前已经确认：

- Python：`3.12.9`
- PyTorch：`2.6.0+cu124`
- PyTorch CUDA Runtime：`12.4`
- vLLM：`0.8.4`
- FlashAttention：`2.7.4.post1`
- PyTorch CXX11 ABI：`False`
- GPU：NVIDIA L40S，共 8 张

当前仓库的 `setup.py` 声明支持 `vllm>=0.7.3,<=0.8.5`，所以 vLLM 0.8.4 位于声明范围内。

执行进度：

- [x] 核对 Python、Torch、CUDA 和 ABI
- [x] 安装并成功导入 FlashAttention
- [ ] 安装当前 verl 仓库
- [ ] 检查 Python 包依赖冲突
- [ ] 准备 GSM8K 数据
- [ ] 完成单卡、单 step 冒烟测试
- [ ] 扩展到 8 卡训练

# 2. 每次登录服务器后的基础检查

进入项目根目录：

```bash
cd /home/h50061831/learn-verl-v0.5.0
```

检查 GPU：

```bash
nvidia-smi
```

检查当前命令使用的 Python 和 pip，避免把依赖安装进另一套环境：

```bash
which python3
python3 -V
python3 -m pip --version
```

检查核心运行环境：

```bash
python3 -c "import torch,vllm; print('torch:',torch.__version__); print('cuda:',torch.version.cuda); print('vllm:',vllm.__version__); print('gpu_count:',torch.cuda.device_count()); print('ABI:',torch._C._GLIBCXX_USE_CXX11_ABI)"
```

当前环境的预期关键输出为：

```text
torch: 2.6.0+cu124
cuda: 12.4
vllm: 0.8.4
gpu_count: 8
ABI: False
```

# 3. FlashAttention 安装记录

## 3.1 正确的 wheel

当前服务器必须使用同时匹配以下条件的 wheel：

```text
Python 3.12
Torch 2.6
CUDA 12
CXX11 ABI False
Linux x86_64
```

正确文件名是：

```text
flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

在网络较快的本地计算机下载：

```bash
curl -L \
  -o 'flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl' \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl'
```

从本地上传到服务器，替换命令中的用户名、服务器地址和目标目录：

```bash
scp 'flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl' \
  用户名@服务器地址:/服务器目标目录/
```

在服务器上安装时使用 `python3 -m pip`，并加入 `--no-deps`，避免 pip 顺带替换现有 Torch：

```bash
python3 -m pip install --no-deps \
  './flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl'
```

## 3.2 安装验证

```bash
python3 -c "import torch,flash_attn; print('torch:',torch.__version__); print('cuda:',torch.version.cuda); print('flash_attn:',flash_attn.__version__); print('ABI:',torch._C._GLIBCXX_USE_CXX11_ABI)"
```

已经得到的正确结果是：

```text
torch: 2.6.0+cu124
cuda: 12.4
flash_attn: 2.7.4.post1
ABI: False
```

进一步验证 CUDA 扩展入口可以加载：

```bash
python3 -c "from flash_attn import flash_attn_func; print('FlashAttention OK')"
```

## 3.3 已解决的 ABI 错误

此前误装过如下 wheel：

```text
flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

它与当前 `ABI: False` 的 PyTorch 不兼容，导入时出现包含以下内容的 `ImportError`：

```text
undefined symbol
__cxx11::basic_string
```

处理方式是卸载 TRUE 版本，再安装真正的 FALSE wheel：

```bash
python3 -m pip uninstall -y flash-attn
python3 -m pip install --no-deps \
  './flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl'
```

不能通过重命名 wheel 文件解决 ABI 不匹配，因为 TRUE/FALSE 对应的是不同的已编译二进制内容。

# 4. 安装当前 verl 仓库

确认仍在项目根目录：

```bash
pwd
```

预期路径：

```text
/home/h50061831/learn-verl-v0.5.0
```

以 editable 模式安装当前仓库：

```bash
python3 -m pip install -e .
```

这里安装的是当前仓库内的 verl 代码，不要再单独安装另一个不同版本的 verl。

检查依赖冲突：

```bash
python3 -m pip check
```

检查训练链路的核心包：

```bash
python3 -c "import verl,vllm,ray,transformers,tensordict,flash_attn; print('verl: OK'); print('vllm:',vllm.__version__); print('ray:',ray.__version__); print('transformers:',transformers.__version__); print('tensordict:',tensordict.__version__); print('flash_attn:',flash_attn.__version__)"
```

如果 `pip install -e .` 之后 Torch、vLLM 或 FlashAttention 的版本发生变化，先停止训练并重新核对版本，不要直接继续运行。

# 5. 准备 GSM8K 数据

当前仓库提供了 GSM8K 预处理脚本。它会下载数据集，并生成训练所需的 Parquet 文件：

```bash
python3 examples/data_preprocess/gsm8k.py \
  --local_dir "$HOME/data/gsm8k"
```

确认输出文件存在且大小不为 0：

```bash
ls -lh \
  "$HOME/data/gsm8k/train.parquet" \
  "$HOME/data/gsm8k/test.parquet"
```

这两个文件分别用于训练和验证。数据脚本还会把 GSM8K 标准答案转换成规则奖励函数需要的字段。

# 6. 单卡、单 step GRPO 冒烟测试

## 6.1 为什么先用单卡

第一轮的目标不是训练出有效模型，而是用最低排查成本验证完整链路：

```text
GSM8K Parquet
  → Ray 创建 worker
  → FSDP 加载 actor/reference model
  → vLLM 生成 4 个候选回答
  → GSM8K 规则奖励打分
  → GRPO 计算组内相对优势
  → actor 完成一次参数更新
```

0.5B 模型没有必要一开始就占用 8 张 L40S。单卡成功后，再验证 8 卡分布式启动。

## 6.2 启动命令

```bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$HOME/data/gsm8k/train.parquet" \
  data.val_files="$HOME/data/gsm8k/test.parquet" \
  data.train_batch_size=8 \
  data.max_prompt_length=512 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
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
  actor_rollout_ref.rollout.max_model_len=768 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.logger=console \
  trainer.project_name=verl_smoke \
  trainer.experiment_name=qwen25_05b_grpo_lora \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_training_steps=1 \
  2>&1 | tee verl_smoke.log
```

第一次执行可能需要下载 `Qwen/Qwen2.5-0.5B-Instruct`。模型下载时间不属于训练耗时。

## 6.3 成功标准

不能只以“模型加载完成”判断成功。日志中应当出现一个实际训练 step，以及类似以下指标：

```text
step:1
timing/gen
actor/pg_loss
critic/score/mean
response_length/mean
```

训练完成后检查关键日志：

```bash
grep -E "step:|ERROR|Traceback|CUDA out of memory" verl_smoke.log | tail -30
```

成功条件：

- Ray worker 正常启动；
- vLLM 完成 rollout；
- 奖励函数完成打分；
- actor 完成一次更新；
- 日志没有 `Traceback` 或 CUDA OOM；
- 达到 `trainer.total_training_steps=1` 后进程正常退出。

# 7. 扩展到 8 卡

单卡 smoke test 成功后，重新运行同一条训练命令，但进行以下替换：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

```text
data.train_batch_size=64
actor_rollout_ref.actor.ppo_mini_batch_size=256
trainer.n_gpus_per_node=8
trainer.total_training_steps=10
```

其余参数先保持不变，包括：

```text
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.n=4
```

此时每个训练 step 会生成：

```text
64 个 prompt × 每个 prompt 生成 4 个回答 = 256 条 rollout
```

因此设置 `ppo_mini_batch_size=256` 与本轮 rollout 数量对齐。

0.5B 模型的主要用途是验证系统，不适合衡量 8 张 L40S 的训练效率。完整链路稳定后，可以再切换到：

```text
Qwen/Qwen2.5-3B-Instruct
```

仓库中的 3B GRPO-LoRA 参考脚本为：

```text
examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh
```

# 8. 常用排查命令

## 8.1 保存基础环境信息

```bash
python3 -m pip check
nvidia-smi
python3 -c "import torch,vllm,transformers,flash_attn,tensordict,ray; print('torch:',torch.__version__); print('cuda:',torch.version.cuda); print('vllm:',vllm.__version__); print('transformers:',transformers.__version__); print('flash_attn:',flash_attn.__version__); print('tensordict:',tensordict.__version__); print('ray:',ray.__version__); print('ABI:',torch._C._GLIBCXX_USE_CXX11_ABI)"
```

## 8.2 检查日志末尾

```bash
tail -100 verl_smoke.log
```

## 8.3 检查 GPU 占用

```bash
nvidia-smi
```

## 8.4 判断是否安装到了错误的 Python

```bash
which python3
python3 -m pip --version
python3 -c "import torch; print(torch.__file__)"
python3 -c "import flash_attn; print(flash_attn.__file__)"
```

如果 `pip` 与 `python3 -m pip` 指向不同环境，后续统一使用 `python3 -m pip`。

# 9. 一页执行顺序

正常情况下，按照以下顺序操作：

```bash
cd /home/h50061831/learn-verl-v0.5.0

python3 -c "import torch,vllm,flash_attn; print(torch.__version__,torch.version.cuda,vllm.__version__,flash_attn.__version__,torch._C._GLIBCXX_USE_CXX11_ABI)"

python3 -m pip install -e .
python3 -m pip check

python3 examples/data_preprocess/gsm8k.py \
  --local_dir "$HOME/data/gsm8k"

ls -lh \
  "$HOME/data/gsm8k/train.parquet" \
  "$HOME/data/gsm8k/test.parquet"
```

完成这些步骤后，再执行第 6 节的单卡、单 step GRPO 命令。只有单卡日志满足成功标准后，才进入第 7 节的 8 卡验证。
