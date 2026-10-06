# V1：基础 GRPO

当前已经实现训练入口；尚未在远端验证这份 Agentic Search 训练适配。V0 截图基线为 EM=39.50%、F1=49.71%，并不代表 V1 已产生收益。

## 本轮只运行两步

同步本次代码后，在远端仓库根目录执行：

```bash
cd /home/h50061831/learn-verl-v0.5.0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash project/scripts/run.sh \
  bash project/scripts/mainline.sh v1 --mode smoke --gpus 8
```

沿用已有 Python 环境，不需要重新下载资源或重建语料/索引。默认按项目指南使用 8 张 L40S，先确认这些卡可供本任务使用；`--gpus` 是本次作业实际可见卡数，不是模型并行度。若实际只分配 4 张，CUDA_VISIBLE_DEVICES 和 --gpus 同时改为 4 张；2/4 卡配置入口存在，但显存尚未实测。

命令自动完成：校验冻结 train/dev 文件 → 转 Parquet → 校验本地 Qwen3 非思考模板 → 建立共享 CPU 检索器 → 采样并完成 2 步 GRPO → 保存 checkpoint → 输出短摘要。不会启动正式训练。每题 4 条轨迹，每步 16 题，共计划 128 条训练轨迹。

**回传终端最后的 `=== V1 | ... ===` 摘要截图即可。** 同样的内容保存在：

```text
runtime/runs/v1/latest_summary.txt
```

正常训练期间每分钟打印一条存活提示，完整日志不会刷满终端。失败时摘要包含最后几条异常；若摘要不足以定位，再截图该 run 的 `train.log` 末尾 traceback。实际 run 名在摘要第二行，也记录在 `runtime/runs/v1/latest_run.json`。

## 输出位置

```text
runtime/
├── data/processed/2wiki_v1/verl/
│   ├── train.parquet
│   ├── dev.parquet
│   └── manifest.json
└── runs/v1/
    ├── latest_summary.txt
    ├── latest_run.json
    └── smoke_<时间>_<短ID>/
        ├── config.yaml             # 完整已解析 veRL 配置
        ├── agent_loop.yaml
        ├── manifest.json           # 数据/模型来源、项目与核心适配文件 hash
        ├── train.log
        ├── summary.txt
        ├── report.json
        ├── reward_audit/rewards_<进程号>.jsonl
        └── checkpoints/
            ├── global_step_1/actor/
            └── global_step_2/actor/
```

checkpoint 是 veRL 分片训练状态，不是可直接交给 V0 的 Hugging Face 模型目录。reward_audit 保留模型各轮输出和 mask 计数，供远端诊断；不要求回传完整文件。dev Parquet 为 trainer 提供固定验证数据接口，本次关闭内置验证，test 不被读取。

## 本次验证什么

- **真实更新**：两个步骤都要出现有限、非零的梯度与参数抽样变化，并找到最终模型 checkpoint。参数探针每个本地参数张量最多取 32 个位置，观测到非零变化能证明存在参数更新；零变化不能证明整个模型完全没有更新，因此标记 NEEDS_REVIEW。
- **奖励可学习**：按题号汇总同题 4 条轨迹的终局 F1，至少有一组不同分。本指标来自训练采样，不从 V0 greedy EM 推断。
- **工具屏蔽**：工具内容及新插入的聊天模板 mask=0，只有实际采样 token mask=1；奖励只能落在最终模型 token 上。摘要中的 mask 计数是运行诊断，语义边界另有本地测试。
- **奖励**：严格协议可解析的终局答案使用 canonical token F1；动作错误、未完成、超预算或截断为 0。当前无别名扩展、检索奖励、重复查询惩罚、引用奖励。KL loss 系数 0.001 是策略正则项，不是答案奖励。
- **预算**：最多 4 次搜索、每次 top-3、每轮最多 512 新 token、实际上下文不超过 8192。veRL 的 response_length=8192 包含工具与模板 token，不能理解成每轮都生成 8192 个 token。因上游接口只返回 token IDs，恰好生成满 512 token 的边界情况保守归入截断。

看到 PASSED 只说明训练接线通过，不能声称答案质量提升。若全组同分或无参数变化，不自动重复试种子/继续长跑；根据这一次摘要定位原因。

## 启动通过后的主干

预置主训练命令如下；先回传本次两步结果，再安排运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash project/scripts/run.sh \
  bash project/scripts/mainline.sh v1 --mode main --gpus 8
```

固定 125 步，即 2000 题按每批 16 题训练一轮；从原始 Qwen3-4B 起步，不续接 smoke 权重。每 25 步保存，保留最近两个 checkpoint。下一阶段需要合并 checkpoint，再用冻结的 200 题 dev 评测；导出和 V1 评测入口尚未在本轮实现。正式比较前还需明确历史非思考模板 token 的保留方式：训练保留采样上下文，V0 原实现每轮重渲染对话，两者不能未经核查就声称输入逐 token 相同。

## 工程取舍与复盘

困难：V0 为推理重渲染对话，但 RL 要计算原采样动作的概率，重编码会改变历史 token。处理：AgentLoop 保留采样 ID，只编码后续观察/模板，并对这些新增 token 屏蔽损失。下一次可更早将推理、训练统一到 token-preserving 会话接口，减少两套上下文构造逻辑。

困难：上游 veRL 0.5.0 异步服务把剩余上下文全部当作生成预算，直接传每轮 max_tokens 会产生重复参数。处理：仅补充“请求预算与剩余上下文取小值”，无请求上限时保留原行为。

本轮只有三处小型核心扩展，其余代码在 project/training：

1. `verl/workers/rollout/vllm_rollout/vllm_async_server.py`：接受每轮生成上限。
2. `verl/trainer/ppo/reward.py`：可选 external_lib 注册自定义奖励管理器。
3. `verl/workers/actor/dp_actor.py`：可选参数变化探针，默认关闭。

**同步时必须包含这些核心文件，不能仅上传 project/。** 没有挪用旧参考仓库的整个 veRL、升级依赖或引入外部模型服务。

本地验证命令：

```bash
python3 -m unittest discover -s project/tests -v
```

当前 21 个 CPU 测试通过，覆盖 token 前缀保留、工具文本不能冒充奖励答案、严格格式/预算、失败摘要及已有数据/V0 测试。本机缺少 torch/transformers/Hydra，未执行真实模型加载、Hydra 组合或分布式更新；这些由远端两步检查确认。
