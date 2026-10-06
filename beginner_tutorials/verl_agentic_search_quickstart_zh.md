# 用 veRL 做 Agentic Search：初学者使用教程

> 适用环境：本仓库 veRL v0.5.0。目标是学会“怎么用和改哪里”，不是深入研究 veRL 内部实现。

## 1. 先说结论

可以。你已经跑通了单卡和 8 卡 GRPO 冒烟测试，说明下面这条基础训练链路可用：

```text
数据 → 模型生成 → 奖励计算 → GRPO advantage → 参数更新 → rollout 保存
```

把它改成 Agentic Search 后，链路只是多了“模型调用搜索工具并继续思考”：

```text
问题
  → 模型生成搜索请求
  → 搜索工具返回资料
  → 模型阅读资料并继续生成
  → 得到最终答案
  → 奖励函数评分
  → GRPO 更新模型
```

实际开发时，主要掌握下面 6 个部分就够了：

1. 数据集；
2. Agent/rollout 推理逻辑；
3. 搜索工具与外部环境；
4. 奖励函数；
5. 训练配置；
6. 评估、日志和轨迹检查。

前五项决定系统如何运行，第六项决定你能否判断实验真的有效。大多数项目不需要修改 Ray、FSDP、vLLM 权重同步或 `RayPPOTrainer` 的底层实现。

## 2. veRL 在项目中负责什么

可以把 veRL 当成“训练底座”，把你的 Agentic Search 代码当成“任务插件”。

| 模块 | veRL 帮你完成 | 你主要负责 |
|---|---|---|
| 分布式训练 | Ray、FSDP、显卡资源、参数更新 | GPU 数量和 batch 配置 |
| 模型推理 | vLLM/SGLang rollout、权重同步 | Agent 每一轮该做什么 |
| 多轮交互 | Agent Loop、token 轨迹整理 | 搜索、停止和回答策略 |
| 工具调用 | 工具注册、并发调用接口 | 搜索服务和工具实现 |
| 强化学习 | GRPO/PPO loss、advantage | 奖励定义和实验目标 |
| 数据加载 | Parquet、prompt、batch | 数据内容和字段 |

你首先应该把 veRL 当作黑盒使用。只有当研究问题本身是“修改 GRPO/PPO 算法”时，才需要进入训练器内部。

## 3. 一条 Agentic Search 样本如何流动

假设问题是：

```text
2024 年某项赛事的冠军是谁？请给出答案和证据。
```

一条轨迹可能是：

```text
用户问题
→ 模型：需要搜索赛事名称和 2024 冠军
→ search(query)
→ 工具：返回若干文档片段
→ 模型：检查资料是否足够
→ search(second_query)（可选）
→ 工具：返回补充资料
→ 模型：<answer>最终答案</answer>
→ 奖励函数：答案正确性 + 格式 + 搜索成本
```

训练时必须区分两类 token：

- 模型生成的 token：`response_mask=1`，参与策略梯度；
- 搜索工具返回的资料：`response_mask=0`，作为环境观察，不应被当成模型动作训练。

本仓库的 `ToolAgentLoop` 已经处理了这种掩码。初学阶段不要重新把整段对话 decode 后再 tokenize，否则可能改变原始 token，导致训练轨迹和模型实际采样不一致。

## 4. 你真正需要改的地方

### 4.1 数据集

参考：[examples/data_preprocess/gsm8k_tool_agent_loop.py](../examples/data_preprocess/gsm8k_tool_agent_loop.py)

每条数据至少需要表达：问题、使用哪个 Agent、标准答案以及奖励所需的附加信息。概念上类似：

```python
sample = {
    "data_source": "my_search_qa",
    "agent_name": "tool_agent",
    "prompt": [
        {"role": "system", "content": "需要时使用搜索工具，最后输出 <answer>...</answer>。"},
        {"role": "user", "content": question},
    ],
    "reward_model": {
        "style": "rule",
        "ground_truth": {"target": answers},
    },
    "extra_info": {
        "id": sample_id,
        "question": question,
    },
}
```

最重要的字段：

- `prompt`：输入消息；
- `agent_name`：选择 Agent Loop；不填时默认单轮生成；
- `reward_model.ground_truth`：奖励函数使用的标准答案；
- `data_source`：可用于选择不同奖励逻辑；
- `extra_info`：样本 ID、来源、证据等调试信息。

最终保存成 train/test Parquet。先用几十条样本跑通，再扩大数据规模。

### 4.2 Agent/推理逻辑

核心接口位于：[verl/experimental/agent_loop/agent_loop.py](../verl/experimental/agent_loop/agent_loop.py)

现成工具 Agent 位于：[verl/experimental/agent_loop/tool_agent_loop.py](../verl/experimental/agent_loop/tool_agent_loop.py)

如果你的流程只是：

```text
模型 → 调用搜索 → 读取结果 → 继续回答
```

优先直接使用现成的 `tool_agent`，通过 prompt、工具配置和最大轮数控制行为，不必新写 Agent Loop。

只有出现以下需求时，才自定义 `AgentLoopBase.run()`：

- 强制执行“规划 → 搜索 → 反思 → 再搜索”；
- 自己设计停止条件；
- 同时调用搜索、数据库、代码执行器等多种工具；
- 保存额外的轨迹状态；
- 做树搜索、多 Agent 或其他特殊推理流程。

自定义 Agent Loop 最终要返回：

```python
AgentLoopOutput(
    prompt_ids=...,
    response_ids=...,
    response_mask=...,  # 模型 token 为 1，工具结果为 0
    num_turns=...,
    metrics=...,
)
```

你不需要修改 vLLM 的生成实现，只需要通过 `server_manager.generate(...)` 请求下一段模型输出。

### 4.3 搜索工具

工具基类位于：[verl/tools/base_tool.py](../verl/tools/base_tool.py)

现成搜索实现位于：[verl/tools/search_tool.py](../verl/tools/search_tool.py)

一个工具主要有四个生命周期方法：

```python
class MySearchTool(BaseTool):
    async def create(self, instance_id=None, **kwargs):
        # 为一条轨迹创建状态
        ...

    async def execute(self, instance_id, parameters, **kwargs):
        # 调用搜索服务
        return result_text, 0.0, {"latency": latency}

    async def calc_reward(self, instance_id, **kwargs):
        # 可选：工具级奖励
        return 0.0

    async def release(self, instance_id, **kwargs):
        # 清理轨迹状态
        ...
```

工具再通过 YAML 注册。参考：[examples/sglang_multiturn/config/tool_config/search_tool_config.yaml](../examples/sglang_multiturn/config/tool_config/search_tool_config.yaml)

```yaml
tools:
  - class_name: my_project.tools.MySearchTool
    config:
      retrieval_service_url: http://127.0.0.1:8000/retrieve
      timeout: 30
      rate_limit: 32
      type: native
    tool_schema:
      type: function
      function:
        name: search
        description: Search documents for a query.
        parameters:
          type: object
          properties:
            query_list:
              type: array
              items:
                type: string
          required: [query_list]
```

搜索服务本身可以是 BM25、向量数据库、搜索引擎 API 或你自己的 HTTP `/retrieve` 服务。veRL 只关心工具输入和返回的文本，不负责建立检索索引。

### 4.4 奖励函数

Search-R1 风格的答案抽取和 Exact Match 示例位于：
[verl/utils/reward_score/search_r1_like_qa_em.py](../verl/utils/reward_score/search_r1_like_qa_em.py)

自定义奖励文件可以放在你的项目目录中：

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    predicted = extract_answer(solution_str)
    answer_reward = float(predicted in ground_truth["target"])

    # NaiveRewardManager 会把 Agent Loop 的轮数写入 extra_info。
    # 无工具时通常是 2 轮；每完成一轮“工具返回 + 再生成”大约增加 2 轮。
    num_turns = (extra_info or {}).get("num_turns", 2) or 2
    estimated_search_rounds = max(0, (num_turns - 2) // 2)
    cost_penalty = 0.02 * estimated_search_rounds

    return {
        "score": answer_reward - cost_penalty,
        "answer_reward": answer_reward,
        "estimated_search_rounds": estimated_search_rounds,
    }
```

这里用轮数近似搜索轮数，适合第一个冒烟版本。如果一次可能并行调用多个工具，想精确惩罚每次调用，就应在自定义 Agent Loop 或轨迹日志中显式记录调用次数，不能继续依赖这个近似值。

在配置中加载：

```yaml
custom_reward_function:
  path: /absolute/path/to/my_project/reward.py
  name: compute_score
```

第一版奖励建议保持简单：

```text
总奖励 = 最终答案正确奖励 + 格式奖励 - 搜索成本惩罚
```

不要一开始堆很多奖励项。否则训练结果变化后，很难判断究竟是哪一项起作用。

### 4.5 训练配置

Agentic rollout 与普通单轮 GRPO 的主要区别是：

```yaml
data:
  return_raw_chat: true

actor_rollout_ref:
  rollout:
    mode: async
    multi_turn:
      enable: true
      tool_config_path: /absolute/path/to/tools.yaml
      max_assistant_turns: 4
      max_user_turns: 4
      max_parallel_calls: 1
```

同时仍然需要你已经熟悉的 GRPO 配置：

```yaml
algorithm:
  adv_estimator: grpo

actor_rollout_ref:
  rollout:
    n: 4
```

`n=4` 表示每个问题采样 4 条轨迹，GRPO 在同一道题的候选轨迹之间计算相对 advantage。

注意：本版本 Agent Loop 标记为 alpha。你现在验证通过的是普通 vLLM GRPO 链路；切换到异步、多轮和工具调用后，应当把它视为一条新的链路，重新从单卡小样本冒烟测试开始，不能直接假定一定兼容。

### 4.6 更换模型

同类 Hugging Face CausalLM 通常只需更新 `actor_rollout_ref.model.path`，但不能认为所有模型都能随意替换。更换前要确认：

- Transformers 能识别模型架构和 tokenizer；
- vLLM/SGLang 支持该架构；
- chat template 与工具调用 parser 匹配；
- 显存、上下文长度和 batch 参数重新适配。

例如 Qwen3-4B 可以作为下一步基座，但[官方模型卡](https://huggingface.co/Qwen/Qwen3-4B)要求 `transformers>=4.51.0`；vLLM 0.8.4 已有 Qwen3 支持，而 [Qwen 部署文档](https://github.com/QwenLM/Qwen3/blob/main/docs/source/deployment/vllm.md)更推荐 0.8.5。当前 veRL 允许 `vllm<=0.8.5`，因此应先用现有环境做模型加载和纯推理测试，必要时再单独升级。优先使用原始 BF16 模型，确认可用后再考虑量化版本。

模型更换仍按“单独加载 → 纯推理 → 单卡 1-step → 8 卡短训练”重新验收，不能直接沿用 0.5B 模型的 batch 配置。

### 4.7 评估与轨迹检查

Agentic Search 不能只看最终平均 reward。至少记录：

| 指标 | 说明 |
|---|---|
| `answer_accuracy` | 最终答案是否正确 |
| `format_success` | 是否正确输出答案标签 |
| `tool_call_rate` | 有多少样本调用了搜索 |
| `search_success_rate` | 工具调用是否正常返回 |
| `avg_search_count` | 每条轨迹平均搜索次数 |
| `max_turn_hit_rate` | 是否经常达到最大轮数后被强制停止 |
| `mixed_groups` | GRPO 组内是否有不同奖励 |
| `response_length` | 是否出现越来越长的无效输出 |
| `tool_latency` | 搜索耗时是否拖慢训练 |

尤其要人工阅读 rollout：模型可能得到高分，却利用了数据泄漏、错误格式或奖励漏洞。

## 5. 推荐的项目目录

不要把实验代码直接塞进框架包 `verl/`。初学阶段建议在当前仓库根目录创建一个与 `verl/` 平级的 Python 包：

```text
learn-verl-v0.5.0/
├── verl/                         # 框架源码，尽量不改
└── agentic_search_project/       # 你的项目代码
    ├── __init__.py
    ├── preprocess.py             # 原始数据 → Parquet
    ├── agent_loop.py             # 特殊流程才需要
    ├── tools.py
    ├── reward.py
    ├── configs/
    ├── scripts/
    └── evaluation/
```

这样既能直接导入 veRL 的通用设施，也能通过 `agentic_search_project.tools.MySearchTool` 等类路径注册自己的实现。项目需要独立发布或同时维护多个 veRL 版本时，再拆成单独仓库并固定 veRL 依赖。

## 6. 最稳妥的开发顺序

不要一上来运行 8 卡长训练。按下面顺序推进：

1. **数据检查**：读取 3 条 Parquet，确认 prompt、答案和 `agent_name` 正确；
2. **工具检查**：不用模型，单独调用搜索工具，确认输入、结果和超时；
3. **奖励检查**：手写正确、错误、格式错误三种回答，确认分数符合预期；
4. **纯 rollout**：用 10～30 条问题观察模型能否生成工具调用并得到结果；
5. **单卡 1～3 step**：确认存在非零奖励和 mixed groups；
6. **8 卡短训练**：确认分布式链路和吞吐正常；
7. **正式实验**：一次只修改一个变量，并保留对照组。

进入下一步的最低条件是：当前步骤没有异常，而且你能解释保存下来的轨迹发生了什么。

## 7. 哪些代码通常不要改

第一阶段尽量不要修改：

- `verl/trainer/ppo/ray_trainer.py`；
- FSDP worker；
- Ray 资源调度；
- vLLM/SGLang 权重同步；
- GRPO loss 的底层张量计算。

这些是 veRL 提供的训练基础设施。修改它们会扩大排错范围，而且通常与 Agentic Search 的研究问题无关。数据、Agent Loop、工具、奖励和配置应优先写在 `agentic_search_project/` 中，通过注册接口接入。

只有当你的研究问题是下面这些方向时，才进入训练逻辑内部：

- 新的 advantage 估计方法；
- token 级或步骤级 credit assignment；
- off-policy 修正；
- 新的 KL 或 clipping 形式；
- 特殊的多轮 loss mask。

## 8. 适合从这里开展的研究题目

完成最小闭环后，可以从以下方向一次选择一个：

- **是否搜索**：让模型学习什么时候无需搜索；
- **查询生成**：提升 query 的准确性和多样性；
- **多跳搜索**：根据第一轮证据生成第二轮查询；
- **停止策略**：证据充分后尽早停止，减少成本；
- **证据质量**：奖励答案正确性之外，再检查引用是否支持结论；
- **成本约束**：在准确率与搜索次数、延迟之间权衡；
- **奖励设计**：比较结果奖励、过程奖励以及两者组合；
- **失败恢复**：搜索超时、空结果或冲突证据时如何继续。

## 9. 初学阶段需要记住的三句话

1. veRL 负责高效生成和训练，你负责定义 Agent 如何行动以及什么行为值得奖励。
2. 工具返回内容是环境观察，不是模型动作，因此训练 mask 必须正确。
3. 先证明“工具能用、轨迹正确、奖励有效”，再增加数据规模和 GPU 数量。
