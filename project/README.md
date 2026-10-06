# Agentic Search 主干实施计划

下载资源后执行：[数据与模型基本测试：命令、结果路径与回传要求](RESOURCE_SMOKE.md)。

基础测试通过后执行：[正式数据准备、建索引与 V0：三步命令及截图回传](MAINLINE.md)。当前新增代码采用 `project/configs/main.json` 保存实验配置，机器路径仍由 `.env` 管理。

V0 已有远端截图结果：开发集 200 题，EM=39.50%、F1=49.71%。下一步：[V1 GRPO 两步启动检查：命令、摘要与回传要求](TRAINING.md)。V1 的 125 步训练已有远端 PASSED 截图；下一步按 TRAINING.md 的“V1 主训练完成后的同集评测”导出权重并评测相同 200 题 dev。

更新：2026-10-06。依据 `project_resource/Agentic_Search项目指导.md` 与本地参考代码制定。

## 1. 当前决定

- 主模型：`Qwen/Qwen3-4B`，原始 BF16 权重，固定 revision；统一 `enable_thinking=False`。
- 主数据：2WikiMultiHopQA，优先官方 `data_ids_april7.zip`（修复句子切分一致性）。
- 初始检索器：`intfloat/e5-base-v2`，服务于由 2Wiki context 构建的共享库。与策略训练顺序使用 GPU，训练时优先 CPU 检索，按实际延迟决定资源安排。
- 本地开发代码；远端下载权重、处理完整数据、建索引和运行训练。机器配置写根目录 `.env`，由 `project/scripts/run.sh` 加载为环境变量；默认资源根目录为仓库下 `runtime/`。
- 第一目标：50 题以内完成“问题→搜索→证据→下一轮→答案→评分”，随后跑通少量 GRPO 更新。
- 正式实验只有 V0 提示词 Agent、V1 基础 GRPO、V2 纠错增强、V3 最终系统。V2 是最小交付，V3 后置。不增加消融、多种子、模型大小对比或多数据集实验。

## 2. 五个工作的具体取舍

| 工作 | 项目采用内容 | 实施时机与边界 |
|---|---|---|
| Search-R1 | 生成搜索动作、执行检索、拼接观察、继续生成；观察 token 不参与策略损失 | V0/V1。借鉴循环和工具协议，接入本仓库 AgentLoop；仅补充三处可选/兼容性核心扩展，见 TRAINING.md |
| ReSeek | 显式判断当前证据是否有用、指出缺口、补查；纠错训练 | V2。首先使用困难 episode 与终局 F1；必要时再引入可核验判断奖励 |
| Agentic-R | 让检索输入包含原问题与当前查询；从搜索任务构造正负例 | V3。冻结策略，只训练一次检索器并重建索引，省去强模型 sub-answer、逐候选 global utility 与交替训练 |
| RE-TRAC | 一次尝试后压缩证据、未解关系、失败查询和下一步方向，再继续探索 | V3。使用用户后续提供的 API；最多一次压缩续搜，全部计入总预算 |
| MiroFlow | 工具管理、执行循环、超时、有限重试、轨迹保存 | V0 起保留简单运行能力；最终整理文本演示。暂不引入分层多 Agent 或 heavy reasoning |

### ReSeek：可以采用，但不直接复制奖励公式

本地 `ReSeek/README.md` 的方法说明包含 JUDGE 动作与 instructive reward，所以它同时涉及动作设计和奖励反馈，不只是训练一个额外 reward model。

`reseek_regex.py` 有用答案字符串是否出现在返回中判断信息价值的逻辑；`reseek_rerank.py` 有答案与返回相似度评分，并依赖外部服务。这些是本地实现的具体选择，不能直接当作多跳证据效用的可靠标签。

例如“甲就读学校的创办者出生在哪里”：第一跳查出学校，非常有用，但返回未必包含最终城市。直接按最终答案重合度奖励，会错罚这类桥接证据。

V2 先采用判断动作与两类困难返回，继续用终局 F1。若实际出现持续乱判且妨碍任务完成，再用训练集的支持句和关系映射，为明确可核验的判断提供监督；未匹配 gold 的片段不能一律判为无用。报告称为“借鉴 ReSeek 的纠错增强”，不称完整复现。

### Agentic-R：简化成一次任务适配的检索器训练

本地 README 第 3.3 节明确先用 Qwen-72B-Instruct 生成 sub-answer，再算 local utility，另算 global utility，然后训练检索器。

项目采用以下替代流程：

1. 完成 V2 后，冻结策略，收集**训练 split** 的查询及返回。
2. 训练检索输入使用 `原问题 [SEP] 当前查询`。V0–V2 也提前采用同一输入构造，减少后续接口变化。
3. 对能明确对齐关系和支持句的查询，支持段落作正例；检索出的同实体、不同关系段落，经规则筛选及少量抽查后作困难负例。
4. 查询无法可靠对齐时跳过；成功轨迹里的所有文档不自动视为正例，未命中支持标签也不自动视为负例。
5. 做一次对比学习微调，重建索引，进入 V3；不再反复训练 Agent。

这里学习的是**支持证据监督下的任务检索能力**，没有测量原论文定义的因果/下游效用，不能宣称等价复现 Agentic-R。省去 sub-answer 的依据是已有人工标注和关系信息，而非凭空换一个廉价裁判。

### MiroFlow：负责把 Agent 运行起来

可以把 Qwen3 理解为决策模型，veRL 负责训练决策，MiroFlow 负责运行时的工具编排和执行。模型输出一个搜索请求后，需要代码执行请求、处理失败、把结果送回模型并记录全过程，这就是本项目需要的部分。

论文还包含 Agent Graph、可选 heavy reasoning 和鲁棒工作流；本地主要入口是 `src/core/orchestrator.py`、`src/tool/manager.py`、`src/logging/task_tracer.py`。无需为了引用第五篇论文完整安装另一个 Agent 框架。优先借鉴/提取简单机制，训练和评测共享同一动作协议、停止规则和预算。

RE-TRAC 则强调**跨尝试传递状态**：压缩一条探索轨迹，再基于它继续探索；只缩短一段上下文不等于完整实现。API 压缩器只能看到 Agent 已经获得的资料，不能读取答案或支持标签；记录 API 模型、提示词、输出和成本，不让它代替 Qwen3 完成答题。

## 3. 数据处理为什么做、做到哪里

| 步骤 | 处理逻辑与目的 | 产物 |
|---|---|---|
| 划分与去重 | 官方 train 中取项目 train/dev，官方 dev 中取项目 test；先按 ID、规范化问题去重，派生样本跟随原题，避免同题两边出现 | split 清单和保留/删除原因 |
| 建共享库 | context 的标题与句子去重，保留原始句子边界和稳定 passage ID；保留同名不同文本的冲突记录，避免只按标题覆盖 | corpus.jsonl、源记录到 passage 的映射 |
| 证据映射 | supporting_facts 的 title/sent_id 定位句子；evidences 三元组用于检查关系，无法明确落到文本的记录单独处理 | 私有证据/关系映射、缺失统计 |
| 筛选任务 | 保留有完整证据、关系链可靠、长度可处理的多跳题；先抽查 30–50 题，发现问题再修规则 | 首批 QA 和筛选报告 |
| 固定搜索环境 | 选定样本全部 context，加固定抽样背景段落，建一份共享库；搜索面对全库，不能按当前题悄悄过滤到 gold 文档 | 语料和索引版本、固定背景抽样种子 |
| V2 困难 episode | 相关但无用、首次缺关键证据两类；后续仍能通过正常检索获得证据，让模型练习识别缺口并补查 | 扰动规则、原题 ID、种子 |
| 轨迹整理 | 存查询、返回 ID、判断、终止原因、最终答案与成本；若需要 SFT，只收格式正确、答案正确且证据支持的轨迹 | rollout JSONL、接受/拒绝统计 |

起步规模：50 题以内调通；随后固定训练 2,000 题、开发 200 题、测试 300 题。按题型做简单分层抽样，冻结测试不参与选 checkpoint。数据质量或有效样本数量不足时调整并记录，不为了凑数放入标签错误的样本。

测试事实可以在共享语料中，但测试问题、答案、支持标签、参考分解不能混入工具返回。语料覆盖检查不等于检索已经能找到证据；这两个失败原因分别记录。结果属于项目共享语料设置，不是原数据集榜单成绩。

## 4. 与当前 veRL 的接入点

- 本仓库 `verl/experimental/agent_loop/agent_loop.py` 的 `AgentLoopOutput.response_mask` 已明确区分模型 token=1、工具 token=0。
- `tool_agent_loop.py` 可参考已有工具执行与掩码逻辑；若采用 Search-R1 标签协议，写项目自己的 AgentLoop 适配器，避免并存两套执行协议。
- 当前 `ToolAgentLoop` 的 `apply_chat_template` 调用没有显式传 `enable_thinking=False`；初始提示及后续轮次都需统一模式。数据处理参数不能假定会自动传到此处。
- Actor 在 `verl/workers/actor/dp_actor.py` 中消费 `response_mask`。只需核验一条真实搜索轨迹掩码和一次更新，不另起底层框架验证项目。
- Reward 采用自定义评分入口：仅提取模型最终答案，使用官方规则兼容的 EM/F1；格式无效得 0。不可误从工具返回中提取答案标签。
- 本地根提交核对值为 `bf1ea26`。源码入口存在不代表 Qwen3 + 多轮异步 rollout 已在服务器跑通。
- 复制参考实现时保留原许可证、版权头和来源路径；不安装参考仓库自带的旧版 veRL 来替换当前框架。

## 5. 远端现在准备什么

建议下载三个资源：Qwen3-4B、2Wiki、e5-base-v2。暂不下载全量 Wikipedia、五套论文训练数据、72B 模型或 MiroVerse。

以下命令均在**远程服务器**执行。默认资源放在仓库的 `runtime/`，该目录不进 Git；如果代码盘空间不足，只在 `.env` 中把 `AGENTIC_ROOT` 改为数据盘绝对路径。命令尚未在远端执行；下载链接来源已核对，但实际网络与文件内容需下载后确认。

```bash
# 在仓库根目录执行，仅首次复制，保留已有配置。
test -f .env || cp .env.example .env
bash project/scripts/run.sh --print-paths

# 打开已加载配置的准备终端，后续下载命令在这个终端中执行。
bash project/scripts/run.sh bash
mkdir -p "$AGENTIC_ROOT/models" "$AGENTIC_ROOT/data/raw/2wiki" "$AGENTIC_ROOT/runs"

# 在现有训练环境记录一次版本，不先升级整套环境。
python3 -c 'import sys; from importlib.metadata import version; print(sys.version); print({p: version(p) for p in ["torch", "vllm", "transformers", "huggingface-hub"]})'
nvidia-smi
```

下载模型使用已有 `huggingface_hub`；缺失时可在单独下载环境安装，不影响训练环境。下面每个模型先固定一次实际 revision，重跑会复用该 revision：

```bash
python3 - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

root = Path(os.environ["AGENTIC_ROOT"])
for repo in ["Qwen/Qwen3-4B", "intfloat/e5-base-v2"]:
    name = repo.rsplit("/", 1)[1]
    manifest = root / "models" / f"{name}.revision.json"
    if manifest.exists():
        meta = json.loads(manifest.read_text())
        assert meta["repo_id"] == repo
    else:
        meta = {"repo_id": repo, "revision": HfApi().model_info(repo).sha}
        manifest.write_text(json.dumps(meta, indent=2) + "\n")
    snapshot_download(repo_id=repo, revision=meta["revision"],
                      local_dir=str(root / "models" / name))
    print(meta)
PY
```

官方修复版数据下载及解压：

```bash
curl -fL --retry 3 \
  'https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1' \
  -o "$AGENTIC_ROOT/data/raw/2wiki/data_ids_april7.zip"
unzip -n "$AGENTIC_ROOT/data/raw/2wiki/data_ids_april7.zip" \
  -d "$AGENTIC_ROOT/data/raw/2wiki"
sha256sum "$AGENTIC_ROOT/data/raw/2wiki/data_ids_april7.zip" \
  > "$AGENTIC_ROOT/data/raw/2wiki/archive.sha256"
find "$AGENTIC_ROOT/data/raw/2wiki" -maxdepth 3 -type f
```

需要保留 train/dev 原文件以及存在的实体 ID/别名文件。官方 test 无公开答案，本项目冻结测试取官方 dev。若修复版压缩包缺少 `id_aliases.json`，从官方 README 的 `data_ids.zip` 获取别名，不用旧包覆盖修复版 train/dev。解析代码以实际解压目录和字段为准。

Qwen3 模型卡要求 Transformers ≥4.51.0，并给出 vLLM ≥0.8.5 的部署建议；本仓库的 vLLM 可选依赖上限为 0.8.5。现有 0.8.4 先跑一次短生成和工具往返，若确有兼容错误，再在独立环境试 0.8.5，不反复修补无关依赖。成功标准包含 tokenizer/模板、生成、工具续写及后续训练权重同步，而不只是 import 成功。

## 6. 后续推进与停止标准

| 阶段 | 只做的主干工作 | 进入下一阶段的证据 |
|---|---|---|
| 0 | 远端资源、数据解析、固定库、search、短生成 | 30–50 题抽查、证据可定位、工具往返可回放 |
| V0 | 原始 Qwen3 多轮搜索；固定最多 4 次 search、总上下文约 8K | 有开发集分数及代表失败；工具格式可用 |
| V1 | GRPO：建议 16 题/批、每题 4 条、8 卡 FSDP、BF16、微批次 1，按实测显存调整 | 少量更新确认 mask、奖励有差异、真实更新后扩大一次训练 |
| V2 | 两类困难 episode、判断/补查动作，继续训练 | 与 V1 同协议评测，保存成功与失败纠错案例 |
| V3 | 一次检索微调、一次压缩续搜、文本演示 | 四版本最终统一评测与报告 |

SFT 只在工具格式确实不稳定时补一次小规模示范训练。首轮调参只处理阻塞项：格式错误、奖励无区分、检索找不到、截断、显存不足。依据日志改对应环节，不展开网格搜索。训练失败或无收益也保留记录，不循环优化直到数字好看。

统一指标：答案 EM/F1、平均搜索次数、总 token、固定并发下端到端耗时；失败样本计入分母。自然/困难 episode 分组展示。V3 的 API 压缩、所有尝试与重试均记录成本，搜索预算共享；因使用外部压缩模型，其成绩属于系统效果。

## 7. 每阶段保留一条困难与复盘记录

只记录真实遇到的问题，采用以下六行模板即可：

```text
现象：哪类题、哪个环节出了什么问题？
证据：对应样本 ID、日志/轨迹路径、配置。
原因判断：为什么先检查这里？还有什么不确定？
采取行动：改了什么，为什么选择这个最小修复？
结果：是否解决，是否有代价；失败尝试也保留。
重做优化：若再做一次，哪个接口/数据规则/步骤应提前准备？
```

当前已识别、尚待远端验证的风险：Qwen3 模板模式未贯通；以最终答案重合度误判中间证据；多仓库依赖冲突。不要将这些预判写成已经发生并解决的实验经历。

## 8. 来源

- [项目指导](../project_resource/Agentic_Search项目指导.md)
- [Qwen3-4B 官方模型卡](https://huggingface.co/Qwen/Qwen3-4B)
- [2Wiki 官方数据与修复版本](https://github.com/Alab-NII/2wikimultihop)
- [ReSeek 本地奖励实现](../project_resource/ReSeek/verl/utils/reward_score/reseek_regex.py)
- [Agentic-R 本地训练流程](../project_resource/Agentic-R/README.md)
- [RE-TRAC 本地说明](../project_resource/InfoAgent/retrac/README.md)
- [MiroFlow 论文](https://arxiv.org/html/2602.22808v1)与[本地运行入口](../project_resource/MiroFlow/src/core/orchestrator.py)

## 9. 目录与配置约定

默认布局如下。标有“后续”的目录尚未实现或生成；解压包内的层级以实际文件为准，不手工改动原始数据。

```text
learn-verl-v0.5.0/
├── .env.example                  # 入 Git：机器配置模板
├── .env                          # 不入 Git：本机配置
├── verl/                         # 当前训练框架
├── project_resource/             # 五个参考项目和论文
├── project/                      # 本项目代码
│   ├── README.md
│   ├── scripts/run.sh            # 统一环境加载入口
│   ├── configs/                  # 后续：V0–V3 实验配置，入 Git
│   ├── data_pipeline/            # 后续：数据处理
│   ├── retrieval/                # 后续：建索引、检索服务、微调
│   ├── agent/                    # 后续：搜索循环和状态
│   ├── training/                 # 后续：veRL 适配、奖励和启动
│   └── evaluation/               # 后续：评分和案例整理
└── runtime/                      # 不入 Git：默认 AGENTIC_ROOT
    ├── models/
    │   ├── Qwen3-4B/             # config、tokenizer、权重分片等
    │   ├── Qwen3-4B.revision.json
    │   ├── e5-base-v2/
    │   └── e5-base-v2.revision.json
    ├── data/
    │   ├── raw/2wiki/            # 压缩包、SHA256、解压后 train/dev 等
    │   ├── processed/2wiki_v1/   # 后续：split、Parquet、私有证据标签
    │   └── corpus/2wiki_v1/      # 后续：corpus.jsonl、版本清单
    ├── indexes/2wiki_v1/
    │   ├── e5-base-v2/           # 后续：初始检索索引
    │   └── e5-task-v1/           # 后续：微调后新索引，保留旧索引
    └── runs/<run_id>/            # 后续：每次运行独立目录
        ├── resolved_config.yaml # 实际参数、绝对路径；过滤 API key
        ├── manifest.json        # commit、模型 revision、数据/索引版本
        ├── checkpoints/
        ├── trajectories/
        ├── metrics.json
        └── notes.md             # 困难、处理、结果、重做思路
```

配置分工：

- `.env`：机器差异（资源根目录、后续服务地址与密钥）。它是本地持久配置，但不会自动生效，必须通过启动入口加载。仅使用 Bash 赋值语法，不在此文件中写命令。
- `run.sh`：定位仓库、加载 `.env`、从一个根目录派生资源路径、切回仓库再启动命令。无论调用时位于哪个目录，默认路径都一致；不会创建资源目录或下载文件。
- `project/configs/*.yaml`：后续保存版本、批次、搜索预算、奖励、数据与索引选择等实验参数。配置必须入 Git；临时实验覆盖使用启动参数，保存最终解析值。
- `runs/<run_id>/`：后续保存真实运行配置与产物，不能仅靠 `.env` 回忆一次实验使用了什么。密钥不写入记录。

路径优先规则：`.env` 中的 `AGENTIC_ROOT` 赋值覆盖继承的同名环境变量；若 `.env` 未赋值则保留继承值；最终空值回落到 `<repo>/runtime`。其余上述路径统一派生，不在 `.env` 中重复维护。V3 的新模型/索引通过实验配置明确选择，不覆盖 V0–V2 资源。

日常入口示例：

```bash
bash project/scripts/run.sh --print-paths
bash project/scripts/run.sh python3 your_script.py
```

第二条中的 `your_script.py` 是调用方式示例，不是已实现的训练脚本。Python 进程通过 `os.environ` 读取路径；后续 Ray 训练入口将路径解析进传给 worker 的配置，不依赖 worker 隐式继承整个 `.env`。

本地同步远端时同步源码，保留远端 `.env` 与 `runtime/`。若使用 rsync，明确排除这两项；Git 忽略规则不会自动成为 rsync 排除规则。
