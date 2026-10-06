# 下一步：固定数据 → E5 索引 → V0 多轮搜索

## 已确认与本轮目标

用户提供的远端摘要确认：Qwen3-4B 在 Torch 2.6.0 / Transformers 4.51.3 / vLLM 0.8.4 / 单卡 L40S 下能生成具体查询，E5 可编码 404 段资料，模型可以读取真实检索返回。基础 smoke 到此结束。

本轮完成数据准备与 V0 提示词 Agent；尚未进入 veRL 参数训练。V0 从现有 Qwen3-4B 出发，最多调用 search 4 次，允许根据中间实体继续查下一跳。固定单并发、非思考模式、8K 上下文，回答预算每轮最多 512 token。参考 Search-R1 的搜索/观察/回答协议，未安装参考仓库的旧版 veRL。

## 1. 数据为什么这样处理

- **划分**：官方 train 中选项目 train=2000、dev=200；官方 dev 中选冻结 test=300。整个官方 dev 的 ID 和规范化问题都从训练候选中排除；项目各 split 不共享 ID/同一规范化问题。此处不声称去除了所有语义改写或预训练污染。
- **题型**：固定种子 42，四类题交替抽取，优先接近平衡。让小规模训练能覆盖推理/组合/比较/桥接比较，不直接继承原始类别比例。这是项目子集，不是官方榜单设置。
- **筛选**：支持句可定位、至少两个支持标题、核心字段有效；过长单句无法完整放入片段时剔除候选并记录原因。不以关键词规则证明题目一定需要真实多跳推理。
- **切段与证据映射**：按完整句子切段，每片含标题和 E5 前缀不超过 384 个 E5 token；稳定 ID 对应完整文本与原句子编号。同名但内容不同的段落保留不同 ID。支持事实映射到 `passage_id + sentence_id`，不靠标题覆盖。
- **关系信息**：保留原 evidences 三元组供后续纠错与检索对构造；当前不伪称完成了自动关系语义验证或标准子问题分解。
- **共享语料**：选定 train/dev/test 的全部 context 去重，再加 3000 个固定抽样背景段落。所有题都查同一语料，工具不接收题号或 gold 标签，不按题筛选候选。
- **信息边界**：Agent 可读 question JSONL 和 corpus；答案/支持事实/关系放在 `.labels.jsonl`，只用于评分与后续训练侧监督。测试事实可以在检索库中，测试 QA 标签不进入 Agent。
- **冻结**：生成后保留 manifest 和内容 SHA256。再次 prepare/index 不覆盖已有正式版本，不必反复运行。

训练 Parquet、困难 episode 和检索正负例在 V1/V2/V3 对应阶段加入；本轮先保存可复用的任务、标签和语料。

## 2. 远端三个命令

先同步最新 `project/`，在原有训练环境运行。机器路径继续使用现有 `.env`，默认资源根为 `/home/h50061831/learn-verl-v0.5.0/runtime`。无需新增模型或数据下载。

### A. 数据准备（CPU）

```bash
cd /home/h50061831/learn-verl-v0.5.0
bash project/scripts/run.sh bash project/scripts/mainline.sh prepare
```

成功显示 `=== PREPARE | PASSED ===`，截图该摘要即可。失败则截图 FAILED 摘要，暂不运行后续命令。

摘要位置：`runtime/runs/prepare_v1/summary.txt`

数据产物：

```text
runtime/data/processed/2wiki_v1/
├── train.jsonl / dev.jsonl / test.jsonl       # 问题、ID、类型
├── train.labels.jsonl / dev.labels.jsonl / test.labels.jsonl
├── rejected.jsonl                           # 候选剔除原因
└── manifest.json                            # 划分、来源哈希、规模和限制
runtime/data/corpus/2wiki_v1/
├── corpus.jsonl                             # 文本及原句子编号，无 gold 标签
└── manifest.json
```

注意：读取 raw JSON 一次一个 split；标准 JSON 解析需要 CPU 内存，原始文件保持不变。

### B. 建立 E5 索引（一张空闲 GPU）

```bash
CUDA_VISIBLE_DEVICES=0 bash project/scripts/run.sh \
  bash project/scripts/mainline.sh index
```

GPU 0 换成实际空闲卡。此阶段只有 E5 使用 GPU，结束退出后再启动 V0，因此不与 Qwen3 争抢显存。小型固定语料使用归一化向量的精确内积搜索，无需 FAISS 新依赖。

成功显示 `=== INDEX | PASSED ===`，截图该摘要即可。

摘要位置：`runtime/runs/index_v1/summary.txt`

索引位置：`runtime/indexes/2wiki_v1/e5-base-v2/embeddings.npy` 与 `manifest.json`。

### C. V0 开发集评测（一张空闲 GPU）

```bash
CUDA_VISIBLE_DEVICES=0 bash project/scripts/run.sh \
  bash project/scripts/mainline.sh v0
```

默认评完项目 dev 的 200 题，每 10 题打印进度。每题最多 4 次搜索和一次最后回答机会；上下文不足、动作格式错误、生成截断等均记录终止原因，不从分数分母中删除。E5 查询在 CPU、Qwen3 在 GPU，复用已建立索引。

不需要再次跑资源 smoke，也不先做多组小规模对照。完成后截图 `=== V0 DEV | PASSED ===` 开始的摘要；PASSED 表示评测执行完毕，不表示质量达到目标。

摘要包含 EM/F1、平均搜索次数/token/题耗时、终止原因计数，以及一条失败/成功案例（若存在）。简短来源 ID 由模型生成，未知 ID 单独记录；答案评分与引用诊断分开，不用引用规则悄悄修改 EM/F1。

固定查看入口：

```bash
cat runtime/runs/v0/latest_summary.txt
```

若 AGENTIC_ROOT 使用其他目录，则把上面的 runtime 替换为该资源根。

每次 V0 独立保存到：

```text
runtime/runs/v0/dev_<UTC时间>_<短ID>/
├── summary.txt
├── report.json
├── trajectories.jsonl       # 每题完成立即落盘；保留失败和完整检索返回
├── manifest.json            # 代码哈希、模型文件清单、数据/索引版本、提示词
└── resolved_config.json
```

运行日志：`runtime/runs/mainline_logs/`。出错先截简短摘要，后续再根据具体问题截日志；不用传服务器上的 JSON 文件。

## 3. 指标解释与边界

- EM：规范化后的短答案完全匹配率；F1：答案 token 的重叠 F1。分母含本轮所有已执行题，包括预算耗尽和无效动作。
- 当前别名文件缺失，统一按 canonical answer 评分，摘要明确标注 canonical only；不要把这组数当官方含实体别名的成绩。所有版本须保持相同评分器，若后续补别名则统一重评分。
- 总 token 包含每轮重复读入的 prompt 和模型输出；不是最终一条上下文长度。8K 限制每次生成的上下文窗口，所有 search 受 4 次总上限约束。
- 题耗时包含该题生成和检索，排除模型/索引初始化；固定并发为 1。建索引时间不混入题耗时。
- 运行级故障导致中断时标记 FAILED，摘要明确 completed/planned，部分结果不能当完整基线。
- 测试集此次不运行。V0 先用开发集建立可解释基线，测试仅用于最终 V0–V3 对比。

## 4. 只保留必要检查

本地执行：

```bash
python3 -m unittest discover -s project/tests -p 'test_*.py' -v
bash -n project/scripts/mainline.sh
```

CPU 用合成小样本核验 ID/问题不跨 split、切段后支持句可定位、公共语料不带答案字段、最多 4 次搜索的边界、占位动作拒绝、答案评分及失败分母。真实完整数据准备、E5 GPU 建索引及 V0 GPU 评测尚待服务器执行，本地通过不代表远端性能或任务质量。

本轮收到摘要后：围绕少量失败轨迹确定格式、检索、补查或证据理解的主要问题，然后接入 veRL GRPO。只有阻塞流程的错误才修复，不追加消融。
