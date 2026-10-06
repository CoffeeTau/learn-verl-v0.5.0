# 下载完成后的两步测试

## 1. 路径一次配置清楚

假设服务器仓库在 `/home/h50061831/learn-verl-v0.5.0`，在仓库根目录的 `.env` 中设置：

```bash
AGENTIC_ROOT=/home/h50061831/learn-verl-v0.5.0/runtime
```

这是**资源目录**，不是代码目录，也不是某个模型目录。留空时 `run.sh` 会自动采用仓库下 `runtime/`，效果相同。脚本不会自动从别的目录寻找或下载缺失模型。

资源应位于：

```text
/home/h50061831/learn-verl-v0.5.0/
├── .env
├── project/
└── runtime/
    ├── models/
    │   ├── Qwen3-4B/       # config.json、tokenizer、权重分片等
    │   └── e5-base-v2/     # config.json、tokenizer、模型权重等
    └── data/raw/2wiki/
        ├── data_ids_april7.zip
        └── <压缩包原有目录>/
            ├── train.json
            ├── dev.json
            └── ...
```

测试会递归定位唯一的 train.json/dev.json，允许压缩包多一层目录。如果发现两套同名文件，会明确报错，不悄悄选一套。模型目录要直接包含 config.json，不要再嵌套一层同名模型目录。

在已经安装 Torch/vLLM/Transformers 的训练环境执行：

```bash
cd /home/h50061831/learn-verl-v0.5.0
test -f .env || cp .env.example .env
bash project/scripts/run.sh --print-paths
```

不用安装五个参考项目的依赖，也不用先升级当前训练环境。资源测试会强制离线读取本地权重。

## 2. 第一步：数据基本测试，无需 GPU

**运行命令：**

```bash
bash project/scripts/run.sh bash project/scripts/smoke.sh data
```

做什么：依次读取 train/dev JSON，每个随机抽查 50 题（固定种子 42），检查核心字段、支持句能否按标题/索引定位、关系三元组格式、抽样内重复 ID；生成训练题上下文的小型去重检索库。读取完整 JSON 后抽样，一次只保留一个 split，需要足够 CPU 内存。这不是全量清洗或正式训练/测试划分。

别名文件缺失只记录 warning，暂不阻塞基础测试；三元组格式检查不代表已经核验它的语义或完成关系到句子的映射。

**成功终端提示：** `DATA: PASSED`。失败时退出码非零，报告保存错误样本位置。

**请发回：**

- `runtime/runs/resource_smoke/data_report.json`
- 若失败，再发 `runtime/runs/resource_smoke/data.log`；不用发整个数据集。

同时生成 `sample_corpus.jsonl` 和 `sample_questions.json`，供下一步本地读取，正常情况下不用发回。语料仅含 ID/标题/文本，模型问题文件仅含 ID/问题，不携带答案或支持标签。

## 3. 第二步：E5 + Qwen3 基本测试，占一张空闲 L40S

先确认第一步通过，选择空闲 GPU；示例使用物理编号 0。如果 0 正在用，把下面的 0 换成空闲卡编号。

**运行命令：**

```bash
CUDA_VISIBLE_DEVICES=0 bash project/scripts/run.sh bash project/scripts/smoke.sh models
```

做什么：

1. 记录实际包版本、GPU 和模型路径/revision。
2. E5 在 CPU 编码小型共享库，检查向量有限且已归一化；沿用 Search-R1 的 masked mean pooling、query/passage 前缀和归一化逻辑。
3. 检查 Qwen3 的非思考模板，用 vLLM 单卡 BF16、8K 上下文、并发 1 做短生成。采用 eager 模式降低首次测试编译成本，显存占用比例起点为 0.65。
4. 给模型一道抽样训练题，要求生成 `<search>`；E5 从抽样题共享库真实检索 Top-3，把文本作为 `<information>` 送回模型，再生成 `<answer>`。格式失败会报错，不以硬编码搜索请求替代模型生成。

这次固定一次搜索，是资源/协议连通性测试。答案可以错误或说明证据不足；不据此判断多跳任务效果。它也不证明 GRPO 更新、masking、权重同步或多卡训练已通过。基础测试成功后才开发正式多轮流程。

**成功终端提示：** `MODELS: PASSED`。

**请发回：**

- `runtime/runs/resource_smoke/models_report.json`
- `runtime/runs/resource_smoke/model_trace.json`
- 若失败，再发 `runtime/runs/resource_smoke/models.log`。

报告包含失败阶段和 traceback；轨迹保存每轮输入输出，便于直接检查工具协议。进程若被系统强杀，JSON 可能仍是 running；此时发日志及终端最后几行，不能把旧的成功结果当作本次结果。

## 4. 文件位置与重跑约定

上述路径均相对于仓库根目录，默认绝对目录为：

```text
/home/h50061831/learn-verl-v0.5.0/runtime/runs/resource_smoke/
├── data_report.json
├── data.log
├── sample_corpus.jsonl
├── sample_questions.json
├── models_report.json
├── models.log
└── model_trace.json
```

这组 smoke 文件保存各阶段最近一次结果，重跑相同阶段会覆盖；发回后再按排错建议重跑即可。正式实验以后使用独立 run_id。若改了 AGENTIC_ROOT，报告就位于新根目录的 `runs/resource_smoke/`，启动命令会打印实际绝对路径。

如果没有生成报告（例如 Python 都未能启动），请发终端报错；能够生成日志时一并发对应 `.log`。不用只截取 PASSED/FAILED，JSON 对定位问题更有用。

## 5. 本地已验证的范围

开发机无本次远端权重/GPU，未运行真实 E5 或 Qwen3。已提供 CPU 自检，验证坏支持索引、同名冲突段落、重复数据包、动作格式，以及失败报告覆盖旧成功状态：

```bash
python3 -m unittest discover -s project/tests -p 'test_resource_smoke.py' -v
```

复用来源与改动见 [THIRD_PARTY.md](THIRD_PARTY.md)。不通过 sys.path 导入参考仓库，避免加载其旧版 veRL。
