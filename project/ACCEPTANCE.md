# 冻结测试集验收操作

当前代码已通过本地检查，尚未在服务器执行测试集。开发集已选定 V2 为交付候选；这次只验收，不训练、不通过测试成绩改参数。

## 固定范围

- V0–V3 使用各自正式 dev run 的权重、提示词和检索索引，配置均从保存的 manifest 读取。V3 使用修复后完整动作历史＋补充状态。
- 每版 300 道自然 test，再评测其中固定的 75 道困难任务，共 1500 条轨迹。困难任务和自然任务分别报告，不能混为 375 道独立题。
- 困难任务由 seed=42 的任务 ID 哈希选择：38 道 withhold_one、37 道 related_distractors，仅干预首次实际检索，后续全部正常。复用 V2 扰动函数，不增加搜索次数。
- test 的支持标签只在私有环境中定义扰动，答案只用于评分，均不进入 Agent 消息。自然组完全不按标签过滤返回。
- 各版本使用相同困难题 ID 和规则，但查询/检索器不同，实际删改的返回可能不同；汇总记录 applied 次数。未实际改变返回的任务仍保留在困难组分母，不能按结果重新筛选。
- 原始 test 文件和语料 hash、权重文件清单、索引 hash、运行代码 hash 在首次生成前核对/固定。权重清单使用项目原有文件元数据检查，不冒充全量权重内容哈希。

## 开始运行

同步代码，在远端项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 bash project/scripts/run.sh \
  bash project/scripts/mainline.sh acceptance
```

只需一张 L40S，自动按 V0→V1→V2→V3 顺序运行。每版模型加载一次，先自然组再困难组；版本间进程退出后释放资源。冻结清单保存在 `frozen.json`，包括 V2 候选决策与困难计划，之后不按分数更换版本。

终端显示各版本开始、每分钟运行提示及阶段汇总；运行提示不是完成了一题或一次训练更新。按已有 dev 每题约 2–3 秒，1500 条轨迹可粗略预留 1–2 小时，但困难组、加载和机器负载会改变时间；这不是实测承诺。

## 回传什么

最终回传 `=== FROZEN TEST | COMPLETE ===` 开始的表格截图，通常一屏即可。固定文件：

```text
runtime/runs/acceptance/latest_summary.txt
```

表中自然/困难分别列 EM、F1、搜索、token、秒数、协议失败及实际干预次数。COMPLETE 只表示四版本都执行完并保存完整结果，不表示模型全部答对，也不预设 V3 必须超过 V2。

```text
runtime/runs/acceptance/
  latest_run.json
  latest_summary.txt
  test_<时间>_<ID>/
    frozen.json
    summary.txt
    V0/  # V1、V2、V3 同结构
      run.log
      natural.jsonl
      hard.jsonl
      report.json
```

不覆盖此前 dev 文件。完整原文和逐题标签留在服务器，截图交换只需摘要。

## 中断或报错

不要再次启动一个新的验收批次。先回传报错摘要；需要继续时使用最初终端显示的 `test_<时间>_<ID>`：

```bash
CUDA_VISIBLE_DEVICES=0 bash project/scripts/run.sh \
  bash project/scripts/mainline.sh acceptance \
  --resume runtime/runs/acceptance/test_<时间>_<ID>
```

将最后的目录替换为实际路径。已完整完成的版本校验后跳过；未完成版本的目录自动保留为 `Vx_interrupted_<ID>`，再从该版本开始运行，不混合两次部分轨迹。恢复前拒绝运行代码、数据、模型或索引变化；如必须修复程序，应先记录具体变更与重跑范围，不手工改 hash 绕过检查。

## 验证与后续

本地执行：`python3 -m unittest discover -s project/tests -q`，39 项通过。新增检查覆盖固定困难子集、首次检索后恢复、私有扰动不进入检索请求、失败题保留分母。CLI、Python 编译、shell 语法检查通过。本机没有实际执行 GPU/vLLM 的冻结测试集验收。

拿到结果后更新阶段验收表和实验日志，保留全部失败。之后整理文本运行/轨迹回放交付；不依据 test 再调提示词、增加训练或筛选模型。
