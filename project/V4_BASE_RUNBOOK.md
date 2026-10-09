# V4-base：目标保持的第一轮可执行实验

2026-10-09。代码已实现，本地 CPU 检查；服务器 GPU 训练及真实数据上的收益待验证。

## 这轮具体做什么

从 `v4/main_20261007T095646Z_3f4cdb` 的选中 step100 初始化 actor 和 KL reference，固定 V3 E5/index。每次工具结果后，程序重复不可变的原问题；judge 在原有40词限制内说明 `target / known / missing`，要求已完成原关系链就停止、中间实体已知但属性缺失则继续。不要求最终实体另有独立页面。

这是**提示及终局奖励 GRPO 的可行性实验**，不是已完成的证据状态 SFT。进度字段仅为提示要求，不增加格式拒绝规则；模型写出的 known/sufficient 不是真值，不给额外过程分。没有自动制作弱标签 SFT，因为标题/答案词匹配不能可靠证明关系成立。若本轮没有改善，下一步仍是审核 train-only 状态→动作示例，不能把失败解释成多跑几轮必然有效。

训练/独立评测共享系统提示及 observation 构造，继续保留实际生成 token；原始证据不截断重写。新增提醒计入原8192上下文，可能压缩后续可用空间。保留4次搜索、512单轮生成、16题×4采样、首次检索干预、严格终局F1、学习率1e-6和KL系数0.001。不加第二次干预，不修改检索器，不放宽 judge/answer 冲突。

新输出隔离到 `runs/v4_base`、`runs/v4_base_eval`、`tensorboard/v4_base_*`。新 step0 是“旧step100权重＋新提示”，不是原V4训练的step0；新增125更新相对其编号。选模沿用V4的自然EM/F1优先、协议失败不增加、hard不低于本轮step0，可回退step0。选中step0不代表训练改善。

## 服务器操作

先同步本次代码；从仓库根目录执行。无需重新准备语料或训练检索器。

可先只读最新 aligned 轨迹（不运行推理）：

```bash
bash project/scripts/run.sh python3 -m project.scripts.cases_v4 \
  --stage aligned --eval-run dev_20261009T064354Z_7f3ed6
```

祖父题可按 ID 查看；生日题 ID 为 `199fbba20bdc11eba7f7acde48001122`。旧 rerender 案例不是当前仍失败的证明。

先跑2步冒烟（训练及验证都使用新提示）：

```bash
bash project/scripts/run.sh bash project/scripts/mainline.sh v4-base \
  --mode smoke --gpus 8
```

确认完整 PASSED、没有新格式/截断大面积失败后，再跑正式训练：

```bash
bash project/scripts/run.sh bash project/scripts/mainline.sh v4-base \
  --mode main --gpus 8
```

正式训练自动重新从原V4 step100初始化，**不从 smoke 续训**。125更新；step0及每25步验证；TensorBoard继续读取现有根目录。训练 PASSED 只验收链路，不承诺能力提升。

完成后将下面 `MAIN_RUN_ID` 替换为这次正式训练摘要中的 Run：

```bash
CUDA_VISIBLE_DEVICES=0 bash project/scripts/run.sh \
  bash project/scripts/mainline.sh v4-base-eval --train-run MAIN_RUN_ID
```

评测强制 token-continuation，读取新训练同一提示/预算/验证采样参数。不要指定旧step100为本轮选中步；本轮选中步由 selection.json 决定。

## 怎样判断是否继续

- 同环境 step0→选中checkpoint：判断继续训练是否改善；不能把新提示收益全部称为参数训练收益。
- 独立同集确认：与V4 aligned自然80%/F1 85.78%、matched82%、hard62%/F1 62.57%比较。自然不退化、困难改善、失败不增加，再检查 token/搜索/拒答代价；50题中1题=2pp，小差异不作稳定显著提升承诺。
- 复查完整轨迹：生日类是否在找回实体后补属性；祖父类是否在原关系链闭合时停止。以当前aligned的实际失败及整体配对为准，不只以两道熟悉题为验收。
- 如果新增格式错误、过度拒答或成本明显增加，保留原V4 step100，记录负结果，转向审核后的状态→动作监督。不能通过放松评分获得“修复”。

评测完成后，可把 `EVAL_RUN_ID` 换为新评测 Run，做只读配对及案例提取：

```bash
bash project/scripts/run.sh python3 -m project.scripts.review_v4 --base --eval-run EVAL_RUN_ID
bash project/scripts/run.sh python3 -m project.scripts.cases_v4 --stage base --eval-run EVAL_RUN_ID
```

旧冻结test不参与这轮调参。
