# V4-base 证据引用奖励实验

先在服务器项目根目录运行两步冒烟（8 卡）：

```bash
bash project/scripts/run.sh python3 -m project.training.v4_base_evidence --mode smoke --gpus 8
```

检查 `PASSED`、`Evidence proxy` 中 `observations_saved=128`、`reward_records=128`，以及奖励差异组非零。开发集仍报告原始 EM/F1。若没有任何 `correct_with_support_bonus`，先检查标注与引用映射，不直接进行正式训练。

冒烟通过后正式训练（默认重新从原 V4-base main 的 step125 初始化，不从 smoke 继续）：

```bash
bash project/scripts/run.sh python3 -m project.training.v4_base_evidence --mode main --gpus 8
```

默认父模型：`v4_base/main_20261009T100336Z_3e49ab` step125。新结果目录：`runtime/runs/v4_base_evidence/`，与旧实验分开。125 updates，原 2000 训练题、4 samples、8 GPUs、原检索器/预算/干预/提示词。

正式训练结束后独立评测（替换 run ID）：

```bash
bash project/scripts/run.sh python3 -m project.evaluation.v4_base_evidence --train-run MAIN_RUN_ID
```

训练审计（同样替换 run ID）：

```bash
bash project/scripts/run.sh python3 -m project.scripts.audit_v4_training --stage v4_base_evidence --train-run MAIN_RUN_ID --examples 1
```

奖励：`0.8 * canonical_answer_F1 + 0.2 * EM * observed_citation_support_F1`。支持分按训练标签的支持段落计算，只承认实际 observation 中出现的引用；无效引用附加分为零；重复段落不重复计分，无关段落降低精度。标签仅在奖励端读取，不插入模型输入。验证与独立评测保持原始严格 EM/F1。全零组仍可能没有奖励差异；本实验不声称实现语义验证或解决全部纠错问题。

训练与验证 reward_audit 保存同一记录内的 model_outputs + observations，可还原模型实际看到的证据；不是重新检索。保留 score=原始 F1，training_reward=优化实际使用的奖励。

本地 41 项 CPU 测试通过，尚未在目标 GPU 服务器执行。
