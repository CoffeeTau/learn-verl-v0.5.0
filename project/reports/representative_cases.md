# Agentic Search 代表案例分析

更新：2026-10-07。仅记录已收到截图支持的案例，不补写截断输出或未见到的证据正文。下列全部来自开发集，不能作为独立测试泛化证据。主结果见 [阶段验收](mainline_acceptance.md)。

每个案例可用 ID 在对应 run 的 `trajectories.jsonl` 查找；公共字段包含问题、模型逐轮输出、检索 hits 与终局，gold 只用于离线评分。本文不把模型自述当作事实核验。

## 1. 祖父问题：路径更合理，身份和关系核对仍未解决

ID：`2f761f100bb011ebab90acde48001122`。
问题：Who is Iseabail Ní Mheic Cailein's paternal grandfather?
标准答案：Archibald Campbell, Master of Campbell。

| 版本 | 可见行为 | 终局 |
|---|---|---|
| V0 | 4 次搜索 | Colin Campbell，EM=0 |
| V1 | 3 次搜索，后续查询转向 Áed Findliath | Niall Caille，EM=0 |
| V2 | 先查父亲，再查 Colin Campbell, 1st Earl of Argyll 的父亲；2 次搜索 | Archibald Campbell, 4th Earl of Argyll，EM=0 |
| V3 修复后 | 查询依次包含 Iseabail 的父亲、Colin 的父亲、Archibald Campbell, Master of Campbell 的父亲；4 次搜索 | insufficient evidence，EM=0 |

V2 的首轮检索仍出现 Áed Findliath/Niall Caille，但未沿它们继续；它明确表达“还缺父亲的父亲”，说明本例的多跳拆解与中间查询更合理。第二轮出现另一个 Archibald 的标题后，模型判定 sufficient 并回答错误身份。姓名相同不能替代人物限定词和具体关系。

V3 查询里已经出现标准答案人物，却进一步查他的父亲，提示可能多走一跳或未确认关系。查询里出现答案字符串不等于已经获得支持关系的证据。最终拒答符合协议，不算答题修复，也不证明停止判断校准良好。

**可得结论：** 搜索路径改善和最终答对是两件事；本例仍是持续失败案例。未见完整证据正文，不能把原因确定为漏召回、关系方向误读或实体消歧中的某一项。

**复盘：** 状态应保留完整身份、关系方向及来源原文，不能将未经核实的模型判断压缩为确定事实。

## 2. 乐队国家问题：答案命中、协议合规和搜索成本需分开看

ID：`1f53f012087711ebbd67ac1f6bf848b6`。
问题：Are both bands, Transitshop and The Acid, from the same country?
标准答案：no。

| 版本 | 搜索 | 结果 |
|---|---:|---|
| V0 | 1 | no，EM=1 |
| V1 | 3 | no，EM=1；后两次查询相同 |
| V2 | 3 | judge 为 insufficient，理由为 The Acid 的来源国家不清楚，却输出 no；协议拒绝，记录空答案，EM=0 |
| V3 修复后 | 2 | 分别查 Transitshop origin、The Acid origin，正常输出 no，EM=1 |

V1 提醒我们：最终答对仍可能冗余搜索。V2 的失败不是“没有生成答案”，而是生成了与判断冲突的答案。V3 在本例恢复了合规正确回答，并比 V1/V2 少查一次，但仍未优于 V0 的搜索次数。

**可得结论：** 这是 V3 的局部成功，不代表 V3 整体优于 V2；不能将 V2 的原始 no 悄悄恢复为得分答案，也不能仅凭命中 gold 宣称推理有据。

## 3. Law Or Loyalty：多跳答对仍伴随重复查询

ID：`2a819d500bdc11eba7f7acde48001122`。
问题：Who is the child of the director of film Law Or Loyalty?
V2 标准答案与预测均为 John Derek，EM=1，搜索 4 次。

可见查询顺序：

1. Who directed Law Or Loyalty
2. Children of Lawson Harris
3. Children of Lawson Harris
4. John Derek's parent

**可得结论：** 模型能够围绕导演和子女生成连续查询并最终命中答案，但重复查询仍存在。最后一次反向查父母可能用于验证，也可能是冗余动作；没有完整返回正文不能确定。不是“发现错误后成功纠正”的充分证据。

**复盘：** 答案 F1 奖励不直接约束每次搜索是否新增信息。当前按预算和真实成本报告，不为单个成功案例增加新奖励项。

## 4. 国籍比较：判断标签与说明、动作不一致

ID：`ec9fe92f08e311ebbda4ac1f6bf848b6`。
来源：V3 正式 run 的离线复盘截图；原始问题与 gold 未在该截图完整展示，不补写。

搜索 2 次后，模型说明为 “Debashish Mohanty is Indian, Fahrudin Jusufi was Yugoslav”，却使用 insufficient 标签，并输出 no。状态为 judge_action_mismatch。

**可得结论：** 冲突在预算用尽前就出现，不能单靠“最后一轮提醒”解决。模型的文字说明似乎在做比较，但是否回答了实际问题、引用原文是否可靠，尚缺完整材料。不能称其为被评分器误杀的正确答案。

**复盘：** 输出判断标签不等于具备可靠判断能力。应分别考察标签、说明、动作和原始证据，不能让自述代替事实核对。

## 案例共同支持的结论

- V2 有整体答案收益和重复搜索下降，但祖父题仍错，判断—动作约束仍有失败。
- V3 有局部修复，也有更多退化；完整集的 16 修复/22 退化比单个成功例更能支持版本选择。
- 目前没有足够案例证据证明稳定的“发现错误路径后纠正”能力；可以陈述已经实现判断与补查机制，并展示其成功边界和失败方式。
- 本文先保留 4 个有具体材料的案例，后续仅从冻结版本的真实轨迹补充，不为了达到某个数量重复包装同一现象。

## 取证位置

## V4 待核对：先看错误行为，再决定是否调整推理入口

2026-10-08：暂缓推理路径对齐，固定 step100，只读取已有开发轨迹。

- **祖父题（V0–V3均错）：** V1偏离实体，V2形成两跳查询但混淆同名人物，V3出现正确人物名称后继续多跳并拒答。核对V4的父子关系原文、人物限定词和停止位置；不能仅凭查询中出现gold认定修复。
- **乐队题（V0/V1对、V2协议冲突、V3对）：** 核对V4是否保持正确答案，judge理由是否真的由两支乐队的证据支持，是否仍重复补查。
- **国籍比较（V3协议冲突）：** 核对V4的标签、说明和动作是否一致，以及来源是否支持实际问题。原始答案命中不恢复官方得分。
- **V4自然→困难：** 15道退化中14道answered、1道协议冲突；先从实际干预生效的answered退化题取一个完整案例，区分错误作答与拒答，再追踪干预后是否改写查询、是否取得新证据、是否忽略矛盾。统计本身不支持“主要是格式问题”。

只读入口 `project.scripts.cases_v4 --eval-run <run>`：输出跨版本短摘要，并在对应run生成 `badcase_evidence.md`（完整逐轮输出及hits正文）。固定3个历史案例，另按ID选1个干预退化及1个协议冲突样本；样本不是原因占比估计。历史轨迹缺失显式标注，不补造数据。本地仅验证合成轨迹读取与问题/gold对齐拒绝，真实V4案例尚待服务器取证。

| 版本 | 相对项目根目录的轨迹文件 |
|---|---|
| V0 | `runtime/runs/v0/dev_20261006T043535Z_7da4f2/trajectories.jsonl` |
| V1 | `runtime/runs/v1_eval/dev_20261006T151952Z_111fcb/trajectories.jsonl` |
| V2 | `runtime/runs/v2_eval/dev_20261007T024032Z_0bed4f/trajectories.jsonl` |
| V3 修复后 | `runtime/runs/v3_eval/dev_20261007T045730Z_081068/trajectories.jsonl` |

截图确认了本文列出的输出；本地未直接访问远端完整轨迹。案例的原因分析若要进一步定论，须读取对应 step 的 hits 原文，而不是仅凭标题或 judge 解释。
