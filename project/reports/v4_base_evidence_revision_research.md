# V4-base：证据支持、旧判断修正与回查调研

2026-10-10。核查官方论文入口、官方仓库说明及下列实现；仅静态阅读，未运行这些外部项目。本文是下一步设计依据，未修改训练代码或启动训练。

> 方向修订：用户指出应优先研究近期agentic search的自主搜索学习机制，不能从几条badcase直接推出固定动作SFT。下文早期“优先SFT”建议暂缓；最新研究与判断见末节。SFT并不只训练格式，也不等同规则，但目前尚未证明本项目需要它。

## 本项目要解决的动作

结合用户提供的已保存轨迹：

- 生日hard：其他歌曲→Glenn Miller被错接到原歌曲；下一轮正确歌曲→Eddie Rabbitt已到达，但模型不修正known。正确人物生日尚未检索到。
- 导演hard：其他电影→Eric Byler错接到原电影；随后Éric Rohmer的日期又错接到Byler。正确电影/导演证据没有到达，且模型未回查第一跳。
- 两者需要核对“哪个主体的什么关系由哪段证据支持”，并使判断改变下一步动作。标题相似、人物同名、检索排名、日期字符串出现均不能替代关系支持。

## 已有工作的思考和实现

### 1. Self-RAG：相关性与支持性分开训练

来源：[项目与论文](https://selfrag.github.io/)、[官方仓库训练说明](https://github.com/AkariAsai/self-rag#training)。

其关键区分是检索需求、文档相关性、生成内容的证据支持程度及用途。训练链是GPT-4标注critic数据→训练critic→critic辅助构造生成器数据→训练生成器；不是只在提示里加“自我检查”。

[run_short_form.py](https://github.com/AkariAsai/self-rag/blob/main/retrieval_lm/run_short_form.py)的`call_model_rerank_w_scores_batch`读取反思token概率，分别计算relevance/groundness/utility，支持分数使用fully与partially supported的归一化概率，组合后选择候选。判断实际参与候选选择。

对本项目的借鉴：导演/日期这类文本主题相关，也可能完全不支持所声称的主体关系。应该监督“关系—证据”的支持性。局限：这些代码不是我们四次检索多跳任务的现成回退器；不能把Self-RAG结果当作本项目成功率。完整移植还涉及特殊token、训练数据和多候选生成成本。

### 2. RARR：先判断是否需要修正，再编辑旧说法

来源：[论文](https://aclanthology.org/2023.acl-long.910/)、[官方仓库](https://github.com/anthonywchen/RARR)。

RARR对已生成文本提出核验问题、搜索证据，再逐条判断和修改。公开实现主要是提示驱动的外部模型调用，不是现成4B强化学习训练配方。

- [utils/agreement_gate.py](https://github.com/anthonywchen/RARR/blob/main/utils/agreement_gate.py)：输入claim/query/evidence，判断依据两侧回答是否存在不一致，返回是否允许编辑。
- [utils/editor.py](https://github.com/anthonywchen/RARR/blob/main/utils/editor.py)：根据同组输入产生修改后的claim。
- [run_editor_sequential.py](https://github.com/anthonywchen/RARR/blob/main/run_editor_sequential.py)：遍历证据；门打开时调用编辑器，再更新claim，记录revision_steps。存在编辑距离条件，但默认max_edit_ratio=100，不能把默认配置描述为严格的小改动保障。

对生日例的借鉴：旧“原歌曲由Glenn Miller演唱”必须接受正确歌曲证据复核，更新人物后重新查生日。局限：原方法主要做输出后的文本修订，未替我们实现多跳依赖失效与重新规划。对于导演例，其他电影的文档只是“不支持原电影关系”，不必然构成明确反证，所以仅做矛盾检测还不够；应区分支持、不支持、冲突，不能把没发现冲突当作已证实。

### 3. CRAG：检索质量判断驱动后续检索动作

来源：[官方仓库](https://github.com/HuskyInSalt/CRAG)、[论文](https://arxiv.org/abs/2401.15884)。

框架按检索质量区分correct/incorrect/ambiguous，采用内部知识精炼、外部搜索或组合。实现发布了评估器训练及知识准备代码：

- [train_evaluator.py](https://github.com/HuskyInSalt/CRAG/blob/main/scripts/train_evaluator.py)：基于T5-large单输出头训练评分，二值标签映射到-1/+1。
- [internal_knowledge_preparation.py](https://github.com/HuskyInSalt/CRAG/blob/main/scripts/internal_knowledge_preparation.py)：段落分解后调用select_relevants筛选片段。
- [data_process.py](https://github.com/HuskyInSalt/CRAG/blob/main/scripts/data_process.py)：PopQA部分按文档标题是否等于主体百科标题构造标签；README说明训练标签收集方法与此类似。这是任务相关近似，不能直接当成多跳语义支持真值。

对导演例的借鉴：结果不支持原电影导演时，应改变查询路径；不能因为返回了某位导演就继续查其日期。本项目可以保留固定语料，借鉴“质量判断→改写/回查”的控制逻辑，无需引入外网搜索。原方法没有完整解决已形成错误known后的撤销问题。

### 4. RE-TRAC：旧状态允许被质疑，而非不可变事实

来源：[官方说明](https://github.com/microsoft/InfoAgent/tree/main/retrac)、[配置实现](https://github.com/microsoft/InfoAgent/blob/main/retrac/retrac/deep_research.yaml)。本地同路径代码已读。

`deep_research.yaml`要求摘要区分事实来源及验证状态、推断、缺口；continue_prompt明确提醒旧摘要可能包含未验证假设和过早结论，允许忽略不可靠部分并重新规划。

本地`graph.py`的end_cycle产生摘要，start_cycle重建原问题并带入最新摘要。它是跨轨迹重启/压缩，不是逐跳关系验证器。Verified字段仍由模型生成，不是程序证明。

对本项目最有用的是状态语义：原问题固定，known可修正。我们四次搜索短轨迹无需照搬长摘要和多轮深度研究；只需在错误关系被撤销时停止沿它推导，并保留仍有证据的事实。

### 5. ReSeek：Judge–Replan方向相符，但必须检查奖励近似

来源：[官方仓库](https://github.com/TencentBAC/ReSeek)、[论文](https://arxiv.org/abs/2510.00568)。

本地`project_resource/ReSeek/verl/utils/reward_score/reseek_regex.py`的`compute_content_overlap_reward`中，最终答案正确且judge=Yes会直接得到正奖励；否则调用`check_label_in_information`检查gold字符串是否包含于观察。仅对已核查的这一实现作判断，不将其概括为仓库所有奖励路径。

对当前案例：正确日期出现在错误人物段落中，仍可能通过字符串检查；第一跳正确但不包含最终日期，也不能因此视为无用。这种奖励不可直接作为我们的关系支持监督。Judge–Replan架构可借鉴，语义标签须另行保证可靠性。

## 收敛到本项目的一条主线

建议继续归在V4-base证据利用修复阶段：保留原问题提醒，学习“按证据修正关系状态→执行正确下一动作”。这是上述机制结合本项目的设计推论，不是某篇论文已完整实现的同名算法。

| 可见状态 | 期望判断与动作 |
| --- | --- |
| 正确中间人物已证实，最终属性未到达 | 保留人物，查该人物属性 |
| 只有相似作品/相似姓名 | 不把其他实体关系写进known；回查所需主体关系 |
| 旧known缺乏支持，新证据给出正确关系 | 修正关系，重新评估依赖它的后续步骤 |
| 新文档谈另一个人 | 不能把其属性套给旧人物；必要时回查第一跳 |
| 两跳证据足够且主体关系一致 | 停止搜索并答题，保留祖父题收益 |

实现层面优先研究经过审核的train-only局部状态→判断＋下一动作SFT，再决定接回原GRPO。输入保留原问题、实际可见证据和此前模型判断；监督短判断及search/answer动作。不需要先增加长期记忆、大JSON或第二次检索干预。

教师模型可帮助生成示范，但必须核查证据主体/关系/方向/属性；来源ID存在和原句匹配只能核查来源，不能证明语义支持。状态中撤销“原歌曲→Glenn Miller”后，Glenn Miller的生日可以仍是独立事实，但失去回答原问题的资格，不能只更改人物名却继承旧日期。

训练样本必须来自train，按题隔离诊断集；当前祖父、生日、导演开发例只用于分析与验收，不进入训练或构造其定制答案模板。保留足够完整证据即可作答的正例，避免教成一律不信证据、一律多搜或一律拒答。

## 怎样验证它是否有效

先在train内部小规模检查：错误绑定率、纠错证据到达后的关系修正与正确下一动作、已齐证据时的停止行为。若仅judge文字变好而后续仍搜错/答错，不认为机制成功。

再按冻结配置比较原V4-base step125与候选模型：同自然200/hard50开发题、同固定检索器与预算、同token续接、同严格评分。主看自然EM/F1是否守住、hard净修复数与成本；同时核查实际干预集合变化。hard50题每题2个百分点，不能把少数题净变化当作稳健统计结论。全部回归例按“正确证据未到达/已到达未使用/错误关系或属性错接/拒答或协议”分类。

这些开源工作证明相关机制已有可执行实现，并提供具体监督设计依据；没有证据能给出本项目4B配置下的成功概率。本轮最有根据的假设是增加关系支持与修正动作的监督会比继续只调终局奖励更直接，但仍须上述验证。

## 补充调研：直接面向Agentic Search的近期工作

2026-10-10。优先考察策略自主选查询、证据利用与搜索恢复；不将一般RAG文献作为主方案依据。不把论文、发布说明、静态源码和实际复现混为一谈。以下均未在本项目运行。

### CaRR / C-GRPO（2026-01）：最贴近错误证据链问题

[论文](https://arxiv.org/html/2601.06021v1)、[官方仓库](https://github.com/THUDM/CaRR)。面向deep search，将任务拆成可验证约束，检查隐藏实体是否识别、引用是否支持约束、受支持约束是否与预测答案连通。策略自行生成查询，奖励检查完成质量而非指定唯一搜索顺序。基于已有SFT模型再做RL；不能描述成从未使用SFT的实验。

已读[launch_server.py](https://github.com/THUDM/CaRR/blob/main/deepsearch_rm_with_rubrics/launch_server.py)的get_rubric_reward：模型提取实体、从history解析引用、模型判断填充后陈述是否被证据支持，再从E0做BFS筛选连通约束。BFS只查结构，语义可靠性仍依赖前面的LLM判断。

论文C-GRPO对答案正确轨迹增加rubric分量；错误轨迹保持零，因此不能单独解决全错组无对比信号。服务端get_reward的直接加权返回与训练时组级重算须区分，仓库README也提醒这点。论文对“所有轨迹都加rubric奖励”的消融较差，提示不能随意奖励似是而非的局部进度。其4B是Thinking版本、开放网页环境，与我们的非thinking固定语料不同。

本项目意义：生日日期或导演日期来自真实文档，仍可能与原问题断开。只查source ID有效、答案字符串出现、judge格式正确均不够。优先研究这种证据链奖励能否可靠地区分现有rollout，而不是先规定纠错措辞。

### MR-Search（2026-03）：通过RL学习如何利用上次尝试

[论文](https://arxiv.org/html/2603.11327v2)、[仓库](https://github.com/tengxiao1/MR-Search)。将多次尝试组成meta-episode，把反思作为后续尝试的上下文，使用按尝试/turn划分并向前传播未来收益的优势估计。策略通过结果学习调整搜索。

源码入口已确认：[generation.py](https://github.com/tengxiao1/MR-Search/blob/main/meta-search/llm_agent/generation.py)、[core_algos.py](https://github.com/tengxiao1/MR-Search/blob/main/verl/trainer/ppo/core_algos.py)。存在多种优势估计路径，尚未完成端到端配置映射；README所列根目录train_grpo_step.sh本次访问404，不能称为已验证可直接运行。

本项目意义：让纠错行为因后续成功而得到优化，而不是奖励一句“我发现错了”。限制是跨尝试与我们单条轨迹内四次搜索不同，直接引入会增加预算并改变实验；不能声称无需修改就解决当前第二轮证据未利用。

### EviBack（2026-07）：证据约束下给全错组补充学习信号

[论文](https://arxiv.org/html/2607.23955v1)、[作者发布说明](https://huggingface.co/chery-nextai/eviback/blob/main/README.md)。训练时仅对整组无EM命中的rollout调用Teacher；先不看gold判断原问题的可见证据是否充分，再在允许分支做gold-aware答案校准；后者不能推翻先前的证据不足判断。归一化后缩小fallback优势，推理只部署Actor和检索器。

这是训练奖励辅助，不是推理时Teacher指定下一查询，也不是固定动作SFT。适配Qwen3、E5、Search-R1环境，值得优先关注。作者发布说明有代码结构与复现边界，但本次未核验其具体训练源码，不能与CaRR奖励代码的核查深度等同。

我们的221/2000有奖励差异不等于其余全部是零奖励：须分清全错、全对、相同部分F1。只有确认全错组占比及证据充分率，才能判断该方法是否切中瓶颈。它本身也不保证教会自主撤销旧关系。

### ERL / ESearch（2025-10，ICLR 2026）：相关，但有强制回退边界

[论文方法](https://arxiv.org/html/2510.00861v1)、[ICLR记录](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9ffc16880822e689abcf6801e11f7f53-Abstract-Conference.html)。按错误类型擦除子答案、后续查询或初始计划再生成。v1方法用gold证据相似度和gold子答案F1构造过程分数，阈值触发擦除。

它明确研究错误中间状态传播，与案例相关；但这种训练时外部触发不能等同模型自主发现错误并回退。当前未确认作者代码入口，不列作已核验开源实现，更不建议照搬成推理规则。

## 修订后的下一步判断

撤回“下一步优先SFT”的默认建议。优先验证训练信号和探索覆盖两个问题：

1. 现有训练rollout里，能自主修正并答对的轨迹是否已经出现？若有但与猜对/错接链同分，优先研究CaRR式证据链质量区分；若几乎没有成功恢复，单改奖励也可能学不到，需要改善训练探索而非只延长训练。
2. 对现有训练组区分全零、全满分、同部分分、有差异；若全错组大量包含“已搜齐但不会用”，再评估EviBack式证据约束fallback。当前统计不足以证明。
3. 奖励应评价是否正确且有据，不奖励固定搜索次数、指定实体查询顺序或固定纠错措辞；推理时保持Actor自主决定搜索与停止。
4. 先用训练集保存轨迹审核评分器能否识别错主体、错关系、断链，同时不惩罚合法替代证据链。通过后选择一条最小RL改动，不将CaRR、MR-Search、EviBack全部拼接。

本轮只修订研究结论，不启动SFT、RL或新增推理规则。
