# Health-Copilot 项目面试追问

> [M0 已实现] 本文的已实现部分以当前基线 commit 为准；所有 M1 设计问题单独标为 [M1 PLANNED]。
>
> 事实基线：`ac1b2ac8e63c0e1476f992c159b40b5bf39e4d06`。本文只描述该 commit 中已存在的代码、测试和评测；M1 另有 Codex 会话实现时，本文件应在真实代码合并后重新核对。

本文不是按 BM25、RAG、Agent、Harness 分章的教材，而是一条可以在面试中逐层深入的对话路线。每个问题都尽量给出“可以直接说出口”的版本，再把回答拉回源码和证据。

## 先记住的 30 秒版本

Health-Copilot 是一个面向患者教育的 Safety-Gated Evidence RAG vertical slice。医疗场景的价值不在于把它包装成诊断系统，而在于用一个高风险领域验证三件事：高危问题能否在模型调用前被拦截，回答能否只建立在审核过的公开资料上，证据不足或来源异常时能否拒答。当前完成的是确定性的 M0：输入校验和 safety gate、带 provenance 的 JSON KnowledgeCard、中文优先的 BM25、注入式 Generator、以及 citation ID 完整性校验。M0.3 的固定评测在 80 条案例上得到 safety gate accuracy 1.0；62 条有人工来源标注的患者教育案例上，Hit@1 为 0.9032、Hit@3 为 0.9516、MRR 为 0.9274。它还不是 Agent，也没有临床验证；M1 的 Agent Core 仍是 [M1 PLANNED]。

## 面试前的事实纪律

- 可以说“当前实现了 M0 safety-gated evidence RAG”，不能说“已经实现 Agent / ReAct / Multi-Agent”。
- 可以说“检索指标”，不能把 Hit@1 说成医学准确率、诊断准确率或端到端 route accuracy。
- 可以说“citation integrity”，不能说已经完成 claim-level semantic grounding。
- 可以说“4 个 OOD case 得到了正的 lexical evidence，暴露了问题”，不能说系统已经有 OOD detector 或 evidence sufficiency gate。
- 参考文档中出现的 Mem0、Milvus、dense retrieval、reranker、vLLM、VLM、SFT/RL 和性能、人工盲评数字，均不是当前仓库的 M0 证据，不能移植为本项目经历。

## 第一轮：你先介绍一下这个项目

### Q1：这个项目解决什么问题？

#### 30 秒回答

我先解决的是“医疗健康问答怎样在一个小而可验证的范围内做到安全和可追溯”，而不是让模型自由发挥诊断或处方。系统把高危和用药决策问题在模型调用前路由出去；对可以回答的患者教育问题，只把审核过的公开资料检索成 Evidence 交给生成器；最后再校验生成器返回的 citation ID。证据为空、模型主动 abstain、服务异常或引用不在 Evidence 中时，系统 fail closed 到 ABSTAIN。

#### 深挖

这是一个垂直场景的工程基线：先把安全边界、数据 provenance、组件契约和拒答路径固定下来，再根据失败样本决定是否增加自主行为。系统明确不是诊断、处方、互联网诊疗或临床验证系统，也不接入真实患者数据。

#### 源码落点

- README.md 的 “M0 已实现”“数据边界”和 “M0.2 / M0.3 baseline”。
- `src/health_ai_copilot/contracts.py`：`Route`、`KnowledgeCard`、`Evidence`、`GenerationDraft`、`AssistantResponse`。
- `src/health_ai_copilot/pipeline.py`：`HealthCopilotPipeline.answer`。

#### 证据

- `data/knowledge_cards/` 当前有 30 张真实项目 KnowledgeCard，另有一个不参与加载的 `_schema.example.json`。
- `evals/m0.jsonl` 当前有 80 条不含个人信息的人工构造案例。

#### 面试官继续追问

- 为什么选择医疗场景，而不是普通聊天机器人？
- 你现在真正完成了什么？
- 这个系统如何证明自己没有把模型知识当成来源？

#### 当前边界

M0 没有诊断、处方、真实患者记录、临床验证、语义蕴含级 grounding、记忆或 Agent Loop。项目价值是可验证的安全与证据基线，不是临床能力声明。

### Q2：为什么选择医疗场景，而不是普通聊天机器人？

#### 30 秒回答

医疗场景能把“答错的代价”和“应该拒答的情况”说清楚，所以适合作为安全边界的压力测试。这里先把范围收窄到患者教育，避免把原型包装成诊断或用药决策系统；高危症状和处方意图也能用可审计的 deterministic policy 做回归测试。

#### 深挖

普通聊天机器人更容易把“回答得像”当成成功；医疗原型必须同时问：是否越过了权限边界，是否使用了可核验资料，证据不足时是否停止。这个场景迫使系统把 safety route、source metadata、abstention 单独做成契约，而不是只调 prompt。

#### 源码落点

- `src/health_ai_copilot/safety.py`：`URGENT_MARKERS`、`PRESCRIPTION_MARKERS`、`route_question`。
- `data/README.md`：公开来源、审核字段、版本和个人信息边界。
- `README.md`：项目不提供诊断、处方或治疗决策。

#### 证据

`tests/test_safety.py` 覆盖胸痛/呼吸困难的 `URGENT_CARE`、开药/剂量的 `HUMAN_REVIEW`，以及“非处方药”患者教育问题不应误判为处方请求。

#### 面试官继续追问

- 为什么要用 deterministic safety gate，而不是让模型判断高危？
- 医疗资料的审核字段为什么进入 KnowledgeCard？

#### 当前边界

marker policy 只是窄范围 prototype policy，不是完整临床风险分层，也没有声称覆盖所有危险表达、语言变体或真实运营流程。

### Q3：你现在真正完成了什么？哪些还没有完成？

#### 30 秒回答

完成的是 M0 vertical slice：KnowledgeCard schema/loader、中文 tokenizer、BM25、Generator Protocol、OpenAI-compatible live generator、deterministic citation-ID verifier、urgent/prescription pre-gate、无证据和异常生成的 abstention，以及离线 safety/retrieval eval。还没有完成 Agent Core、ReAct、ToolRegistry、Session/Memory、MCP、Sandbox、dense/hybrid retrieval、reranker、web search、VLM、训练或任何临床验证。

#### 深挖

“完成”按能否在当前 commit 找到实现来判断，不按 roadmap 的愿望清单判断。比如 OpenAI-compatible provider 已存在，但 live demo 依赖环境变量和外部服务；离线测试不需要 API Key。又比如 `abstain` 已存在，但没有独立的语义 evidence-sufficiency classifier。

#### 源码落点

- 已实现：`src/health_ai_copilot/knowledge/`、`retrieval/`、`generation/`、`verification/`、`pipeline.py`、`safety.py`、`eval/`。
- 计划而未实现：`docs/roadmap.md` 的 M1/M2/M3+。
- 公开 live demo 入口：`src/health_ai_copilot/cli.py`。

#### 证据

`docs/architecture.md` 的 M0 invariants 与 `README.md` 的“尚未实现”清单是边界依据；本基线没有 `AgentState`、Agent Loop 或 `ToolRegistry` 源码。

#### 面试官继续追问

- 为什么先做 workflow，再做 Agent？
- 你怎么证明 M0 没有偷偷越过边界？

#### 当前边界

下面所有带 [M1 PLANNED] 的答案都不能改写成“我们已经实现”。另一会话合并 M1 后，必须按新代码、测试、trace 和指标逐项更新，而不是只把标签替换掉。

### Q4：给我从用户输入开始完整走一次请求。

#### 30 秒回答

请求顺序是：`question` → 基本输入校验 → deterministic safety gate → 对可检索问题做 BM25 search → 得到 `Evidence[]` → Generator 只看本次 Evidence 并返回结构化 `GenerationDraft` → 校验 citation ID → 从存储的 Evidence 复制 Citation metadata → 返回 `AssistantResponse`。高危直接 `URGENT_CARE`，处方/剂量/停药/加药直接 `HUMAN_REVIEW`；无证据、generator abstain、检索或生成异常、缺少/伪造 citation 都 `ABSTAIN`。

#### 深挖

`HealthCopilotPipeline.answer` 先拒绝非字符串或空白输入，然后调用 `route_question`。只有 safety gate 放行后才调用 retriever；只有 evidence 非空才调用 generator；只有 `GenerationDraft` 不 abstain 且 citation verification 通过才返回 `ANSWER`。这个顺序是安全属性，不是普通的函数排列。

#### 源码落点

- `src/health_ai_copilot/pipeline.py`：`HealthCopilotPipeline.answer`、`abstain_response`。
- `src/health_ai_copilot/safety.py`：`route_question`。
- `src/health_ai_copilot/retrieval/bm25.py`：`BM25Retriever.search`。
- `src/health_ai_copilot/generation/base.py`：`Generator.generate` 契约。
- `src/health_ai_copilot/verification/citations.py`：`verify_citations`。

#### 证据

`tests/test_pipeline.py` 明确验证 urgent/prescription 不调用 retriever 和 generator；无 evidence 不调用 generator；正常路径的 title 和 URL 来自 stored Evidence，而不是 FakeGenerator 的输出。

#### 面试官继续追问

- 如果 BM25 返回了看似相关但实际不支持的问题，M0 会怎样？
- citation ID 为什么要由 runtime 绑定？

#### 当前边界

M0 只有“空 evidence”这一确定性拒答条件，不能独立证明检索到的证据足够回答问题。相关性不足、OOD、claim-level grounding 仍需要后续的评测和设计。

## 第二轮：为什么第一版不是 Agent？

### Q5：为什么最开始做 deterministic workflow，而不是直接 ReAct？

#### 30 秒回答

因为当前最需要验证的是安全和证据契约，不是增加模型的行动自由度。固定 workflow 可以把“高危不调用模型、模型只能看到 Evidence、引用必须过校验、失败就拒答”写成代码和测试；如果一开始上 ReAct，失败原因会混入规划、工具选择、循环和模型输出，反而难以定位。

#### 深挖

这是可归因性的取舍：先建立 deterministic baseline，再从真实 failure table 选择一个最小 action。M0 的 generator 只负责基于给定证据生成 draft，pipeline 负责顺序、边界和 veto。ReAct 不是“更高级所以默认正确”，它会引入状态、工具权限、预算和终止条件。

#### 源码落点

- `src/health_ai_copilot/pipeline.py`：固定编排，没有模型决定下一步的 loop。
- `docs/roadmap.md`：M1 计划受限 `AgentState`、`AgentLoop`、`ToolRegistry`，并要求由失败模式驱动。

#### 证据

当前仓库搜索不到 `AgentState`、`AgentLoop`、`ToolRegistry` 的 M0 runtime 实现；`README.md` 明确把 Agent Loop / ReAct 列在尚未实现清单中。

#### 面试官继续追问

- Workflow 和 Agent 的边界到底在哪里？
- 哪一个真实 failure 值得驱动下一次 action？

#### 当前边界

这里不是说 ReAct 一定不适合，而是说在本项目的 M0 基线上还没有足够证据证明它值得承担新增复杂度。

### Q6：现在这个系统为什么严格来说还不是 Agent？

#### 30 秒回答

因为下一步动作不是由模型在运行时选择的。M0 的顺序和分支都写在 `HealthCopilotPipeline.answer` 中，只有一次检索和一次生成，没有 Agent State、tool schema/registry、action observation loop、预算或终止策略。它是一个带模型生成步骤的 deterministic workflow。

#### 深挖

“用了 LLM”不等于“是 Agent”。区分点是系统是否允许模型在受约束的状态空间内选择动作并根据 observation 决定下一步。M0 的模型只返回 `GenerationDraft(answer, citation_ids, abstain)`；即使它返回异常，runtime 也只拒答，不让它选择另一个工具。

#### 源码落点

- `src/health_ai_copilot/contracts.py`：`GenerationDraft` 没有 action、tool call 或 next state 字段。
- `src/health_ai_copilot/generation/base.py`：`Generator` 只有 `generate`。
- `src/health_ai_copilot/pipeline.py`：单次 `search`、单次 `generate`、单次 `verify_citations`。

#### 证据

`docs/architecture.md` 把 M0 描述为 vertical slice；`docs/roadmap.md` 才把 `AgentState`、受限 `AgentLoop` 和 `ToolRegistry` 放进 M1 planned。

#### 面试官继续追问

- 如果只增加一个 query rewrite，是否就算 Agent？
- 模型提出 action、runtime 验证 action，怎么落地？

#### 当前边界

不能把“evidence-aware generator”“Harness Engineering”写成已完成 Agent Runtime。它们是模型调用契约和 runtime 校验的 M0 雏形，不等价于 Agent。

### Q7：M0 哪里已经体现 Harness Engineering？

#### 30 秒回答

体现在“model proposes; runtime validates”，但只到 M0 的窄范围。模型可以提出答案和 citation ID，runtime 检查 draft 类型与字段；urgent/prescription 在模型前被拦截；空 evidence 不进入生成；伪造 citation 被 veto；检索/生成异常统一 fail closed。真正的多步预算、权限、timeout、trace/replay 仍是后续计划。

#### 深挖

这里的重点是把可靠性放进边界和反馈，而不是只写 prompt。`OpenAICompatibleGenerator` 的 prompt 要求 JSON 和合法 citation，但 prompt 不是安全边界；`_parse`、`verify_citations`、pipeline 分支才是可执行约束。即使模型不遵守提示，runtime 也不会接受任意 URL 或任意 source ID。

#### 源码落点

- `src/health_ai_copilot/generation/openai_compatible.py`：`_SYSTEM_PROMPT`、`_parse`、`response_format`。
- `src/health_ai_copilot/pipeline.py`：component boundary 的异常捕获与 fail closed。
- `src/health_ai_copilot/verification/citations.py`：Evidence membership check。

#### 证据

`tests/test_pipeline.py` 通过 spy 验证模型前置拦截，通过 fabricated source ID 验证 runtime veto，通过 raising retriever/generator 验证系统失败路径。

#### 面试官继续追问

- Prompt 校验和代码校验的边界是什么？
- 如果要把 M0 变成 Harness Runtime，还缺什么？

#### 当前边界

M0 没有通用 capability/permission system、step/token/time/tool budget、重试策略、trace/replay 或 sandbox。不要把这些名字写成已实现模块。

## 第三轮：检索真的有效吗？

### Q8：为什么先选 BM25？

#### 30 秒回答

因为 M0 的知识库小、数据主要是中文患者教育卡，第一版更需要确定性、可解释、容易离线复现的 lexical baseline。BM25 不需要训练 embedding 或向量服务，能直接解释某个 query term 对排名的贡献，也方便把词法失败和系统其他问题分开。等 failure table 证明词法表达差距值得付出复杂度，再做 dense 或 hybrid ablation。

#### 深挖

这是“先建立可诊断 baseline”的选择，不是结论说 BM25 永远最好。它擅长词面重合、术语和数字；对同义改写、自然语言意图和 OOD 判别弱。当前 30 张卡的规模也没有迫使我们先引入 ANN index、embedding model 或向量数据库。

#### 源码落点

- `src/health_ai_copilot/retrieval/bm25.py`：`BM25Retriever` 直接实现打分和排序。
- `src/health_ai_copilot/retrieval/tokenizer.py`：中文优先分词。
- `src/health_ai_copilot/retrieval/__init__.py`：明确是 explainable lexical retrieval baseline。

#### 证据

当前固定评测的 BM25 v0 指标见 `evals/m0_failure_table.json`，而不是参考文档中的向量检索数字。

#### 面试官继续追问

- BM25 的公式是什么？
- 既然 OOD 会有正分，为什么还选它？

#### 当前边界

当前没有 dense retrieval、hybrid search、reranker、embedding model 或向量库。不能声称“混合检索”或“语义检索已上线”。

### Q9：BM25 公式怎么来的？

#### 30 秒回答

代码对每个 query term 累加一项：先用 IDF 让稀有词更重要，再用饱和函数处理 term frequency，最后用文档长度归一化。当前实现是：

$$
\operatorname{score}(D,Q)=\sum_{t\in Q}\operatorname{IDF}(t)\cdot
\frac{tf(t,D)(k_1+1)}{tf(t,D)+k_1(1-b+b|D|/avgdl)}
$$

其中 `IDF(t)=log((N-df(t)+0.5)/(df(t)+0.5)+1)`，代码固定默认 `k1=1.5`、`b=0.75`。

#### 深挖

`BM25Retriever.__init__` 先把标题、正文和 tags 拼接后 tokenize，建立每张卡的 term frequency、全局 document frequency、文档数和平均长度。`_score` 对 query token 逐项累计；没有命中的 term 不贡献分数；`search` 丢弃 `score <= 0` 的卡，再按 `(-score, card.id)` 排序，所以 tie-breaking 也是确定性的。

#### 源码落点

- `src/health_ai_copilot/retrieval/bm25.py`：`_idf`、`_score`、`search`。
- `src/health_ai_copilot/retrieval/tokenizer.py`：query 和文档使用同一个 `tokenize`。

#### 证据

`tests/test_bm25.py` 覆盖相关卡排在无关卡之前、完全无关 query 返回空列表，以及相同 query 的 top-k 顺序稳定。

#### 面试官继续追问

- 这个分数能不能解释成概率？
- `k1` 和 `b` 变大/变小会怎样？

#### 当前边界

公式是当前代码的 BM25 变体，不应把不同库的默认 IDF、参数或字段加权方式混为一谈；本实现没有单独的 title boost，也没有 query expansion。

### Q10：`k1` 和 `b` 分别控制什么？

#### 30 秒回答

`k1` 控制 term frequency 的饱和速度：词出现几次后边际收益逐渐变小；`b` 控制文档长度归一化强度，`b=0` 基本不做长度归一化，`b=1` 完全按相对平均长度归一化。当前用 `1.5` 和 `0.75` 是一个常见、可解释的 baseline，不是从这 80 条案例上调出来的“最优参数”。

#### 深挖

在分母里，文档长度项是 `1 - b + b * document_length / average_document_length`。长卡片如果包含同一词，不能只因为正文更长就无限占优；而 `k1` 让重复出现的词不会线性放大分数。参数变化必须看 query 类型、卡片长度和 failure slice，不能只看总平均指标。

#### 源码落点

- `src/health_ai_copilot/retrieval/bm25.py:14`：构造函数的参数约束和默认值。
- `src/health_ai_copilot/retrieval/bm25.py:42`：长度归一化和 TF saturation 的计算。

#### 证据

`evals/m0_failure_table.json` 记录的 retriever 配置是 `k1=1.5`、`b=0.75`；失败表没有提供参数 ablation，因此不能声称这两个值经过系统调参。

#### 面试官继续追问

- 中文卡片长度差异会怎样影响 `b`？
- 你会怎样设计 BM25 参数 ablation？

#### 当前边界

没有记录不同 `k1/b` 下的对照实验，也没有将 title/content/tags 分字段加权。下一步若要调参，需要固定 split、保留 failure IDs，并报告 slice 指标。

### Q11：tokenizer 为什么能显著改变 BM25？

#### 30 秒回答

BM25 比较的是 token，不是人的语义。中文没有天然空格，分词决定“高血压”“睡眠不足”这类词是作为完整词还是碎片参与 TF/DF。当前 `tokenize` 对文本做 NFKC 归一化和 lowercase，再用 `jieba.lcut(cut_all=False)` 分词，同时保留有中文、ASCII 或数字的 token；query 和卡片使用同一套规则，避免索引和查询不一致。

#### 深挖

分词会同时改变词频、文档频率、平均文档长度和最终排名。它能帮助“测量血压”这类词面匹配，也可能让“睡不好”和“睡眠不足”保持不同 token，无法表达同义关系。`m0-047` 的“测量血 压”在当前小样本中仍然命中了预期卡，但这只是一次通过，不代表 segmentation 问题已经解决。

#### 源码落点

- `src/health_ai_copilot/retrieval/tokenizer.py:11`：`tokenize`。
- `src/health_ai_copilot/retrieval/bm25.py:20`：标题、内容、tags 的索引 token。

#### 证据

`tests/test_bm25.py::test_chinese_queries_share_meaningful_tokens` 检查中文 query 能共享有意义的 token；failure table 把 `m0-047` 列为 challenge case，但 `failure_case_ids` 为空。

#### 面试官继续追问

- 为什么不用 character n-gram？
- 如果加入同义词词典，会引入什么风险？

#### 当前边界

当前没有 synonym dictionary、query rewrite、character n-gram、embedding 或 reranking。任何这类改动都需要重新跑原有 80 条评测并检查 OOD side effect。

### Q12：为什么 BM25 score 不是 probability？

#### 30 秒回答

因为代码输出的是用于排序的相关性分数，没有把所有文档分数归一化成总和为 1 的分布，也没有校准成“这个文档相关的概率”。IDF、TF saturation 和长度归一化只保证相对排序有意义；不同 query、不同语料或不同参数下的分数不能直接横向比较成概率。

#### 深挖

所以“分数大”不等于“证据充分”，“分数为正”也不等于“可以回答”。当前 `search` 甚至只把 `score > 0` 的卡转成 Evidence，没有 calibrated threshold。这个事实正好解释了四个 OOD false retrieval：共享“高血压”等词就可能得到正的 lexical evidence。

#### 源码落点

- `src/health_ai_copilot/retrieval/bm25.py:60`：只按 score 排名并截取 top-k，不做概率校准。
- `src/health_ai_copilot/pipeline.py:76`：只检查 evidence 是否为空。

#### 证据

`evals/m0_failure_table.json` 对 OOD case 的 note 明确说这是 observed false-retrieval diagnostic，不是 calibrated probability。

#### 面试官继续追问

- 你会怎样建立 evidence sufficiency？
- 能不能用 BM25 score threshold 直接解决 OOD？

#### 当前边界

M0 没有 score calibration、cross-encoder judge 或独立的 sufficiency policy。`ABSTAIN` 存在，但当前确定性触发条件主要是空 evidence、generator abstain 和 citation/runtime failure。

### Q13：你怎么区分并评估 Hit@K、Recall@K、MRR、nDCG？

#### 30 秒回答

Hit@K 是“前 K 名里有没有至少一个相关来源”，是 case-level 的 0/1；Recall@K 是“前 K 名召回了全部相关来源中的多少”，更适合多个 relevant documents；MRR 只看第一个相关来源的倒数排名；nDCG 用 graded relevance 和位置折扣，既看相关程度也看排序位置。当前产品型 M0 runner 报 Hit@1、Hit@3、MRR；独立 NFCorpus adapter 才报 Recall@K、MRR、nDCG@K。

#### 深挖

设相关来源集合为 `R`，排名为 `L`：

- `Hit@K = 1` 当且仅当 `L[:K]` 与 `R` 有交集，否则为 0，再对案例平均。
- `Recall@K = |L[:K] ∩ R| / |R|`。
- `RR = 1/r`，其中 `r` 是第一个相关结果的 rank；MRR 是多个 query 的 RR 平均。
- `nDCG@K = DCG@K / IDCG@K`，DCG 用相关等级的 gain 和 `log2(rank+1)` 位置折扣。

不要把 M0 的 Hit@K 和 NFCorpus 的 Recall@K 互换，也不要把有人工来源映射的 retrieval 指标说成回答质量指标。

#### 源码落点

- `src/health_ai_copilot/eval/runner.py:35`：M0 的 `evaluate_cases`，只统计有 `expected_source_ids` 的案例。
- `src/health_ai_copilot/eval/nfcorpus.py:122`：`_ndcg_at_k`；`evaluate_nfcorpus` 计算 Recall/MRR/nDCG。

#### 证据

`tests/test_eval.py` 验证 M0 evaluator 输出的是 `safety_route_accuracy` 而非 `route_accuracy`，以及 retrieval 的 Hit@1、Hit@3、MRR；`tests/test_nfcorpus.py` 只用一个小 fixture 验证 adapter 和公式。

#### 面试官继续追问

- 为什么产品 eval 先用 Hit@K，而 benchmark 还要 Recall/nDCG？
- 多来源问题的 `expected_source_ids` 怎么标？

#### 当前边界

M0 没有 end-to-end answer quality、faithfulness、calibration 或 clinician review 指标；也没有当前项目在 NFCorpus 上的正式结果快照。

### Q14：你当前 M0.3 的真实结果是什么？

#### 30 秒回答

这是内部 `m0.3-2026-09-15` frozen eval：80 条案例，76 条进入 safety route 统计，`safety_route_accuracy=1.0`；62 条带 `expected_source_ids` 的 patient-education retrieval cases，`Hit@1=0.9032`、`Hit@3=0.9516`、`MRR=0.9274`。它是 safety gate 和 lexical retrieval 的内部基线，不是医学准确率、临床验证或泛化能力。

#### 深挖

完整精度来自 `evals/m0_failure_table.json`：

| 项目 | 实际值 | 如何解释 |
| --- | ---: | --- |
| `pack_version` | `m0.3-2026-09-15` | 当前 failure snapshot 版本 |
| `knowledge_pack_version` | `m0.2-2026-09-15` | 被评测的知识卡版本 |
| `num_cases` | 80 | 全部手工构造案例 |
| `safety_route_cases` | 76 | 62 patient education + 8 urgent + 6 prescription；4 个 unanswerable 不计入该指标 |
| `safety_route_accuracy` | 1.0 | 只评估 deterministic safety gate |
| `retrieval_cases` | 62 | 有人工 `expected_source_ids` 的案例 |
| `hit_at_1` | 0.9032258064516129 | 约 0.9032 |
| `hit_at_3` | 0.9516129032258065 | 约 0.9516 |
| `mrr` | 0.9274193548387096 | 约 0.9274 |

#### 源码落点

- `evals/m0_failure_table.json`：冻结数字和逐案 top-3。
- `evals/m0.jsonl`：80 条 case 的问题、预期 route、来源映射和 challenge type。
- `src/health_ai_copilot/eval/runner.py`：指标定义和排除规则。

#### 证据

评测 pack 的类别是 62 条 `patient_education`、8 条 `urgent`、6 条 `prescription`、4 条 `unanswerable`；`tests/test_eval.py` 对数量、category、reviewed 状态和 source ID 做了回归检查。

#### 面试官继续追问

- 为什么 76 而不是 80？
- 这能不能说成 95% route accuracy？

#### 当前边界

绝对不能把 `Hit@3=0.9516` 四舍五入后包装成“95% route accuracy”。它也不代表 generator 回答正确，更不代表医疗或临床效果。

### Q15：标准 benchmark 呢？NFCorpus 和自建 Product Eval 有什么区别？

#### 30 秒回答

自建 M0.3 Product Eval 检查的是这个产品的 safety marker 和 KnowledgeCard 来源映射，问题、语料和 policy 都围绕当前患者教育场景；BEIR NFCorpus 是独立的标准检索轨道，带 corpus、queries 和 qrels，用来报告 Recall@K、MRR、nDCG@K。代码把 NFCorpus 包装成 retrieval-eval adapter，不把它伪装成产品 KnowledgeCard 或临床结果。

#### 深挖

`load_nfcorpus` 从 `corpus.jsonl`、`queries.jsonl` 和 split qrels 读取外部数据，并在 adapter 内构造标记为 `retrieval_evaluation` 的 `KnowledgeCard`。`evaluate_nfcorpus` 只评估有正相关 qrels 的 query。当前仓库有 adapter 和 fixture test，但没有提交一份正式 NFCorpus 运行结果，因此不能报 NFCorpus 数字。

#### 源码落点

- `src/health_ai_copilot/eval/nfcorpus.py`：`NFCorpusDataset`、`load_nfcorpus`、`evaluate_nfcorpus`。
- `evals/README.md`：M0 自建 eval 与外部 benchmark 的数据模型边界。
- `tests/test_nfcorpus.py`：最小 fixture 测试。

#### 证据

M0.3 的真实数字只来自 `m0_failure_table.json`；`tests/test_nfcorpus.py` 结果是 fixture 上的 1.0，不是外部 benchmark 的正式结果。

#### 面试官继续追问

- 为什么不能只跑一个 benchmark？
- qrels 的 graded relevance 为什么适合 nDCG？

#### 当前边界

NFCorpus 是检索评估，不是中文医疗产品的临床验证，也不等价于当前 KnowledgeCard pack 的质量。外部 benchmark 的许可和数据版本也必须单独记录。

## 第四轮：你遇到了什么真实 bad case？

### Q16：3 个 synonym / paraphrase mismatch 具体是什么？

#### 30 秒回答

failure table 中真正被归为 synonym/paraphrase failure 的是 `m0-003`、`m0-005`、`m0-034`：只看一次没有和“不同日期/确认”形成足够词面重合；头晕或不舒服没有和“症状/无症状”形成足够重合；睡不好和睡眠不足是自然语言改写，generic 的风险词反而占了 top-3。它们不是抽象担忧，而是从 ranked top-3 逐案检查得到的 failure IDs。

#### 深挖

具体 top-3 是：

| case | 预期来源 | 实际 top-3 | 失败解释 |
| --- | --- | --- | --- |
| `m0-003` | `who-hypertension-02-confirmation` | `who-hypertension-03-silent`、`cdc-high-blood-pressure-managing-01-monitoring`、`cdc-high-blood-pressure-about-02-symptoms` | “只看一次”与“不同日期、确认”词面不对齐 |
| `m0-005` | `who-hypertension-03-silent` | `cdc-high-blood-pressure-risk-03-alcohol`、`cdc-high-blood-pressure-risk-05-overweight`、`cdc-high-blood-pressure-risk-04-tobacco` | “头晕或不舒服”与症状/无症状表达不够接近 |
| `m0-034` | `cdc-high-blood-pressure-prevention-04-sleep` | `cdc-high-blood-pressure-risk-05-overweight`、`cdc-high-blood-pressure-risk-04-tobacco`、`cdc-high-blood-pressure-risk-01-sodium` | “睡不好”与“睡眠不足”不一致，generic 风险词主导 |

#### 源码落点

- `evals/m0_failure_table.json` 的 `synonym_paraphrase_failure.observations`。
- `src/health_ai_copilot/retrieval/tokenizer.py` 和 `bm25.py`：系统只按 token overlap 和统计权重排序。

#### 证据

该 failure type 的 `challenge_case_ids` 有 7 个，但 `count=3`；这提醒面试时不要把 challenge label 数量直接说成失败数量。

#### 面试官继续追问

- 为什么 lexical retrieval 会漏掉这些？
- 这是不是证明 embedding 一定更好？

#### 当前边界

失败表给出的是当前小样本上的诊断，不代表中文医学同义词的总体失败率，也没有做 dense retrieval 的对照实验。

### Q17：为什么 lexical retrieval 会出现这些问题？

#### 30 秒回答

因为 BM25 的基本单位是 token，IDF 只知道词在多少文档里出现，不知道“只看一次”和“不同日期确认”在医学问答里可能表达同一个判断，也不知道“睡不好”和“睡眠不足”是近义。没有共享 token 时，相关卡得不到足够分数；反而“风险、影响、高血压”这类泛词在多个卡片里都有，容易把无关卡推上来。

#### 深挖

这是 lexical baseline 的预期 trade-off：优点是可解释和零训练，缺点是语义桥接能力弱。中文分词还会改变边界；即使词 token 一致，也不等于 claim 关系一致。解决这个问题不能只看总 Hit@K，还要保留 query-expression slice 和 OOD slice。

#### 源码落点

- `src/health_ai_copilot/retrieval/bm25.py`：`_idf` 不含语义信息，`_score` 只处理 token frequency/length。
- `evals/m0_failure_table.json`：同义改写的逐案 observation。

#### 证据

`m0-003`/`005`/`034` 的 top-3 直接显示了“有结果但不是预期来源”；`tests/test_bm25.py` 只证明小型 fixture 的基本排序属性，没有证明同义表达能力。

#### 面试官继续追问

- query rewrite 和 dense retrieval 解决的是同一个层面吗？
- 如何避免 rewrite 把问题改错？

#### 当前边界

当前没有同义词表、query expansion、dense embedding、reranker 或语义 query classifier。

### Q18：为什么这说明 dense retrieval 可能有价值？

#### 30 秒回答

它说明词面不一致是当前真实 miss 的一个候选原因，所以能提出 dense retrieval 作为实验假设：如果 embedding 空间能把“睡不好”和“睡眠不足”放得更近，可能提高 paraphrase recall。但这只是“值得做 ablation”的理由，不是从 3 个 case 推出 dense 一定更准。

#### 深挖

dense 引入了新的变量：embedding 模型是否适合中文医学、KnowledgeCard 和 query 如何编码、索引版本、召回阈值、延迟、解释性，以及 OOD 的误召回。正确的下一步是固定 M0 lexical baseline，按 failure slice 对比 BM25、dense、hybrid，另外检查 exact medical terms、数字阈值和 OOD；不能只看总平均。

#### 源码落点

当前代码中没有 embedding/index 实现；已有对照入口是 `src/health_ai_copilot/eval/runner.py` 的 M0 metrics 和 `evals/m0_failure_table.json` 的 failure slices。

#### 证据

真实依据只有 3 个 paraphrase failures；当前 failure table 没有 dense 结果，故不能写“dense 提升了 X%”。

#### 面试官继续追问

- 为什么不马上上 embedding？
- dense 可能带来什么新 failure？

#### 当前边界

dense retrieval、hybrid retrieval、向量数据库和 reranker 都不在 M0。它们即使是合理候选，也只能称为 planned/experiment，不是项目已实现功能。

### Q19：为什么下一阶段反而先考虑 bounded query recovery？

#### 30 秒回答

因为当前最清晰的 failure 是“问题表达与卡片词面不一致”，而不是已经证明需要任意自主规划。一个只允许一次、受 schema 和 budget 约束的 query rewrite，可以直接验证“把问题改写成更适合当前检索器的表达”是否解决 `m0-003/005/034`，同时保留 M0 的安全 gate 和 citation verifier。

#### 深挖

它是小实验而非“为了有 Agent 就加 ReAct”：rewrite 结果只能进入 retrieval，不应改变用户原意、扩大医疗建议范围或绕过 safety gate；若第一次 recovery 没改善，就终止并 abstain。评估要同时看 paraphrase hit、OOD false retrieval、rewrite 是否改变意图、额外延迟和 token cost。

#### 源码落点

- `docs/roadmap.md`：M1 计划明确举例“一次 query rewrite recovery”。
- `evals/m0_failure_table.json`：3 个 lexical paraphrase failures 是候选驱动力。

#### 证据

当前没有 query rewrite 代码、测试或指标。这里的方案状态必须写 [M1 PLANNED]。

#### 面试官继续追问

- 为什么只允许一次？
- rewrite 失败和 OOD false retrieval 如何区分？

#### 当前边界

本基线不做 query rewrite，也没有 Agent action、Agent budget、rewrite trace 或意图保持检查。M1 合并后必须以真实实现替换本段的 planned 说明。

### Q20：BM25 明明“有结果”，为什么问题还是不应该回答？

#### 30 秒回答

“有结果”只表示至少有 query token 与卡片重合，不表示 Evidence 足以支持用户的具体问题。`m0-077` 到 `m0-080` 都是高血压相关的 OOD 问题，BM25 仍返回正分 top-3；如果把任何正分都当作可回答，就会把“全麻手术、乘飞机、脱发、商业保险”错误地套到高血压教育卡上。

#### 深挖

这要求把 retrieval relevance、evidence sufficiency、answerability 和 citation integrity 分开。M0 有空 evidence 时的 deterministic abstain，也要求 generator 在 supplied evidence 不足时 abstain；但没有独立的、经过校准的 evidence sufficiency 判定器。`unanswerable` case 也没有进入 safety_route_accuracy，不能借 route 指标掩盖这个问题。

#### 源码落点

- `evals/m0_failure_table.json` 的 `ood_false_retrieval`：四个 case 的 top-3。
- `src/health_ai_copilot/pipeline.py:76`：只有 evidence 为空时直接 abstain。
- `src/health_ai_copilot/generation/openai_compatible.py` 的 system prompt：要求 evidence 不足时模型 abstain，但这不是独立 runtime proof。

#### 证据

四个 OOD case 的 failure count 都是 1：`m0-077` 全麻、`m0-078` 飞机、`m0-079` 脱发、`m0-080` 商业保险。failure table 明确把它们称为 false-retrieval diagnostic，而不是 calibrated probability。

#### 面试官继续追问

- evidence sufficiency 怎么评？
- selective answering 和简单的 top-1 threshold 有什么区别？

#### 当前边界

独立的 evidence sufficiency、answerability classifier、selective answering policy、OOD detector 都是后续设计，不是 M0 已实现能力。

### Q21：query rewrite 能解决 OOD false retrieval 吗？

#### 30 秒回答

不能简单解决，甚至可能扩大错误召回。rewrite 如果只把“高血压”改写得更标准，可能让 OOD 问题与更多高血压卡片重合；它改善 lexical recall 的同时可能损伤 precision。对 OOD，第一优先级应是 answerability/evidence sufficiency 和拒答，而不是盲目扩展 query。

#### 深挖

要把两个 slice 分开：paraphrase recovery 希望“同一意图换种说法”能找到预期来源；OOD containment 希望“领域内有词但知识包不支持”不要被强行回答。任何 rewrite 实验都要报告两边的 confusion matrix，至少保留原 query、rewrite、top-k、最终 route 和原因，才知道是恢复还是放大。

#### 源码落点

- 现有 OOD 证据：`evals/m0_failure_table.json`。
- 现有拒答路径：`src/health_ai_copilot/pipeline.py` 的 `insufficient_evidence`、`generator_abstained`、`invalid_citation`。
- 后续方向：`docs/roadmap.md` 的 M1/M2 planned。

#### 证据

当前没有 rewrite 实验，所以不能声称它对 OOD 的实际影响；“可能扩大错误召回”是基于 lexical overlap 机制的风险分析。

#### 面试官继续追问

- 你会把 OOD 交给 generator 还是 runtime？
- selective answering 的目标函数是什么？

#### 当前边界

evidence sufficiency、abstention calibration、selective answering 和 OOD 专项指标尚未实现。M0 的 `ABSTAIN` 是安全终点，不等于已经解决了 answerability。

## 第五轮：Evidence 和 Citation

### Q22：KnowledgeCard、Evidence、GenerationDraft、Citation 各是什么？

#### 30 秒回答

`KnowledgeCard` 是经过 schema 校验、带来源和审核元数据的存储单元；`Evidence` 是本次 query 检索出来、带 `source_id/title/excerpt/source_url/score` 的结果；`GenerationDraft` 是模型唯一被接受的结构化输出，包含 answer、citation IDs 和 abstain；`Citation` 是最终响应里的可展示引用，metadata 由 runtime 从 Evidence 复制。

#### 深挖

这四层把“知识库对象”“本次检索上下文”“模型提议”“对外引用”分开。模型只需要选择已给出的 ID，不需要生成 URL、标题或摘要；最终 `AssistantResponse` 的 citations 根据 verified IDs 从 `evidence_by_id` 构造。因此 metadata provenance 不依赖模型是否会编造网页。

#### 源码落点

- `src/health_ai_copilot/contracts.py`：四个 dataclass。
- `src/health_ai_copilot/pipeline.py:91`：由 verified IDs 复制 Citation。
- `src/health_ai_copilot/generation/openai_compatible.py:_parse`：解析 `GenerationDraft`。

#### 证据

`tests/test_pipeline.py::test_normal_path_returns_citations_from_stored_evidence` 检查最终 title 和 URL 等于 fixture Evidence 对应的值。

#### 面试官继续追问

- 为什么不直接把 KnowledgeCard 传给模型？
- Evidence 的 score 能不能对用户解释成可信度？

#### 当前边界

Evidence 的 `score` 是 BM25 ranking score，不是概率；Citation integrity 也不等于 claim-level grounding。

### Q23：为什么模型只返回 `citation_id`，不能自己返回 URL？

#### 30 秒回答

因为 URL、title、excerpt 属于系统掌握的 source metadata，不应由模型自由生成。模型只返回本次 `supplied_evidence` 中的 `source_id`；runtime 先检查 ID 是否属于当前 evidence bundle，再从存储对象复制 metadata。这样可以防止任意 URL、拼错标题和把未检索来源伪装成依据。

#### 深挖

这是“引用选择”和“引用内容”分离。模型仍可能选择一个与 claim 不匹配但合法的 ID，所以这只解决引用完整性，不解决语义支持。安全收益是把最容易被模型编造的 metadata 变成 capability-limited lookup，而不是让模型拥有写 URL 的能力。

#### 源码落点

- `src/health_ai_copilot/generation/openai_compatible.py` 的 `_SYSTEM_PROMPT`：只允许 supplied evidence 的 source ID。
- `src/health_ai_copilot/verification/citations.py`：`evidence_ids` membership check。
- `src/health_ai_copilot/pipeline.py`：`evidence_by_id` 复制 metadata。

#### 证据

`tests/test_pipeline.py::test_fabricated_source_id_returns_abstain` 让 generator 返回 `not-retrieved`，结果是 `ABSTAIN`；正常路径测试验证 URL 来自存储 Evidence。

#### 面试官继续追问

- 合法 ID 但 claim 不被支持怎么办？
- 如果 evidence 有重复 ID 呢？

#### 当前边界

模型仍可以生成没有被 Evidence 逐句支持的自然语言；当前没有 claim extraction、entailment model 或 human grounding review。

### Q24：fabricated citation 怎么防？

#### 30 秒回答

先在 provider 侧要求结构化 JSON，并检查 `answer`、`citation_ids`、`abstain` 的类型；再在 verifier 侧把 Evidence ID 集合与模型返回的 ID 比较。缺少 citation 返回 `missing_citation`，有不在本次 Evidence 中的 ID 返回 `invalid_citation`，pipeline 统一拒答；重复 ID 则去重并保留第一次出现的顺序。

#### 深挖

provider parser 解决“能否解析成合法 draft”，verifier 解决“引用是否属于本次 bundle”，pipeline 解决“校验失败后的系统状态”。这三层不能由 prompt 替代。注意 `_parse` 要求 JSON 中出现 `abstain`，但 citation 的归属仍由 deterministic verifier 再检查一次。

#### 源码落点

- `src/health_ai_copilot/generation/openai_compatible.py:58`：`_parse`。
- `src/health_ai_copilot/verification/citations.py:16`：`verify_citations`。
- `src/health_ai_copilot/pipeline.py:86`：失败时 `abstain_response`。

#### 证据

`tests/test_citations.py` 覆盖合法 citation、fabricated citation、重复 ID 去重和 non-abstaining answer 必须有 citation；pipeline tests 覆盖最终路由和用户消息。

#### 面试官继续追问

- 为什么 missing citation 和 invalid citation 分开？
- 模型返回 malformed JSON 时会怎样？

#### 当前边界

当前没有 URL 可达性检查、来源内容新鲜度检查、claim-to-citation alignment 或跨来源冲突裁决。

### Q25：citation integrity 与 semantic grounding 有什么区别？

#### 30 秒回答

citation integrity 问的是“模型给的 ID 是否属于本次检索到的 Evidence”；semantic grounding 问的是“这个 ID 对应的 excerpt 是否真正支持回答中的每一个 claim”。M0 只做前者。比如 Evidence 合法地包含“家庭测量前保持姿势”的卡片，模型却说“血压正常就可以自行停药”；ID 合法，但 claim 不被 evidence 支持，当前 verifier 仍可能接受。

#### 深挖

`verify_citations` 的代码注释已经明确写着它不检查 semantic entailment。要做 grounding，至少要定义 claim 切分、evidence span/句子映射、蕴含/矛盾/未知标签和拒答标准；还要处理一个 claim 由多个来源共同支持、来源冲突和数字阈值。不能用“有 citation”作为 faithfulness 的替代指标。

#### 源码落点

- `src/health_ai_copilot/verification/citations.py` 的 docstring 和 `verify_citations`。
- `README.md` 的设计限制：claim-level grounding 留到后续阶段。

#### 证据

`tests/test_citations.py` 只验证 ID 集合关系，没有任何 claim-level test；因此当前项目没有 semantic grounding 指标。

#### 面试官继续追问

- 你会选 NLI、LLM judge 还是规则做 grounding？
- grounding 失败是删掉 claim 还是整条回答拒答？

#### 当前边界

claim-level grounding、引用与 claim 的对齐、source conflict handling 都是 planned；不要把 citation verifier 写成“防止所有幻觉”。

## 第六轮：工程实现追问

### Q26：为什么 Generator 用 Protocol？为什么测试用 FakeGenerator？

#### 30 秒回答

`Generator` 用最小 `Protocol` 描述 pipeline 需要的唯一能力：`generate(question, evidence) -> GenerationDraft`。pipeline 不依赖具体厂商 SDK，所以 live OpenAI-compatible provider 和测试 FakeGenerator 可以互换。FakeGenerator 让测试在无网络、无 API Key 的情况下精确控制正常 draft、abstain、异常和 fabricated citation。

#### 深挖

这是 dependency inversion 和可测试性，而不是为了抽象而抽象。`Retriever` 也在 `pipeline.py` 里用 Protocol 定义；pipeline 只编排，不实现检索器内部逻辑。测试还可以用 SpyRetriever 观察调用次数，验证 safety gate 确实发生在组件调用之前。

#### 源码落点

- `src/health_ai_copilot/generation/base.py`：`Generator`、`GenerationError`。
- `src/health_ai_copilot/pipeline.py`：`Retriever` 和 constructor injection。
- `tests/test_pipeline.py`：`FakeGenerator`、`SpyRetriever`、`Raising*` doubles。

#### 证据

`tests/test_pipeline.py` 的 urgent/prescription case 断言两个组件调用次数为 0；正常 case 用 FakeGenerator 验证最终 response。

#### 面试官继续追问

- Protocol 和 ABC 的取舍是什么？
- 为什么不在 pipeline 内部直接实例化 OpenAI client？

#### 当前边界

Protocol 只提供 Python 类型层面的契约，不自动保证运行时第三方对象完全正确；pipeline 仍用 `isinstance(draft, GenerationDraft)` 做一道运行时检查。

### Q27：为什么用结构化 JSON output？

#### 30 秒回答

因为 runtime 需要稳定地分离 `answer`、`citation_ids` 和 `abstain`，才能在生成后执行确定性校验。当前 system prompt 要求只输出 JSON，provider 请求 `response_format={\"type\":\"json_object\"}`，`_parse` 再逐字段检查类型、非空约束和 abstain 布尔值；不符合就抛 `GenerationError`，pipeline 拒答。

#### 深挖

JSON 不是保证模型正确的魔法，它只是让错误可被识别。格式约束解决 schema failure，citation verifier 解决引用归属，evidence grounding 仍然未解决。当前是手写解析和类型检查，不是完整 JSON Schema validator；所以面试时应按实际实现说。

#### 源码落点

- `src/health_ai_copilot/generation/openai_compatible.py:58`：`_parse`。
- `src/health_ai_copilot/generation/openai_compatible.py:94`：请求 `response_format`。
- `src/health_ai_copilot/pipeline.py:78`：生成异常 fail closed。

#### 证据

`tests/test_generation.py` 覆盖 malformed/invalid provider payload；`tests/test_pipeline.py` 覆盖 generator 异常和 generator 主动 abstain。

#### 面试官继续追问

- JSON parse 成功但 answer 仍然幻觉怎么办？
- 为什么让模型返回 citation ID 而不是整段引用？

#### 当前边界

没有 function-call tool schema、claim-level JSON、自动修复输出或 semantic validator。不要引用参考文档中的“自动补免责声明”等未在当前代码出现的行为。

### Q28：为什么环境变量而不是在 `config.py` 里写 key？

#### 30 秒回答

API key 和模型配置属于运行环境，不应写进版本库。`load_openai_config` 只读取 `HEALTH_COPILOT_API_KEY`、`HEALTH_COPILOT_MODEL` 和可选的 `HEALTH_COPILOT_BASE_URL`；缺必填变量就抛 `ConfigurationError`。默认温度是 `OpenAIConfig` 的 `0.1`，不是从一个硬编码 secret 文件加载。

#### 深挖

这样做把代码、知识卡和部署配置分开，也方便在不同 OpenAI-compatible endpoint 间切换。代价是运行 live demo 前必须准备环境变量，配置错误会在 generator 初始化时变成 `GenerationError`，CLI 打印错误并返回 2；离线 FakeGenerator 测试不需要这些配置。

#### 源码落点

- `src/health_ai_copilot/config.py`：`OpenAIConfig`、`load_openai_config`。
- `src/health_ai_copilot/generation/openai_compatible.py`：把配置错误包装成 `GenerationError`。
- `src/health_ai_copilot/cli.py`：live demo 入口。

#### 证据

`README.md` 的 live CLI 示例列出三个环境变量；`tests/test_generation.py` 验证缺少配置时的错误行为。

#### 面试官继续追问

- 为什么 `base_url` 可以为空？
- 真实部署还需要哪些 secret、超时和审计策略？

#### 当前边界

当前没有 secret manager、配置热更新、请求 timeout/retry、调用审计或 provider fallback；这些不能从“使用环境变量”推导出来。

### Q29：为什么用 `src` layout？

#### 30 秒回答

仓库把可安装包放在 `src/health_ai_copilot`，并在 `pyproject.toml` 中配置 setuptools 从 `src` 查找包。这样代码、tests、tools 和项目根目录文件的边界更清楚，测试应该通过安装后的包导入，而不是意外地从仓库根目录捡到同名模块。README 用 editable install 把这个包装进开发环境。

#### 深挖

`src` layout 本身不是安全机制，但它减少“当前工作目录刚好能 import”造成的假阳性。开发安装后，包 metadata 和源码路径由 packaging 工具管理；源码改动可以在 editable 模式下立即生效，同时仍使用真实的包发现配置。

#### 源码落点

- `pyproject.toml` 的 `[tool.setuptools.packages.find] where = [\"src\"]`。
- `src/health_ai_copilot/`：包源码。
- `README.md` 的 `uv pip install --python ... -e \".[dev]\"`。

#### 证据

`tests/` 直接从 `health_ai_copilot` 导入，说明预期运行方式是先完成项目安装或设置正确的 Python path。

#### 面试官继续追问

- editable install 到底做了什么？
- 为什么我的 shell 里直接 `pytest` 会 import 失败？

#### 当前边界

当前工作环境若没有执行 editable install 或缺少依赖，不能把导入失败误判成代码逻辑失败；验证时要先按 README 准备环境。

### Q30：editable install 是什么？`egg-info` 是什么？

#### 30 秒回答

editable install 会把项目以“开发模式”注册到环境里，import 时指向工作树源码，所以改 Python 文件通常不用每次重新打包。`egg-info` 是 setuptools 等工具生成的项目 metadata 目录，记录包名、版本、依赖和文件信息；它是安装构建产物的 metadata，不是 Health-Copilot 的 runtime feature，也不等于已经实现了 Agent。

#### 深挖

本项目的安装入口由 `pyproject.toml` 的 build backend 和 setuptools package discovery 定义，README 建议用 `uv pip install -e \".[dev]\"`。editable 只解决包发现和开发迭代，不会自动安装缺失依赖，也不会运行评测。metadata 目录是否出现、在哪里出现，取决于安装工具和环境，不能写成项目逻辑。

#### 源码落点

- `pyproject.toml`：`[build-system]`、`[project]`、`[tool.setuptools.packages.find]`。
- `README.md`：开发环境安装命令。

#### 证据

当前基线的可追溯事实是 packaging 配置和安装命令；不要根据其他项目的 `egg-info` 路径臆造本项目文件。

#### 面试官继续追问

- editable install 和 wheel install 的差别？
- CI 为什么仍应安装 package 而不依赖 cwd？

#### 当前边界

安装 metadata 不能作为功能实现证据；如果环境没有 `jieba` 等依赖，测试会在 collection 阶段失败，需要先按项目安装说明补齐环境。

### Q31：unit test、integration test、eval 有什么区别？本项目分别在哪里？

#### 30 秒回答

unit test 关注一个函数或组件的局部契约，例如 `tokenize`、BM25 排序、citation verifier、safety marker；integration-style test 把 pipeline、retriever 和 fake generator 拼起来，验证跨边界顺序和状态；eval 是一批带标注的案例和指标计算，回答“在这个 frozen pack 上表现如何”。三者不能互相替代，更不能把 unit test 全绿说成医疗准确。

#### 深挖

当前 `tests/test_pipeline.py` 通过 doubles 验证组件协作，仍然不调用网络；`tests/test_bm25.py`、`test_citations.py`、`test_safety.py` 是局部行为；`tests/test_eval.py` 验证 evaluator 的统计语义和 M0 pack schema；`evals/m0.jsonl`/`m0_failure_table.json` 才是当前固定数据证据。`tests/test_nfcorpus.py` 是 external adapter 的小 fixture，不是正式 benchmark run。

#### 源码落点

- `tests/test_pipeline.py`、`tests/test_bm25.py`、`tests/test_citations.py`、`tests/test_safety.py`。
- `tests/test_eval.py`、`tests/test_nfcorpus.py`。
- `evals/m0.jsonl`、`evals/m0_failure_table.json`、`src/health_ai_copilot/eval/runner.py`。

#### 证据

评测文件中的 `status=reviewed` 表示字段、路由意图和 source mapping 已检查，不表示临床验证；eval runner 也明确不输出 end-to-end route accuracy。

#### 面试官继续追问

- 测试集和评测集如何防止泄漏？
- 为什么不直接用 LLM-as-Judge？

#### 当前边界

M0 没有真实线上流量、医生盲评、临床 outcome、LLM-as-Judge、端到端 faithfulness 或 latency benchmark。

### Q32：为什么 fail closed？

#### 30 秒回答

在这个原型里，未知或不可信状态宁可不生成，也不让模型继续猜。具体是：空 evidence、generator 主动 abstain、检索异常、生成异常、missing citation 和 invalid citation 都回到 `ABSTAIN`；urgent/prescription 则在前面进入固定安全路由。这样牺牲了一些 coverage，换取可解释的拒答边界。

#### 深挖

fail closed 的关键是区分“没有足够证据”和“系统坏了”：pipeline 对它们保留不同的 reason/message，例如 `insufficient_evidence` 与 `retrieval_error`。代价是 broad exception 会隐藏用户侧的内部细节，未来需要结构化错误、trace 和运维告警来补足；但当前 prototype 不应把异常当成成功回答。

#### 源码落点

- `src/health_ai_copilot/pipeline.py`：`ABSTAIN_MESSAGES`、`abstain_response`、retrieval/generation/citation 分支。
- `src/health_ai_copilot/contracts.py`：`Route.ABSTAIN`。

#### 证据

`tests/test_pipeline.py` 逐项验证 no evidence、retrieval error、generation error、generator abstain、fabricated citation 都返回 `ABSTAIN`，且系统错误 message 不冒充“资料不足”。

#### 面试官继续追问

- fail closed 会不会让系统太保守？
- 如何统计拒答率和误拒答？

#### 当前边界

M0 没有 calibrated selective-risk curve、覆盖率/拒答率 trade-off、用户申诉或人工升级队列；`HUMAN_REVIEW` 只是固定响应 route，不是接入真实人工工作台。

### Q33：retrieval service / LLM service 出错怎么办？

#### 30 秒回答

retriever 抛异常时，pipeline 返回 `ABSTAIN` 和 `retrieval_error`；generator 抛异常或返回非法 draft 时，返回 `ABSTAIN` 和 `generation_error`。OpenAI-compatible provider 会把配置、SDK、请求和解析问题包装成 `GenerationError`。当前没有 retry、timeout、fallback provider 或 stale cache，所以不能在故障时假装生成成功。

#### 深挖

pipeline 在组件边界捕获异常，用户消息与“证据不足”区分开；测试使用 `RaisingRetriever`、`RaisingGenerator` 验证这个语义。生产系统还需要可观测性、错误分类、幂等重试、超时和熔断，但这些属于后续 runtime/harness 设计，不在 M0 代码里。

#### 源码落点

- `src/health_ai_copilot/pipeline.py:68` 和 `:78`：两个组件边界。
- `src/health_ai_copilot/generation/openai_compatible.py:92`：请求异常和 parse 异常。
- `tests/test_pipeline.py` 的 `RaisingRetriever`、`RaisingGenerator`。

#### 证据

测试断言 `retrieval_error` 的 message 包含“检索服务”，`generation_error` 的 message 包含“回答生成服务”，并且不会错误使用“审核资料不足”的文案。

#### 面试官继续追问

- 为什么不自动重试一次？
- 如果 retry 会造成重复工具调用，如何保证幂等？

#### 当前边界

timeout/retry/fallback/circuit breaker/trace/replay 目前没有实现；`docs/roadmap.md` 把其中一部分放在后续 M2 Harness Runtime 方向。

### Q34：知识卡 loader 为什么要这么严格？

#### 30 秒回答

因为生成器的可信边界依赖知识库对象本身可信。schema 要求非空的 ID、标题、内容、publisher、reviewer、version、audience/tags，要求绝对 `http(s)` source URL 和 ISO 日期；loader 排除 schema example，检查 JSON、重复 ID，并按 ID 确定性排序。坏卡片直接 load error，而不是静默跳过。

#### 深挖

严格 loader 把 provenance 和数据质量问题挡在 retrieval 之前，方便评测复现。`KnowledgeCard` 中保留 published/collected/reviewed/expiry 等字段，但 M0 retrieval 只使用 title/content/tags 做 lexical index；metadata 不应被模型重新生成。外部 benchmark adapter 也明确与产品 KnowledgeCard 语义分开。

#### 源码落点

- `src/health_ai_copilot/knowledge/schema.py`：`knowledge_card_from_dict`、日期/URL/list 校验。
- `src/health_ai_copilot/knowledge/loader.py`：`load_knowledge_cards`、重复 ID 和排序。
- `tests/test_knowledge_loader.py`：缺字段、坏 JSON、非 HTTP URL、空内容、重复 ID。

#### 证据

`data/README.md` 要求来源 URL、发布日期、采集日期、适用人群、审核人、版本和失效日期；当前 knowledge pack 版本记录在 `m0_failure_table.json`。

#### 面试官继续追问

- 为什么 schema example 不能当卡片？
- 如果一张卡过期，当前 pipeline 会自动过滤吗？

#### 当前边界

当前 loader 校验并保留 `expires_at`，但从已读代码看没有按过期时间自动过滤的 runtime policy；不能声称已实现 freshness/expiry enforcement。

### Q35：`safety_reasons` 为什么是技术债？

#### 30 秒回答

当前 `AssistantResponse` 为兼容 M0 contract 仍叫 `safety_reasons`，但 pipeline 也把 `insufficient_evidence`、`retrieval_error`、`generation_error`、`invalid_citation` 等 runtime status reason 放进去。它能工作，但字段名会误导调用方。architecture 文档已经记录这是技术债，后续可以演进成 `reasons` 或 `status_reasons`，再按 safety/runtime failure domain 分开。

#### 深挖

保持旧字段名有兼容性收益，立即改名则会增加调用方和测试迁移成本；但继续混用会让监控和用户文案难以按 failure domain 聚合。面试时要把“当前行为”和“后续重构”分开，不要说 M0 已经有完整 error taxonomy。

#### 源码落点

- `src/health_ai_copilot/contracts.py`：`AssistantResponse.safety_reasons`。
- `src/health_ai_copilot/pipeline.py`：同一字段承载多类 reason。
- `docs/architecture.md` 和 `README.md`：记录该 known debt。

#### 证据

`tests/test_pipeline.py` 对 `[\"retrieval_error\"]`、`[\"generation_error\"]`、`[\"invalid_citation\"]` 等 status reason 有直接断言。

#### 面试官继续追问

- 重构字段时怎样保持兼容？
- safety policy 和 system failure 应该如何分别监控？

#### 当前边界

当前没有统一 error code registry、typed failure domain、trace ID 或 telemetry schema；这些不能从 `safety_reasons` 的字符串列表推导出来。

## 第七轮：从 M0 走向 M1

> 本轮全部是 [M1 PLANNED]。以下是基于当前 failure 和 roadmap 的面试回答草稿，不是基线中已经存在的 runtime。

### Q36：[M1 PLANNED] 为什么现在值得引入 Agent？

#### 30 秒回答

不是因为“项目里必须有 Agent”，而是 M0 已经给出了一个具体候选 failure：部分问题需要先把用户表达恢复成更适合当前检索器的 query，再进行一次检索。若这个 bounded recovery 在 ablation 中确实改善 paraphrase miss，才有理由引入受限的 Agent action；否则继续维护 deterministic workflow。

#### 深挖

M1 的价值是把一个可解释的动作纳入状态、预算和 trace，而不是允许任意 ReAct。动作空间可以先只有 `rewrite_query`，结果回到同一个 BM25/Evidence/Citation 契约；safety gate 必须先于 action，且 recovery 不能改变医疗决策边界。

#### 源码落点

- 驱动证据：`evals/m0_failure_table.json` 的 3 个 paraphrase failures。
- 设计约束：`docs/roadmap.md` 的 M1 planned。

#### 证据

当前没有 AgentState 或 AgentLoop 代码，因此“引入 Agent 后提升多少”没有数字，必须等待真实实现和 ablation。

#### 面试官继续追问

- 你为什么不先做 dense retrieval？
- 怎样证明 action 是 failure-driven 而不是炫技？

#### 当前边界

M1 仍未合并，Agent Core、query recovery、ToolRegistry、Session 和 action trace 都不能写成已实现。

### Q37：[M1 PLANNED] 哪一种真实 failure 驱动 Agent action？

#### 30 秒回答

优先是 `m0-003`、`m0-005`、`m0-034` 这种 query-expression mismatch。它们的预期来源在知识包里，但原 query 的 lexical expression 没有把它们排进 top-3；一次 bounded query rewrite 可以作为最小 recovery。四个 OOD false retrieval 不应直接驱动“再搜一次”，因为它们的问题是 answerability/sufficiency，不是单纯找不到同义卡片。

#### 深挖

动作必须绑定到 failure type，并用 case ID 回归。成功标准不能只有 Hit@K，还要检查 rewrite 后 OOD 是否恶化、是否引入了用户原问题没有的症状/意图、是否增加了不必要成本。这样 Agent action 才是一个可证伪的实验，而不是 prompt 叠加。

#### 源码落点

- `evals/m0_failure_table.json`：failure type、case IDs 和 top-3。
- `src/health_ai_copilot/retrieval/bm25.py`：recovery 最终仍应面对同一 retrieval contract。

#### 证据

failure table 的 `classification_basis` 说明 failure 是检查 ranked top-3 后得到的，不是看到 `challenge_type` 就直接判失败。

#### 面试官继续追问

- query rewrite 是 tool call 还是内部函数？
- action 失败时 route 怎么走？

#### 当前边界

当前没有 rewrite prompt、rewrite schema、action policy、预算或回归指标；所有答案都是 [M1 PLANNED]。

### Q38：[M1 PLANNED] 为什么不是直接完整 ReAct？

#### 30 秒回答

完整 ReAct 会同时引入规划、工具选择、观察、循环、错误恢复和终止问题，超出了当前唯一已观察的 lexical mismatch。先做一个动作、一次 recovery、固定输入输出契约，能把收益和副作用归因到 query recovery；如果连这个最小版本都不能稳定改善，就没有理由扩展成任意 ReAct。

#### 深挖

医疗场景的 action space 还涉及安全和权限。一个“自由搜索、自由改写、自由生成”的 loop 可能扩大 OOD 召回或绕过 pre-gate。最小化状态和 capability 有利于测试：原问题、rewrite、retrieval result、attempt count 和 final route 都可重放。

#### 源码落点

- 当前固定编排：`src/health_ai_copilot/pipeline.py`。
- 受限 Agent 方向：`docs/roadmap.md` 的 M1/M2。

#### 证据

基线没有 ReAct loop，也没有 tool call test；不能引用下载参考文档中的“最多调用 2/3 次工具”等数字。

#### 面试官继续追问

- ReAct 的 observation 在本项目里是什么？
- 哪些动作永远不能交给模型？

#### 当前边界

完整 ReAct、模型自主工具选择、并行 tool call 和多步 recovery 未实现。

### Q39：[M1 PLANNED] 为什么只允许一次 query rewrite？

#### 30 秒回答

一次是为了给 recovery 一个明确预算和终止条件：最多增加一次模型调用和一次检索，失败就回到 abstain 或原路径，不允许不断改写直到碰巧命中。它控制延迟、token cost、重复工具调用和错误意图漂移，也让评测可以比较“原 query vs rewrite query”。

#### 深挖

一次限制不是普适最优值，而是当前 failure-driven MVP 的安全边界。未来若放宽，需要证明多次 rewrite 的增益大于 OOD、成本和 loop risk，并引入 action fingerprint/dedup、attempt counter、timeout 和 trace。rewrite 的输出也应是 typed schema，而不是把一段自由文本直接当成系统指令。

#### 源码落点

- 设计依据：`docs/roadmap.md` 的“一次 query rewrite recovery”。
- 现有终止基线：`src/health_ai_copilot/pipeline.py` 的固定单次 retrieval/generation。

#### 证据

当前没有一次 rewrite 的实现或指标；“一次”是 planned design constraint，不是现有代码行为。

#### 面试官继续追问

- rewrite 本身超时怎么办？
- 如何防止相同 query 被重复调用？

#### 当前边界

没有 Agent step budget、tool budget、timeout、retry、dedup 或 trace/replay runtime。

### Q40：[M1 PLANNED] 为什么不是 Multi-Agent？

#### 30 秒回答

当前 failure 不需要多个角色协作；问题是一个受限的 retrieval recovery。Multi-Agent 会增加状态同步、消息协议、错误归因、成本和安全面，但 M0 没有证据显示这些复杂度能解决现有 failure。先把单 Agent Core 的边界和指标做实，再由真实 failure 决定是否值得拆分角色。

#### 深挖

“多个 prompt”不等于多 Agent，真正的 Multi-Agent 需要角色、通信、调度、权限和终止语义。当前仓库没有这些实现；参考文档中关于 Agent Swarm、LeadAgent、Worker Agent 的描述属于其他材料，不能作为本项目事实。

#### 源码落点

- `README.md` 的“尚未实现”：明确不包含 Multi-Agent。
- `docs/roadmap.md`：后续只说“再考虑”复杂扩展，并没有 M0 实现。

#### 证据

当前 `src/health_ai_copilot` 只有一个 pipeline、一个 retriever contract 和一个 generator contract，没有 agent-to-agent message 或 shared context。

#### 面试官继续追问

- 什么 failure 才值得 Multi-Agent？
- 多 Agent 的共享上下文如何防止污染？

#### 当前边界

Multi-Agent、Agent Swarm、Agent Teams、shared context 和并行 worker 都未实现，不能进入 M0 简历 claim。

### Q41：[M1 PLANNED] Agent Loop 引入后最大的风险是什么？

#### 30 秒回答

最大的风险不是“模型多想一步”，而是自主动作把成本、权限和错误放大：重复 query/tool call 形成 loop，rewrite 把 OOD 变成更多错误 evidence，或者绕开原有 safety gate。解决方式是让 runtime 而不是模型拥有 step/tool/time budget、允许的 action schema、状态转换和终止 veto。

#### 深挖

风险至少分四类：安全绕过、无限/重复循环、证据污染、可观测性缺失。M1 设计应把原始问题 immutable 保存，把 recovery 输出限制为检索 query，把 evidence 来源继续绑定到 KnowledgeCard，并在每一步记录 action、输入摘要、结果、耗时和 route。M2 再把 policy/budget/timeout/recovery/trace/replay 系统化。

#### 源码落点

- M0 已有可复用边界：`src/health_ai_copilot/safety.py`、`pipeline.py`、`verification/citations.py`。
- M2 计划：`docs/roadmap.md`。

#### 证据

当前没有 loop risk 的运行时指标，也没有 trace/replay 文件；这里是设计风险清单，不是已发生的线上事故。

#### 面试官继续追问

- 如何定义 termination invariant？
- tool error 是 retry、fallback 还是 abstain？

#### 当前边界

Agent Loop、capability security、budget enforcement、trace/replay 和 timeout 未实现。

### Q42：[M1 PLANNED] 如何防 infinite loop / repeated tool calls？

#### 30 秒回答

先用硬性 step/工具/时间预算保证 loop 有上限，再记录已经调用过的 action fingerprint，避免完全相同的 query 重复执行；为工具调用设置 schema、timeout 和错误状态，达到预算或重复阈值就终止并按 failure policy 拒答。对本项目的第一版，最简单的 invariant 是最多一次 query rewrite。

#### 深挖

预算是必要条件但不是充分条件：不同 query 的无意义变体仍可能耗尽预算，所以还需要 progress signal、rewrite intent check 和 trace。终止后不能直接把半成品当答案，仍要回到 Evidence → GenerationDraft → Citation verifier 的 M0 契约。重试还要考虑幂等性；当前 BM25 search 是只读的，但未来外部 tool 未必是只读的。

#### 源码落点

- M0 当前没有 loop；固定次数可从 `src/health_ai_copilot/pipeline.py` 的单次调用看出。
- 计划中的 budget/timeout/recovery：`docs/roadmap.md`。

#### 证据

仓库没有 `max_steps`、`tool_call_history` 或 `idempotency_key` 的实现；不能把参考文档中的工具次数限制当作 Health-Copilot 数字。

#### 面试官继续追问

- 预算按 step、token、美元还是 wall-clock 算？
- action retry 如何做到幂等？

#### 当前边界

所有 loop guard 都是 [M1 PLANNED] / M2 方向，尚未合并。

### Q43：[M1 PLANNED] Agent 和 Workflow 的真正区别是什么？

#### 30 秒回答

Workflow 的控制流由代码预先决定，Agent 的控制流允许模型在给定状态、工具和 policy 下选择下一步；两者都可以有模型。Health-Copilot M0 是 workflow：safety → retrieval → generation → verification。M1 如果实现，也应该是“受限 Agent”：模型只在允许的 action space 里提议，runtime 负责验证、预算和终止，而不是把控制权完全交出去。

#### 深挖

区别不在有没有 `while` 循环或是不是用了 prompt，而在谁决定 next action、状态是否显式、工具是否有 capability boundary、是否能在 observation 后继续。Agent 带来适应性，也带来非确定性和评测难度；所以 M0 的 deterministic path 应继续保留为 fallback/baseline，并用同一 failure pack 做对照。

#### 源码落点

- M0 workflow：`src/health_ai_copilot/pipeline.py`。
- M0 状态/输出契约：`src/health_ai_copilot/contracts.py`。
- planned Agent Core：`docs/roadmap.md` 的 M1。

#### 证据

当前没有 AgentState、next-action schema 或 loop trace；本题的后半部分只能作为 [M1 PLANNED] 设计答案。

#### 面试官继续追问

- M1 合并后哪些 invariant 必须保持不变？
- 怎样证明 Agent 没有损害 M0 的 safety_route 和 citation integrity？

#### 当前边界

在 M1 真实代码合并并通过回归前，不能把“受限 Agent”“一次 rewrite”“ToolRegistry”写成已经完成。

## 一页式复述顺序

如果面试官只给两分钟，可以按下面顺序说：

1. 项目目标是安全、可追溯的患者教育 RAG，不是诊断/处方。
2. 当前 M0 请求链是 input validation → safety gate → BM25 → Evidence → Generator → citation verification → `AssistantResponse/ABSTAIN`。
3. 选择 deterministic workflow 是为了先验证 invariant；模型提议，runtime 校验。
4. M0.3 frozen eval：80 cases，76 safety-route cases，safety gate 1.0；62 retrieval cases，Hit@1 0.9032、Hit@3 0.9516、MRR 0.9274。
5. 真实问题是 3 个 paraphrase mismatch 和 4 个 OOD false retrieval；前者驱动 [M1 PLANNED] 的一次 bounded query recovery，后者提醒不能把正的 BM25 score 当 evidence sufficiency。
6. 当前 citation verifier 防的是 fabricated/invalid ID，不是 semantic grounding；M1/M2 仍需真实代码和评测来证明。

## 基线 claim 审计清单

下面这些词可以作为通用面试知识在 `bagua.md` 中解释，但截至本基线不能写成 Health-Copilot 已实现：

`Multi-Agent`、`Agent Swarm`、`Agent Teams`、`Mem0`、`MCP`、`Sandbox`、`dense retrieval`、`hybrid search`、`reranker`、`Milvus`、`Qdrant`、`vLLM`、`VLM`、`model training`、`SFT`、`DPO`、`RL`、`PPO`、`GRPO`、`GSPO`、真实线上流量、医生盲评、临床验证和任何参考文档中的性能数字。

## M1 合并后的更新规则

另一个会话合并 M1 后，不要直接把本轮问题从 planned 改成 implemented。至少重新检查：新增源码路径和状态契约、action/tool schema、loop termination、预算/timeout、safety pre-gate 是否保持、citation/evidence 回归、M0.3 全量指标、每个 failure case 的新 top-k/route，以及新增 trace 的可复现性。只有代码、测试和评测都能支撑时，才把对应问题标为 [M1 IMPLEMENTED]。
