# Health-Copilot 项目面试追问

> 事实基线：`main@1bea0f37f91aaff7dead8e9a01acecfba360af3e`。
>
> 本文是项目面试追问地图，不是产品说明书。`[M0 已实现]`、`[M1 已实现]` 和 `[M2 PLANNED]` 只表示这个基线中能否回到代码、测试或已保存评测核对。M1 已经合并，但 M1 的评测是 focused diagnostic，不是临床验证，也不是泛化能力声明。

## 先记住的 30 秒版本

Health-Copilot 是一个面向患者教育的 Safety-Gated Evidence RAG vertical slice。医疗场景的价值不是把它包装成诊断系统，而是用高风险领域验证三件事：高危问题能否在模型调用前被拦截，回答能否只引用审核过的公开知识卡，证据不足或来源异常时能否拒答。

M0 先做了一个确定性的 workflow：输入校验 → safety gate → BM25 → Evidence → Generator → citation-ID 校验。M0.3 在 80 条案例上记录了 `safety_route_accuracy=1.0`；62 条有来源标注的患者教育案例上，`Hit@1=0.9032`、`Hit@3=0.9516`、`MRR=0.9274`。失败表显示了 3 个 lexical paraphrase mismatch 和 4 个 OOD false retrieval。

M1 没有把系统改成任意 ReAct，而是在 M0 的安全门和证据契约之内增加一个单 Agent、单恢复动作的运行时：初始检索后让模型选择直接回答或调用一次 `search_knowledge(query)`，工具结果作为 observation 回到第二个 model turn，最后仍由运行时验证 citation。M1 focused run 是 12 个 case、3 次 trial、36 条 trajectory；安全 case 在 Agent 之前短路，实际 Agent run 是 30 条。恢复诊断的 9/9 次命中预期 source，但 OOD case 仍有 11/12 次工具激活；它说明恢复动作能覆盖已知表达不匹配，却没有解决 answerability/evidence sufficiency。

当前不能声称临床效果、医学准确率、通用 autonomous agent、long-horizon planning、Multi-Agent、dense retrieval、reranker、训练或线上吞吐。后续 M2 只应从真实 OOD 和运行时约束问题继续推进。

## 面试前的事实纪律

- 当前已实现：M0 safety-gated Evidence RAG；M1 bounded single-agent recovery；OpenAI-compatible tool-calling adapter；离线 Agent mechanics tests；保存过的 M1 focused run。
- 当前未实现：Multi-Agent、Agent Swarm、长期 Memory、MCP、Sandbox、dense/hybrid retrieval、reranker、web search、VLM、流式 UI、并行工具、持久化 trace/replay、SFT、DPO、RL 和临床验证。
- `runs/m1/20260916T192210+0800/config.json` 的 `commit_sha` 是运行时 closeout `285b4af4e26009aca03883a38607744761d84bc7`；这是实验运行时的可复现版本，不应改写成合并 commit。最终仓库事实基线是 `1bea...`。
- M1 artifact 记录的 `model` 是 `deepseek-flash`，`base_url` 是 `https://api.deepseek.com`；仓库没有保存更细的服务端版本信息，因此不把别的版本名当作本仓库事实，也不把结果推广到所有模型。
- `safety_route_accuracy` 只评估 safety gate；它不是 end-to-end route accuracy，也不是医学准确率。

## 第一轮：你先介绍一下这个项目

### Q1：这个项目解决什么问题？

#### 30 秒回答

我想解决的是患者教育问答里的三个工程问题：高危问题不要进入模型，普通回答要能回到审核过的证据，证据或引用不可靠时要拒答。于是我先做了一个确定性的 Evidence RAG，再用 M0 的失败案例驱动 M1：只给模型一次受限的检索恢复机会。

#### 深挖

这里的“解决”是把风险边界和失败处理做成可测试的代码契约，不是声称系统能诊断疾病。M0 的生成器只接收本次检索到的 `Evidence`，模型只返回 `citation_ids`；标题、摘要和 URL 由运行时从证据复制。M1 允许模型提议一个检索动作，但工具名、参数、调用次数和最终 citation 仍由 runtime 控制。

#### 源码落点

- `src/health_ai_copilot/pipeline.py`：`HealthCopilotPipeline.answer`、`_answer_with_agent`、`_response_from_draft`。
- `src/health_ai_copilot/safety.py`：`route_question`。
- `src/health_ai_copilot/retrieval/bm25.py`：`BM25Retriever`。
- `src/health_ai_copilot/verification/citations.py`：`verify_citations`。

#### 证据

M0 的结果来自 `evals/m0_failure_table.json` 和 `evals/baselines/m0.json`；M1 的结果来自 `runs/m1/20260916T192210+0800/metrics.json`，不是参考项目或简历中的外部数字。

#### 面试官继续追问

- 为什么医疗场景适合验证这个边界？
- 你说的“拒答”是模型自己说不确定，还是运行时强制拒答？

#### 当前边界

当前系统是患者教育原型，不接真实患者记录，不提供诊断、处方或治疗决策，不能把这组指标说成医疗质量。

### Q2：为什么选择医疗场景，而不是普通聊天机器人？

#### 30 秒回答

因为医疗场景会迫使我先回答“什么时候不应该回答”。在普通聊天 demo 里，模型答得流畅就很容易被当成成功；在这里，高危和处方请求必须在模型前被拦截，普通问题也必须有证据和合法 citation。这个场景让安全边界、数据 provenance 和 abstention 变成一等工程问题。

#### 深挖

医疗场景提高的是验证要求，不是系统能力。当前 safety policy 只是源码里的窄规则：`胸痛`、`呼吸困难` 等 urgent marker 路由 `URGENT_CARE`，`处方`、`剂量`、`停药` 等路由 `HUMAN_REVIEW`。这不是临床 triage，也没有医生团队或线上患者流量可以引用。

#### 源码落点

- `src/health_ai_copilot/safety.py`：`URGENT_MARKERS`、`PRESCRIPTION_MARKERS`、`route_question`。
- `tests/test_safety.py`、`tests/test_pipeline.py`、`tests/test_agent.py`：验证安全门先于 retriever、generator、Agent model。

#### 证据

M0 评测中 safety cases 是 76 条，`safety_route_accuracy=1.0`；M1 focused pack 每个 trial 有 1 个 urgent 和 1 个 prescription case，`safety_short_circuit_accuracy=1.0`。

#### 当前边界

marker policy 有漏报和误报风险，不等于医学风险分层；不能把 `1.0` 说成临床安全率。

### Q3：你现在真正完成了什么？哪些还没有完成？

#### 30 秒回答

我完成了 M0 和 M1。M0 是知识卡 loader、中文 tokenizer、BM25、Generator Protocol、OpenAI-compatible generator、safety pre-gate、结构化输出和 citation-ID 校验。M1 在它之上增加了 typed message、`AgentSession`、`AgentState`、`ToolRegistry`、单一 `search_knowledge` 工具、最多两次模型 turn 和一次工具调用的 `AgentLoop`，并保存了 focused run。

还没有完成的是 OOD answerability 判定、claim-level grounding、完整 Harness Runtime、持久化 trace/replay、dense/hybrid retrieval、reranker、Multi-Agent、训练和临床验证。

#### 源码落点

- `src/health_ai_copilot/agent/`：M1 typed runtime。
- `src/health_ai_copilot/tools/search_knowledge.py`：唯一产品工具。
- `src/health_ai_copilot/eval/m1.py`：阶段分离的 M1 指标。
- `.github/workflows/ci.yml`：ruff、pytest 和 M0 baseline assertion。

#### 证据

M1 合并 commit 是 `1bea0f3...`；M1 run 有 36 条 trajectory 和 0 条 hard failure。实现 closeout、运行 artifact 和 merge checkpoint 的关系见本文开头的事实纪律。

#### 当前边界

只有能回到当前代码和 artifact 的能力才算已实现。M2 是 `[M2 PLANNED]`，不能因为 M1 有 Agent loop 就声称有生产级 Harness、Memory 或 Sandbox。

### Q4：给我从用户输入开始完整走一次请求。

#### 30 秒回答

完整路径是：

```text
question
  → input validation
  → deterministic safety gate
      → urgent / prescription: terminal response, no retrieval, no model
      → normal question
  → initial BM25 retrieval
      → empty evidence: ABSTAIN, no Agent
  → M1 Agent turn
      → FinalTurn: citation verification
      → ToolCallTurn(search_knowledge): validate and execute once
          → ToolResult observation
          → second Agent turn
  → initial + recovery Evidence union
  → citation-ID verification
  → AssistantResponse / ABSTAIN
```

#### 深挖

M0 路径中 `Generator.generate(question, evidence)` 返回 `GenerationDraft`。M1 路径中，模型返回 `FinalTurn` 或 `ToolCallTurn`；工具执行结果不是模型自己拼接的文本，而是 typed `ToolResultMessage` observation。 如果模型给出未观察到的 citation、没有 citation、模型异常、预算耗尽或 runtime 没有得到可靠 final turn，就 fail closed。

#### 源码落点

- `pipeline.py`：输入、safety、initial retrieval、M0/M1 分流和最终响应。
- `agent/loop.py`：`AgentLoop.run` 的 turn/tool 计数和终止。
- `agent/messages.py`：`FinalTurn`、`ToolCallTurn`、`ToolResultMessage`。
- `verification/citations.py`：最终 ID 校验。

#### 证据

`tests/test_agent.py::test_one_recovery_executes_tool_then_calls_model_again` 验证了两次 model turn 和事件顺序；`test_safety_short_circuits_agent_and_tool_calls` 验证 safety pre-gate；`test_recovered_evidence_is_used_for_final_citation_verification` 验证恢复证据会进入最终校验。

#### 当前边界

“第二次模型 turn”不是长对话记忆，也不是多轮规划；M1 的硬上限是 2 个 model turn、1 个 tool call。

## 第二轮：第一版为什么不是 Agent，以及 M1 到底是什么

### Q5：为什么最开始做 deterministic workflow，而不是直接 ReAct？

#### 30 秒回答

因为第一步要先建立可解释的安全和证据 baseline。M0 的路径固定，任何失败都能定位到 safety、retrieval、generation 或 citation verification；如果一开始就让模型自由选择工具，很难知道是检索本身错了，还是策略、工具调用或终止错了。M0 的 failure table 给出真实失败后，才有理由引入一个很小的 Agent action。

#### 深挖

ReAct 的价值是模型可以根据 observation 调整下一步，但它同时扩大 action space、延迟、成本、非确定性和测试面。这里的 M1 只把“表达不匹配时换一个检索 query”纳入 action space；原始问题、审核知识卡和最终 citation 契约保持不变。

#### 源码落点

- M0：`src/health_ai_copilot/pipeline.py` 的 generator path。
- M1：`src/health_ai_copilot/agent/loop.py` 和 `tools/search_knowledge.py`。
- `evals/m0_failure_table.json`：M1 action 的 failure driver。

#### 证据

M0 观察到 3 个 synonym/paraphrase mismatch，而不是“所有问题都需要 Agent”。这就是 bounded recovery 的范围依据。

#### 面试官继续追问

- 为什么不直接做 query expansion？
- 你如何证明 Agent 没有绕过 safety gate？

#### 当前边界

M1 仍不是通用 autonomous medical agent；它是为一个已知 retrieval failure slice 加上的受限 runtime。

### Q6：当前系统为什么严格来说仍不是通用 Agent？

#### 30 秒回答

M1 已经具备“模型在显式状态和工具集合中选择下一步”的单 Agent loop，但它不是通用自主系统。模型只有一个只读工具，最多两次 turn、一次调用，不能改变安全路由、不能写外部状态、不能自行添加工具，也不能长程规划。因此我会说“实现了 bounded single-agent recovery”，不会说“实现了通用 Agent 平台”。

#### 深挖

Agent 的关键不在于调用了 LLM，而在于是否有 decision step 和可执行 action。M1 第一 turn 可以选择 `FinalTurn` 或 `ToolCallTurn`；runtime 验证工具和预算，工具结果回到 transcript，第二 turn 再决定 final/abstain。权限、状态和终止权仍在代码侧。

#### 源码落点

- `src/health_ai_copilot/agent/model.py`：`AgentModel.respond` 和 provider adapter。
- `src/health_ai_copilot/agent/loop.py`：受限控制流。
- `src/health_ai_copilot/agent/tools.py`：显式 registry。

#### 当前边界

没有 memory、parallel tools、planner、sub-agent、MCP、sandbox 或长期 session；这些不能从 M1 代码推导出来。

### Q7：M0/M1 哪里体现了 Harness Engineering？

#### 30 秒回答

它体现为“model proposes; runtime validates”。M0 已有 safety pre-gate、evidence-only context、citation verifier 和 fail-closed；M1 再把 tool schema、注册表、预算、结构化 observation、状态计数和 Agent lifecycle events 加进来。模型提议调用什么、用什么 query，不能决定是否越过 safety、能调用几次或哪些 citation 合法。

#### 深挖

这不是说项目已经有完整生产 Harness。当前 runtime 的可执行边界是有限的：`AgentLoopConfig` 只允许 `max_model_turns<=2`、`max_tool_calls<=1`；`ToolRegistry` 明确注册 `search_knowledge`；最终响应必须再次走 citation verification。持久化 trace、timeout、权限系统、重放和更广泛 policy 是 M2 方向。

#### 源码落点

- `agent/loop.py`：硬预算和终止。
- `agent/tools.py`：显式 capability registry。
- `pipeline.py`：pre-gate 和 fail closed。
- `agent/events.py`：元数据 lifecycle events。
- `.github/workflows/ci.yml`：离线测试和 M0 baseline regression gate。

#### 证据

M1 focused run 的 `budget_exhaustion_rate=0.0`、`citation_integrity_pass_rate=1.0`，但这只是在 36 条 trajectory 上成立；不能扩大为生产可靠性。

#### 当前边界

事件目前只在内存中，且不含问题、工具参数或回答内容；没有持久化 trace/replay、token/time budget 或 sandbox。

## 第三轮：检索真的有效吗？

### Q8：为什么第一版先选 BM25？

#### 30 秒回答

因为 M0 的知识库只有 30 张中文优先患者教育卡，第一步最需要的是确定性、可解释和离线可复现。BM25 不需要训练 embedding 或向量库，能直接观察 query term、TF/DF 和排序；这样 failure table 才能告诉我到底是词面表达问题，还是后面的回答/引用问题。

#### 深挖

BM25 擅长术语和词面重合，但不理解“只看一次”和“不同日期确认”是近似意图，也不能自动判断 OOD。选择它是 baseline，不是断言 sparse 永远优于 dense。

#### 源码落点

- `src/health_ai_copilot/retrieval/bm25.py`：直接实现 `_idf`、`_score` 和 deterministic sort。
- `src/health_ai_copilot/retrieval/tokenizer.py`：中文优先分词。
- `tests/test_bm25.py`：token、相关卡排序、空结果和 tie-breaking。

#### 证据

M0.3 的 62 个 retrieval cases 上 `Hit@3=0.9516`，但仍有 3 个 paraphrase miss 和 4 个 OOD false retrieval，说明总平均不能替代 failure slices。

#### 当前边界

当前没有 dense retrieval、hybrid search、ANN index 或 reranker；不能用这些名词描述 Health-Copilot 已实现能力。

### Q9：BM25 公式怎么来的？当前代码具体算什么？

#### 30 秒回答

当前实现对 query 中每个 token 累加：

```text
score(D, Q) = Σ IDF(t) · [ tf(t,D)·(k1+1) /
  (tf(t,D) + k1·(1-b+b·|D|/avgdl)) ]
IDF(t) = log((N-df(t)+0.5)/(df(t)+0.5)+1)
```

它用 `k1=1.5`、`b=0.75`，对标题、正文和 tags 拼接后统一 tokenize；只保留正分文档，并按分数降序、`card.id` 升序打破平局。

#### 深挖

`tf` 是 term 在一张卡里出现的次数，`df` 是包含 term 的卡片数，`N` 是卡片总数，`|D|/avgdl` 做长度归一化。IDF 让稀有词更有区分度，TF saturation 防止重复出现把分数无限推高。

#### 源码落点

`src/health_ai_copilot/retrieval/bm25.py` 的 `BM25Retriever.__init__`、`_idf`、`_score`、`search`。

#### 面试官继续追问

- 这里的 IDF 为什么加 `0.5` 和 `+1`？
- 你有没有做 title boost 或字段权重？

#### 当前边界

这是本项目的 BM25 变体，不应把其他库的 IDF、字段 boost 或默认参数混进答案；当前没有 query expansion。

### Q10：`k1` 和 `b` 分别控制什么？

#### 30 秒回答

`k1` 控制 TF 增长的饱和速度：越大，重复出现同一词还能带来更多增益；`b` 控制文档长度归一化：`b=0` 不做长度归一化，`b=1` 更充分惩罚相对长文档。当前值是 `k1=1.5`、`b=0.75`，是一个可解释的 baseline 配置，不是通过大规模 tuning 得出的最优值。

#### 深挖

长度归一化不是“短文档必然更相关”；它只修正词频在不同长度文档之间的可比性。中文分词改变文档 token 数，所以 `b` 的效果和 tokenizer 强相关。

#### 源码落点

- `BM25Retriever.__init__` 校验参数范围。
- `_score` 中的 `term_frequency + k1 * (...)`。
- `tests/test_bm25.py`。

#### 当前边界

没有参数 ablation 或统计显著性结论；如果继续做，应固定 eval pack，按 paraphrase、数字阈值和 OOD 分 slice 比较。

### Q11：tokenizer 为什么能显著改变 BM25？

#### 30 秒回答

BM25 比较的是 token，不是人的语义。中文没有天然空格，`jieba.lcut` 决定“高血压”“睡眠不足”是完整 token 还是碎片；query 和 KnowledgeCard 必须使用同一规则。当前 tokenizer 先做 NFKC 和 lowercase，再保留包含中文、ASCII 或数字的 token。

#### 深挖

分词既影响 TF/DF，也影响文档长度 `|D|` 和平均长度 `avgdl`。分得太碎，词的区分度下降；分得太粗，同义或变体更难共享词面。M0 的三个 paraphrase miss 不是把 tokenizer 当成唯一原因，而是表明纯 lexical matching 没有语义桥接。

#### 源码落点

- `src/health_ai_copilot/retrieval/tokenizer.py`：`tokenize`。
- `tests/test_bm25.py::test_chinese_queries_share_meaningful_tokens`。

#### 当前边界

当前没有医学词典、自定义 synonym table 或 embedding；不能声称 tokenizer 已解决同义词。

### Q12：为什么 BM25 score 不是 probability？

#### 30 秒回答

BM25 score 是用于排序的相关性分数。它没有被归一化成所有候选文档和为 1，也没有校准成“这条证据支持问题的概率”。当前代码只根据 `score > 0` 保留文档，所以正分只能说明有词面贡献，不能直接说明可回答。

#### 源码落点

- `BM25Retriever.search`：丢弃 `score <= 0`，不做 softmax 或 probability calibration。
- `contracts.py::Evidence.score`：保存 ranking score。

#### 面试官继续追问

- 能不能用 score threshold 解决 OOD？
- score 很高但 claim 不被支持时怎么办？

#### 当前边界

本项目没有经过标定的 evidence sufficiency threshold；M1 也没有把 BM25 score 当作 answerability classifier。

### Q13：Hit@K、Recall@K、MRR、nDCG 怎么区分？

#### 30 秒回答

在每个 query 有一个或多个相关文档时：

- `Hit@K`：top-K 是否至少命中一个相关文档，按 query 计 0/1。
- `Recall@K`：top-K 找回了多少相关文档，分母是该 query 的全部相关文档。
- `MRR`：第一个相关文档的倒数排名，第一名是 1，第二名是 0.5。
- `nDCG@K`：按相关性等级给位置折扣，适合多个相关文档和 graded relevance。

M0 当前 runner 计算的是 safety route accuracy、Hit@K 和 MRR；NFCorpus adapter 另外输出 Recall@K、MRR 和 nDCG@K。

#### 深挖

单相关 source 的 Product Eval 中 Hit@K 很直观，但它不等于回答质量；一个 query 可能有多个可接受 source，或者 top-1 命中但 excerpt 仍不支持具体 claim。指标分母和标注 schema 必须先说清楚。

#### 源码落点

- `src/health_ai_copilot/eval/runner.py`：M0 case evaluation。
- `src/health_ai_copilot/eval/nfcorpus.py`：NFCorpus 指标适配。
- `src/health_ai_copilot/eval/m1.py`：M1 的阶段指标。

#### 当前边界

这些是 retrieval metrics，不是 end-to-end medical accuracy，也不直接衡量 citation 的 semantic grounding。

### Q14：当前 M0.3 的真实结果是什么？

#### 30 秒回答

M0.3 frozen eval 有 80 条 reviewed cases，其中 76 条进入 safety-route evaluation，`safety_route_accuracy=1.0`；62 条患者教育 retrieval cases 的 `Hit@1=0.9032258064516129`、`Hit@3=0.9516129032258065`、`MRR=0.9274193548387096`。这组数字来自 `evals/baselines/m0.json`，不是医学准确率。

#### 深挖

安全分母是 76，不是全部 80；retrieval 分母是有 `expected_source_ids` 的 62 条。失败表明确记录 3 个 synonym/paraphrase failure 和 4 个 OOD false retrieval；其他 challenge type 在这个小样本中没有形成 failure case。

#### 源码落点

- `evals/m0_failure_table.json`：失败分类和 top-3。
- `evals/baselines/m0.json`：冻结数字。
- `src/health_ai_copilot/eval/runner.py`：指标定义。

#### 证据

失败 case 是 `m0-003`、`m0-005`、`m0-034`；OOD 是 `m0-077` 到 `m0-080`。`.github/workflows/ci.yml` 通过 `tools/check_m0_baseline.py` 对这些 M0 数字做回归断言。

#### 当前边界

80 条是内部小型 Product Eval，不支持泛化、临床有效性、医生替代或线上质量声明。

### Q15：标准 benchmark 呢？NFCorpus 和自建 Product Eval 有什么区别？

#### 30 秒回答

自建 Product Eval 贴近 Health-Copilot 的中文 KnowledgeCard、safety route 和真实 bad case；NFCorpus 是公开的 BEIR biomedical retrieval benchmark，有自己的 query、corpus 和 qrels。前者回答“本产品契约下的已知问题表现如何”，后者回答“在公开标准检索任务上如何”，不能混成一个数字。

#### 深挖

NFCorpus 适合比较 retrieval pipeline 的 Recall、MRR、nDCG，但它没有本项目的 urgent/prescription pre-gate，也不等于患者教育回答安全。Product Eval 的 OOD 和 paraphrase slice 也不应被 benchmark 平均值掩盖。

#### 源码落点

- `src/health_ai_copilot/eval/nfcorpus.py`：独立 adapter。
- `evals/m0.jsonl`、`evals/m0_failure_table.json`：产品评测。

#### 当前边界

仓库只记录了 Product Eval 和 benchmark adapter；当前文档不填未经运行核验的 NFCorpus 数字。

## 第四轮：真实 bad case 如何驱动迭代

### Q16：3 个 synonym / paraphrase mismatch 具体是什么？

#### 30 秒回答

M0 failure table 里的三个真实 miss 是：`m0-003` 用“只看一次”问高血压确认，目标卡写的是“不同日期测量确认”；`m0-005` 用“头晕或不舒服”问是否一定有症状，目标卡写的是“通常没有明显症状”；`m0-034` 用“睡不好”问风险，目标卡写的是“睡眠不足”。这些不是凭空设计的 Agent demo，而是 M0 top-3 没命中的表达差异。

#### 深挖

对应 top-3：

| case | 目标 source | M0 top-3 | failure 解释 |
| --- | --- | --- | --- |
| `m0-003` | `who-hypertension-02-confirmation` | `who-hypertension-03-silent`、`cdc-high-blood-pressure-managing-01-monitoring`、`cdc-high-blood-pressure-about-02-symptoms` | “只看一次”与“不同日期/确认”共享词面少 |
| `m0-005` | `who-hypertension-03-silent` | `cdc-high-blood-pressure-risk-03-alcohol`、`cdc-high-blood-pressure-risk-05-overweight`、`cdc-high-blood-pressure-risk-04-tobacco` | “头晕或不舒服”没有强匹配 symptom/no-symptom 词 |
| `m0-034` | `cdc-high-blood-pressure-prevention-04-sleep` | `cdc-high-blood-pressure-risk-05-overweight`、`cdc-high-blood-pressure-risk-04-tobacco`、`cdc-high-blood-pressure-risk-01-sodium` | “睡不好”和“睡眠不足”不共享足够词面，泛化 risk 词占优 |

#### 源码落点

- `evals/m0_failure_table.json` 的 `synonym_paraphrase_failure.observations`。
- M1 对应的 `evals/m1_recovery.jsonl`：`m1-001`、`m1-002`、`m1-003`。

#### 证据

这三个 case 组成 M1 的 `recovery_expected=true` slice；每个 trial 的恢复 top-3 都单独记录在 `runs/m1/.../trajectories.jsonl`。

#### 当前边界

这里只能说观察到词法表达不匹配，不能说这三个 case 覆盖了所有医学同义词。

### Q17：为什么 lexical retrieval 会出现这些问题？

#### 30 秒回答

BM25 的基本单位是 token。它知道某个 token 在多少文档里出现，却不知道“只看一次”和“不同日期确认”可能在同一个医学判断上相关，也不知道“睡不好”和“睡眠不足”是近义。共享词面不足时，目标卡得不到分数；“高血压、风险、影响”等泛词却可能把其他卡推上来。

#### 深挖

这不是 BM25 算错了，而是目标定义不同：BM25 做 lexical ranking，面试里我们关心的是语义相关性和可回答性。tokenizer 能缓解分词不一致，但不能凭规则覆盖开放式 paraphrase。

#### 源码落点

`retrieval/bm25.py` 的 `_score` 只对 query token 的 TF/IDF 累加；没有 synonym map、embedding 或 reranker。

#### 面试官继续追问

- 为什么 dense retrieval 可能帮助？
- dense 召回会不会引入新的 OOD 误召回？

#### 当前边界

当前没有 dense 实验，不能把“可能有价值”说成“已经提升”。

### Q18：为什么这些 failure 说明 dense retrieval 可能有价值？

#### 30 秒回答

因为目标卡和 query 可能没有完全相同的 token，但有相近语义。dense bi-encoder 可以把 query 和 card 编成向量，用相似度召回词面不同的内容。可是它也会引入 embedding 模型、中文医学领域适配、索引版本、阈值、延迟和 OOD 校准问题，所以应该按 M0 failure slice 做 BM25/dense/hybrid 对比，而不是只报平均值。

#### 深挖

dense 解决的是表示空间问题，不自动解决 evidence sufficiency；一个语义相近但不回答具体问题的卡仍可能被召回。医疗数字阈值和范围也需要检查 lexical exact match、scope 和 source conflict。

#### 源码落点

当前源码只有 `retrieval/bm25.py` 和 `eval/nfcorpus.py`；没有 embedding、向量库或 reranker 实现。

#### 当前边界

`dense retrieval`、`hybrid search` 和 `reranker` 都不是 Health-Copilot 已实现能力，属于后续实验方向。

### Q19：为什么没有立刻上 embedding，而是先做 bounded query recovery？

#### 30 秒回答

因为当前有一个更窄、可验证的假设：在已观察到的 3 个 paraphrase mismatch 上，如果模型把问题转换成更贴近知识卡的短 query，一次额外 BM25 可能找回目标 source。这个改动保留现有 retriever 和 citation 契约，复杂度比同时引入 embedding 服务、索引和阈值小，适合先做 focused diagnostic。

#### 深挖

这不是认为 query recovery 比 dense 更强，而是按失败驱动最小增量。M1 的工具不是 `rewrite_query` 加 `search` 两个工具，而是一个 `search_knowledge(query)`：模型直接提供检索 query，工具只读执行现有 BM25。

#### 源码落点

- `src/health_ai_copilot/tools/search_knowledge.py`：恢复检索工具。
- `src/health_ai_copilot/agent/model.py` 的 system prompt：允许最多一次更合适的 query。
- `src/health_ai_copilot/agent/loop.py`：一次 tool budget。

#### 证据

M1 recovery pack 保留了 3 条 paraphrase case、3 条 direct-hit control、4 条 OOD control 和 2 条 safety control，共 12 条 case。

#### 当前边界

M1 只验证了 focused recovery；没有和 dense/hybrid 做 ablation，也没有证明 query recovery 对新领域有效。

### Q20：4 个 OOD false retrieval 是什么？

#### 30 秒回答

M0 的 `m0-077` 到 `m0-080` 分别问高血压与全麻手术、乘飞机、脱发、商业保险。它们都不是当前知识包应回答的目标，但因为共享“高血压”等词，BM25 仍返回正分 top-3。failure table 把它们记录为 false retrieval，而不是把“有结果”当成功。

#### 深挖

top-3 分别是：

- `m0-077`：`cdc-high-blood-pressure-managing-01-monitoring`、`cdc-high-blood-pressure-risk-04-tobacco`、`who-hypertension-03-silent`。
- `m0-078`：`nhc-hypertension-day-03-home-monitoring`、`cdc-high-blood-pressure-managing-02-care-plan`、`who-hypertension-01-definition`。
- `m0-079`：`cdc-high-blood-pressure-risk-04-tobacco`、`cdc-high-blood-pressure-risk-05-overweight`、`nhc-hypertension-day-02-weight`。
- `m0-080`：`cdc-high-blood-pressure-prevention-01-diet`、`cdc-high-blood-pressure-managing-01-monitoring`、`cdc-high-blood-pressure-risk-04-tobacco`。

#### 源码落点

- `evals/m0_failure_table.json` 的 `ood_false_retrieval.observations`。
- `src/health_ai_copilot/retrieval/bm25.py`：正分就会进入 `Evidence`。

#### 证据

M1 将这类问题复制为 `m1-007` 到 `m1-010`，用三次 trial 观察 Agent 是否错误回答。

#### 当前边界

它们是当前小知识包里的 OOD controls，不是对所有医疗 OOD 的完整覆盖。

### Q21：BM25 明明“有结果”，为什么问题还是不应该回答？

#### 30 秒回答

“有结果”只说明至少有 query token 与卡片重合，不说明证据支持用户的具体 claim。比如“高血压能不能坐飞机”拿到的是饮食或测量卡，就不能把一般高血压教育拼成航空医学结论。系统需要区分 retrieval hit、evidence sufficiency 和 answerability。

#### 源码落点

- `Evidence.score` 只保存排名分数。
- `pipeline.py` 当前只在空证据、模型 abstain 或 citation 不合法时拒答。
- `verification/citations.py` 明确声明不做 semantic entailment。

#### 面试官继续追问

- M1 是否已经实现 evidence sufficiency classifier？
- 你准备如何构造负例？

#### 当前边界

evidence sufficiency、answerability 和 claim-level grounding 目前没有实现；M1 的 OOD abstain 来自模型在当前 observation 下选择 abstain，不是独立的证据充分性判定器。

### Q22：query rewrite 能解决 OOD false retrieval 吗？

#### 30 秒回答

不能简单解决，甚至可能扩大错误召回。rewrite 如果把“全麻手术”压成泛化的“高血压管理”，会让 BM25 更容易命中高血压卡，却更远离用户真正的问题。它适合修复已知表达不匹配，不等于 answerability 判断。

#### 深挖

恢复策略至少要区分：初始证据没命中但问题属于知识包；初始证据命中但 claim 不被支持；问题本来就是 OOD。M1 只有 bounded recovery 和最终 abstain，没有显式的意图保持检查或 evidence sufficiency policy，所以诊断中要单独报告 OOD tool activation。

#### 证据

M1 的 OOD 指标是 `ood_tool_activation_rate=0.9166666666666666`、`ood_answer_rate=0.0`、`ood_abstain_rate=1.0`。这代表安全结果尚可，但 action policy 过于愿意尝试工具。

#### 当前边界

answerability classifier、query intent preservation、retrieval threshold calibration 和 selective answering 是 `[M2 PLANNED]` / 后续实验，不能写成 M1 已解决。

### Q23：这如何自然引出 evidence sufficiency、abstention 和 selective answering？

#### 30 秒回答

检索系统不是必须对每个问题给答案。selective answering 的目标是只在证据足够且风险可接受时回答，其余拒答或转人工。当前代码已经有 fail-closed 的 `ABSTAIN` 结果，但还没有独立的“证据是否足够”判定；M1 只证明模型在 OOD focused cases 上最终没有回答。

#### 深挖

可以把决策拆成：`retrieval relevance`、`evidence sufficiency`、`claim grounding`、`safety route`。这四个维度不能用一个 BM25 score 代替。后续评测应该记录 tool activation、answer、abstain、unsupported claim 和 calibration，而不只记录 top-K。

#### 源码落点

- `pipeline.py::abstain_response`：已有拒答 contract。
- `verification/citations.py`：只有 citation-ID integrity。
- `src/health_ai_copilot/eval/m1.py`：分别计算 OOD tool/answer/abstain rates。

#### 当前边界

M1 没有 answerability model、claim-level verifier 或 calibrated abstention threshold。

## 第五轮：M1 Agent Core 是怎样工作的

### Q24：M1 到底增加了什么真实行为？

#### 30 秒回答

M0 在初始 BM25 后直接进入 generator。M1 在初始 evidence 非空后启动 `AgentLoop`：第一轮模型可以直接给 `FinalTurn`，也可以给一个原生 `ToolCallTurn`；只有注册的 `search_knowledge(query)` 能执行。工具结果变成 `ToolResultMessage` observation，再进行第二轮模型调用，最后用 initial evidence 和 recovery evidence 的 union 做 citation verification。

#### 深挖

恢复 query 不是 runtime 根据 case ID 硬编码的字符串，而是 provider tool call 的参数。工具包装现有 retriever，返回 compact data 和 `observed_evidence`。运行时同时保留 ranking stage 和去重后的 union，避免把 recovery 结果混进初始指标。

#### 源码落点

- `agent/loop.py::AgentLoop.run`。
- `tools/search_knowledge.py::SearchKnowledgeTool`。
- `pipeline.py::_answer_with_agent`。
- `agent/messages.py` 的 `FinalTurn`、`ToolCallTurn`、`ToolResultMessage`。

#### 证据

`tests/test_agent.py::test_one_recovery_executes_tool_then_calls_model_again`、`test_recovered_evidence_is_used_for_final_citation_verification`；M1 trajectories 保存 `model_turns_used`、`tool_calls_used` 和三种 evidence list。

#### 当前边界

只有一个只读检索工具；没有任意 web search、写操作、计划器或多 Agent。

### Q25：为什么没有单独的 `rewrite_query` 工具？

#### 30 秒回答

因为 M1 的真实 action 是“用更合适的 query 做一次检索”，不是把 rewrite 文本当成独立产物。让模型直接调用 `search_knowledge(query)` 少一个工具和一层 schema，工具边界更小；runtime 仍能校验 query 非空、只允许 `query` 字段，并记录检索结果。

#### 深挖

如果把 rewrite 和 search 拆开，就要决定 rewrite 是否必须经过额外的意图保持检查、谁负责把 rewrite 交给 retriever、如何处理 rewrite 成功但 search 失败。M1 暂时不引入这些分支，但代价是 query recovery 的意图约束较弱，这正是 OOD tool activation 偏高的原因之一。

#### 源码落点

- `SearchKnowledgeTool.spec`：唯一 model-facing tool schema。
- `SearchKnowledgeTool.validate_arguments`：只允许非空 `query`。
- `AgentModel` system prompt：允许最多一次 `search_knowledge`。

#### 当前边界

没有独立 rewrite artifact、rewrite quality metric 或 intent preservation verifier。

### Q26：`AgentSession`、`AgentState`、`AgentRunResult` 分别是什么？Session 是 Memory 吗？

#### 30 秒回答

`AgentSession` 是一次运行的 typed transcript，保存 `UserMessage`、assistant tool call 和 `ToolResultMessage`；它不是长期记忆。`AgentState` 是这次 bounded run 的可变执行状态，包括 model/tool 计数、stop reason、draft 以及 initial/recovery/observed evidence。`AgentRunResult` 把 state 和内存事件一起返回给 pipeline。

#### 深挖

把 transcript 和 execution state 分开，能避免把“消息历史”误当成“预算和状态”。当前 `AgentSession` 没有跨进程 persistence、retrieval memory 或用户画像；每次 pipeline answer 都是一次 run。

#### 源码落点

- `src/health_ai_copilot/agent/session.py::AgentSession`。
- `src/health_ai_copilot/agent/state.py::AgentState`。
- `src/health_ai_copilot/agent/loop.py::AgentRunResult`。
- `tests/test_agent.py::test_session_distinguishes_transcript_from_execution_state`。

#### 证据

`AgentState` 当前字段包括 `model_turns_used`、`tool_calls_used`、`stop_reason`、`final_draft`、`initial_ranked_evidence`、`recovery_ranked_evidence` 和 `observed_evidence`。

#### 当前边界

“session”在这里是运行级容器，不可作为实现了 Memory 的简历 claim。

### Q27：AgentLoop 的消息和响应协议是怎样的？

#### 30 秒回答

输入消息是 `UserMessage`；模型响应只能落到 `FinalTurn` 或 `ToolCallTurn`。前者转为 `AssistantFinalMessage`，后者转为 `AssistantToolCallMessage`；工具执行后追加 `ToolResultMessage`。OpenAI-compatible adapter 把这些 typed objects 映射成 system/user/assistant/tool provider messages，再把原生 tool call 映射回 typed `ToolCall`。

#### 深挖

最终 JSON 只接受 `answer`、`citation_ids`、`abstain` 这类结构化 draft；工具调用不靠解析“请帮我搜索……”这类 prose。provider 返回 malformed JSON 或空 message 时，adapter 抛 `AgentModelError`，loop 把它转成 `MODEL_ERROR`，pipeline 再 fail closed。

#### 源码落点

- `agent/messages.py`：typed message/turn。
- `agent/model.py::OpenAICompatibleAgentModel.respond`、`_provider_messages`、`_provider_tool`。
- `generation/openai_compatible.py`：复用 draft JSON parser。
- `tests/test_agent_adapter.py`：tool call、tool result、final JSON 和 malformed input。

#### 当前边界

provider adapter 针对 OpenAI-compatible Chat Completions 形状；没有通用 provider abstraction、streaming 或 retry。

### Q28：为什么要显式 `ToolRegistry`？

#### 30 秒回答

显式 registry 把 capability 列表、名字、schema 和执行边界集中管理。M1 只注册 `search_knowledge`，重复注册直接报错；模型拿到的是 `ToolSpec`，runtime 再查名字、校验参数、捕获异常并返回结构化 `ToolResult`。这样模型不能通过目录扫描或名字猜测获得额外能力。

#### 深挖

当前 schema validator 只实现 M1 需要的 object/string 子集：required、additionalProperties 和 string type；代码明确写了不是完整 JSON Schema。显式边界的代价是以后增加工具要手动注册和测试，但换来了可审计的 capability surface。

#### 源码落点

- `agent/tools.py::ToolSpec`、`ToolResult`、`ToolRegistry`、`_validate_against_schema`。
- `tests/test_agent.py::test_registry_returns_structured_errors_for_unknown_and_malformed_calls`。
- `tests/test_agent.py::test_registry_rejects_duplicate_tool_names`。

#### 证据

unknown tool 返回 `error.code=unknown_tool`；空 query 返回 `invalid_arguments`；两者都不会调用 retriever。

#### 当前边界

没有插件发现、权限模型、版本协商或完整 JSON Schema 支持。

### Q29：工具失败怎么办？为什么不是立即终止？

#### 30 秒回答

工具失败会被 registry 转成 `ToolResult.failure`，再追加为 `ToolResultMessage` observation；它不是一个 terminal `StopReason`。第二次模型可以看到错误并返回 `abstain=true`。如果最终没有安全的 final turn，AgentLoop 会因为 model error、budget 或其他受控原因停止，pipeline 统一 fail closed。

#### 深挖

这样区分“工具失败”和“整个 run 立即崩掉”：模型仍有一次机会把外部失败解释为不能可靠回答。但 observation 不是成功证据，失败结果不会加入 `observed_evidence`。M1 没有 retry，因为只有一次 tool budget，且没有证明重试同一个只读查询能改善结果。

#### 源码落点

- `agent/tools.py::ToolRegistry.execute`：参数和执行异常转 `ToolResult`。
- `agent/loop.py`：追加 `ToolResultMessage` 后继续 turn。
- `tests/test_agent.py::test_tool_exception_becomes_observation_and_second_turn_can_abstain`。

#### 当前边界

没有 timeout、retry、backoff、circuit breaker、fallback provider 或持久化错误 trace；这些是后续 Harness 工作，不要写成 M1 已完成。

### Q30：为什么是 `max_model_turns=2`、`max_tool_calls=1`？

#### 30 秒回答

这是由 failure scope 推出来的最小预算：一次初始判断/动作，最多一次检索恢复，再一次基于 observation 的回答。`AgentLoopConfig` 对模型 turn 上限和工具上限做硬校验；如果预算耗尽却没有可靠 final，就返回 `ABSTAIN`，不会为了“完成回答”再放宽预算。

#### 深挖

预算同时限制成本、延迟和循环风险，但它不是正确性的证明。M1 的 one-step recovery 不需要长程 planner；把预算固定在代码里，才能在 eval 和 CI 中断言 termination。配置对象允许小于默认值，但不能超过 M1 的 2/1 上限。

#### 源码落点

- `agent/loop.py::AgentLoopConfig.__post_init__`。
- `AgentLoop.run` 的 `while state.model_turns_used < ...` 和 tool-call budget 判断。
- `tests/test_agent.py::test_second_tool_call_is_not_executed_and_fails_closed`。

#### 证据

M1 run 的 `max_model_turns=2`、`max_tool_calls=1`；`budget_exhaustion_rate=0.0`。测试还验证第二个 tool call 不会执行，route 是 `ABSTAIN`、reason 是 `max_tool_calls`。

#### 当前边界

没有 token budget、time budget、deadline、cost accounting 或 progress signal；M2 才讨论更完整的预算策略。

### Q31：如果模型连续两次返回 tool call，系统怎么处理？

#### 30 秒回答

第一次调用已经把 `tool_calls_used` 加到 1。第二轮再返回 tool call 时，`state.tool_calls_used + len(response.tool_calls) > max_tool_calls`，loop 不执行第二次工具，停止为 `MAX_TOOL_CALLS`，pipeline 返回 `ABSTAIN`。它不会偷偷把第二次调用改成回答。

#### 源码落点

`agent/loop.py::AgentLoop.run` 的 `max_tool_calls` 判断；`tests/test_agent.py::test_second_tool_call_is_not_executed_and_fails_closed`。

#### 当前边界

M1 的响应类型可以承载一个 list of tool calls，但产品配置不允许并行或多次执行；这不是已经支持 parallel tool calls。

### Q32：安全门如何保证 Agent 不能覆盖？

#### 30 秒回答

`HealthCopilotPipeline.answer` 先做 input validation 和 `route_question`，只有返回 `None` 才进行 initial retrieval 和 Agent loop。因此 urgent 或 prescription 请求的 retriever、Agent model、tool 都不会被调用。Agent 没有一个“允许忽略安全门”的 tool 或状态分支。

#### 源码落点

- `pipeline.py::answer`：safety gate 位于 retrieval 之前。
- `safety.py::route_question`：确定性 marker policy。
- `tests/test_agent.py::test_safety_short_circuits_agent_and_tool_calls`。

#### 证据

M1 focused run 中安全 case 每 trial 2 条、共 6 条 short-circuit；metrics 记录 `safety_short_circuit_accuracy=1.0`，且 trajectory 的 `agent_ran=false`、model/tool count 为 0。

#### 当前边界

安全门是窄规则原型，不覆盖所有紧急表达；指标只证明这组标注案例的 pre-gate 行为。

### Q33：initial、recovery、observed evidence 为什么要分开？

#### 30 秒回答

initial ranking 是用户原问题的 BM25 top-k；recovery ranking 是工具 query 的 top-k；observed evidence 是两者按 `source_id` 去重后的 union。最终 citation verifier 可以看 union，但 recovery metric 只能看 recovery ranking，不能把初始命中算成恢复成功。

#### 深挖

如果只保存一个 evidence list，`post_recovery_hit` 会混淆“本来就命中”和“工具找回”；这会夸大 Agent 的贡献。state 还保留 first-seen order，恢复结果若与初始 source 重复，不会产生两个 citation metadata。

#### 源码落点

- `agent/state.py::add_evidence`、`add_recovery_evidence`。
- `agent/loop.py::AgentRunResult` properties。
- `eval/m1.py::_initial_hit`、`_recovery_hit`、`_initial_miss_and_recovery_hit`。
- `pipeline.py::_response_from_draft`。

#### 证据

`tests/test_eval.py::test_m1_metrics_keep_initial_and_recovery_stages_separate` 明确断言 recovery case 的初始和恢复 source 分开；`test_observed_evidence_is_deduplicated_by_source_id` 断言 union 去重。

#### 当前边界

Evidence union 仍不等于 semantic grounding；同一个 source 被观察到，不代表它支持回答里的每个 claim。

### Q34：M1 的事件能不能算 trace/replay？

#### 30 秒回答

当前只有轻量的内存 lifecycle events：`agent_start`、`turn_start`、`model_response`、`tool_start`、`tool_end`、`turn_end`、`agent_end`。事件只带 session、turn、tool name、成功状态和 stop reason 等元数据，不携带问题、工具参数或回答内容。它帮助测试顺序，但还不是持久化 trace/replay 系统。

#### 源码落点

- `src/health_ai_copilot/agent/events.py`：`AgentEventType`、`AgentEvent`。
- `agent/loop.py`：事件发射和 observer 异常隔离。
- `tests/test_agent.py::test_one_recovery_executes_tool_then_calls_model_again`。

#### 当前边界

没有耗时、token、成本、完整输入输出、脱敏策略、持久化、replay ID 或跨进程查询；M2 planned 的 Harness Runtime 才覆盖这些问题。

## 第六轮：M1 评测到底证明了什么

### Q35：M1 focused eval 怎么设计？为什么是 12 × 3？

#### 30 秒回答

M1 eval pack 有 12 个 reviewed cases：3 个需要 recovery 的 paraphrase、3 个 direct-hit control、4 个 OOD false-retrieval control、1 个 urgent、1 个 prescription。每个 case 跑 3 次，共 36 条 trajectory；安全 case 在 Agent 前短路，所以真正进入 Agent loop 的是 30 条。

#### 深挖

这是针对一个假设的 diagnostic，不是通用 benchmark。三次 trial 用来观察 live provider 的重复行为；metrics 同时报告 initial、recovery、post-recovery、tool activation、route、预算和 citation integrity，避免只看最终 answer。

#### 源码落点

- `evals/m1_recovery.jsonl`：12 个 case 和 category。
- `src/health_ai_copilot/eval/m1.py::summarize_m1_runs`：阶段指标。
- `runs/m1/20260916T192210+0800/config.json`：trials、模型和预算。
- `runs/m1/.../trajectories.jsonl`：逐条轨迹。

#### 证据

artifact 的 `pack_case_count=12`、`trial_count=3`、`trajectory_count=36`、`run_cases=30`。

#### 当前边界

样本很小，模型和知识包固定，没有跨模型、跨领域、长期运行或线上流量证据。

### Q36：M1 的真实指标是什么？

#### 30 秒回答

直接引用 `runs/m1/20260916T192210+0800/metrics.json`：`initial_hit@3=0.5`，`recovery_attempt_rate=1.0`，`recovery_success@3=1.0`，`post_recovery_hit@3=1.0`，`unnecessary_recovery_rate=0.0`；OOD 的 `ood_tool_activation_rate=0.9166666666666666`、`ood_answer_rate=0.0`、`ood_abstain_rate=1.0`；`expected_answer_rate=1.0`、`unexpected_abstain_rate=0.0`、`safety_short_circuit_accuracy=1.0`、`citation_integrity_pass_rate=1.0`。

另外，`mean_model_turns=1.6666666666666667`、`mean_tool_calls=0.6666666666666666`、`budget_exhaustion_rate=0.0`，hard acceptance failures 是 0。

#### 深挖

`initial_hit@3=0.5` 的分母是 18 个 expected-source trajectory；`recovery_success@3=1.0` 的分母是 9 个 recovery-expected trajectory。它不应该被简化成“Agent retrieval accuracy 100%”。准确说法是：在 3 个 frozen paraphrase recovery case × 3 trials 的 focused diagnostic 中，9/9 recovery attempts 的 recovery top-3 命中预期 source。

#### 源码落点

- `runs/m1/20260916T192210+0800/metrics.json`。
- `src/health_ai_copilot/eval/m1.py`：分母和 stage semantics。
- `tests/test_eval.py`：指标定义和安全分母测试。

#### 面试官继续追问

- 为什么 initial_hit 只有 0.5？
- recovery success 为什么不能叫 end-to-end retrieval accuracy？

#### 当前边界

这次 run 是 DeepSeek 配置下的 focused regression/diagnostic；不是临床准确率、不是 general Agent benchmark，也不证明 OOD evidence sufficiency。

### Q37：为什么说 recovery success 不是“Agent retrieval accuracy 100%”？

#### 30 秒回答

因为指标只在预先指定的 9 条 recovery-expected trajectory 上计算，并且要求 initial top-3 先 miss、调用过 `search_knowledge`、recovery top-3 命中 expected source。它回答的是“针对这三个已知 paraphrase slice 的恢复动作是否命中”，不回答所有 query 的检索准确率。

#### 源码落点

`eval/m1.py::_initial_miss_and_recovery_hit` 同时检查 initial miss、tool attempted 和 recovery hit；`_recovery_hit` 只看 `recovery_ranked_evidence[:3]`。

#### 证据

`recovery_expected_cases=9`、`recovery_success@3=1.0`；这 9 条来自 3 case × 3 trials。

#### 当前边界

没有大规模随机 query、跨知识包、跨模型或与 dense retrieval 的对照，因此不能使用“100% 检索准确率”这样的表述。

### Q38：M1 的 OOD 结果应该如何解读？

#### 30 秒回答

4 个 OOD case × 3 trials = 12 条 OOD trajectory，其中 11 条激活了 `search_knowledge`，所以 `ood_tool_activation_rate=11/12=0.9166666666666666`；但 `ood_answer_rate=0.0`、`ood_abstain_rate=1.0`。这说明当前策略对 OOD 过于愿意尝试恢复，但最终没有把不相干证据变成回答。

#### 深挖

后半段是安全结果，前半段暴露了 policy 问题。不能因为最终全 abstain 就说 OOD 已解决：工具调用本身有成本和风险，且当前 abstain 主要来自模型的最终选择，没有独立 answerability verifier。下一步要测工具激活条件、证据充分性和错误拒答，而不是继续增加自由度。

#### 源码落点

- `runs/m1/.../metrics.json`：OOD 三项 rates。
- `runs/m1/.../trajectories.jsonl`：每条 OOD 的 initial/recovery/observed evidence 和 final route。
- `src/health_ai_copilot/eval/m1.py`：`ood_tool_activation_rate`、`ood_answer_rate`、`ood_abstain_rate`。

#### 当前边界

OOD 只覆盖 4 个 frozen control；没有 general OOD detector、calibration 或 claim-level checker。

### Q39：M1 证明了什么，没证明什么？

#### 30 秒回答

它证明了 bounded model → tool → observation → model loop 能运行，工具 contract 会校验，2/1 硬预算会终止，safety pre-gate 仍然生效，recovery evidence 能进入 citation verifier，provider-native tool call adapter 可映射，M0 baseline 没有被破坏。它没有证明通用 Agent 能力、临床疗效、OOD evidence sufficiency、claim-level grounding、long-horizon planning 或 Multi-Agent 优势。

#### 证据

M1 run 的 hard failure records 为 0，M0 baseline 文件仍是 80/76/62 和原有三项 retrieval 数字；CI 还会重放 M0 baseline。

#### 当前边界

“M1 已实现”指这些具体机制已存在并有测试/运行证据，不等于所有 Agent/医疗问题已解决。

## 第七轮：Evidence、Citation 与安全输出

### Q40：KnowledgeCard、Evidence、GenerationDraft、Citation 各是什么？

#### 30 秒回答

`KnowledgeCard` 是经过 schema 校验、带 provenance 的存储知识单元；`Evidence` 是某次 retrieval 的结果，带 source、excerpt 和 ranking score；`GenerationDraft` 是模型允许返回的 answer、citation IDs 和 abstain；`Citation` 是运行时从实际 Evidence 复制的对外来源 metadata。它们分开是为了不让模型自己发明来源。

#### 源码落点

- `src/health_ai_copilot/contracts.py`：四个 dataclass。
- `src/health_ai_copilot/knowledge/loader.py`、`schema.py`：KnowledgeCard 校验。
- `pipeline.py::_response_from_draft`：从 Evidence 建立 Citation。

#### 当前边界

`Evidence.score` 是 ranking score，不是概率；`Citation` 的存在也不代表 claim 被语义蕴含。

### Q41：为什么模型只返回 citation IDs，不能自己返回 URL？

#### 30 秒回答

因为 URL、title 和 excerpt 属于受信任的知识卡 metadata，不应由模型自由生成。模型只返回 `citation_ids`；runtime 检查 ID 是否属于本次 observed evidence，再从对应 `Evidence` 复制 metadata。这样可以防 fabricated URL，也让来源版本集中在 loader/card 数据里。

#### 源码落点

- `agent/model.py` 的 system prompt 和 `_provider_messages`。
- `verification/citations.py::verify_citations`。
- `pipeline.py::_response_from_draft`。
- `tests/test_pipeline.py::test_fabricated_source_id_returns_abstain`。

#### 当前边界

只控制来源完整性，不控制回答文本是否真的支持每一个 claim。

### Q42：fabricated citation 怎么防？

#### 30 秒回答

`verify_citations` 先去重模型返回的 IDs，再和本次 Evidence 的 `source_id` 集合比较。非 abstain 回答没有 citation 会返回 `missing_citation`；出现未观察到的 ID 会返回 `invalid_citation`；pipeline 看到 invalid result 就 route 到 `ABSTAIN`。Citation metadata 不信任模型，而是从 Evidence 复制。

#### 证据

`tests/test_citations.py` 覆盖合法、伪造、重复和缺失 citation；`tests/test_agent.py::test_unobserved_source_is_rejected_after_recovery` 验证 recovery 后仍不能引用未观察 source。M1 focused run 的 `citation_integrity_pass_rate=1.0` 是在该 run 的 answer attempts 上统计的完整性结果。

#### 当前边界

合法 ID 只能证明 source 被观察到，不能证明 claim 与 excerpt 之间有 entailment。

### Q43：citation integrity 和 semantic grounding 有什么区别？

#### 30 秒回答

integrity 问的是“这个 ID 是否属于本次实际观察到的证据”；semantic grounding 问的是“回答中的具体 claim 是否被该证据支持”。例如模型看到合法的 `who-hypertension-03-silent`，却回答“只要不头晕就一定没有高血压”。ID 合法，但 excerpt 只支持“高血压常常没有明显症状”，并不支持这个反向结论。

#### 源码落点

- `verification/citations.py` 的 docstring 明确写的是 citation integrity，不做 semantic entailment。
- `pipeline.py::_response_from_draft` 只调用 `verify_citations`。

#### 当前边界

claim-level grounding、entailment verifier 和 evidence sufficiency 都没有实现；这是重要的 `[M2 PLANNED]` 边界。

## 第八轮：工程实现追问

### Q44：为什么 Generator 和 AgentModel 用 Protocol？为什么测试用 Fake？

#### 30 秒回答

Protocol 让 pipeline 依赖行为契约而不是某个 provider SDK：M0 要一个 `generate`，M1 要一个 `respond`。测试可以注入 `FakeGenerator`、`FakeAgentModel`、`SpyRetriever`，精确控制 immediate final、recovery、异常和预算，不需要 API key 或网络。

#### 源码落点

- `generation/base.py::Generator`。
- `agent/model.py::AgentModel`。
- `tests/test_pipeline.py`、`tests/test_agent.py` 的 Fake/Spy classes。

#### 证据

M1 mechanics tests 可以验证模型调用次数、工具参数、事件顺序和 evidence union；它们与 live DeepSeek focused run 分开，避免把网络可用性当成单元测试。

#### 当前边界

Protocol 不会自动提供 provider compatibility、重试或 schema 兼容；这些仍由 adapter 和 runtime 负责。

### Q45：为什么用结构化 JSON output？

#### 30 秒回答

因为 runtime 需要可靠区分 final answer、citation IDs 和 abstain，也需要区分 final turn 与 tool call。M1 adapter 对 final content 使用 JSON object parser；malformed JSON 直接变成 `AgentModelError`，不会靠字符串正则猜测模型意图。

#### 源码落点

- `agent/model.py::respond`：`response_format={"type":"json_object"}` 和 `FinalTurn` 解析。
- `generation/openai_compatible.py`：`_parse`。
- `tests/test_agent_adapter.py::test_openai_adapter_rejects_malformed_final_json`。

#### 当前边界

JSON valid 不等于内容 truthful；schema 约束不能替代 evidence、citation 和 safety verifier。

### Q46：为什么配置放环境变量，而不是把 key 写在 `config.py`？

#### 30 秒回答

API key 是部署 secret，不应进入源码、commit 或文档。`load_openai_config` 从环境变量读取 key、base URL、model 和 temperature；测试可以构造 fake client 或 fake model，不需要真实凭据。

#### 源码落点

- `src/health_ai_copilot/config.py`：配置加载。
- `src/health_ai_copilot/generation/openai_compatible.py`、`agent/model.py`：provider 初始化。
- `runs/m1/.../config.json`：只记录 `configured-but-not-recorded`，不保存 key。

#### 当前边界

环境变量解决 secret 不入库，不等于完整 secret management；没有声称有 Vault、KMS 或生产部署。

### Q47：为什么用 `src` layout？editable install 和 `egg-info` 是什么？

#### 30 秒回答

`src` layout 让测试和运行必须通过安装后的 import path 找到包，减少“从仓库根目录误导入同名目录”的风险。editable install 会把环境中的包指向工作树，所以改源码后不用反复复制；`*.egg-info` 是构建后端记录 package metadata、依赖和文件清单的生成目录，不是业务源码。

#### 源码/工程落点

- `pyproject.toml`：package source、editable install 和 dev dependencies。
- `src/health_ai_copilot/`：实际包路径。
- `.github/workflows/ci.yml`：`pip install -e ".[dev]"` 后运行 ruff/pytest。

#### 当前边界

editable install 只影响本地开发/测试导入，不代表部署方式；`egg-info` 是否出现取决于安装工具，不是运行时能力。

### Q48：unit test、integration test、eval 有什么区别？

#### 30 秒回答

unit test 检查局部契约，例如 tokenizer、BM25、citation verifier、safety marker；integration-style test 把 pipeline、retriever、fake model 和 tool registry 拼起来，检查边界顺序和状态；eval 用带标注的 case pack 和指标回答“在这个固定数据集上表现如何”。三者都不能互相替代。

#### 源码落点

- `tests/test_bm25.py`、`test_citations.py`、`test_safety.py`：局部契约。
- `tests/test_pipeline.py`、`test_agent.py`、`test_agent_adapter.py`：跨组件行为。
- `evals/m0.jsonl`、`evals/m1_recovery.jsonl`、`src/health_ai_copilot/eval/`：评测。

#### 证据

`.github/workflows/ci.yml` 在离线测试后执行 M0 baseline replay，防止代码“没崩”但检索指标回退。

#### 当前边界

当前没有 human preference eval、LLM judge calibration、线上监控或真实患者数据评测。

### Q49：为什么 fail closed？retrieval/LLM/tool 出错怎么办？

#### 30 秒回答

医疗教育场景里，无法证明可靠比输出未经支持的回答更安全。retrieval exception → `retrieval_error`；generator/model exception → `generation_error` 或 `model_error`；tool exception → structured observation，后续没有可靠 final 就 abstain；缺 evidence、缺 citation、fabricated citation、预算耗尽也都不强行回答。

#### 源码落点

- `pipeline.py::abstain_response` 和 `answer` 的边界异常处理。
- `agent/tools.py::ToolRegistry.execute`。
- `agent/state.py::StopReason`。
- `tests/test_pipeline.py`、`tests/test_agent.py` 的 error cases。

#### 当前边界

fail closed 不等于系统永不误拒答，也不等于已有完整 reliability engineering。timeout、retry、fallback、circuit breaker 和 operator escalation 仍是 `[M2 PLANNED]`。

### Q50：知识卡 loader 为什么严格？

#### 30 秒回答

因为 Evidence 和 Citation 的可信度从数据源开始。loader 校验 JSON、必填 provenance、版本、审核字段、URL、重复 ID 和 schema；测试使用 synthetic cards，不把它们说成医学资料。没有通过 loader 的卡不能进入检索。

#### 源码落点

- `src/health_ai_copilot/knowledge/schema.py`。
- `src/health_ai_copilot/knowledge/loader.py`。
- `tests/test_knowledge_loader.py`。

#### 当前边界

schema/provenance 校验不等于医学内容事实核验；当前知识卡是公开来源的人工整理原型，没有声称有临床审核流程。

### Q51：`safety_reasons` 为什么是技术债？

#### 30 秒回答

为了保持已有 `AssistantResponse` contract，当前字段同时承载 safety reason 和 pipeline status reason，例如 `urgent_marker:胸痛`、`retrieval_error`、`invalid_citation`。这会混淆两个 failure domain；M1 为兼容性继续保留，后续再拆成更准确的 `reasons` 或 `status_reasons`。

#### 源码落点

- `contracts.py::AssistantResponse`。
- `pipeline.py::abstain_response`。
- `docs/architecture.md` 的 compatibility note。

#### 当前边界

这是已知 contract debt，不要把字段名解释成已经完成的分层错误 taxonomy。

## 第九轮：从 M1 走向 M2

### Q52：为什么现在值得继续做 Agent，而不是停在 M0？

#### 30 秒回答

因为 M0 的 failure table 给出了一个低风险、可复现的动作机会：3 个 query-expression mismatch。M1 证明一次 bounded recovery 能在这个 slice 上找回目标 source，同时没有越过 safety gate 或 citation verifier。下一步不是扩大自由度，而是处理 M1 暴露的 OOD tool activation 和 answerability 边界。

#### 证据

M0 3 个 paraphrase failure → M1 9/9 focused recovery success；M1 OOD 11/12 激活工具但 0/12 回答 → 继续做 evidence sufficiency/policy 才有真实驱动。

#### 当前边界

没有证据表明更长 loop、更复杂 planner 或 Multi-Agent 会改善这些问题。

### Q53：M1 最大的未解决风险是什么？

#### 30 秒回答

不是“模型不会调用工具”，而是模型对 OOD 过于愿意调用工具。一次 query recovery 可能把问题改写得更泛，增加错误召回；即使最终 abstain，仍会增加 latency 和 provider cost。另一个边界是合法 citation 仍不代表 claim-level grounding。

#### 深挖

要把 action policy 和 answer policy 分开评估：何时值得调用工具，工具结果是否覆盖问题，最终 claim 是否被支持。M1 当前只有硬预算和最终 citation-ID verifier，没有独立的 answerability classifier、query intent check 或 semantic entailment verifier。

#### 证据

`ood_tool_activation_rate=0.9166666666666666` 与 `ood_answer_rate=0.0` 的组合就是这个诊断信号。

#### 当前边界

后续的 policy、evidence sufficiency、claim grounding、timeout、trace/replay 和更完整 harness 仍是 `[M2 PLANNED]`，不能写成当前能力。

### Q54：为什么不直接做完整 ReAct 或 Multi-Agent？

#### 30 秒回答

因为当前问题只需要一次检索恢复，完整 ReAct 会把验证面扩大到多步规划、更多工具和更多失败模式；Multi-Agent 还会引入角色分工、消息协调、共享状态和成本。没有 failure evidence 证明这些复杂度能改善 M1 的 OOD 或 grounding，我会先保持一个可测的 single-agent boundary。

#### 源码落点

- `agent/loop.py`：当前只有顺序、一工具、2/1 budget。
- `tools/search_knowledge.py`：唯一只读工具。
- `docs/roadmap.md`：M2/M3+ 的后续方向。

#### 当前边界

Multi-Agent、Agent Teams、Agent Swarm、MCP、Sandbox 都不在当前实现中。

### Q55：Agent 和 Workflow 的真正区别是什么？M1 属于哪一种？

#### 30 秒回答

Workflow 的主要控制流由代码预先决定；Agent 多一个模型选择下一步的 decision step。M0 是 workflow：safety → retrieval → generation → verification。M1 是 bounded Agent：模型只能在显式 messages 和一个 tool spec 内选择 final 或 search，runtime 负责 schema、预算、执行、观察和终止。

#### 深挖

“用了 LLM”不等于 Agent；M0 的 generator 只生成回答，不改变控制流。M1 的 Agent 性质来自 `FinalTurn`/`ToolCallTurn` 分支，但自主性被限制在 recovery query 上。工程上，二者可以共存：固定的安全门和 citation verifier 包住中间的 Agent decision。

#### 当前边界

不要把 bounded Agent 夸大为 autonomous planner；也不要因为控制流有模型选择就声称有生产级 agent platform。

### Q56：如何防 infinite loop / repeated tool calls？

#### 30 秒回答

当前用两个硬 guard：`max_model_turns=2` 和 `max_tool_calls=1`。第二次工具调用不执行，预算耗尽且没有 final 就 `ABSTAIN`。这能保证终止和 fail closed，但不能判断一次 rewrite 是否有实际进展，也没有 token/time deadline。

#### 源码落点

`agent/loop.py::AgentLoopConfig`、`AgentLoop.run`；`tests/test_agent.py::test_second_tool_call_is_not_executed_and_fails_closed`、`test_no_final_answer_before_model_budget_fails_closed`。

#### 当前边界

progress signal、重复 query 检测、timeout、retry policy、token/cost budget 和持久化 trace 是 `[M2 PLANNED]`，当前不要写成已实现。

### Q57：下一阶段为什么应该先做 Harness/answerability，而不是再加工具？

#### 30 秒回答

因为 M1 的主要新信号不是“工具不够多”，而是 OOD 工具激活偏高、citation integrity 和 semantic grounding 之间有缺口。下一阶段应先把 policy、evidence sufficiency、timeout、trace/replay 和回归指标做成可测约束，再根据新的 failure 决定是否增加 retrieval 或其他 capability。

#### 当前状态

`docs/roadmap.md` 的 M2 是 `[planned]`。当前文档不把 M2 写成已实现，也不启动 M2 runtime code。

## 一页式复述顺序

1. Health-Copilot 是患者教育用的 Safety-Gated Evidence RAG，不是诊断或处方系统。
2. M0 先做 deterministic baseline：input validation → safety gate → BM25 → Evidence → Generator → citation verification → `AssistantResponse/ABSTAIN`。
3. M0.3 的 80 cases 中，76 条 safety route accuracy 为 1.0；62 条 retrieval cases 的 Hit@1/Hit@3/MRR 是 0.9032/0.9516/0.9274。
4. failure table 观察到 3 个 paraphrase mismatch 和 4 个 OOD false retrieval。
5. 3 个 paraphrase miss 驱动 M1 的一次 bounded `search_knowledge(query)` recovery。
6. M1 通过 typed session/state/messages、显式 registry、2 model turns/1 tool call 和 observed-evidence union 把模型提议包在 runtime 约束内。
7. M1 focused run 是 12 cases × 3 trials，36 trajectories、30 Agent runs；9/9 frozen recovery attempts 命中 recovery top-3。
8. OOD 仍有 11/12 工具激活，但 0/12 错误回答、12/12 abstain；这说明最终 fail closed 尚可，answerability policy 仍不足。
9. citation verifier 防 fabricated/invalid ID，不防 claim-level semantic grounding。
10. 下一步应先做 M2 的 Harness/answerability 约束；Multi-Agent、dense retrieval、训练等都不能从当前仓库声称已实现。

## 简历 claim 审计清单

可以说：

- “实现了 M0 safety-gated evidence RAG baseline。”
- “在 M0 failure table 的 paraphrase mismatch 上实现了 bounded single-agent retrieval recovery。”
- “实现 typed Agent state/session/message、显式 tool registry、原生 tool call adapter 和 2-turn/1-tool hard budget。”
- “对 M1 recovery、OOD tool activation、safety short-circuit、citation integrity 和 M0 baseline 做了分阶段评测。”

不要说：

- “实现通用医疗 Agent、Multi-Agent 或 Agent Swarm。”
- “M1 让 OOD retrieval 或 clinical accuracy 达到 100%。”
- “citation verifier 证明了答案语义被证据支持。”
- “Health-Copilot 使用了 dense retrieval、reranker、MCP、Sandbox、vLLM、SFT、DPO、RL、GRPO 或 GSPO。”
- “有线上流量、医生团队验证、患者数据或生产 QPS。”
