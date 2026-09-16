# Agent / LLM 岗通用八股

本文是独立的通用面试知识地图。它不改变 Health-Copilot 的实现状态：遇到项目映射时，必须回到 [project.md](project.md) 和当前仓库证据。每个主题按“面试官 → 短答 → 原理 → 公式/数量级 → trade-off → 常见追问”组织；没有必要把历史综述背成教材。

## 先用这套答题模板

回答一个通用概念时，可以按五句走：

1. 一句话定义它解决的工程问题。
2. 解释最小工作机制。
3. 给一个公式、数据流或数量级。
4. 说一个主要 trade-off 和失败模式。
5. 最后说明在当前项目中是“已实现、未使用，还是 planned”。

## A. Agent Runtime

### 面试官：Agent 和 Workflow 有什么区别？

**短答**：Workflow 的控制流主要由代码预先决定；Agent 允许模型在显式状态、工具和 policy 约束下选择下一步。两者都可以调用 LLM，所以“用了 LLM”不等于“是 Agent”。

**原理**：Workflow 通常是固定 DAG/状态机，例如 validate → retrieve → generate → verify。Agent 多一个受约束的 decision step：读取 state 和 observation，提出 action，runtime 验证并执行，再把结果写回 state。真正可用的 Agent 仍然需要代码拥有权限、预算和终止权。

**公式/数量级**：可以把一次 loop 抽象成 state_t → model(action_t) → policy-check → tool → observation_(t+1)。如果最多允许 S 步、每步最多 T 个工具调用，则最坏模型/工具迭代数是有限的，而不是无限 while loop。

**trade-off**：Agent 的适应性更强，但 action space、成本、延迟、非确定性和评测难度都变大；Workflow 更容易测试、缓存和复现，但遇到未预先编码的路径会僵硬。

**常见追问**：如何定义“自主性”？如何给 Agent 设置 fallback？Health-Copilot M0 是 workflow；M1 Agent Core 在当前基线仍是 [M1 PLANNED]。

### 面试官：ReAct 是什么？

**短答**：ReAct 把 reasoning 和 acting 交错起来：模型提出下一步动作，工具返回 observation，模型根据 observation 再决定继续还是回答。它解决的是仅靠一次生成无法动态获取外部信息的问题。

**原理**：典型轨迹是 Thought/plan → Act(tool, args) → Observation → … → Final。工程实现不能只展示 prompt，还要定义 action schema、工具执行边界、错误状态、最大步数和最终输出校验。

**公式/数量级**：若每轮有 m 个可能工具、最多 S 轮，未经约束的路径数会随 m^S 增长；这不是说真实系统一定穷举，而是说明 action space 和预算会迅速成为复杂度来源。

**trade-off**：动态检索和错误恢复更灵活，但循环、重复调用、prompt injection、工具副作用和 latency 更难控制。对于只需一次检索的任务，ReAct 可能是过度设计。

**常见追问**：ReAct 与 chain-of-thought 有何不同？如何隐藏内部 reasoning 但保留 trace？原始 ReAct 论文见 [Yao et al., 2022](https://arxiv.org/abs/2210.03629)。

### 面试官：tool calling / function calling 到底是什么？

**短答**：模型不直接执行函数，而是返回一个结构化的 tool name 和 arguments；runtime 校验 schema、权限和预算后才真正调用工具，再把结果以 observation 送回模型。

**原理**：工具描述至少包括 name、用途、参数 schema、返回 shape、错误语义和权限。模型输出是 proposal，执行是 runtime 的责任；不能因为 JSON 能解析就信任参数。参数还要做长度、枚举、资源范围和敏感字段检查。

**公式/数量级**：一次调用至少有 model proposal → schema validation → policy/capability check → execution → result validation 五个边界；每增加一个外部工具，就增加失败模式和潜在权限面。

**trade-off**：结构化调用比解析自然语言稳定，但 schema 设计过宽会放大风险，过窄又限制能力。只读检索工具通常比写入/支付/发消息工具容易做安全控制。

**常见追问**：工具错误要不要重试？如何保证幂等？串行和并行调用怎么选？可参考 [OpenAI function calling guide](https://platform.openai.com/docs/guides/function-calling) 和 [Anthropic tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview)。

### 面试官：Agent State、Context、Session、Memory 怎么区分？

**短答**：工程上可以这样分：State 是运行中可变的结构化状态；Context 是某一步实际喂给模型的材料；Session 是一次会话的边界和身份；Memory 是跨步骤或跨会话被保存、检索和复用的信息。

**原理**：例如 state 里可以有 question、attempt、route、evidence_ids、last_tool_error；context 是为下一次模型调用挑出的相关字段；session 记录 user/session ID 和生命周期；memory 可能是经过摘要、权限过滤和 TTL 的历史事实。把全部历史都塞进 context 不等于设计了 memory。

**公式/数量级**：上下文 token 近似为 system + tools + history + retrieved evidence + current input；多轮增长会推高每次 prefill 成本，压缩/检索 memory 是在控制这项增长，而不是免费增加记忆。

**trade-off**：保存更多信息提高可回忆性，但会带来 token cost、陈旧信息、隐私和错误记忆污染。状态要可序列化、版本化，memory 要有来源、删除和权限策略。

**常见追问**：短期 history 和长期 memory 的一致性怎么处理？如何防止记忆注入？当前 Health-Copilot M0 没有 Session/Memory。

### 面试官：message protocol、tool schema、Tool Registry 为什么需要显式化？

**短答**：它们把 Agent 与工具之间的隐式约定变成可验证接口。message protocol 规定消息类型和顺序，tool schema 规定参数/返回值，Tool Registry 负责发现、版本、权限和生命周期。

**原理**：至少要区分 user、system/policy、assistant proposal、tool result、final response；工具结果还要标记 success/error、source 和可重试性。Registry 不只是一个 dict，还要处理 namespace 冲突、版本兼容、capability 和下线。

**公式/数量级**：接口组合复杂度大致随“消息类型 × 工具数 × 状态版本”增长；稳定 schema 可以把变化限制在边界内，避免每个 Agent 都理解每个工具的私有格式。

**trade-off**：显式协议增加前期设计和迁移成本，但降低解析歧义、调试成本和跨 Agent/服务耦合。schema 过于自由会把验证压力推回 prompt。

**常见追问**：tool result 是否直接可信？Registry 如何做权限过滤？当前 M0 只有 Generator/Retriever Protocol，没有 Agent Tool Registry。

### 面试官：串行和并行 tool calls 如何选择？

**短答**：有数据依赖或共享可变状态时串行；多个独立、只读、可合并的查询可以并行。并行降低 wall-clock latency，但增加 fan-out、限流、聚合、错误处理和成本峰值。

**原理**：如果 B 必须读 A 的结果，不能并行；如果 A/B/C 都是独立 retrieval，可以并行后用 deterministic merge 或 rerank。并行结果要保留调用 ID、来源和完成状态，不能因为最快返回的工具先到就改变语义。

**公式/数量级**：串行延迟近似 Σ latency_i；理想并行延迟接近 max(latency_i)，但真实值还要加调度、队列和聚合开销。fan-out 为 n 时，失败概率和资源峰值通常都会上升。

**trade-off**：串行更容易审计和限流；并行适合独立只读操作，但要有 per-request fan-out、global rate limit、partial failure policy 和 deterministic merge。

**常见追问**：部分工具失败是整条失败还是降级？同一工具并行调用是否幂等？当前 Health-Copilot M0 没有多工具调用。

### 面试官：idempotency、timeout、retry 怎么设计？

**短答**：幂等保证同一个逻辑请求重复执行不会产生额外副作用；timeout 给每步和全局请求设上限；retry 只能对可重试错误执行，并且要有退避、次数和预算。三者要一起设计，不能只写一个 try/except。

**原理**：读操作通常天然更接近幂等，写操作需要 idempotency key 或去重表；timeout 需要区分 connect/read/tool/global deadline；retry 要区分 transient、validation、permission 和 business error。重试前要把原始 action、attempt 和结果写入 trace。

**公式/数量级**：指数退避常写成 delay = min(cap, base × 2^attempt) + jitter；总请求时间必须受 global deadline 约束，而不是每次 retry 都重新获得完整 timeout。

**trade-off**：retry 能提高瞬时故障成功率，但会放大流量、延迟和非幂等副作用；timeout 太短会误杀慢但正确的操作，太长则拖垮并发。

**常见追问**：HTTP 429 和 schema error 是否同样重试？工具调用如何去重？Health-Copilot M0 尚未实现 timeout/retry/fallback。

### 面试官：Agent Loop 怎么终止？

**短答**：至少需要硬上限和语义终止：max steps、tool-call budget、token/time budget、显式 final/abstain、重复 action 检测和异常终止。终止条件必须由 runtime 执行，不能只相信模型说“我完成了”。

**原理**：每一步更新 attempt_count、cost、elapsed、seen_actions、last_error；若达到上限、没有 progress、policy deny 或工具连续失败，就进入可解释的 terminal state。最终答案还要经过独立 validator。

**公式/数量级**：一次请求的成本上限可以写成 C ≤ C_model + S × (C_tool + C_model_step)，其中 S 由 runtime 硬限制；没有 S，成本不具备上界。

**trade-off**：硬上限可能截断复杂但有价值的任务；没有硬上限则容易无限循环或成本失控。应该用 failure slice 调整预算，而不是凭感觉放大。

**常见追问**：重复 query 怎么判定？遇到 partial result 如何收尾？Health-Copilot M0 没有 loop，M1 termination 是 [M1 PLANNED]。

### 面试官：Context Engineering 和 Prompt Engineering 有什么差别？

**短答**：Prompt Engineering 主要调指令和示例；Context Engineering 关注在每一步给模型什么信息、以什么顺序、什么格式、哪些信息不该给。它包含检索、状态摘要、工具结果、权限过滤、上下文压缩和新鲜度。

**原理**：模型效果不仅取决于 system prompt，还取决于 context 的信噪比、时间顺序、来源和冲突。一个简单的 context assembly 可以是 policy + task + selected state + evidence + output schema，而不是把所有 history 和所有工具描述无差别塞进去。

**公式/数量级**：context token budget B 固定时，增加无关历史就会挤出 evidence 或 schema；优化目标通常是“在 B 内最大化 task-relevant signal”，而非无限加长。

**trade-off**：更多上下文可能减少遗漏，但增加成本、冲突和 distraction；压缩可能丢失否定词、数值和条件。需要保留来源和可回放摘要。

**常见追问**：如何做 progressive disclosure？如何评估 context 的有效性？Health-Copilot M0 的 generator 只接收当前 Evidence，未实现通用 context runtime。

### 面试官：Agent Skills 是什么？和普通工具有什么区别？

**短答**：工程语境下，Skill 通常是一个可复用、带说明和输入输出契约的能力单元；工具偏执行接口，Skill 可以进一步包含使用条件、工作流建议、资源和示例。这个词没有一个跨框架完全统一的标准，必须看具体实现。

**原理**：Skill 的价值在于能力复用和渐进式披露：先告诉模型有哪些能力，再按需要加载某个能力的详细说明，避免一次把所有文档/工具 schema 放进 context。Skill 仍然要经过权限、schema 和 runtime 验证。

**公式/数量级**：如果每个 Skill 说明平均 t tokens、共有 n 个能力，全部预加载是 O(nt)；按需加载可以把单次 context 控制在 O(kt)，其中 k 远小于 n，但多一次发现/加载步骤。

**trade-off**：渐进式披露节省 context，但会引入发现失败、版本漂移和额外一次调用；Skill 描述写得含糊会让模型选错能力。

**常见追问**：Skill 和 MCP tool 如何分工？Health-Copilot 当前没有 Skills registry；参考文档中“7 个 Skills”不是本仓库事实。

### 面试官：MCP 是什么？

**短答**：MCP（Model Context Protocol）是一种让 host/client 与外部 server 以标准方式暴露和调用模型可用能力的协议。它解决的是连接和发现的互操作性，不自动解决业务正确性、权限或安全。

**原理**：MCP server 可以声明工具及其 input schema，client/host 负责把可用能力呈现给模型并执行调用；真正的系统还要加身份认证、授权、数据范围、timeout、审计和结果校验。协议标准化的是消息/能力边界，不是“给模型无限访问”。

**公式/数量级**：一次 MCP 工具链路仍然是 model → host policy → client → server → downstream；链路节点越多，trace、timeout 和 error propagation 越重要。

**trade-off**：标准协议降低集成成本、方便复用工具，但扩大了连接面和供应链/权限面；生产环境应做 allowlist、namespace、最小权限和版本治理。

**常见追问**：MCP 和 function calling 的关系？MCP tool 如何鉴权？参考 [MCP tools specification（2026-07-28）](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)。Health-Copilot M0 未使用 MCP。

### 面试官：Harness Engineering 是什么？

**短答**：它不是一个单独算法，而是一种围绕 Agent 建立可执行边界和反馈回路的工程方式：让模型有清晰的环境、工具、权限、状态、测试、日志、回放和质量规则，并把关键 invariant 放进 runtime，而不是只写 prompt。

**原理**：一个 harness 通常负责准备 context、暴露 capability、执行工具、限制预算、保存 trace、处理错误、验证输出和驱动 eval。目标是让 Agent 的行为可观察、可复现、可纠错；业务代码仍负责领域规则。

**公式/数量级**：一次 agent task 的可靠性不是单次模型准确率，而是多个边界共同决定：P(success) ≈ P(policy) × P(tool) × P(context) × P(model) × P(verification)（只是工程分解，不是严格独立概率模型）。任一关键项为 0，最终任务可能失败。

**trade-off**：约束越多，安全和可复现性越好，但能力和实现成本可能下降；应优先机械化 enforce 高风险 invariant，给低风险部分保留灵活性。

**常见追问**：Harness 与 prompt/context engineering 的边界？如何知道规则应放文档还是代码？可参考 [OpenAI Harness Engineering](https://openai.com/index/harness-engineering/)。Health-Copilot M0 只有窄范围的“model proposes; runtime validates”雏形。

### 面试官：sandbox / permission / capability security 为什么是 Agent 的一等问题？

**短答**：因为 Agent 不只是生成文本，它可能通过工具影响文件、网络、数据库或现实业务。Sandbox 限制执行环境，permission 决定主体能否做某件事，capability security 则让系统只把必要能力的引用交给 Agent，并在执行点再次校验。

**原理**：最小权限、只读默认、路径/域名 allowlist、资源配额、隔离凭据和审计是常见控制。不要把“工具描述里写着请谨慎”当作权限；模型输出必须经过 runtime policy。

**公式/数量级**：风险面可粗略看作工具数 × 每工具权限范围 × 每 loop 步数；减少工具、缩小资源范围和限制步数都会降低组合风险。

**trade-off**：强隔离可能降低开发便利和工具能力；弱隔离则一旦 prompt injection 或模型误判发生，损害半径很大。读操作、写操作和外部副作用应分级。

**常见追问**：如何隔离不可信 tool result？如何设计 human approval？Health-Copilot M0 没有 sandbox、MCP 或通用 capability system。

### 面试官：trace / replay / evaluation 为什么重要？

**短答**：单看最终答案无法解释 Agent 是在哪一步错的。Trace 记录状态、action、tool input/output、耗时、token、policy decision 和 final route；replay 用固定版本和脱敏输入重放轨迹；eval 把轨迹转成可比较的回归指标。

**原理**：要区分模型错误、检索错误、工具错误、策略拒绝和聚合错误，必须记录足够的中间事件，同时避免保存敏感数据。replay 还要固定 prompt、模型版本、工具版本、知识库版本和随机性，否则比较不公平。

**公式/数量级**：长任务的诊断成本随着步骤数增加；trace 的最小事件数通常至少是 2 × steps + terminal（proposal/result 加终态），实际还要记录 policy、timing 和 error events。

**trade-off**：更完整的 trace 提高可诊断性，但增加存储、隐私和合规成本；脱敏或摘要过度又可能无法复现。

**常见追问**：如何做 deterministic replay？哪些字段不能记录？Health-Copilot M0 有离线 eval，但没有完整 agent trace/replay。

## B. RAG / Retrieval

### 面试官：RAG 解决什么问题？

**短答**：RAG 在生成前从外部或非参数知识中检索相关内容，再把它作为模型上下文，从而支持知识更新、来源追溯和领域资料注入。它不能保证检索到的内容正确，也不能自动防止模型误读。

**原理**：经典流程是 query → retrieve → select/rerank → construct context → generate → verify。RAG 把“找什么”和“怎么说”拆开，检索 recall 不足会限制答案上限，证据冲突或过期也会传给生成器。

**公式/数量级**：如果相关证据被 top-k 召回的概率是 r_k，在没有额外信息时，回答有机会基于正确证据的上限受 r_k 约束；但高 r_k 不等于高 faithfulness。

**trade-off**：外部知识增加可更新性和 provenance，却引入检索延迟、chunk/index 维护、上下文噪声和 prompt injection 风险。应分别评估 retrieval、generation 和 grounding。

**常见追问**：RAG 和 fine-tuning 的边界？为何要 abstain？参考 [Lewis et al., 2020](https://arxiv.org/abs/2005.11401)。Health-Copilot M0 是 evidence RAG，但只实现 lexical retrieval 和 citation integrity。

### 面试官：sparse 和 dense retrieval 怎么选？

**短答**：Sparse 主要依赖词项匹配和稀疏权重，解释性、exact term 和部署简单性较好；dense 把 query/document 编成向量，能更好处理部分语义改写，但需要 embedding 模型、向量索引、版本管理和阈值校准。没有统一的“dense 一定更好”。

**原理**：sparse 的相似度来自 token statistics；dense 的相似度来自表示空间。医学数字、专有名词、否定和来源范围可能更适合保留 lexical signal，因此常用 hybrid 或 reranker 做互补，但必须实测。

**公式/数量级**：sparse BM25 逐 term 累加；dense 常对向量 q,d ∈ R^h 计算相似度，索引存储规模约与 N×h 成正比（还不含压缩和索引开销）。

**trade-off**：dense 有语义 recall 潜力，但 embedding 领域适配、OOD 误近邻、内存、GPU/服务依赖和解释性是成本；sparse 对 paraphrase 有明显盲点。

**常见追问**：为什么 hybrid？embedding 如何评测？Health-Copilot M0 只有 BM25，dense/hybrid 都未使用。

### 面试官：BM25、TF、DF、IDF 怎么讲？

**短答**：TF 是词在一个文档中的频率，DF 是包含该词的文档数，IDF 让稀有词权重大、常见词权重小；BM25 在此基础上加入 TF saturation 和 document-length normalization。

**原理**：一个词在当前文档出现，说明可能相关；如果它在所有文档都出现，区分度低。BM25 的 TF 不线性增长，长度归一化避免长文档仅因词多而占优。不同实现对 IDF 平滑和字段加权可能不同。

**公式/数量级**：当前 Health-Copilot 实现的 IDF 是 log((N-df+0.5)/(df+0.5)+1)；TF 项是 tf(k1+1)/(tf+k1(1-b+b|D|/avgdl))。

**trade-off**：统计方法简单可解释，但不理解同义、否定和复杂关系；分词、停用词、字段拼接和参数都会改变结果。

**常见追问**：BM25 score 是 probability 吗？k1/b 怎么调？原理来源可看 [Robertson & Zaragoza, BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)。

### 面试官：embedding、bi-encoder、cross-encoder 分别是什么？

**短答**：embedding 把文本映射为向量；bi-encoder 分别编码 query 和 document，能提前缓存文档向量，适合大规模召回；cross-encoder 把 query 和 document 一起输入模型，交互更充分但每个候选都要重新算，适合 top-k reranking。

**原理**：bi-encoder 的离线/在线分工是 encode(D) 预计算、encode(q) 在线检索；cross-encoder 直接学习 pair relevance，通常放在召回后缩小候选。长文档需要 chunk、聚合或 late interaction 设计。

**公式/数量级**：bi-encoder 大规模查询成本近似 O(encode(q)+ANN(q,N))；cross-encoder 对 K 个候选约需 O(K × pair_model)，所以常见 K 远小于 N。

**trade-off**：bi-encoder 快但表示空间的独立编码可能丢 pair-specific 关系；cross-encoder 精度潜力更高但延迟和成本大。领域 embedding 未必在 OOD 上更稳。

**常见追问**：如何训练 hard negatives？如何把多个 chunk 合并成文档分数？Health-Copilot M0 没有 embedding 或 reranker。

### 面试官：cosine、inner product、L2 有什么区别？

**短答**：cosine 只看夹角，q·d/(||q||||d||)；inner product 直接看点积，长度也会影响分数；L2 是欧氏距离，越小越近。若向量都做 L2 normalization，最大化 cosine 等价于最大化 inner product，且与最小化平方 L2 单调相关。

**原理**：选择哪种度量要和 embedding 训练目标、是否归一化、向量库 index 和阈值校准一致。不能拿 cosine 的阈值直接套到未归一化 inner product。

**公式/数量级**：对单位向量，||q-d||² = 2 - 2(q·d)；这只在两者都单位归一化时成立。

**trade-off**：cosine 对向量尺度不敏感；inner product 可表达幅度信息但更依赖训练和归一化；L2 直观但高维距离可能集中，需要实测分布。

**常见追问**：为什么 ANN index 对 metric 敏感？score 如何校准成 retrieval threshold？当前项目 BM25 不涉及这些向量度量。

### 面试官：HNSW 和 IVF 是什么？

**短答**：HNSW 用多层近邻图做近似搜索，查询沿图导航；IVF 先把向量分到 coarse clusters，查询时只探查部分簇。两者都用少搜一部分候选换速度和召回。

**原理**：HNSW 的关键参数通常包括图连接度和搜索 ef；IVF 常见 nlist、查询探查的 nprobe，还可以叠加 PQ 压缩。参数越偏向速度，越可能漏掉真实近邻；索引构建和更新策略也不同。

**公式/数量级**：暴力搜索约为 O(Nh)；ANN 目标是显著减少候选数量，但真实复杂度取决于图/簇结构、硬件和参数，不能只背一个 Big-O 就声称延迟。

**trade-off**：HNSW 在线召回和低延迟常见，但内存/更新成本可能较高；IVF 更容易控制扫描比例和压缩，但 coarse partition 错误会影响 recall。

**常见追问**：如何选择 nprobe/ef？动态更新会怎样？Health-Copilot M0 没有向量索引；参考 [HNSW paper](https://arxiv.org/abs/1603.09320) 可继续阅读。

### 面试官：hybrid search 和 RRF 怎么讲？

**短答**：hybrid search 融合 sparse 与 dense 等多个检索信号；RRF（Reciprocal Rank Fusion）按各路结果的排名倒数相加，避免不同检索器的 raw score 不在同一尺度上。

**原理**：分别得到多个 ranked lists，RRF 常见形式是 score(d)=Σ_i 1/(k + rank_i(d))，没出现在某一路的文档该路贡献 0。融合后还可以交给 reranker。k 是平滑常数，不是 BM25 的 k1。

**公式/数量级**：一个文档在多个列表都排得靠前会累积高分；融合的计算量通常是各路 top-k 合并，而不是重新比较全库。

**trade-off**：能兼顾 exact term 和语义改写，但增加两个索引、两套版本和权重/候选治理；如果某一路 OOD 很差，fusion 可能把错误结果抬高。

**常见追问**：为什么不用 raw score 加权？如何做 ablation？Health-Copilot M0 没有 hybrid/RRF。

### 面试官：reranker 为什么放在召回之后？

**短答**：先用便宜的 retriever 从大库取一个候选集，再用更贵但交互更充分的 reranker 精排。这样把 cross-encoder 等高成本模型的计算限制在 top-k，而不是全库。

**原理**：召回阶段优先保证 recall，精排阶段优先优化 precision/顺序；如果第一阶段漏掉 relevant document，reranker 无法凭空恢复它。因此要同时看 recall 和 reranked metrics。

**公式/数量级**：全库 pair scoring 约 N 次；先召回 K 个候选后约 K 次，通常 K 远小于 N，但每个 pair 的模型计算更重。

**trade-off**：精度潜力和解释能力可能提高，代价是延迟、GPU、批处理和模型版本管理；reranker 也可能过拟合 query style 或把 OOD 结果排得更自信。

**常见追问**：reranker 的 hard negative 怎么来？如何设置 K？当前项目未实现 reranker。

### 面试官：Chunking 为什么重要？

**短答**：chunk 是检索和上下文的基本粒度。太大导致噪声和 token 浪费，太小会丢失条件、否定、表格和跨句关系；好的 chunk 需要同时服务 recall、grounding 和可引用性。

**原理**：切分应尊重标题、段落、列表、表格和语义单元，保留 document/chunk ID、来源位置和版本。overlap 能缓解边界断裂，但会增加重复和 index size；不是 overlap 越大越好。

**公式/数量级**：若文档长度为 L、chunk size 为 c、overlap 为 o，粗略 chunk 数约 L/(c-o)；overlap 增大时存储和召回候选也上升。

**trade-off**：小 chunk 提升定位精度却可能缺上下文；大 chunk 更完整却增加 distraction。应在 query-level recall、answer grounding、context tokens 和跨边界 case 上做评测。

**常见追问**：如何处理跨页表格？如何定位到原文？Health-Copilot M0 使用的是一张卡一个检索单元，没有实现通用 chunker。

### 面试官：Recall@K、Hit@K、MRR、nDCG 怎么比较？

**短答**：Hit@K 看前 K 是否至少命中一个；Recall@K 看相关集合被召回的比例；MRR 只关心第一个相关结果的 rank；nDCG 还考虑 graded relevance 和位置折扣。选择指标取决于一个 query 是需要一个答案、多个来源，还是高质量排序。

**原理**：Hit@K 适合“有一个可用证据就够”的 case-level 产品问题；Recall 更适合多来源覆盖；MRR 惩罚第一个 relevant 排得靠后；nDCG 适合不同相关等级的 qrels。指标定义和标注协议必须一起写。

**公式/数量级**：RR=1/r；Recall@K=|retrieved∩relevant|/|relevant|；nDCG=DCG/IDCG。没有相关文档的 query 要明确排除或定义分母，不能偷偷改分母。

**trade-off**：单一指标容易被优化偏；只看 Hit 可能忽略覆盖率，只看 Recall 可能忽略首条体验，只看 nDCG 依赖 qrels 质量。

**常见追问**：多标签 query 如何处理？M0 为何同时有 Product Eval 和 NFCorpus？当前项目的准确数字见 evals/m0_failure_table.json。

### 面试官：grounding、hallucination、abstention 怎么区分？

**短答**：grounding 是回答 claim 能否被给定证据支持；hallucination 是回答包含无证据、错误或与证据冲突的内容；abstention 是系统在证据不足/风险过高时主动不回答。citation 存在不等于 grounding 成功。

**原理**：要评估 grounding，需定义 claim 粒度、支持/矛盾/未知标签和证据范围；要评估 abstention，还要同时看 coverage、selective risk、误拒答和漏拒答。检索空结果只是最简单的 abstention trigger。

**公式/数量级**：selective answering 可按 coverage P(answer) 与 selective risk P(error | answer) 画曲线；目标通常是在可接受 risk 下最大化 coverage，而不是追求 100% 回答率。

**trade-off**：更保守的 abstain 降低错误风险但损失可用性；更激进的回答提高 coverage 但可能放大 OOD 和 hallucination。阈值必须用独立验证集校准。

**常见追问**：如何做 claim-level evaluator？如何处理证据冲突？Health-Copilot M0 只有 citation ID integrity 和若干 fail-closed 路径。

## C. LLM 推理与后训练（通用知识；不代表 Health-Copilot M0 已使用）

> 本节覆盖常见 Agent/LLM 岗追问。KV Cache、PagedAttention/vLLM、分布式并行、FSDP/ZeRO、SFT、PPO、GRPO、GSPO 都不是当前 Health-Copilot M0 的已实现能力。

### 面试官：Transformer 的 self-attention 在做什么？

**短答**：self-attention 让序列中每个位置根据其他位置的表示计算加权聚合，从而建立长距离依赖；Transformer 用多头 attention、位置表示和前馈网络堆叠成可并行训练的序列模型。

**原理**：输入通过线性层得到 Q、K、V，先计算 query 对 key 的匹配，再用 softmax 得到权重，对 value 加权：Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V。多头让模型在不同子空间建模关系，残差和归一化帮助深层训练。

**公式/数量级**：标准 self-attention 的序列交互矩阵大小是 L×L，单层计算/显存随长度近似二次增长；这也是长上下文优化的根源之一。

**trade-off**：全局依赖强但长序列成本高；稀疏 attention、滑窗、检索和压缩可以降成本，却可能丢远程信息。

**常见追问**：prefill 和 decode 有什么区别？GQA/MQA 如何影响 KV cache？参考 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。

### 面试官：为什么需要 KV Cache？

**短答**：自回归 decode 时，新 token 的 attention 需要读取前面 token 的 key/value；如果每一步都重新计算历史 K/V，会重复计算。KV cache 把历史 K/V 保存起来，下一步只算新 token 的 Q/K/V。

**原理**：prefill 阶段一次处理输入 prompt 并建立 cache；decode 阶段每次追加一个 token 的 K/V，并用新 Q 访问历史 cache。cache 不是“缓存最终答案”，而是每层 attention 的中间状态。

**公式/数量级**：对 MHA，单层 KV cache 的元素量近似 2 × L × n_kv_heads × head_dim；乘以层数、batch 和 bytes 才是显存。GQA/MQA 通过减少 n_kv_heads 降低 KV cache，而不一定减少 query heads。

**trade-off**：KV cache 用显存换 decode 计算，长上下文和高并发时会成为瓶颈；cache sharing、量化、分页和 eviction 可以节省显存，但会增加实现复杂度或精度风险。

**常见追问**：为什么 prefill/decode 的瓶颈不同？PagedAttention 解决了 cache 的哪一层问题？当前项目没有自建模型推理服务。

### 面试官：PagedAttention / vLLM 解决什么？

**短答**：它主要解决 LLM serving 中 KV cache 的动态内存管理和碎片问题：把 cache 按 block/page 管理，而不是给每个请求预留一大片连续空间，并配合调度提高批处理利用率。它不是一个新的语言模型架构。

**原理**：请求的序列长度会动态增长、不同请求结束时间不同；连续分配会有内部/外部碎片。PagedAttention 借鉴虚拟内存分页，把逻辑 token block 映射到非连续物理 block，同时由 serving engine 调度请求和复用空闲块。

**公式/数量级**：cache 需求仍随 batch × sequence_length × layers × KV width 增长；分页主要减少浪费和提高可分配性，不会让模型权重或每个 token 的基本 K/V 信息凭空消失。

**trade-off**：吞吐和并发通常更好，但引入 block table、调度、抢占、版本和硬件 kernel 的复杂度；单请求低并发场景的收益可能不同。

**常见追问**：continuous batching 是什么？PagedAttention 是否降低单 token FLOPs？参考 [Kwon et al., 2023](https://arxiv.org/abs/2309.06180)。Health-Copilot M0 未使用 vLLM。

### 面试官：prefill、decode、throughput、latency 怎么讲？

**短答**：prefill 处理已有 prompt，通常计算密集且可以并行；decode 逐 token 生成，常受 memory bandwidth、KV cache 和调度影响。latency 是单请求响应时间，throughput 是单位时间处理的 token/request 数，不能只优化一个。

**原理**：首 token 延迟常包含 queue + prefill + first decode；后续 token 受每步 decode 影响。batching 能摊薄 kernel/权重读取，但会让短请求等待更久；continuous batching 在请求到达和结束时动态调整 batch。

**公式/数量级**：端到端 latency 近似 queue + prefill + output_tokens × decode_step + postprocess；吞吐常用 generated tokens / second 或 requests / second，必须同时报告 batch、输入/输出长度和硬件。

**trade-off**：追求高吞吐会牺牲 tail latency；追求低 P99 可能降低 GPU 利用率。Agent 多步调用会把每个模型/工具 latency 累加，不能只报单次 token speed。

**常见追问**：TTFT 和 TPOT 有什么区别？如何定位 P99？当前项目没有线上 latency/QPS claim。

### 面试官：DP、TP、PP、CP、SP 分别是什么？

**短答**：DP 复制模型到不同 rank、分不同数据；TP 把一个层内的张量/矩阵切到多个 rank；PP 把不同层放到不同 stage；CP 把上下文/序列维度切分到多个 rank；SP 通常指对 sequence 维度或激活做切分以配合 tensor parallel。CP/SP 的具体语义会随框架变化，要先说上下文。

**原理**：DP 需要梯度同步；TP 需要层内 all-reduce/all-gather 等通信；PP 需要 micro-batch 和 pipeline schedule，可能有 bubble；CP/SP 试图降低长序列激活/attention 的单卡内存，但通信和实现更复杂。

**公式/数量级**：理想 DP 吞吐随副本数近似增加但通信也上升；PP 的 pipeline bubble 与 stage 数和 micro-batch 数有关；TP/CP 的收益受通信带宽和模型切分均衡限制，不能只按 GPU 数线性外推。

**trade-off**：DP 工程简单但单卡必须容纳模型状态；TP/PP/CP 能放大模型或上下文，却有通信、负载均衡和调度成本；组合并行需要清楚说明每个维度切的是什么。

**常见追问**：训练和推理中的 TP 有何不同？为什么 pipeline bubble 会产生？Health-Copilot M0 没有分布式推理或训练。

### 面试官：FSDP 和 ZeRO 在解决什么问题？

**短答**：两者都通过切分/分片模型状态来降低单卡训练内存。FSDP 是 PyTorch 的 sharding API；ZeRO 是一组逐阶段消除参数、梯度和 optimizer state 冗余的设计。它们相关但不是可以不加条件互换的名词。

**原理**：完整 sharding 通常让参数、梯度和 optimizer state 分布在 rank 上，在 forward/backward 需要时 all-gather，梯度再 reduce-scatter。不同 stage/strategy 的通信、峰值显存和实现支持不同。

**公式/数量级**：Adam 类训练的显存不只包含参数，还包括梯度、master weights 和 optimizer states；分片的目标是把每张卡承担的模型状态从接近完整副本降到约 1/world_size 的量级（还要加通信 buffer、激活和碎片）。

**trade-off**：显存下降能训练更大模型，但 all-gather/reduce-scatter、checkpoint、wrap policy 和调试复杂度增加；小模型或低带宽集群未必值得。

**常见追问**：FSDP full shard 的通信点？ZeRO stage 1/2/3 差别？参考 [PyTorch FSDP docs](https://docs.pytorch.org/docs/stable/fsdp.html) 和 [ZeRO paper](https://arxiv.org/abs/1910.02054)。Health-Copilot M0 没有模型训练。

### 面试官：SFT 和 RL 有什么区别？

**短答**：SFT 用示范答案做 supervised next-token learning，目标是模仿数据分布；RL 根据 reward 优化策略产生结果的期望回报，可以直接优化规则、结果或偏好，但训练不稳定、奖励设计难。

**原理**：SFT 常见目标是负的 Σ log πθ(y_t | x, y_<t)；RL 先从当前 policy rollout，再由 reward model/rule 给分，用 policy gradient 或其变体更新策略。SFT 依赖 label quality，RL 依赖 reward quality 和 exploration。

**公式/数量级**：SFT 是 token-level likelihood；RL 目标可写为 J(θ)=E[y~πθ(.|x)][R(x,y)]，实际还需要 clipping、KL 或其他稳定项。

**trade-off**：SFT 稳定、实现简单但受示范覆盖限制；RL 能针对 outcome 优化但可能 reward hacking、分布漂移和 entropy collapse。两者都不能替代独立 eval。

**常见追问**：什么时候只做 SFT？RL 的 reward 如何防作弊？当前 Health-Copilot M0 没有 SFT/RL。

### 面试官：PPO 在 LLM 后训练里做什么？

**短答**：PPO 通过限制新旧 policy 的概率比，避免一次更新把策略推得过远；在 LLM RLHF/RLVR 中，通常采样回答、得到 reward/advantage，再对 token log-prob 做 clipped policy update，并常配 KL 约束参考模型。

**原理**：核心 clipped surrogate 常写成 L_CLIP = E[min(r_t(θ)A_t, clip(r_t(θ),1-ε,1+ε)A_t)]，其中 r_t=πθ/πold，A_t 是 advantage。LLM 里还要处理序列长度、token mask、reward 分配和 value/critic。

**公式/数量级**：每个 rollout 可能包含 prompt、completion、old log-prob、reference log-prob、reward、advantage 和 value；显存和通信通常比单纯 SFT 更复杂。

**trade-off**：clipping 提升稳定性但可能限制有效更新；critic/reward model 增加显存和误差；KL 太强学不动，太弱容易偏离 reference 或 exploit reward。

**常见追问**：GAE 是什么？PPO 与 GRPO 的区别？参考 [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)。Health-Copilot M0 未使用 PPO。

### 面试官：GRPO 是什么？

**短答**：GRPO（Group Relative Policy Optimization）通过对同一个 prompt 采样一组回答，用组内 reward 的相对值构造 advantage，减少对单独 value/critic model 的依赖；它是 PPO 家族的变体，不是“无需 reward 的训练”。

**原理**：对同一个问题得到 rewards r_1...r_G，常见组内标准化是 A_i=(r_i-mean(r))/(std(r)+ε)，再将它放进类似 clipped policy objective。具体实现还会加 KL、长度处理和过滤规则。

**公式/数量级**：group size G 增大，组内相对排序估计可能更稳定，但每个 prompt 的 rollout 成本也近似增加；如果一组 reward 全相同，relative signal 会很弱。

**trade-off**：减少 critic 显存和训练组件，但需要多样、可比较的 group samples；reward 噪声、组大小和 sampling temperature 会影响梯度质量。

**常见追问**：为什么组内 baseline 能工作？GRPO 如何处理长回答？参考 [DeepSeekMath](https://arxiv.org/abs/2402.03300)。Health-Copilot M0 未使用 GRPO。

### 面试官：GSPO 对 GRPO 做了什么变化？

**短答**：GSPO（Group Sequence Policy Optimization）把重要性比率、clipping、reward/optimization 的粒度放到 sequence 层面，而不是直接对每个 token 使用独立的重要性比率；目标是改善长序列/大模型训练的稳定性与效率。具体性质要按论文和实现版本说明。

**原理**：token-level ratio 在长序列上可能产生很多 noisy/local updates；sequence-level likelihood ratio 让一个 completion 作为整体参与权重和裁剪。它不是简单把 GRPO 的 group size 改名，序列概率、数值稳定和归一化都要重新处理。

**公式/数量级**：sequence ratio 可写成 r_seq=exp(log πθ(y|x)-log πold(y|x))；实现通常需要对 completion token log-prob 聚合，长序列会带来数值和长度偏置问题。

**trade-off**：序列粒度更贴近整条回答的 reward，但 credit assignment 更粗，长短回答的比较和 clipping 设计更敏感；不能只凭“更稳定”就推断所有任务更好。

**常见追问**：token-level 和 sequence-level reward 有何差别？GSPO 如何处理长度 bias？参考 [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)。Health-Copilot M0 未使用 GSPO。

### 面试官：什么是 reward hacking？

**短答**：模型优化的是 reward proxy，而不是我们真正想要的目标；当 proxy 有漏洞时，模型会找到高分但不合意的行为。例如只奖励格式，模型可能输出漂亮但没有内容的答案。

**原理**：reward 可能由规则、模型 judge、相似度、人工偏好等组合而成；每个信号都有盲区。医疗场景中“引用格式正确”不等于“claim 被证据支持”，“拒答很多”也不等于安全性高。

**公式/数量级**：若总 reward R=Σ w_i r_i，模型会寻找让高权重项上升而其他目标不受约束的路径；权重不是目标定义本身。

**trade-off**：更多 reward components 能覆盖更多目标，但增加冲突、标注成本和优化难度；规则越容易被模型看到并利用，越要保留 adversarial/hidden eval。

**常见追问**：如何发现 reward hacking？如何设计 held-out adversarial set？Health-Copilot M0 没有 reward training。

### 面试官：什么是 entropy collapse？

**短答**：策略输出分布的 entropy 过快下降，模型越来越只选择少数模式，探索和表达多样性减少；它可能是训练收敛的一部分，也可能是 reward over-optimization、KL/temperature 不当或数据覆盖不足的信号。

**原理**：离散分布 entropy 是 H(π)=-Σ_a π(a)logπ(a)；在语言模型里通常看 token entropy、response diversity、长度和独立质量指标的联合变化。不能只看 entropy 单曲线下结论。

**公式/数量级**：若某个 token 的概率趋近 1，entropy 趋近 0；对于 V 个等概率 token，最大 entropy 是 log V。真实 LLM 还会受上下文、temperature 和 tokenization 影响。

**trade-off**：更低 entropy 可能表示策略更确定、格式更稳定；过低则可能导致模板化、短回答、拒答模式或对新问题泛化变差。应联合 held-out quality、coverage 和 refusal correctness 监控。

**常见追问**：temperature 调节和训练 entropy regularization 有何不同？如何缓解 collapse？Health-Copilot M0 没有训练过程或 entropy 指标。

## 通用面试中的“项目映射”规则

- 能在 project.md 找到源码、测试和数字，才说“Health-Copilot 已实现”。
- 只在 bagua.md 出现的概念，默认是通用知识，不自动变成项目经历。
- “当前项目未使用”不是缺点；能解释为什么 M0 先不用、哪一个真实 failure 可能驱动引入，通常比罗列名词更可信。
- 任何性能数字都必须同时说明数据集、split、分母、top-k、模型/索引版本和是否为内部 frozen eval。

## 参考源

- Agent： [ReAct 原论文](https://arxiv.org/abs/2210.03629)、[OpenAI Harness Engineering](https://openai.com/index/harness-engineering/)、[MCP tools specification（2026-07-28）](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)。
- RAG/检索： [RAG 原论文](https://arxiv.org/abs/2005.11401)、[BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)、[BEIR 原论文](https://arxiv.org/abs/2104.08663)、[HNSW 原论文](https://arxiv.org/abs/1603.09320)。
- Transformer/推理： [Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[PagedAttention/vLLM 原论文](https://arxiv.org/abs/2309.06180)。
- 分布式/后训练： [PyTorch FSDP 文档](https://docs.pytorch.org/docs/stable/fsdp.html)、[ZeRO 原论文](https://arxiv.org/abs/1910.02054)、[PPO 原论文](https://arxiv.org/abs/1707.06347)、[DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300)、[GSPO 原论文](https://arxiv.org/abs/2507.18071)。
