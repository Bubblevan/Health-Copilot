# Roadmap

## M0 — Safety-Gated Evidence RAG

当前里程碑，目标是稳定的、非自主的 vertical slice：

- KnowledgeCard schema/loader；
- 中文 tokenizer 和直接实现的 BM25 baseline；
- evidence-aware Generator Protocol；
- deterministic citation-ID verification；
- urgent / prescription pre-gate；
- no-evidence / malformed generation / fabricated citation 的 abstention；
- synthetic tests 和离线 route/retrieval eval。

## M1 — Agent Core（planned）

只在 M0 的检索、证据和安全契约稳定后实现 `AgentState`、受限 `AgentLoop`、ToolRegistry
和 Session。Agent Loop 应由失败模式驱动，例如只允许一次 query rewrite recovery，而
不是为了“有 Agent”而添加 ReAct。

## M2 — Harness Runtime（planned）

把 policy、capability/permission、step/token/time/tool budget、timeout、fallback、
trace/replay 做成可测试的运行时约束。YAML 可以是配置格式，但不应替代代码级 invariant、
测试和反馈闭环。

## M3+ — Evaluation and system extensions（planned）

后续可按真实失败案例依次增加：evaluation harness、plugin/provider boundaries、复杂证据
研究的 Agent Team、context/memory、dense/hybrid retrieval + reranker，以及视觉证据输入。
任何 post-training 都必须建立在固定评测集、trajectory/failure 数据和合规数据许可之上。

M0 不实现医疗诊断、处方、真实患者记录或临床验证。
