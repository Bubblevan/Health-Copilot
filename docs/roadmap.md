# Roadmap

## M0 — Safety-Gated Evidence RAG（implemented）

当前里程碑，目标是稳定的、非自主的 vertical slice：

- KnowledgeCard schema/loader；
- 中文 tokenizer 和直接实现的 BM25 baseline；
- evidence-aware Generator Protocol；
- deterministic citation-ID verification；
- urgent / prescription pre-gate；
- no-evidence / malformed generation / fabricated citation 的 abstention；
- synthetic tests 和离线 route/retrieval eval。

## M1 — Bounded Agent Core + One-Step Retrieval Recovery（implemented）

M1 已在 M0 契约之上实现：

- typed `AgentState`、typed messages、in-memory `AgentSession`；
- 显式 `ToolSpec`、参数校验、`ToolRegistry` 和结构化 `ToolResult`；
- `max_model_turns=2`、`max_tool_calls=1` 的顺序 `AgentLoop`；
- 唯一只读 `search_knowledge(query)` 工具，以及初始/恢复 Evidence union citation 校验；
- 生命周期事件与离线 FakeAgentModel 测试矩阵；
- M0 path 仍可通过旧 Generator 重放，CLI 提供 `--mode m0` 和 `--mode m1`。

M1 的 Agent action 只针对 M0 观察到的 lexical query-expression mismatch；它不解决 OOD
false retrieval，也不声称解决 evidence sufficiency 或临床安全性。

## M2 — Evidence Policy & Grounding Harness（implemented）

M2 在 M1 的单步 recovery 前设置 EvidencePolicy runtime veto，并在最终回答前做 claim
coverage 与 claim-level grounding。policy 和 verifier 都是 runtime authority，不是 Agent tool；
提案次数与真实工具执行次数分别计数，所有 policy/verifier/provider 失败都会 fail closed。
M0/M1 的旧 final contract 保持可重放。

更广泛的 time/token/cost budget、permission、sandbox、持久化 trace/replay 移到后续 Harness
Runtime 里程碑；M2 不因此扩展 Agent action space。

## M3+ — Evaluation and system extensions（planned）

后续可按真实失败案例依次增加：evaluation harness、plugin/provider boundaries、复杂证据
研究的 Agent Team、context/memory、dense/hybrid retrieval + reranker，以及视觉证据输入。
任何 post-training 都必须建立在固定评测集、trajectory/failure 数据和合规数据许可之上。

M0 不实现医疗诊断、处方、真实患者记录或临床验证。
