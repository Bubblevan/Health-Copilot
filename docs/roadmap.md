# Roadmap

## Harness Macros

宏观路线与里程碑路线分开记录，历史 M0–M5 artifact path 不改名：

| Harness Macro | 主题 | 当前/计划映射 |
| --- | --- | --- |
| H0 | Vertical Slice | M0 |
| H1 | Agent Core | M1 |
| H2 | Harness Runtime | M2 + M3 + M4 |
| H3 | Extensible / Plugin Runtime | M5 + M6 |
| H4 | Agent Eval System | M7 |
| H5 | Agent Team | M8 |
| H6 | MCP / Sandbox / Permission | M9 |
| H7 | Context / Memory | M10 |
| H8 | Post-training | M11 |
| H9 | Multimodal | M12（可选） |

M7 已完成 H4；M8/H5 已实现为实验性、受限的 Agent Team。M9 及之后的条目仍是 future work。

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

## M3 — Capability-Aware Harness & Claim-First Runtime Materialization（implemented）

M3 基于 M2 观察到的 OOD false recovery 与 coverage-missing false accept，增加 reviewed closed-corpus
KnowledgeScope、scope-aware policy validation、claim-first final wire contract、claim-support-only verifier
和 deterministic materializer。它仍只有一个 `search_knowledge` tool、最多两次模型 turn 和一次工具调用；
不包含 web search、memory、MCP、Agent Team、dense retrieval 或新的 plugin framework。M2 的 coverage
path 保持可重放，M3 的用户可见事实只来自 verified claims。

M3 empirical closeout 应按 `runs/m3/` 的 capability、claim-support 与 focused M2-vs-M3 artifacts
解读：claim-first 是消除自由 answer/claim coverage mismatch 的结构性路径改动，不是独立 verifier 或
完美 semantic claim support 的声明。

M3 runtime implementation 已冻结。最新 3-trial focused diagnostic 观测到 M3 OOD tool execution `0/12`、
expected answers `18/18`、unexpected abstains `0/18`、mean model turns `1.30` 和 mean tool executions
`0.30`；较早、同配置的小样本曾观测到 `15/18` expected answers。这些随机的小包诊断仅用于 regression，
不是稳定 answer-rate 或泛化能力声明。当前 freeze 与 evaluation-QA 状态见 `docs/m3_evaluation_freeze.md`。

## M4 — Budgeted & Replayable Harness Runtime（implemented）

M4 添加 all-live-adapter provider boundary、pre-side-effect deadline/call/observed-cumulative-token guards、guarded tool dispatch、
privacy-tiered JSONL trace、strict public-evaluation provider/tool replay，以及 deterministic offline failure
injection。它保留 M0–M3 contracts，未增加 Agent capability。固定六条 M4 record/replay pack 仅是 regression
diagnostic，不是 generalization result。

M4 FINAL FROZEN：`main@51f0ee5a9352c5cabd360d302a2cef1eb9b5da25`。M4.1 明确固定 SDK
`max_retries=0` 与所有 adapter 的 30-second provider timeout；任何未来 retry 都必须由 Harness
显式实现并计入 trace/budget，不能依赖 SDK hidden retry。

## M5 — Hybrid Retrieval & Evaluation Scale（frozen）

M5 has introduced a generic `RetrievalDocument`, frozen BM25 baseline, optional dense retrieval, RRF hybrid fusion,
optional reranking, index provenance, an 80-case component-derived reviewed retrieval suite, retrieval-only ablation
tooling, and a four-arm M3/M4 focused end-to-end diagnostic. Hybrid+CrossEncoder wins these retrieval diagnostics, but
BM25 remains the default because of latency and focused-pack limits. See `docs/m5_retrieval.md`; M6 now formalizes these
frozen alternatives as explicit runtime profiles.

## M6 — FROZEN

H3 Extensible / Plugin Runtime — COMPLETE

Final freeze checkpoint: `main@3872dd94fa048de3007677fc9e63373488e99955`.

M6 是 H3 的第二阶段，也是 M5 之后的第一个通用 subsystem replacement runtime：

- `RuntimeProfile` 只包含 declarative component ID 与 JSON 配置；
- `ComponentRegistry` 仅使用源码显式注册的 trusted in-process factory，不扫描、安装或动态导入；
- `RuntimeBuilder` 一次构造 provider、retriever、policy、verifier、tool、trace 组件；
- `ComponentIdentity`、`LearnedArtifactIdentity` 与 canonical `ComponentManifest` 提供 profile/component provenance；
- manifest hash 进入 `RunIdentity.config_hash`、`RUN_START` metadata 与 replay compatibility；
- role-aware model routing 通过同一个 `ProviderExecutor` 明确区分 Agent/Generator、Policy 与 Verifier；
- `RuntimeComponents.answer()` 为每次请求创建新 `RunContext`，组件不保存可变的 per-run runtime；
- code commit 与 replay 初始 Evidence 内容哈希纳入 provenance，profile/manifest/code commit 不匹配时 replay fail closed；
- `--profile` 区分 hashing/token-overlap demo 与 SentenceTransformer/CrossEncoder learned profile；缺失依赖/模型、显式 revision 或 index 不匹配只会 fail closed，未解析 revision 明确标记为 non-frozen；
- `search_knowledge` 仍是唯一 product Agent tool，`ToolRegistry` 仍负责运行期 dispatch，未被 ComponentRegistry 替换。

M5 的 80-case suite 仍是 component-derived regression/evaluation suite，不是独立 external generalization test；
6-case E2E arm comparison 仍是 compatibility diagnostic，不能从一次随机小样本把 provider call count 或端到端
latency 差异因果归因给 retrieval。M5 metrics 与 BM25 default 决策保持冻结。

组件生命周期、单次 RunContext 生命周期和未来的 Session/Memory 生命周期明确分离；Memory 尚未实现。

## M7 — FROZEN

H4 Agent Eval System — COMPLETE。M7 在 M6 运行时之外建立统一、显式注册的评测平面：

- `EvalSuiteRegistry`、`EvalRunSpec`、`EvalCase` 和 offline/live/replay 执行模式；
- 无 LLM judge 的确定性 graders、可检查 numerator/denominator 的 `MetricResult`；
- `trajectory_v1`、分阶段 FailureRecord、metadata-only 默认隐私策略；
- M0 parity、M5 retrieval parity 与 M4 recorded-exchange replay 的统一 artifact bundle；
- `health-eval list/run/compare`，不扫描、不动态导入、不安装 eval plugin。

M7 不改变 M0–M6 运行时、gold、检索配置、policy 或 replay 语义；M5 metrics 仍冻结。
M0–M5 的历史 evaluator 和 artifact path 保留。M7.1 已完成 target-specific dispatch、
proposal/execution provenance、trajectory_v1、eval-run/execution-run identity、
standalone grounding parity 与 executable budget closeout；功能冻结 checkpoint 为
`main@0d4656c9fc920f1a1c8639bac33c49d5d27f44aa`。
## M8 — FROZEN

H5 Bounded Agent Team 已实现：

- `m8-team-bm25-v1`：Team Lead、Evidence Worker、Guideline Worker 的单轮顺序编排；
- runtime-owned `TaskStore`、typed `Mailbox`、worker context isolation 和 `TeamEvidenceLedger`；
- Lead 两次调用、最多两个任务/worker、worker 两轮/一次工具调用的硬边界；
- Lead/Worker 与既有 M4 ProviderExecutor、ToolRunner、EvidencePolicy、KnowledgeScope、RunBudget 共用父级运行上下文；
- worker citation provenance 以及 M3 claim-support final boundary；
- `m8-agent-team-focused-v1` 的 L0 workflow / L1 frozen single Agent / L2 team 比较定义；
- candidate annotation manifest、Team-specific trace events 和 cost/topology metrics；
- M8.3 Team-Lead contract diagnosis、fail-closed failure taxonomy 与 v2 frozen comparison。

H5 Agent Team — COMPLETE。M8 仍不替换 `m3-bm25-default` 产品默认配置；Team worker
调度有意保持顺序。M9 MCP/Sandbox/Permission、M10 Memory、M11 post-training 和
M12 multimodal 均未启动。v1 失败审计与 v2 closeout 见
`docs/m8_empirical_v1_failure.md`、`docs/m8_empirical_v2_closeout.md` 和
`docs/m8_agent_team.md`。

M0 不实现医疗诊断、处方、真实患者记录或临床验证。

## M9.1 — H6 FINAL CLOSEOUT

M9 增加了严格 pinned 的 MCP `2026-07-28` capability boundary，使用官方
`mcp==2.2.0` Python SDK，并保持 `m3-bm25-default` 为产品默认。新增的
`m9-mcp-search-bm25-v1` 仅通过显式 opt-in 使用 read-only MCP-backed
`search_knowledge`；AgentLoop、M3 policy/verifier、KnowledgeScope 和 BM25
语义保持不变。

H6 的三个边界分别实现并验证：

- MCP：`server/discover`、`tools/list`、`tools/call`、structured schema/result、catalog hash/TTL/cache provenance，以及 Streamable HTTP routing headers；
- Permission/Approval：trusted local default-deny `PermissionPolicy`、`ALLOW`/`DENY`/`REQUIRE_APPROVAL`、argument-hash-bound `ApprovalProvider`；
- Sandbox：真实 WSL2 Ubuntu 24.04 Bubblewrap stdio fixture，独立验证 allowed read、workspace write、outside filesystem deny 和 network deny；required backend 缺失时 fail closed。

M9.1 完成了 H6 closeout：RuntimeProfile 显式选择 MCP/Permission/Sandbox
组件，保留 M0–M8 profile hash，修正 sandbox failure taxonomy，并将
`m9-mcp-security-v1` 注册为 M7 offline security suite。GitHub Actions 仅
验证平台无关的 protocol/permission/unit contract；WSL-specific
containment tests 在 CI 中 skip，并由本地受支持主机上的 WSL2/Bubblewrap
smoke 单独证明。完整边界与 source alignment 见
`docs/m9_mcp_security.md`。

M9 不实现 remote OAuth（文档明确为 NOT IMPLEMENTED）、dynamic topology、
decentralized MAS、A2A、Memory、post-training 或 multimodal。M10 Memory、
M11 post-training、M12 multimodal 均未启动。
