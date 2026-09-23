# Health-Copilot Repository Map — H-Ref Audit

Pinned baseline: `https://github.com/Bubblevan/Health-Copilot.git`, `main@7a7703edf6f85d61c432ffba00e61d750c216048`. M0–M10.1 and E0/E0.1 are frozen at the audit boundary. The map below describes the existing implementation; it does not authorize a runtime change.

## Trust and lifetime map

Health keeps four state planes separate: `RunContext` for one execution, ephemeral `AgentSession` for the provider-facing bounded transcript, persistent `SessionStore` events for cross-run interaction history, and `MemoryStore` for explicitly materialized typed state. This separation is documented in `docs/m10_context_memory.md` and implemented across `runtime/context.py`, `agent/session.py`, `runtime/session.py`, and `runtime/memory.py`.

| Concern | Health source and symbols at `7a7703edf6f85d61c432ffba00e61d750c216048` | Owner/lifetime/authority | Failure or safety behavior |
| --- | --- | --- | --- |
| Agent loop ownership | `src/health_ai_copilot/agent/loop.py:29-107`, `AgentLoop`, `AgentLoopConfig`, `AgentRunResult` | One explicit, sequential product loop. The loop owns bounded model/tool turns; the pipeline owns safety, retrieval and final verification around it. | `max_model_turns <= 2`, `max_tool_calls <= 1`; model/tool/budget failures stop or abstain. |
| Message/state model | `src/health_ai_copilot/agent/messages.py`, `agent/session.py`, `agent/state.py` | `AgentSession` is an ephemeral typed transcript. `AgentState` owns counters, stop reason, draft and evidence observations. Neither is long-term memory. | Typed message/result validation and bounded stop reasons prevent prose tool invocation and unbounded execution. |
| Provider abstraction | `src/health_ai_copilot/runtime/provider.py`, `generation/base.py`, `generation/openai_compatible.py`, `ProviderExecutor` | Shared transport side effects are behind a provider executor; model/domain adapters receive `RunContext` explicitly. | Live/replay/offline providers are explicit; replay mismatch and provider failures fail closed. |
| Runtime profile | `src/health_ai_copilot/runtime/profile.py:12-119`, `RuntimeProfile` | Frozen JSON-compatible selection of provider, retriever, policy, verifier, tools, trace, MCP, permission, sandbox, session, context and memory IDs. | Non-empty IDs and JSON-compatible config are validated; profile hash is stable. |
| Component registry | `src/health_ai_copilot/runtime/registry.py:12-126`, `ComponentRegistry`, `ComponentRegistration` | Static trusted factories keyed by typed component kind and ID. It does not scan directories, import arbitrary plugins or load YAML at runtime. | Duplicate/unknown/construction errors fail at build time. |
| Runtime builder/components | `src/health_ai_copilot/runtime/builder.py:118-220, 391-465`, `RuntimeBuildConfig`, `RuntimeComponents`, `RuntimeBuilder`, `SessionAnswer` | Long-lived components are built once; `RunContext`, budget, trace, AgentState and ephemeral session are run-local. `answer_in_session()` is opt-in. | Profile/component manifest mismatch and unsupported combinations fail closed. |
| Execution context | `src/health_ai_copilot/runtime/context.py:14-73`, `RunIdentity`, `RunContext` | Run-local control-plane state: budget, trace, profile/component identity and metadata. It is not memory or transcript. | Run budget and metadata identity are propagated explicitly. |
| Tool registry/runner | `src/health_ai_copilot/agent/tools.py`, `runtime/tools.py`, `tools/search_knowledge.py`; `ToolRegistry`, `LiveToolRunner`, `ReplayToolRunner` | Explicit `search_knowledge` registration, narrow schema, runner boundary for live/replay calls. | Unknown/duplicate/malformed calls and tool exceptions become structured observations; Health has no discovery loader. |
| Evidence policy | `src/health_ai_copilot/policy/evidence.py`, `EvidencePolicy`, `validate_assessment`; `knowledge/scope.py` | Evidence and citation authority remain in reviewed `KnowledgeCard`/`Evidence` objects. Memory, tool output, MCP output and web text are observations, not Evidence. | Invalid/unsupported/conflicting evidence and unsupported claims abstain. |
| Context selection | `src/health_ai_copilot/runtime/context_manager.py:17-42, 46-220, 226-506`, `ContextManager`, `ContextPlan`, `StructuredCompactorV1` | Deterministic selection under `ContextBudget`; protected current user, Evidence, system pins and unresolved tool exchange. Manager does not call provider/tool/memory write. | Budget exhaustion and unmatched tool groups raise typed context failures. |
| Context projection | `src/health_ai_copilot/runtime/projector.py:28-122, 124-345`, `ContextProjector`, `SessionContextProjector` | Sole boundary from a fresh plan to provider-visible `AgentMessage` tuple. It records plan/session/memory identities in run metadata. | Projection mismatch and atomicity violations fail closed; provider cannot receive an unprojected session transcript. |
| Persistent session | `src/health_ai_copilot/runtime/session.py:53-272, 272-424, 428-715`, `SessionEvent`, `SessionStore`, `InMemorySessionStore`, `SQLiteSessionStore` | Append-oriented interaction log with opaque ID, revision checks, batch turn commit, resume and prefix fork. SQLite uses schema `v1`, foreign keys and transactions. | Revision conflict, unknown schema, closed session and commit failure produce explicit errors/abstain paths; hidden reasoning is rejected. |
| Memory | `src/health_ai_copilot/runtime/memory.py:27-565, 752-966`, `MemoryRecord`, `MemoryPolicy`, `MemoryStore`, `SQLiteMemoryStore` | Narrow typed memory with source/run/session provenance, versioning, expiry, tombstones, scope and snapshot identity. Explicit operation boundary supports ADD/UPDATE/DELETE/NOOP. | Automatic writes from assistant/tool/MCP/retrieved text are denied; sensitive health writes require consent; bad schemas and conflicts fail closed. |
| Safety route | `src/health_ai_copilot/safety.py`; `RuntimeComponents.answer_in_session()` | Raw current question is routed before session resume, memory query, retrieval or provider call. | Urgent-care/human-review route is terminal and independent of potentially corrupt state. |
| MCP client | `src/health_ai_copilot/mcp/client.py`, `mcp/server.py`; `McpClient`, `McpToolAdapter` | Explicit client/server/tool adapter with protocol headers, TTL and structured server/tool identity. It enters the explicit tool registry. | Permission and protocol checks happen before external call; unknown/denied/expired capability fails closed. |
| Permission/approval | `src/health_ai_copilot/mcp/permissions.py`, `PermissionPolicy`, `PermissionGuard`, `ApprovalProvider` | Capability class and `(server_id, tool_name)` binding control allow/deny/require-approval. Approval is bound to request ID and argument hash. | Unknown capability denies; missing approval provider denies; binding mismatch denies; trace records decision metadata. |
| Sandbox | `src/health_ai_copilot/mcp/sandbox.py`, `SandboxProfile`, `SandboxPolicy`, `SandboxBackend`, `BubblewrapSandboxBackend` | Backend contract separates requested filesystem/network policy from real enforcement. Linux Bubblewrap and WSL backend are explicit; dev no-sandbox is not real containment. | Unavailable backend, missing roots, denied network and process failures are typed; real enforcement can be required. |
| Team orchestrator | `src/health_ai_copilot/team.py`, `AgentTeamOrchestrator`, `TaskStore`, `Mailbox`, `TeamEvidenceLedger`; `docs/m8_agent_team.md` | Optional M8 bounded lead/worker topology with isolated worker `AgentSession`s, parent budget, explicit task states and point-to-point mailbox. Team state is ephemeral, not Memory. | Lead/provider/JSON/contract failures have typed stop reasons; workers share policy/tool authority but not each other's sessions. |
| Trace | `src/health_ai_copilot/runtime/trace.py`, `RunTrace`, `TraceEventType`, `TraceContentPolicy` | Metadata-only by default; content recording is explicit and policy-scoped. Trace is run-lifetime evidence, not a hidden transcript store. | Medical content is rejected under metadata-only policy; close status is explicit. |
| Replay | `src/health_ai_copilot/runtime/replay.py`, `RecordingProviderExecutor`, `ReplayProviderExecutor`, `RecordingToolRunner`, `ReplayToolRunner` | Canonical request fingerprints and optional session revision/memory snapshot/plan hashes bind recorded exchanges to state. | Missing exchange, mismatch, extra live call or changed state fails closed. |
| Eval runtime | `src/health_ai_copilot/eval/registry.py`, `eval/schema.py`, `eval/system.py`, `EvaluationRunner` | Explicit suite registry separate from component registry; offline, live-opt-in and replay modes produce case records, trajectories, deterministic graders, metrics and failures. | Offline sentinel rejects accidental provider calls; metrics keep separate numerators/denominators and content policies. |

## Current Health execution traces

### Normal bounded execution

`Pipeline.answer()` performs input validation and safety route, retrieves initial Evidence, builds the explicit agent/tool graph, and invokes `AgentLoop.run()`. The loop may produce a structured final or one structural `search_knowledge` call followed by a tool observation and a second model turn. Final claims are checked against observed Evidence before materialization. `docs/architecture.md` M0/M1 sections and `agent/loop.py` define this as the frozen bounded product path.

### M10.1 session execution

`RuntimeComponents.answer_in_session()` first routes the raw question, then resumes the persistent session, queries typed memory, retrieves Evidence, creates `SessionContextProjector`, creates an ephemeral `AgentSession`, and runs the existing M3 loop. Each provider turn gets a new plan; the projector returns the exact messages sent to the provider. The turn commits `USER_INPUT`, tool observations and final output in one expected-revision batch (`runtime/builder.py:220-390`; `runtime/projector.py:81-122`; `runtime/session.py:247-267`).

### Authority boundaries

- The `RuntimeProfile` selects trusted components; it does not grant a model authority.
- The `ComponentRegistry` constructs components; it does not discover arbitrary code.
- The `ContextManager` selects context; `ContextProjector` is the only provider-facing projection boundary.
- `EvidencePolicy` and reviewed `KnowledgeScope` decide what can support a medical claim.
- `MemoryPolicy` decides what can become durable memory; observations do not auto-promote.
- `PermissionGuard` decides MCP capability access before the client call.
- `SandboxBackend` reports/enforces process confinement; a requested policy is not treated as proof of containment.
- `EvaluationRunner` is outside the product execution graph and does not silently change production components.

## Intentional differences already supported by the source

1. Health uses a static trusted registry instead of DSH's dynamically composable plugin tree. This is intentional: Health profile hashes, replay identity and clinical authority depend on a small reviewable trust root.
2. Health treats Evidence as an authority-bearing object with citation and claim-support validation. Pi and DSH treat general context/session messages as model input; their context mechanisms cannot be considered Evidence mechanisms.
3. Health fails closed when protected context cannot fit, a tool exchange is incomplete, a safety state is unavailable, or replay identity changes. General coding harnesses may continue with recovery text or external sandbox assumptions; that is not acceptable evidence for a medical answer.
4. Health's Team runtime is bounded, sequentially coordinated and tied to the existing EvidencePolicy/KnowledgeScope/ToolRunner. DSH's child agents and Pi's harness lanes are more general execution substrates, not proof that dynamic swarm behavior improves this product.
5. Health's Eval runtime is explicit, deterministic and offline by default. Interactive telemetry or a successful live coding-agent run is not a replacement for the frozen M10.1 metrics.

## Audit status

This map is an architectural record only. No Health runtime code, tests, benchmarks, eval inputs, profile IDs, component registrations, MCP implementation, memory implementation or Team runtime was changed for H-Ref. E1, E2, E3 and M11 remain unstarted at this audit boundary.
