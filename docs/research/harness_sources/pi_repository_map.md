# Pi Repository Map — H-Ref Audit

Pinned source: `https://github.com/earendil-works/pi.git`, `main@898ab804050730e9dcefb4443875d5a932aa6a32`, MIT. The source identity and retrieval evidence are recorded in [`source_manifest.json`](source_manifest.json). All observations below are static source observations at that commit.

## Architectural map

| Concern | Source location and symbols | Verified behavior | H-Ref classification |
| --- | --- | --- | --- |
| Agent loop | `packages/agent/src/agent-loop.ts:37-149`, `agentLoop`, `runAgentLoop`, `runLoop` | The loop owns the message-to-provider turn cycle and emits typed lifecycle events. It transforms `AgentMessage` to provider `Message[]` only at the provider boundary. | ALREADY_EQUIVALENT in role, different lifetime model |
| Durable harness | `packages/agent/src/harness/agent-harness.ts:111-169, 538-622`, `OperationRequest`, `AgentLane`, `AgentHarness` | A harness accepts operations, drives them, exposes lanes, queues, abort/resume, compaction, navigation, hooks, events, and resource registries. Acceptance and execution are separate. | USEFUL_GAP for future E2/E3 durability; not a current Health replacement |
| Provider abstraction | `packages/ai`, `packages/agent/src/stream-fn.ts`, `AgentLoopConfig` and `StreamFn` | Provider/model identity and streaming are outside the loop. The loop receives a `streamFn`; model requests are normalized at the call boundary. | ALREADY_EQUIVALENT at the seam |
| Message/state model | `packages/agent/src/types.ts`, `packages/agent/src/agent.ts` | `AgentMessage` is the in-loop vocabulary; mutable agent state and pending-message queues are separate from the durable session. | INTENTIONAL_DIFFERENCE: Health separates AgentSession and AgentState explicitly for medical bounded execution |
| Session storage | `packages/agent/src/harness/session/session.ts:224-475`, `StorageBackedSession`; `session/types.ts` | Session is a storage-backed append-only entry tree with named branches, values/lists, mutation serialization, and explicit close. `Context` is required on async public operations. | USEFUL_GAP for E2 branch lineage; Health's event store remains the authority for product history |
| Branch/fork | `packages/agent/src/harness/session/fork.ts:5-101`, `createForkSnapshot`; `createBranch` in `StorageBackedSession` | A branch moves a named tip over shared immutable entries. Fork snapshots select durable content and bound values for a new destination. | EXPERIMENT_LATER for research workers; not a substitute for Health's revision-checked session fork |
| Context construction | `packages/agent/src/harness/agent-harness.ts:514-536`, `AgentHarnessOptions`; `agent-loop.ts` | Context contains messages, model, active tools, resources, stream options, retry and compaction settings, tool context, and a caller-supplied conversion hook. | ALREADY_EQUIVALENT for provider context seams; Health adds stronger evidence/protection semantics |
| Compaction | `packages/agent/src/harness/compaction/compaction.ts:147-249, 614-780`, `CompactionSettings`, `prepareCompaction`, `compactWithRequest` | Compaction estimates tokens, chooses a cut point, preserves a retained tail, supports split turns, records a summary/retained tail as a session entry, and can call a model for summary generation. | USEFUL_GAP as a bounded long-context strategy; model summarization is ANTI_PATTERN_FOR_HEALTH until evaluated and evidence-safe |
| Tool registration | `packages/agent/src/harness/types.ts`, `AgentHarnessOptions.tools`, `AgentHarness.getTools/setTools` | Tools are explicit typed registrations, not directory discovery. | ALREADY_EQUIVALENT with Health's explicit ToolRegistry |
| Tool execution | `packages/agent/src/harness/execution/tools.ts:8-205`, `prepareToolCall`, `applyBeforeToolDecision`, `executeToolCall`, `finalizeToolCall` | Tool calls pass through lookup, argument preparation/validation, pre-decision, an abort-aware effect gate, execution, post-patch, and canonical `ToolResultMessage` creation. | USEFUL_GAP: structured lifecycle hooks could help E2 policy boundaries |
| Error normalization | `execution/tools.ts:60-75, 124-180`; `agent-loop.ts:470-620` | Missing tools, invalid arguments, aborted calls, and thrown tool errors become structured tool results; the model can see the result and decide what to do next. | ALREADY_EQUIVALENT in principle; Health additionally fail-closes product outcomes |
| Extension lifecycle | `packages/agent/src/harness/hooks.ts`, `HookRegistry`; `agent-harness.ts` hooks/events | Hooks are registered on a harness and receive explicit context; the harness exposes passive events and lifecycle hooks. | USEFUL_GAP for future controlled instrumentation; dynamic authority mutation is not implied |
| Context propagation | `packages/agent/src/harness/context.ts:1-37`, `withTelemetryContext`; Pi/Chord `Context` | Async calls carry explicit `Context`; telemetry parentage and cancellation are derived from it rather than retained implicitly by shared receivers. | USEFUL_GAP for worker attribution |
| Telemetry | `packages/agent/src/harness/telemetry.ts:42-622`, `startAiSpan`, `startHarnessSpan`; `packages/telemetry` | Typed schemas cover AI/provider and harness spans/events. A telemetry context can be attached to a call context. | EXPERIMENT_LATER for E2/E3 traces; Health's metadata-only policy remains authoritative |
| Storage/durability | `packages/agent/docs/harness.md:23-54`, `packages/agent/src/harness/session/commit.ts` | Durable operations use atomic commits, complete operation state replacement, intent/effect/settlement for uncertain external effects, and terminal cleanup. | USEFUL_GAP for resumable research workers; complexity is high |
| Sandbox/permissions | Root `README.md` and `packages/agent/docs/harness.md` | Pi does not provide a built-in permission system; its coding-agent deployments run with process permissions and recommend external containerization/sandboxing. | INTENTIONAL_DIFFERENCE / ANTI_PATTERN_FOR_HEALTH as a security authority |
| Subagents/orchestration | `packages/agent/src/harness/pico3/*`, harness docs, agent tools | The pinned repository contains bounded harness/pico3 concepts and tool/lane machinery, but no evidence here that its generic runtime is a clinical evidence orchestrator. | OUT_OF_SCOPE for direct adoption |
| Skills/instructions | `packages/agent/src/harness/skills.ts`, `AgentHarnessResources` | Skills and prompt templates are explicit resources attached to a harness; they are not equivalent to medical Evidence. | INTENTIONAL_DIFFERENCE |
| Configuration | `AgentHarnessOptions`, lane configuration, model identity | Configuration is carried into harness/lane state and may be changed through explicit methods; model identity is resolved through a model registry. | USEFUL_GAP for research variants, but Health profile hashes stay frozen |

## Pi end-to-end turn trace

1. A caller invokes `AgentLane.prompt()` or `accept({kind: "prompt"})`; acceptance creates an operation and queues the input. `AgentLane.drive()` owns the actual execution, so a durable operation can exist without a live process-local driver (`agent-harness.ts:111-169, 538-580`). **FACT.**
2. The lane reconstructs the branch context and prepares the next turn. `runLoop()` receives current messages, pending steering/follow-up messages, model configuration, active tools, and a caller-owned `Context` (`agent-loop.ts:101-169`). **FACT.**
3. The provider adapter streams the assistant response through the supplied `StreamFn`. The loop emits assistant events and appends a settled assistant message to the session. **FACT**, supported by the loop and harness session contracts.
4. Assistant tool calls are prepared and validated; the harness applies pre-tool hook decisions, admits the external effect through an abort-aware gate, executes the tool, applies post-tool changes, and constructs a tool-result message (`execution/tools.ts:77-205`). **FACT.**
5. Tool results are appended to the message sequence. The next turn derives context from the updated session and may execute tools sequentially or in a bounded parallel pool (`agent-loop.ts:502-620`). **FACT.**
6. When the operation settles, the harness writes a terminal result and clears operation-owned state. A crash between transactions is recovered from the complete operation state, not by guessing from missing journal entries (`docs/harness.md:39-54, 82-114`). **FACT.**

## Session, context, extension, provider, and telemetry lifecycle

Pi has two related but distinct persistence layers. The session is the durable conversation tree and storage-backed values/lists; the harness operation is durable execution state. Branching changes the future context path without deleting the shared entry history. Compaction changes the context projection by adding a summary and retained tail; it does not erase the source entries (`docs/harness.md:23-54, 116-124`; `compaction.ts:633-706`). **FACT.**

The extension boundary is hook/event based. A tool can be rejected before the effect, transformed after execution, or normalized into a durable result. Harness-level hooks/events receive the caller context and are tied to the harness lifecycle. **FACT.**

Provider/model identity is resolved separately from the agent loop. The loop is provider-independent, while `pi-ai` owns provider request types, streaming, model identity, usage, and adapter errors. **FACT.**

Telemetry is request-scoped rather than an implicit global. `withTelemetryContext()` attaches a parent to the explicit Chord context, and typed AI/harness telemetry schemas attribute work to operation/tool/provider boundaries. **FACT.**

## Observed design principles

| Label | Principle | Evidence |
| --- | --- | --- |
| FACT | Durable execution is explicit: accepted operation, current total state, effect uncertainty, settlement, terminal result. | `packages/agent/docs/harness.md:27-54, 82-114` |
| FACT | Session context is tree-shaped and branchable; compaction is a projection entry, not deletion. | `packages/agent/docs/harness.md:23-25, 116-124`; `session.ts:349-453`; `compaction.ts:633-706` |
| FACT | The core loop stays provider-independent and delegates capability changes to tools/hooks/resources. | `agent-loop.ts:1-4, 37-149`; `agent-harness.ts:514-536` |
| FACT | Context, cancellation, and telemetry parentage are passed explicitly. | `context.ts:1-37`; `docs/harness.md:29-33` |
| INFERENCE | Pi optimizes for resumable coding work where external effects may be uncertain and context continuity is more valuable than a small runtime. | Inferred from the durable effect protocol and coding-agent sandbox disclaimer; this is not a product claim in source. |
| INFERENCE | Pi's session tree is a strong substrate for research lineage, but its generic context lacks Health's Evidence authority and safety routing. | Compared with Health `ContextProjector`, `EvidencePolicy`, and M10 closeout. |

## H-Ref conclusion

Pi supplies useful mechanisms, not a drop-in Health architecture: explicit operation durability, branchable sessions, lifecycle hooks, and context propagation are candidates for later research experiments. Its absence of a built-in permission authority, reliance on external sandboxing, and model-based compaction cannot be copied into Health's frozen safety/replay path.
