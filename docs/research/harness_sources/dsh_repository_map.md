# DeepSeek Harness Repository Map — H-Ref Audit

Pinned source: `https://github.com/deepseek-ai/deepseek-harness.git`, `master@00102833dfaee1da9f48a3a8eae9d34005a75218`, MIT. The source identity and retrieval evidence are recorded in [`source_manifest.json`](source_manifest.json). The audit is based on static inspection of the pinned checkout; no DSH installation or runtime was executed.

## Cordis kernel and composition model

| Concern | Source location and symbols | Verified behavior | H-Ref classification |
| --- | --- | --- | --- |
| Kernel responsibility | `vendor/cordis/src/context.ts`, `registry.ts`, `fiber.ts`, `service.ts`; `Context`, `RegistryService`, `Fiber`, `Service` | Cordis owns context/service lookup, plugin registration, dependency waiting, event dispatch, fiber ownership, reversible effects, and disposal. It does not own the medical domain, model policy, evidence, or a product-specific agent algorithm. | USEFUL_GAP as a composition mechanism; not a reason to replace Health runtime semantics |
| Plugin contract | `vendor/cordis/src/registry.ts:92-135, 165-176, 300-330`, `Plugin`, `inject`, `Context.plugin` | A plugin can be a function, constructor, or object with `apply`; it may declare injected services. The registry creates a runtime record and a fiber for each mounted plugin. | FACT; dynamic loading must be bounded by a trusted composition plane in Health |
| Lifecycle/effects | `vendor/cordis/src/fiber.ts:403-418`, `Fiber.effect`; `docs/cordis-primer.md:9-13, 42-45` | Effects and event listeners are owned by a fiber and disposed in reverse/lifecycle order. Registrations return or are attached to disposers. | USEFUL_GAP for temporary worker scopes |
| Dependency resolution | `registry.ts`, `Fiber.inject` handling; `docs/cordis-primer.md:9-13` | `inject` expresses service dependencies; a plugin waits for required services before activation. | USEFUL_GAP for E2 capability graphs; static Health registry currently solves startup selection differently |
| Scope/context | `packages/core/scope/src/index.ts`, `store.ts:153-264`; `scopeOf`, `scopeChainOf`, `ScopedLayers` | Contributions can be global or scope-specific. Effective values are materialized through a parent chain, with exact-scope contributions winning and `ctx.effect()` owning removal. | USEFUL_GAP for per-worker tool/prompt policy; security review required |
| Typed event system | `vendor/cordis/src/events.ts`; `packages/core/tools/src/index.ts:139-205`; `docs/cordis-primer.md:12, 17-35` | Events have dispatch modes including emit, waterfall, parallel, serial, and bail. Waterfall middleware must call `next()` to delegate; policy listeners may short-circuit intentionally. | USEFUL_GAP for explicit interception; event ordering must remain deterministic |

## Agent assembly and actual turn path

1. `dsh-agent` defines the public `Agent` handle, `AgentRegistry`, creation/resume contracts, agent-scoped identity, and agent events (`packages/core/agent/src/index.ts:14-18`, `Agent`, `AgentRegistry`, `AgentFactory`). **FACT.**
2. `dsh-agent-loop` mounts the concrete `ReactLoopAgent`. Its constructor binds a session, creates an agent scope, initializes the inbox and projections, and prepares the loop dispatcher (`packages/core/agent-loop/src/agent.ts:96-137`). **FACT.**
3. An agent creation transaction runs setup before publication, then announces session/agent creation; teardown stops and drains the loop, closes session write ownership, unregisters the agent, and unwinds the scope (`packages/core/agent-loop/README.md:109-115`; `packages/core/agent/src/index.ts`, creation contract). **FACT.**
4. `ReactLoopAgent` claims queued input at turn/step boundaries, runs `agent/pre-step`, assembles the prompt and tools, calls `ctx.llm.prepareCall()`, logs request headers/context, streams the provider response, and appends accepted events to the session (`packages/core/agent-loop/README.md:117-127`; `agent.ts:251-260`). **FACT.**
5. Tool calls enter `ctx.tools` and traverse pre-execute, execute, post-execute, and result events. The loop then derives the next model-visible context from the durable session (`packages/core/tools/src/index.ts:139-205`; `packages/core/agent-loop/README.md:73-75`). **FACT.**

The important semantic rule is **model-visible means logged**: the session event log is the source from which requests are reconstructed, while live events are extension and UI signals (`docs/architecture.md:109-125`; `packages/core/session/README.md`, `Session`, `deriveMessages`). This is analogous to Health's “plan then project” rule, but DSH makes the log a broader general-purpose model history rather than an evidence authority.

## Capability packages

| Capability | Source location and symbols | Verified behavior | H-Ref classification |
| --- | --- | --- | --- |
| Provider/model | `packages/llm/llm/src/index.ts`, `assembler.ts`, `assistant-stream.ts`, `error.ts`, `types.ts`; `ctx.llm` | LLM message vocabulary, adapter selection, prepared calls, streaming, usage and adapter failures are a service seam consumed by the loop. | ALREADY_EQUIVALENT at architectural level |
| Session/durability | `packages/core/session/src/types.ts:19-159`, `fork.ts:21-30`, `Session`; `docs/subsystems/session.md` | The session is an append-only typed event log with branded sequence positions, format versioning, derived projections, persistence adapters, fork seed construction, repair and flush. `SessionEventMap` is the durable extension vocabulary. | ALREADY_EQUIVALENT in authority, USEFUL_GAP in richer event/projector semantics |
| Tools | `packages/core/tools/src/index.ts:212-270, 319-480`, `ToolDefinition`, `ToolRuntime`, `ToolRuntimeScheduler` | Tool schemas, outputs, presentation, nested execution, abort signals, scoped dispatch and pre/around/post/result policy hooks are explicit. | USEFUL_GAP for E2 tool policy and attribution |
| System prompt/context | `packages/core/system-prompt`, `packages/core/agent-loop/src/runtime-context.ts`; `ctx.systemPrompt` | Prompt sections, variables and runtime context are composable contributions; request reconstruction records the actual call envelope. | INTENTIONAL_DIFFERENCE: Health memory is data-bearing and cannot become authority or prompt instructions |
| Sandbox | `packages/sandbox/sandbox-policy`, `sandbox-local`, `sandbox-windows-acl`, `packages/shell`; `ctx.sandbox` | Sandbox is a replaceable capability selected by profile/bundle. Consumers wrap subprocesses; the policy/backend boundary distinguishes requested policy from enforcement availability. | ALREADY_EQUIVALENT in M9 separation; DSH has broader production capability surface |
| Permission/approval | `packages/bundle/base`, approval and permission packages, session policy event types; delegated policy in `packages/subagent/subagent/src/child-agent.ts:221-275` | Access and approval are composed capabilities, and delegated children receive pinned policy events. | USEFUL_GAP for future worker authority propagation |
| MCP | `packages/mcp/mcp-client/src/connection.ts`, `server-context.ts`, `tools.ts`, `transport.ts` | MCP servers are capability providers whose tools enter the same tool runtime and lifecycle. | ALREADY_EQUIVALENT in intent; Health has a smaller, explicit client/control plane |
| Presets/bundles | `packages/bundle/*`, `packages/preset/agent-preset-registry/src/*`, `docs/architecture.md:15-31` | Profiles compose ordered bundles and patches. Agent presets can compose per-session prompt/tool restrictions and isolate realms. | USEFUL_GAP for research variants; dynamic plugin install is ANTI_PATTERN for frozen Health runs |
| Subagents | `packages/subagent/subagent/src/child-agent.ts:178-218, 221-275`, `lifecycle.ts`; `packages/subagent/subagent/src/*` | Children join parent composition, receive narrower persona/tool/policy state, have durable parent/child lineage and lifecycle start/end observation, and can be one-shot or continuable. | USEFUL_GAP for E2 worker scopes |
| Scheduling/jobs | `packages/schedule/schedule/src/*`, `packages/jobs/*`, `packages/workflow/*` | Scheduling and workflows are capability packages. The agent loop does not silently become a platform scheduler; the serving layer and plugins own scheduling. | EXPERIMENT_LATER for E3 bounded task execution |
| Telemetry | `packages/telemetry/*`, `packages/agent/src/harness/telemetry.ts` analogue not used by DSH; `docs/architecture.md:76-82, 109-125` | Session events are durable facts; agent/capability events are live interception and observation. | ALREADY_EQUIVALENT in separation; DSH has richer event composition |

## Plugin, preset, and failure traces

### Plugin lifecycle trace

1. A profile resolves ordered bundle rows and patches (`docs/architecture.md:15-31`). **FACT.**
2. Cordis mounts a plugin after injected services are available. The plugin receives a context and config and registers services, events, tools, adapters, or effects (`registry.ts:92-135, 300-330`; `cordis-primer.md:9-13`). **FACT.**
3. The plugin's fiber owns its registrations. `ctx.effect()` and `ctx.on()` attach cleanup to the fiber; disposing or unloading the fiber removes them (`fiber.ts:403-418`; `cordis-primer.md:42-45`). **FACT.**
4. Capability packages expose services through stable context keys. Consumers depend on service roles instead of importing one concrete implementation (`docs/architecture.md:59-80`). **FACT.**
5. A configuration/profile change can replace a plugin row or reload a plugin tree. The effect ownership model provides the rollback/removal mechanism. **FACT**, with exact reload policy owned by the loader package.

### Failure propagation

| Failure | Observed destination | Semantic result |
| --- | --- | --- |
| Missing dependency | Cordis plugin activation / fiber state | Plugin does not become active; the dependency contract fails before normal capability use. |
| Plugin setup/creation failure | Agent creation rollback and fiber disposal | Prepared session/agent scope is unpublished or unwound; partial creation is not exposed (`agent-loop` creation contract). |
| Tool validation/policy failure | `ctx.tools` execution result and tool events | The call becomes a structured tool failure or policy result; the loop can record it and continue or end according to the tool contract. |
| Provider/adapter failure | `agent/request-error`, terminal request outcome, optional retry listener | Retry is an explicit event decision; unhandled request failure terminates the turn (`agent-loop/README.md:123-127`). |
| Lifecycle listener failure | Package-specific contained listener or plugin lifecycle failure | DSH distinguishes contained observer failures from failures that abort creation/teardown; it does not treat every exception as a medical answer. |

## DSH design principles

| Label | Principle | Evidence |
| --- | --- | --- |
| FACT | Cordis kernel owns composition, dependency activation, event dispatch, and reversible lifecycle effects. | `vendor/cordis/src/context.ts`, `registry.ts`, `fiber.ts`, `service.ts`; `docs/cordis-primer.md:7-45` |
| FACT | Model, tools, session, sandbox, MCP, scheduling and subagents are capability packages mounted into the context. | `docs/architecture.md:59-80, 145-162`; package maps above |
| FACT | Per-agent scope controls visibility and teardown of registrations. | `packages/core/scope/src/store.ts:153-264`; `agent-loop/src/agent.ts:102-137` |
| FACT | Durable model-visible state is session-log state; transient stream/lifecycle signals are separate live events. | `docs/architecture.md:76-125`; `core/session` and `core/agent-loop` docs |
| FACT | Presets and bundles make composition configurable, but profile rows and package contracts remain explicit. | `docs/architecture.md:15-31`; `packages/preset/agent-preset-registry/src/*` |
| INFERENCE | “Everything is a Plugin” describes the capability assembly plane, not a claim that the Cordis kernel is itself replaceable by an untrusted model. | The kernel still owns context/fiber/event invariants; capability plugins declare contracts and are loaded by profiles. |
| INFERENCE | DSH's composition model is optimized for a broad coding harness with many capabilities and surfaces, while Health needs a narrower trust root and smaller model-visible vocabulary. | Compared with Health `RuntimeProfile`, `ComponentRegistry`, `EvidencePolicy`, M9 and M10 contracts. |

## H-Ref conclusion

DSH is the stronger reference for scoped capability composition, dependency graphs, reversible lifecycle, event-based interception, and child-agent policy inheritance. It is not evidence that Health should become dynamically extensible at runtime. The safe lesson is to import the *idea of explicit scopes and lifecycle contracts* into later research experiments, while retaining Health's static trusted registry, deterministic profile hash, Evidence authority and fail-closed outcomes.
