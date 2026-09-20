# M8 — Bounded Agent Team

M8 implements Harness Macro H5 as a bounded star-topology Agent Team: a
sequential Team Lead plus two specialized worker roles. It is experimental orchestration, not Swarm,
persistent memory, a diagnostic system, or a replacement for the M3 verifier.

The explicit orchestration identity is `topology=star-supervisor-v1` and
`scheduler=sequential-v1`. The lead is the only coordinator; workers report
only to the lead; no parallel subagent claim is made.

## Runtime topology

```text
question -> safety gate -> BM25 initial evidence -> Team Lead turn 1
                                      |
                          FINAL / ABSTAIN / DELEGATE
                                      |
              Evidence Worker -> report -> mailbox -> Team Lead turn 2
              Guideline Worker -> report -> mailbox -> Team Lead turn 2
                                      |
                         claim-first final verifier
                                      |
                         deterministic materialization
```

Only one delegation wave is allowed. Workers run sequentially because the
existing `RunBudgetState` and `RunTrace` are mutable per-run control-plane
state. M8 validates team semantics before considering concurrency.

## Authority and contracts

The model proposes `LeadDecision` values (`final`, `delegate`, `abstain`) and
worker claim reports. The Harness owns runtime IDs, role allowlisting, task
transitions, message delivery, policy/tool execution, evidence provenance,
budgets, termination, and final verification.

The only roles are `evidence` and `guideline`. A lead task contains only a role
and objective; the runtime creates `task-*`, `worker-*`, and `message-*` IDs.
The default logical bounds are two lead calls, two tasks/workers, one
delegation round, two worker model turns, and one worker tool execution. These
are separate from the parent M4 side-effect budget.

`TaskStore` is ephemeral coordination state. Its valid lifecycle is:

```text
PENDING -> RUNNING -> SUCCEEDED | FAILED | CANCELLED
```

`Mailbox` is typed point-to-point communication. Lead-to-worker assignments,
worker-to-lead reports/failures, and runtime notices are supported. Broadcast
and worker-to-worker messages are rejected. Metadata-only traces store hashed
objectives and claim text, not raw user/task content or hidden reasoning.

Each worker receives the original question, its own runtime-generated
objective, role, initial evidence, and the reviewed `KnowledgeScope` in a new
`AgentSession`. Worker sessions are never concatenated. The Lead sees only
structured `WorkerReport` values. All provider and tool side effects still use
the single parent `RunContext`, including one shared provider/tool/token/
deadline budget.

The two workers may use the same checkpoint, provider executor, and tools, but
they do not share one system contract. The Evidence Worker is constrained to
factual evidence acquisition and source coverage and must avoid recommendation
synthesis. The Guideline Worker is constrained to guideline/recommendation
context and publisher/jurisdiction distinctions and must avoid unsupported
factual expansion. `allowed_roles` is passed into the orchestrator and enforced
when a lead proposal is validated; the profile manifest therefore describes
the executable role boundary, not only an identity label.

## Evidence and finalization

`TeamEvidenceLedger` records every source as `INITIAL` or `WORKER`, with
worker/task provenance when applicable. A worker citation must refer to a
source that worker actually observed; otherwise its task fails and the source
is not added to the ledger. A Lead final citation must be in the ledger.

Team finalization reuses the M3 boundary: normalize claims, check citation
integrity, materialize cited evidence, call `ClaimSupportVerifier`, require
every claim to be supported, then deterministically render the claims. Worker
agreement is never a substitute for claim support.

Urgent and prescription routes short-circuit before team construction, so they
create no Team Lead call, task, worker, or tool execution. A budget denial,
provider failure, invalid delegation, or second delegation fails closed.

## Profiles and evaluation

`m8-team-bm25-v1` adds the explicit `agent-team-v1` orchestration component to
the same provider, BM25 retriever, policy, scope, tool, and claim-support
verifier used by `m3-bm25-default`. The component identity binds lead/worker
models, prompt contracts, allowed roles, scheduler version, and all team
limits. Optional `RuntimeProfile.orchestration` is omitted from old profiles'
canonical JSON so frozen M0–M7 hashes remain unchanged.

The candidate suite `m8-agent-team-focused-v1` contains 12 closed-corpus cases:
4 direct, 4 decomposable, 2 cross-source comparison, and 2 uncovered/OOD. Its
annotation manifest is now `human_reviewed_frozen`; results remain a focused
diagnostic, not a validated clinical benchmark. The
comparison arms are:

| Arm | Profile | Control flow |
| --- | --- | --- |
| L0 | `m8-workflow-bm25-v1` | deterministic retrieval + one final-only call |
| L1 | `m3-bm25-default` | frozen single Agent |
| L2 | `m8-team-bm25-v1` | bounded Agent Team |

The arms share the dataset, BM25, scope, provider model, and final verifier.
Cost metrics (provider calls, tools, tokens, elapsed time, and budget
exhaustion) are reported separately from route, evidence-group coverage,
citation integrity, claim support, OOD behavior, and per-category metrics.
No weighted overall score or production-default switch is implied.

M8 records two evidence views. `worker_evidence_overlap` and
`worker_unique_evidence_contribution` use each worker's full observed context,
which includes the shared initial evidence. The separate
`worker_recovery_evidence_overlap` and
`worker_unique_recovery_contribution` metrics use only
`recovery_ranked_evidence`, so they measure incremental worker contribution
without redefining the full-context indicators. Worker completion remains a
task-state fact: an `ABSTAIN` response may complete execution but is not a
productive report. `m8.worker_completion_rate` therefore measures completed
worker executions over worker reports, while
`m8.worker_productive_report_rate` measures completed reports containing claims
or recovery evidence over completed worker executions.

## Source Alignment

M8 is aligned with the centralized hierarchical family of agent-team designs,
but it is deliberately a closeout-sized control experiment rather than a
general multi-agent platform. The decisions below are recorded so that future
experiments do not silently change the M8 empirical baseline.

| Source | Adopted | Intentionally deferred | Rejected for M8 | Why |
| --- | --- | --- | --- | --- |
| [Anthropic Multi-Agent Research](https://www.anthropic.com/engineering/multi-agent-research-system) | Lead-to-specialist delegation, explicit task boundaries, structured worker findings, and source-aware synthesis. | Parallel workers, iterative research loops, memory-backed plans, and a separate citation agent. | Open-web research, dynamic worker spawning, and unbounded replanning. | M8 must isolate role specialization and provenance before measuring live-agent gains. |
| [Anthropic, Patterns and problems in emerging multiagent systems (2026)](https://www.anthropic.com/research/multiagent-systems) | Treat duplicate evidence, premature consensus, and missing unique information as measurable coordination risks; keep a central authority and auditable reports. | Reputation, richer trust calibration, dissent protocols, and adversarial multi-agent stress tests. | Peer-to-peer forums, long-lived peer goals, shared mutable workspaces, and decentralized coordination. | The reported conformity and coordination failures argue for a small, fail-closed control plane in M8. |
| [Microsoft Agent Framework / Magentic](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-orchestration/magentic) | Manager plus specialized participants as the reference family for centralized hierarchical orchestration. | Dynamic manager replanning, progress-aware agent selection, shared context, and multiple collaboration rounds. | Directly adopting the flexible Magentic workflow as the M8 runtime. | M8 needs fixed topology, sequential scheduling, and one delegation wave for deterministic attribution. |
| [Magentic-One](https://arxiv.org/abs/2411.04468) | Central manager, bounded specialist roles, and manager-owned synthesis as an architectural comparison point. | General-purpose open-ended task solving, checkpointing, plan review, and adaptive recovery. | A direct Magentic-One implementation or its full participant set. | The M8 question is role-specialized health evidence handling, not general open-ended autonomy. |
| [AgentScope / Alibaba AgentTeams](https://github.com/agentscope-ai/AgentTeams) | Manager–Worker vocabulary, explicit worker capability boundaries, and per-role execution metadata. | Human-in-the-loop operations, shared team rooms, external gateways, MCP configuration, and distributed deployment. | Matrix/shared-infrastructure coordination and external Manager/Worker control planes. | M8 remains an in-process, replayable harness with one parent budget and no new external control surface. |
| [ClawArena-Team](https://arxiv.org/abs/2606.31174) | Execution-based management evidence, least-privilege awareness, and per-role trajectory records. | Multimodal, multi-directory, multi-turn dynamic workflows and its full management score. | Using its benchmark or score as an M8 quality claim. | M8 records the interface needed for later comparison without importing a different task domain or judge. |
| [MultiAgentBench](https://aclanthology.org/2025.acl-long.421.pdf) | Topology-aware evaluation language and explicit contribution/coordination accounting. | Chain, tree, graph, competition, discussion, and long-horizon coordination protocols. | Dynamic graphs and non-star topology in the M8 execution path. | M8 freezes `star-supervisor-v1` so topology is not a confound in the L0/L1/L2 comparison. |
| [Agent scaling / Ringelmann effect](https://arxiv.org/abs/2606.02646) | Deterministic `worker_evidence_overlap` and `worker_unique_evidence_contribution` indicators, plus strict worker/task caps. | Scale-law fitting, larger teams, heterogeneous-model ablations, and causal claims about useful team size. | Extrapolating a two-worker diagnostic into a scaling law or training signal. | The metrics expose redundancy and unique evidence now; they are not SHARP, SRPO, or Dr.MAS training data. |

### Frozen M8 boundary

M8 is a **bounded star-topology Agent Team** implementing only
**centralized hierarchical orchestration**. It does not implement dynamic
topology, decentralized MAS, A2A, MCP, Memory, or post-training. Those remain
future work tracked beyond this closeout: M9 for topology/protocol/tool-surface
expansion, M10 for Memory, and M11 for post-training. No M8 result should be
reported as evidence for those later capabilities.

## Explicit non-goals

M8 does not start MCP, sandbox, permissions, web search, memory, persistent
conversation state, multimodal work, post-training, dynamic topology,
decentralized MAS, A2A, dynamic role creation, recursive delegation, worker
parallelism, or diagnostic/doctor agents.

## M8.3 Team-Lead contract diagnosis

M8.3 keeps the Team Lead wire contract at the provider-compatible
`response_format={"type":"json_object"}` boundary. The original v1 traces
show that this mode reached provider responses, but they do not establish that
the configured OpenAI-compatible endpoint supports JSON Schema structured
outputs. M8 therefore does not switch to JSON Schema by assumption; provider
compatibility must be demonstrated separately before any such change.

The repaired contract boundary classifies failures as `provider`,
`empty_response`, `json_decode`, `contract_validation`, or `internal`, while
retaining the normalized provider failure kind when one exists. Metadata-only
traces record only the failure kind, provider failure kind, contract version,
response SHA-256, and response length. Raw model output is permitted only in
the explicitly scoped public three-case diagnostic and is never written to the
normal metadata-only evaluation artifacts.

The targeted public diagnostic found a finite provider wire mismatch: responses
used `type` for the action, `task` for a single delegation, and `claim` for
claim text. M8.3 normalizes only these documented aliases into the canonical
`action` / `tasks` / `text` schema and continues to reject all other fields.

All malformed Lead outputs remain fail-closed: they create no tasks, start no
workers, and cannot become an answer. M7 failure records now expose these
orchestration failures even when the case status is `COMPLETE` with an
`ABSTAIN` route.
