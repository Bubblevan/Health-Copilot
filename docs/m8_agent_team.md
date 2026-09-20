# M8 — Bounded Agent Team

M8 implements Harness Macro H5 as a bounded, sequential Team Lead plus two
specialized worker roles. It is experimental orchestration, not Swarm,
persistent memory, a diagnostic system, or a replacement for the M3 verifier.

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
annotation manifest is explicitly `pending_human_review`; results must be
described as a focused candidate diagnostic, not a validated benchmark. The
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

## Explicit non-goals

M8 does not start MCP, sandbox, permissions, web search, memory, persistent
conversation state, multimodal work, post-training, dynamic role creation,
recursive delegation, worker parallelism, or diagnostic/doctor agents.
