# M10 / H7 — Context Engineering, Persistent Session and Verifiable Memory

M10 is an opt-in harness substrate. It does not add a medical product
capability, clinical longitudinal record, diagnosis memory, prescription
memory, or a claim that the system is clinically validated.

## State planes

The implementation keeps four state planes separate:

```text
RunContext
  └─ one harness execution: budgets, trace and component identity
AgentSession
  └─ ephemeral provider-facing messages for one bounded execution
Persistent Session
  └─ append-oriented cross-run interaction/event history
Context
  └─ the exact selected items placed into one model call
Memory
  └─ selected, typed, provenance-aware state explicitly made available later
```

`SessionStore` owns interaction history. `MemoryStore` owns selected persistent
state. Appending a session event never creates memory. `RuntimeComponents.answer()`
keeps the old memory-off semantics; the experimental `answer_in_session()` facade
loads state, creates a `ContextPlan`, constructs an ephemeral `AgentSession`, runs
the existing M3 pipeline, and then commits only explicit session events.

## Source alignment

| Source | Adopted in M10 | Deferred | Rejected / reason |
| --- | --- | --- | --- |
| [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) | finite context/attention budget, structured note-taking, just-in-time context, progressive disclosure, and deterministic structured compaction | model-based compaction | unlimited transcript replay; context is a finite selection problem |
| [Anthropic, Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) | explicit handoff/persistence boundary and structured session state | richer long-running harness strategies | no claim that M10 reproduces that product harness |
| [AMA-Bench](https://arxiv.org/abs/2602.22769) | trajectory-oriented synthetic fixtures, objective metadata, causal/source event references, and distractors | causal graph retrieval and tool-augmented learned retrieval | no claim that this small pack reproduces AMA-Bench |
| [STITCH / CAME-Bench](https://aclanthology.org/2026.findings-acl.584/) | optional `ContextIntent(goal, action_type, entity_types)` and suppression of context-incompatible memories | learned intent induction and a general ontology | no generic ontology or learned classifier in M10 |
| [Mem2ActBench](https://aclanthology.org/2026.acl-long.370/) | action-grounding fixture shape: memory must affect tool selection/arguments, not only QA recall | learned agent action policy | no product side-effect tool; the fixture is eval-only |
| Memory-R1 | typed ADD/UPDATE/DELETE/NOOP boundary for future work | learned memory-write policy and training | deliberately deferred to M11 |
| MemAct / Memory as Action | separation of proposal from policy/materialization | learned context edit/action policy | deliberately deferred to M11 |

M10 is a deterministic/verifiable substrate. It does not train or optimize a
memory manager, use RL, require embeddings, or claim that BM25 solves memory in
general.

## Persistence and provenance

`SQLiteSessionStore` and `SQLiteMemoryStore` use schema `v1`, enable foreign
keys, and perform writes transactionally. Unknown/newer schemas fail closed;
there is no destructive silent migration. The state directory is supplied by
`RuntimeBuildConfig.state_dir`, never embedded as an absolute path in a
`RuntimeProfile`. Component identity binds backend/schema and a safe path hash;
mutable content belongs to per-run `MemorySnapshotIdentity`, not
`ComponentManifest`.

Sessions use opaque runtime-generated IDs, revisioned appends, optimistic
revision checks, resume, and fork. A fork copies only the selected parent
prefix and records `parent_session_id`/`fork_revision`; the parent is not
mutated.

Memory records are narrow and typed: `PREFERENCE`, `TASK_STATE`,
`USER_ASSERTED_CONTEXT`, and `SESSION_NOTE`. They carry source event/run/session
references, objective/intent hints, temporal validity, sensitivity, and a value
hash. UPDATE creates a new version and marks the old version `SUPERSEDED`.
DELETE removes the value from the active materialized view while retaining a
safe tombstone/history record. Deleting a memory record does not delete the
original session event that contained the text.

By default, `USER_EXPLICIT`, `TRUSTED_APPLICATION`, and deterministic
`SESSION_DERIVED` are accepted sources. Assistant output, tool/MCP output, and
retrieved web text are observations and are rejected as automatic durable
memory. `SENSITIVE_HEALTH` writes are denied unless an explicit trusted
`MemoryConsentProvider` allows them.

## Retrieval and context

Retrieval applies scope, active/time-valid status, exact-key compatibility,
objective/intent compatibility, deterministic lexical ranking, and a recency /
ID tie-break. Expired, deleted, superseded, and cross-scope records do not enter
normal retrieval. Intent fields are optional; M10 never fabricates an intent.

`ContextManager` selects `ContextItem`s under an explicit `ContextBudget` with a
deterministic token estimate. `ContextProjector` is the executable boundary
that turns each selected plan into the exact provider-visible message tuple;
the manager itself never calls a provider. Current user input, current Evidence, system
pins, and unresolved current tool exchanges are protected. Old completed
history and tool exchanges may be replaced by a hash-bearing
`StructuredCompactorV1` summary. Assistant tool calls and matching results are
treated as atomic groups. If protected context cannot fit, M10 fails with
`context_budget_exhausted` rather than dropping the current turn or Evidence.

Memory is emitted as a data-bearing context block, not concatenated into the
system prompt. It is contextual user/session data, not instructions and not
medical Evidence. The M3 `KnowledgeCard`/`Evidence`/`ClaimSupportVerifier`
boundary remains the only citation authority. A remembered preference may help
form a retrieval query or response style, but it can never become
`Evidence[]`, `citation_ids`, or verifier support.

## Security and limitations

Memory text is data. It cannot register tools, grant MCP permission, change the
sandbox, alter `RuntimeProfile`, change `KnowledgeScope`, or change
`MemoryPolicy`. The M9 ApprovalProvider semantics and the raw-current-turn
safety gate remain authoritative. Memory alone cannot trigger a new urgent-care
decision.

SQLite necessarily contains stored content. M10 does not implement encryption
at rest, KMS, multi-tenant production access control, or an erasure guarantee
beyond the documented active-memory/tombstone behavior. Do not store real
patient or personal health information.

## Evaluation boundary

The offline `m10-memory-v1` suite uses deterministic synthetic trajectories and
reports retrieval, supersession, expiry, intent mismatch, deletion, scope,
action-grounding, and context-budget metrics separately. Each rate exposes a
numerator, denominator, and definition version; zero denominators are null.
Retrieval success and correct behavioral utilization are distinct metrics.

The eval-only `prepare_education_output` fixture is not registered in M3, M8,
M9, or either M10 product profile. A live provider smoke is optional and must
use synthetic/public fixtures only. A negative comparison is valid; M10 does
not require memory-on to beat memory-off.

## Interview story

- The whole transcript is not memory: transcript is session history, memory is selected persistent state, and context is the subset for one model call.
- Similarity-only retrieval is insufficient because it does not express supersession, expiry, scope, objective, intent, or provenance.
- Remembered medical text is not medical evidence; reviewed `KnowledgeCard` Evidence remains authoritative.
- The model does not own final memory writes; M10 validates and materializes explicit operations. Learned ADD/UPDATE/DELETE/NOOP policy, Memory-R1/MemAct work, and RL context editing start only in M11.
