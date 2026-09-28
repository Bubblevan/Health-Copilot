# E2 Architecture Protocol

Status: **E2-A capability foundation frozen; no formal architecture comparison run.**

## Architecture IDs

| ID | Meaning | Freeze boundary |
| --- | --- | --- |
| `L0_WORKFLOW` | Existing deterministic workflow baseline. | Existing implementation and its own budget regime. |
| `L1_SINGLE` | Existing single-agent baseline. | Existing implementation and its own budget regime. |
| `L2_HOMOGENEOUS_SEQUENTIAL_FROZEN` | The original homogeneous, sequential M8 Agent Team control. | Exactly the frozen M8 v2 semantics/artifacts in `docs/m8_empirical_v2_closeout.md`, including `runs/m8_empirical_v2_l2/`; this ID does not name a rewritten implementation. |
| `L3_HETEROGENEOUS_SEQUENTIAL` | Fixed-role workers with distinct harness-enforced capability contracts, executed sequentially. | Capability foundation only in E2-A; no formal DEV comparison. |
| `L4_HETEROGENEOUS_PARALLEL` | The identical L3 worker set and assignments under a parallel scheduler. | Reserved for a later stage; no L4 scheduler is implemented here. |

The controlled sequence is `L2 → L3` with capability heterogeneity as the treatment and `L3 → L4` with scheduler as the treatment. Preserve worker-set, model, safety, evidence, and budget controls for the latter comparison.

## E2-A contract

`WorkerCapabilitySpec` is immutable and canonical-hashed. Its fields are:

```text
capability_id
role
allowed_source_families
allowed_capability_domains
tool_ids
retriever_profile_id
authority_scope
objective_contract_id
max_model_turns
max_tool_calls
eligibility
eligibility_reason
```

The resolved source-ID/family/domain catalog has its own canonical hash. Both capability refs and the source-catalog hash enter `ComponentManifest`; the run trace records those hashes with the architecture and scheduler. Contract hash changes when any declared boundary changes. Source-catalog changes also change manifest identity.

The maximum fixed roles are `PUBLIC_HEALTH`, `GUIDELINE`, and `LITERATURE`. Current production eligibility is one role: `PUBLIC_HEALTH`. Guideline and scholarly literature remain frozen as `NOT_YET_ELIGIBLE` until the active retriever has separately reviewed evidence for those surfaces. Synthetic fixtures may use synthetic source families to verify runtime mechanics; they do not change production eligibility.

## Execution and trust rules

- `scheduler = sequential-v1`; no parallel worker calls are allowed.
- Each task receives a fresh `AgentLoop`, `AgentState`, `AgentSession`, scoped tool registry, and local provider/tool/token/time counters.
- Each worker run shares the parent's run identity, deadline/global budget guard, trace sink, and one `EvidenceLedgerV2`.
- The parent hard budget is checked before each provider/tool effect. Worker limits are checked independently. Either denial fails closed and prevents that worker from adding evidence to the ledger.
- Memory reads/writes are `OFF`. No `SessionStore`, `MemoryStore`, `ContextManager`, or memory policy is passed to the runner.
- Worker source-family isolation filters both initial evidence and every retrieval result at the harness boundary. Unknown source IDs are denied. The worker's `ToolRegistry` contains only contract-granted tools.
- A report is `productive` only when it completed, passed citation/source checks, and contributes a new unique source or a claim whose citation was observed by that worker and is present in the ledger.
- The Lead-facing packet is only verified reports plus ledger evidence. Failed, skipped, or provenance-invalid workers contribute no evidence.
- No runtime object accepts task family, required evidence groups, independent width, expected architecture, or other task-profile ground truth.

## Metrics contract

Schema: `e2-metrics-v1`. Ratios preserve numerator and denominator. Annotation-backed coverage is evaluator-only and its annotation is not passed to the runtime.

| Metric | Definition |
| --- | --- |
| `EvidenceGroupCoverage` | Required evidence groups with at least one ledger source / required evidence groups. |
| `SourceFamilyCoverage` | Required source families represented in ledger evidence / required source families. |
| `WorkerCompletionRate` | Completed delegated tasks / delegated tasks. |
| `ProductiveWorkerRate` | Productive completed workers / completed workers. |
| `WorkerEvidenceOverlap` | Mean pairwise intersection-over-union of verified workers' observed source IDs. |
| `WorkerUniqueEvidenceContribution` | Sources observed by exactly one verified worker / the union of verified workers' observed sources. |
| `DelegationPrecision` | Productive delegated tasks / all delegated tasks. Denominator zero is `NA` (`null`). |
| `ProviderCalls`, `ToolExecutions`, `InputTokens`, `OutputTokens` | Observed worker totals; token totals are `null` if any attempted call has unknown usage. |
| `WallClockLatencyMs` | Monotonic run start through all worker outcomes. |
| `SumWorkerExecutionMs` | Sum of per-worker elapsed execution time. |

Existing M8 metric names and historical interpretation remain unchanged.

## Evaluation isolation

The benchmark manifest describes 48 cases split into 24 DEV and 24 TEST. E2-A inspected DEV annotation rows only to audit the schema/source-family mapping; no TEST row was loaded, parsed, or executed. No 24-case comparison is run in E2-A. `task_profiles.jsonl` remains evaluator-only and is not loaded by runtime, runner, prompt, or router. No adaptive router, Jev, oracle lookup, SFT, or RL is permitted.
