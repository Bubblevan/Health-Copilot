# M8.3 Empirical Closeout

Status: **M8 FROZEN / H5 Agent Team COMPLETE**

This closeout records the post-repair comparison required by M8.3. Runtime
results below were produced under code commit
`16a4422b989136c0f01a87bfd020c2cdf00c5c0f` (`team-lead-v2`). The frozen
dataset was unchanged:

```text
evals/m8_agent_team_focused_v1.jsonl
SHA256 78a417bef691892fb0b248911044f0325849ae4a4ae1065ec7ad009968180549
manifest human_reviewed_frozen
```

The three v2 bundles are retained locally and were not written over the v1
bundles:

| Arm | Profile | Bundle |
| --- | --- | --- |
| L0 Workflow | `m8-workflow-bm25-v1` | `runs/m8_empirical_v2_l0/20260921T032412257804+0800/` |
| L1 Single Agent | `m3-bm25-default` | `runs/m8_empirical_v2_l1/20260921T033109000582+0800/` |
| L2 Agent Team | `m8-team-bm25-v1` | `runs/m8_empirical_v2_l2/20260921T033544737024+0800/` |

Each arm used the same 12 frozen cases and 3 trials per case: 36
trajectories, 108 total. Provider, BM25 retrieval, KnowledgeScope, verifier,
and topology-native budget policy were unchanged across arms.

## Original v1 failure and diagnosis

The original v1 L2 run remains at
`runs/m8_empirical_l2/20260921T023642667738+0800/` and is summarized in
`docs/m8_empirical_v1_failure.md`. It contained 36 trials, 1 accepted
delegation, 0 completed workers, `m8.budget_exhaustion_rate = 0`, and 33
`team_stop=lead_error` observations. The existing metadata-only traces did not
contain the raw Lead response, so the exact original wire payload could not be
reconstructed from v1 alone. The audit explicitly does not infer a cause from
the final route.

The allowed three-case diagnostic was then run with the same provider/model,
BM25, KnowledgeScope, Team prompt, temperature, and profile. It exposed the
actual boundary mismatch:

- `type=delegate` with singular `task`;
- `type=final` with claim objects using `claim` rather than canonical `text`;
- one delegate payload with canonical plural `tasks`.

The provider was observed to honor `response_format={"type":"json_object"}`.
JSON Schema support was not established for this OpenAI-compatible endpoint,
so M8 retains `json_object` and uses an explicit finite compatibility adapter.
The raw diagnostic is local-only and is not committed.

This run exposed a Team Lead integration/contract failure and is retained as a
failure artifact. It is not used as evidence that Agent Teams are intrinsically
worse than the single-agent baseline.

## Contract and observability repair

M8.3 added `TeamLeadFailureKind` with `provider`, `empty_response`,
`json_decode`, `contract_validation`, and `internal`, while preserving the
normalized `ProviderFailureKind` where available. `TeamLeadModelError` stores
only contract version, response SHA-256, and response length by default.

Malformed Lead output remains fail-closed: it creates no tasks, starts no
workers, and cannot become an answer. The parser accepts only the finite,
documented observed aliases (`type -> action`, singular `task -> tasks`, and
`claim -> text`) before the existing strict `LeadDecision` validation. Unknown
or runtime-controlled fields remain rejected.

Metadata-only traces and trajectories now retain safe Lead diagnostics,
`topology=star-supervisor-v1`, `scheduler=sequential-v1`, and per-role:

```text
role
provider calls
tool proposals / executions
observed source IDs
final cited source contribution
```

The M7 FailureMapper promotes orchestration failures even when the case is
`COMPLETE` with an internal abstain. The v2 metrics add deterministic
`worker_evidence_overlap` and `worker_unique_evidence_contribution` plus the
recovery-only counterparts.

## Targeted smoke

The three-case post-fix smoke (`m8-simple-001`, `m8-decomposable-002`, and
`m8-cross-source-001`, one trial each) passed the integration boundary:

| Case | Lead actions | Route / harness | Stop | Lead failure |
| --- | --- | --- | --- | --- |
| simple | delegate, final | answer / answer | final | none |
| decomposable | final | answer / answer | final | none |
| cross-source | delegate, final | answer / answer | final | none |

No generic `lead_error`, provider failure, or contract failure occurred in
this smoke.

## Final v2 comparison

Rates are over 36 trials unless noted; expected-answer and unexpected-abstain
rates are conditional on the 10 answer-expected frozen cases (30 trials).
Per-case cost metrics are over the complete-case denominator emitted by the
evaluator.

| Metric | L0 Workflow | L1 Single | L2 Team |
| --- | ---: | ---: | ---: |
| route accuracy | 0.806 | 0.833 | 0.806 |
| expected-answer rate (answer-expected cases only) | 0.767 | 0.833 | 0.767 |
| unexpected abstain rate (answer-expected cases only) | 0.233 | 0.167 | 0.233 |
| evidence-group coverage | 0.333 | 0.444 | 0.333 |
| OOD answer pass rate | 1.000 | 0.833 | 1.000 |
| all-trials-pass | 0.333 | 0.333 | 0.250 |
| route consistency | 0.833 | 0.750 | 0.500 |
| Harness disposition consistency | 0.750 | 0.667 | 0.417 |
| citation integrity | 1.000 | 1.000 | 1.000 |
| budget exhaustion | 0.000 | 0.000 | 0.000 |
| provider calls / case | 1.556 | 2.361 | 2.361 |
| tool executions / case | 0.000 | 0.333 | 0.000 |
| input tokens / case | 1,343.5 | 2,631.1 | 1,835.5 |
| output tokens / case | 1,901.9 | 1,556.5 | 3,613.5 |
| total tokens / case | 3,245.3 | 4,187.6 | 5,449.0 |
| elapsed ms / case | 8,661.5 | 7,247.7 | 19,095.7 |

The pairwise `compare` outputs are preserved in the command history for
L0-vs-L1, L0-vs-L2, and L1-vs-L2. The important deltas are that L2 matched L0
route accuracy but trailed L1 by 0.028, trailed L1 evidence coverage by 0.111,
and added about 1,261 total tokens/case and 11.8 seconds/case versus L1.

### Category analysis

| Category route accuracy | L0 | L1 | L2 | L2 versus L1 |
| --- | ---: | ---: | ---: | ---: |
| `simple_direct` | 0.833 | 1.000 | 0.750 | -0.250 |
| `decomposable` | 1.000 | 0.917 | 0.750 | -0.167 |
| `cross_source_comparison` | 0.167 | 0.333 | 0.833 | +0.500 |
| `ood_uncovered` | 1.000 | 0.833 | 1.000 | +0.167 |

Team helped most on cross-source comparison and matched the L0 OOD result. It
hurt on simple direct and decomposable cases, with lower consistency and higher
latency. This is a small closed-corpus diagnostic, not a general scaling claim.

### Team execution and evidence contribution

L2 recorded 32 final Lead actions and 9 delegate actions across 36 trials;
8 trials delegated at least once, creating 12 tasks and starting 12 workers.
Lead calls totaled 41, or 1.242 per complete Team case with a Lead call.
There were 0 Lead failures of every v2 taxonomy kind and 0 provider failures.

Worker completion was 0/12 and productive reports were 0. The 12 worker
records failed before tool execution: 11 with `max_tool_calls` and 1 with
`model_error`. Consequently, full-context evidence overlap was 1.000 with
unique contribution 0.000 on the 4 comparable multi-worker observations;
recovery-only overlap and unique contribution were both 0.000. The final
citation contribution by role was empty for both Evidence and Guideline in
this run. This preserves the negative result rather than treating failed
worker attempts as productive specialization.

## Regression and verification

- `pytest -q`: **214 passed, 1 warning** (the warning is the existing Windows
  pytest-cache permission warning; tests passed using a repository-local
  basetemp).
- `ruff check .`: passed; the scan emitted only existing permission warnings
  while visiting pytest temporary directories.
- `python -m compileall -q src`: passed.
- `git diff --check`: passed.
- M0 regression: `m0.safety_route_accuracy = 1.0`; frozen Hit@1 remained
  `0.9032258064516129` (Hit@3 `0.9516129032258065`, MRR
  `0.9274193548387096`).
- M4 replay: 6/6 replay cases, route 1.0, replay consistency 1.0, budget
  termination 1.0.
- M7 offline/deterministic and existing live-smoke closeout remained intact;
  the verifier fixture rejected fabricated citation before provider execution.
- Legacy profile hashes remained unchanged except the intentional M8
  Team-Lead contract/version change.

The original v1 negative result and the v2 repair/comparison are both
preserved. The Team profile remains experimental and is not the product
default; `m3-bm25-default` remains the default profile.

M9 has **not** started. MCP/Sandbox/Permission, Memory, and Post-training have
**not** started.
