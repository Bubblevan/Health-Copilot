# RAG-E5-B2 — Frozen Counterfactual Characterization

This report compares OFF, STANDARD, and STRONG over the same 60 frozen tasks (20 users; T0/T1/T2). Scoring separates guideline-content correctness from evidence grounding. No router was trained.

- Task-set SHA-256: `7ac2d95f98efc6fcced665f93977c76c4d219af0cfe7e16f05b1fdd67358979a`
- State-packet-set SHA-256: `860f5da64a6b94790944a5bd2d50aa4977a098bc51d1b6f822321bf4fbbd1d1c`
- Active corpus identity: `9b19ad467f47641032277707cb1cfdb1d05c2fb39c558039b180efc7394692bd`
- Scorer: `e5b2-content-grounding-separated-v2`
- Frozen arms / model calls: 180 / 240 (180 readers, 60 bridges)
- Invalid reader JSON / bridge fallbacks: 80 / 10

## Overall fixed-action matrix

| Metric | OFF | STANDARD | STRONG | ORACLE |
|---|---:|---:|---:|---:|
| E2E quality | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Content-only quality | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| State score (T0/T2) | 0.0000 | 0.0000 | 0.0000 | N/A |
| Guideline content (T1/T2) | 0.0000 | 0.0000 | 0.0000 | N/A |
| Grounding (T1/T2) | 0.0000 | 0.0000 | 0.0000 | N/A |
| Reader JSON valid rate | 1.0000 | 0.3333 | 0.3333 | N/A |
| Mean total latency (ms) | 32597.1226 | 53227.8955 | 79046.4897 | N/A |
| P50 total latency (ms) | 37485.5780 | 48383.3889 | 79420.3623 | N/A |
| P95 total latency (ms) | 57706.8977 | 87520.5616 | 111102.0885 | N/A |

## By task family

| Family | Action | E2E | Content-only | State | Guideline content | Grounding |
|---|---|---:|---:|---:|---:|---:|
| T0 | OFF | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| T0 | STANDARD | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| T0 | STRONG | 0.0000 | 0.0000 | 0.0000 | N/A | N/A |
| T1 | OFF | 0.0000 | 0.0000 | N/A | 0.0000 | 0.0000 |
| T1 | STANDARD | 0.0000 | 0.0000 | N/A | 0.0000 | 0.0000 |
| T1 | STRONG | 0.0000 | 0.0000 | N/A | 0.0000 | 0.0000 |
| T2 | OFF | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| T2 | STANDARD | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| T2 | STRONG | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Paired user-level bootstrap

10,000 resamples; each draw samples 20 users with replacement and carries each user's T0/T1/T2 tasks together.

| Comparison | Mean delta | 95% CI |
|---|---:|---:|
| STANDARD − OFF | 0.0000 | [0.0000, 0.0000] |
| STRONG − OFF | 0.0000 | [0.0000, 0.0000] |
| STRONG − STANDARD | 0.0000 | [0.0000, 0.0000] |
| ORACLE − BEST_FIXED | 0.0000 | [0.0000, 0.0000] |

## Retrieval target coverage (T1/T2)

| Action | Cases | Source @5 | Recommendation @5 | Source @100 | Recommendation @100 |
|---|---:|---:|---:|---:|---:|
| STANDARD | 40 | 1.0000 (40) | 1.0000 (40) | 1.0000 (40) | 1.0000 (40) |
| STRONG | 40 | 1.0000 (40) | 1.0000 (40) | 1.0000 (40) | 1.0000 (40) |

## Oracle and gates

- Best fixed action: OFF (0.0000).
- Per-case oracle E2E: 0.0000; headroom: 0.0000.
- Tie-broken oracle action counts: `{"OFF": 60}`.
- Unresolved cases (best E2E < 1.0): 60.
- Gates: `{"ACTION_DIVERSITY_GATE": "FAIL", "ORACLE_HEADROOM_GATE": "FAIL", "T2_DEPENDENCY_GATE": "PASS"}`.
- E5-C authorized: **False**. E5-C started: **NO**.

## Post-score execution audit

This audit inspects the already-frozen arm artifacts; it does not change the scorer, scores, or gates.

- Retrieval is not the observed bottleneck in this run: STANDARD and STRONG each placed the required source and recommendation in the top 5 for all 40 eligible T1/T2 tasks.
- OFF produced valid JSON for all 60 tasks, but none of the 40 expected state fields matched the frozen field/value contract. The emitted state label was `body_weight_trend`; the runtime packet and teacher contract use the canonical field `body_weight`. This is a model output-contract mismatch, not a post-hoc scoring adjustment.
- STANDARD and STRONG each had 40 invalid reader JSON outputs, all with exactly 256 output tokens—the frozen per-call maximum. Their valid outputs were the 20 T0 tasks; all T1/T2 reader outputs were invalid. This is consistent with output-budget truncation when evidence is present, though the frozen run does not identify truncation as a separate causal variable.
- All measured quality remains zero under the preregistered scorer. STRONG added 60 bridge calls and about 25.8 seconds mean latency versus STANDARD, with no measured quality gain. Given the failed diversity and oracle-headroom gates, these results do not support E5-C or a capability-router experiment on this cohort.

## Interpretation boundaries

Content-only quality isolates answer-content anchors from grounding credit; E2E quality retains the preregistered weights. Retrieval hit@100/hit@5 separates retrieval misses from rank-cutoff misses. Reader misses and grounding misses are counted separately. Content-rescue, grounding-only-gain, and harmful-retrieval flags can overlap. Costs are reported as raw calls, token usage, and latency; no post-hoc cost utility is applied.
