# RAG-E5-B4 CPU — Harness-Native Counterfactual Recovery

State claims are deterministically materialized from frozen longitudinal packets; the same Qwen3-8B CPU guidance generator receives only the case question and the action-specific frozen B2 evidence. The scorer is deterministic and opens teacher labels only after all B4 artifacts pass their freeze checks.

- Protocol lock: `0c56f636290239517fa7002fd6b28bd4a2ee7bf423c12ff4e265ac3fa7576e46`
- Frozen B2 artifact set: `e528263a5141e6ac476252a9bfb04a1a95fc53475c5617bec23fb106980168f1`
- Cases / users / arms / new guidance calls: 60 / 20 / 180 / 120
- New retrieval / bridge calls: 0 / 0
- Measurement health: **PASS**

## Fixed-action matrix

| Metric | OFF | STANDARD | STRONG |
|---|---:|---:|---:|
| E2E quality | 0.5000 | 0.6944 | 0.8611 |
| Content-only quality | 0.5000 | 0.7222 | 0.7778 |
| Guidance quality (T1/T2) | 0.0000 | 0.4167 | 0.5000 |
| State score (T0/T2) | 1.0000 | 1.0000 | 1.0000 |
| Grounding (T1/T2) | 0.0000 | 0.2500 | 1.0000 |
| New model calls | 40.0000 | 40.0000 | 40.0000 |
| Input tokens | 8360.0000 | 52110.0000 | 53900.0000 |
| Output tokens | 1840.0000 | 2180.0000 | 2010.0000 |
| Mean guidance latency (ms) | 7816.2513 | 40719.7753 | 41701.4530 |
| P95 guidance latency (ms) | 12359.0102 | 71070.3175 | 72609.0848 |
| Historical B2 retrieval calls reused | 0.0000 | 60.0000 | 60.0000 |
| Historical B2 bridge calls reused | 0.0000 | 0.0000 | 60.0000 |
| Historical B2 retrieval latency sum (ms) | 0.0424 | 10162.2051 | 18108.6667 |
| Historical B2 bridge latency sum (ms) | 0.0000 | 0.0000 | 1471913.3382 |
| Mean latency per new guidance call (ms) | 11724.3770 | 61079.6630 | 62552.1796 |
| P95 latency per new guidance call (ms) | 12369.3737 | 71667.6124 | 72721.1626 |

## By task family

| Family | Action | E2E | Content-only | State | Guidance | Grounding | Calls |
|---|---|---:|---:|---:|---:|---:|---:|
| T0 | OFF | 1.0000 | 1.0000 | 1.0000 | — | — | 0 |
| T0 | STANDARD | 1.0000 | 1.0000 | 1.0000 | — | — | 0 |
| T0 | STRONG | 1.0000 | 1.0000 | 1.0000 | — | — | 0 |
| T1 | OFF | 0.0000 | 0.0000 | — | 0.0000 | 0.0000 | 20 |
| T1 | STANDARD | 0.5000 | 0.5000 | — | 0.5000 | 0.5000 | 20 |
| T1 | STRONG | 0.7500 | 0.6667 | — | 0.6667 | 1.0000 | 20 |
| T2 | OFF | 0.5000 | 0.5000 | 1.0000 | 0.0000 | 0.0000 | 20 |
| T2 | STANDARD | 0.5833 | 0.6667 | 1.0000 | 0.3333 | 0.0000 | 20 |
| T2 | STRONG | 0.8333 | 0.6667 | 1.0000 | 0.3333 | 1.0000 | 20 |

## Paired user bootstrap

10,000 user-level paired resamples; each user's T0/T1/T2 outcomes remain together.

| Comparison | E2E Δ [95% CI] | Guidance-quality Δ [95% CI] |
|---|---:|---:|
| STANDARD − OFF | 0.1944 [0.1944, 0.1944] | 0.4167 [0.3833, 0.4500] |
| STRONG − OFF | 0.3611 [0.3278, 0.3944] | 0.5000 [0.4333, 0.5667] |
| STRONG − STANDARD | 0.1667 [0.1333, 0.2000] | 0.0833 [0.0500, 0.1167] |

## Oracle and decision gates

- Best fixed action: **STRONG** (0.8611).
- Oracle-3 E2E / headroom: 0.8611 / 0.0000.
- Oracle-2 (OFF|STANDARD) E2E / headroom: 0.6944 / 0.0000.
- Tie-broken oracle counts: `{"OFF": 20, "STANDARD": 10, "STRONG": 30}`.
- Action diversity: **PASS**; two-action candidate: **False**.
- Strong collapse rule: **False**; unique STRONG oracle wins: 30.
- Recommended action space: **OFF|STANDARD**.
- External retrieval conditional value: **YES**.
- Post-training data worth building: **NO**.

## Measurement boundaries

T0 is deterministic state-only materialization and makes zero model calls. T1/T2 guidance outputs are unstructured plain text; `[E1]`–`[E5]` are resolved by the runtime to the original frozen B2 chunk IDs and provenance. Unknown aliases are ignored and counted. No new retrieval or bridge generation was run. The scorer checks the existing deterministic rubric anchors and citations; it does not assess unlisted medical facts or use an LLM judge. Historical B2 retrieval coverage is reused as a frozen diagnostic, not recomputed.
