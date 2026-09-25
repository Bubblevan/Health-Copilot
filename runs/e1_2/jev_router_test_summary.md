# E1.2 QA TEST — Jev hosted router results

## Outcome

The hosted Jev adapter was exercised on the complete, frozen 5,235-case TEST partition. All 5,235 calls were served by `jev-1.13.0`, all returned usage/cost measurements, and there were no Jev fallbacks or provider failures. The service-reported Jev cost was **$1.062125 total** (about **$0.000203 per case**).

The preregistered Jev headline gate **did not pass**. Accuracy was **62.350%** for Jev routing versus **62.445%** for fixed MedCPT RAG (difference **−0.096 percentage points**; paired, stratified 95% bootstrap interval **[−0.344, +0.153] pp**). The interval includes zero. Jev routing also did not clear the preregistered cost-reduction thresholds.

| Frozen TEST arm | Exact-answer accuracy | Retrieval calls | Answer input tokens | Component-latency P95 proxy |
|---|---:|---:|---:|---:|
| Closed-book | 61.987% | 0 / 5,235 | 1,127,913 | 409 ms |
| BM25 RAG | 62.025% | 5,235 / 5,235 | incomplete on one failed answer | 2,328 ms |
| MedCPT RAG | **62.445%** | 5,235 / 5,235 | 6,167,268 | 1,835 ms |
| Random-context control | 60.516% | 5,235 / 5,235 | 6,667,133 | 1,549 ms |
| Deterministic cheap router | 62.235% | 2,311 / 5,235 | 3,640,366 | 2,266 ms |
| Jev router | 62.350% | 4,908 / 5,235 | 5,876,822 | 3,044 ms |

Latency values are component-summed proxies measured in isolated phases, not production end-to-end latency.

## What the routing did

Jev selected closed-book for 327 cases, BM25 for 550, and MedCPT for 4,358. Thus it retrieved on **93.75%** of cases, only 6.25% fewer retrieval calls than always-on retrieval. Its answer-input-token use was 4.7% below fixed MedCPT, short of the 20% gate, and its component-latency P95 proxy was higher. The post-hoc retrieval-opportunity diagnostic found 97.3% recall but only 5.8% precision and a 93.5% false-positive rate. This diagnostic uses TEST outcomes and is for explaining this frozen run only; it was not fed to the router and must not be used to tune on this TEST set.

The deterministic cheap-router control is a promising cost/quality candidate: its point accuracy was 0.210 pp below fixed MedCPT while using 55.9% fewer retrieval calls and 41.0% fewer answer-input tokens. However, its paired 95% interval versus MedCPT was **[−0.917, +0.516] pp**, so this run does not establish non-inferiority within a 0.5 pp margin. It was not the preregistered headline arm. Treat this as a follow-up hypothesis, not a resume claim yet.

## Integrity and next-use boundary

- All six arms completed on the same 5,235 frozen TEST IDs: no missing, extra, or duplicate IDs.
- The DEV selection lock and frozen test configuration were unchanged.
- The aggregate machine-readable report is [`test_analysis.json`](test_analysis.json); it contains no question text, choices, gold labels, or per-case predictions.
- This TEST has now been opened. Do not tune or select a new policy against it. A follow-up headline requires DEV-only policy work followed by a newly sourced or otherwise untouched, preregistered evaluation set.

## Hosted API integration

The client now supports the hosted `https://jevtypesafeai.com/api/v1/decide` service for hosted `jv_live_` credentials while retaining the official TypeSafe direct API path. The adapter smoke test and implementation are recorded in commit `9369b45`. Credentials are not included in the report.
