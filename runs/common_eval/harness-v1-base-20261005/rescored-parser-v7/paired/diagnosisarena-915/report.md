# Harness V1 paired result — diagnosisarena-915

Frozen cases: 915
Model manifest SHA256: `e5466c735d57bd3e32d4607a3e372b1862579edfef7864ca514e709a38853e26`
ID sequence SHA256: `b6c4102a3d2bdcf3d464b77f67df0e02a9f04a5036d7a6e45e34aedb606967e7`

| Metric | B0 Single | B2 Adaptive MDT | Delta |
|---|---:|---:|---:|
| Accuracy | 37.05% | 30.60% | -6.45 pp |
| 95% Wilson CI | 33.98%–40.23% | 27.70%–33.66% | — |
| Parse success | 100.00% | 74.86% | — |
| Safety-route abstentions | 0 | 0 | — |
| Reasoning failures | 0 | 230 | — |
| Explicit model abstentions | 0 | 0 | — |
| Answer-format failures | 0 | 0 | — |
| Provider calls / case | 1.00 | 8.63 | — |
| Input tokens / case | 557.9 | 5560.8 | — |
| Output tokens / case | 17.0 | 1251.1 | — |
| Total tokens / case | 574.9 | 6811.9 | — |
| Mean latency / case | 845.0 ms | 35291.1 ms | — |
| P50 latency / case | 418.6 ms | 32394.7 ms | — |
| P95 latency / case | 4971.3 ms | 59701.1 ms | — |
| Throughput | 4.727 cases/s | 0.130 cases/s | — |
| Provider throughput | 4.727 req/s | 1.234 req/s | — |
| Token throughput | 2717.6 tok/s | 885.7 tok/s | — |

Parsed answer agreement: 407/685 among cases parsed by both arms.
Scoring revision: `deterministic-mcq-parser-v7`.

Serving parity is qualified as `INCOMPLETE_B0_RUNTIME_CAPTURE`; see `runs/common_eval/harness-v1-base-20261005/recovery_inputs/serving-parity-audit.json`. Treat this paired delta as descriptive, not a controlled causal estimate.

## Adaptive complexity slices

| Complexity | N | Correct | Accuracy |
|---|---:|---:|---:|
| advanced | 279 | 28 | 10.04% |
| basic | 124 | 42 | 33.87% |
| failed | 10 | 0 | 0.00% |
| intermediate | 502 | 210 | 41.83% |

Scorer labels were joined by frozen case ID on the evaluator side. Only candidate prompts and answer schema were passed to Harness. Parse failures and Harness safety abstentions remain incorrect in the accuracy denominator.

Adaptive MDT here is the full complexity-routed strategy; B2−B0 does not isolate the moderator by itself.
