# Harness V1 paired result — cmb-common-1024

Frozen cases: 1024
Model manifest SHA256: `e5466c735d57bd3e32d4607a3e372b1862579edfef7864ca514e709a38853e26`
ID sequence SHA256: `ed5816263ee3f20dec4fd6f76f5b8b6ceb6f89a4b01b35716a2b8afc4ffd4fc4`

| Metric | B0 Single | B2 Adaptive MDT | Delta |
|---|---:|---:|---:|
| Accuracy | 62.30% | 67.87% | +5.57 pp |
| 95% Wilson CI | 59.30%–65.22% | 64.95%–70.66% | — |
| Parse success | 94.63% | 93.75% | — |
| Safety-route abstentions | 55 | 55 | — |
| Reasoning failures | 0 | 9 | — |
| Explicit model abstentions | 0 | 0 | — |
| Answer-format failures | 0 | 0 | — |
| Provider calls / case | 0.95 | 3.43 | — |
| Input tokens / case | 199.2 | 1273.3 | — |
| Output tokens / case | 10.2 | 837.6 | — |
| Total tokens / case | 209.4 | 2110.9 | — |
| Mean latency / case | 465.1 ms | 25326.2 ms | — |
| P50 latency / case | 331.8 ms | 23376.5 ms | — |
| P95 latency / case | 1135.6 ms | 47616.2 ms | — |
| Throughput | 8.564 cases/s | 0.155 cases/s | — |
| Provider throughput | 8.104 req/s | 0.633 req/s | — |
| Token throughput | 1793.6 tok/s | 326.7 tok/s | — |

Parsed answer agreement: 732/960 among cases parsed by both arms.
Scoring revision: `deterministic-mcq-parser-v6`.

Serving parity is qualified as `INCOMPLETE_B0_RUNTIME_CAPTURE`; see `runs/common_eval/harness-v1-base-20261005/recovery_inputs/serving-parity-audit.json`. Treat this paired delta as descriptive, not a controlled causal estimate.

## Adaptive complexity slices

| Complexity | N | Correct | Accuracy |
|---|---:|---:|---:|
| advanced | 1 | 0 | 0.00% |
| basic | 890 | 640 | 71.91% |
| failed | 1 | 0 | 0.00% |
| intermediate | 77 | 55 | 71.43% |
| unclassified | 55 | 0 | 0.00% |

Scorer labels were joined by frozen case ID on the evaluator side. Only candidate prompts and answer schema were passed to Harness. Parse failures and Harness safety abstentions remain incorrect in the accuracy denominator.

Adaptive MDT here is the full complexity-routed strategy; B2−B0 does not isolate the moderator by itself.
