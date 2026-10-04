# MA-MVP1 Quantitative Results

## Primary comparison: all DEV

The frozen run evaluated 1,024 paired DEV episodes per system.

| Metric | Strong Single | Routed Medical Team | Δ |
|---|---:|---:|---:|
| Task success | 586/1,024 (57.23%) | 589/1,024 (57.52%) | +0.29 pp |
| Grounded task success | 586/1,024 (57.23%) | 589/1,024 (57.52%) | +0.29 pp |
| Required fact coverage | 72.65% | 69.12% | -3.53 pp |
| Patient-state fact coverage | 58.83% | 55.17% | -3.66 pp |
| External evidence coverage | 100.00% | 100.00% | +0.00 pp |
| Citation validity | 100.00% | 100.00% | +0.00 pp |

Paired outcomes were 42 Team-only successes and 39 Strong-Single-only successes (943 ties). The exact two-sided McNemar test gives **p = 0.824**; the three-episode net gain is not statistically distinguishable on this DEV set.

## Prespecified slices

| Slice | n | Strong Single success | Team success | Δ | Required fact coverage, Single → Team |
|---|---:|---:|---:|---:|---:|
| COMPLEX | 248 | 50/248 (20.16%) | 57/248 (22.98%) | +2.82 pp | 71.14% → 60.00% (-11.14 pp) |
| SIMPLE | 400 | 355/400 (88.75%) | 355/400 (88.75%) | +0.00 pp | 93.88% → 93.88% (+0.00 pp) |
| Other registered families | 376 | 181/376 (48.14%) | 177/376 (47.07%) | -1.06 pp | 51.06% → 48.80% (-2.26 pp) |

Grounded task success matched task success in these slices because every episode passed the grounding and citation checks. Slice comparisons are descriptive: the paired exact McNemar p-values are 0.470 for COMPLEX, 1.000 for SIMPLE, and 0.388 for the other families.

## Routing and execution

- Router chose the Single fast path for 744/1,024 episodes (72.66%) and sent 280/1,024 (27.34%) to the Lead; all 280 activated a Team.
- Exact specialist-set match: 371/1,024 (36.23%).
- Capability micro-precision: 651/1,305 (49.89%); micro-recall: 651/1,034 (62.96%).
- Worker completion: 100%; useful-worker rate: 81.11%; Lead incorporation: 73.98%.
- Routed Team overall latency: mean 23.64 s, P50 8.20 s, P95 72.73 s. Strong Single: mean 9.68 s, P50 8.33 s, P95 16.40 s.
- Among activated Team requests (n=280), P95 latency was 81.69 s. Mean per-request parallel speedup was 1.89×.
- Routed Team used 1,865 provider calls, 2,099 tool calls, and 877,323 tokens; Strong Single used 1,024 provider calls, 2,048 tool calls, and 389,975 tokens.

## TRAIN_DEV

The deterministic TRAIN-only sample contained 256 episodes per system. Task success was 153/256 (59.77%) for Strong Single and 159/256 (62.11%) for Routed Team (+2.34 pp). This was used for engineering selection before the configuration was frozen; final claims use DEV above.

## Reproducibility and scope

- Run ID: `20261003-cpu-r2`.
- Frozen config: `MA_MVP1_CONFIG_V1`, SHA-256 `d0ae79ac5ad550dc255d55c7e65d0fb50448df46eb9d2534d9272c4e5b22a774`.
- Dataset: `health-copilot-owned-longitudinal-v1`, root hash `e28ea9ef9ecae47d3f27f28c68042066e9af297fe808cafebf1d3c8c80fb2134`.
- Each system used all 512 DEV_IID and 512 DEV_STRUCTURAL episodes. Reserved TEST/OOD rows were not accessed or materialized.
- Model: local `Qwen3-8B-Q4_K_M.gguf`, SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`; llama.cpp CPU-only, 24 threads, three server slots, temperature 0.
- The artifact directory contains 2,048 result rows and traces, 1,024 routing records, worker reports, metrics, slice metrics, latency metrics, frozen config, and manifest.

## Closeout

The routed team produced only a three-episode net gain overall, with no detectable paired improvement, lower fact coverage, about 2.44× mean latency, and roughly 2.25× token use. The COMPLEX slice had a nominal +2.82 pp task-success difference, while its required-fact coverage fell by 11.14 pp and the paired result was inconclusive. Keep Routed Medical Team as a research prototype; this run does not support making it the default over Strong Single. The project-owned universe is synthetic research data and does not establish clinical safety or generalization to real patients.
