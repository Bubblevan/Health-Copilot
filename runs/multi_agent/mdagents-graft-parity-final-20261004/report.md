# Health-Copilot MDAgents Graft Parity Sample

Date (UTC): 2026-10-04T06:25:29.567585+00:00

This is a deterministic 128-case sample from the pinned MedQA US test split. The comparison uses the same shuffled `model_question` strings and labels as the frozen local MDAgents adaptive JSONL. It does not rerun the reference arm or the full test split.

| Metric | Frozen MDAgents Adaptive | Health-Copilot graft | Delta |
|---|---:|---:|---:|
| Accuracy | 60.16% (77/128) | 59.38% (76/128) | -0.78 pp |
| Parse success | 87.50% | 99.22% | +11.72 pp |
| Exact parsed-answer agreement | — | 59.38% | 76/128 |
| Agreement when both parse | — | 68.47% | 76/111 |
| Complexity-route agreement | — | 58.59% | 75/128 |

## Graft runtime

- Complexity distribution: `{"advanced": 17, "basic": 90, "failed": 0, "intermediate": 21, "safety_routed": 0}`
- Mean model calls per case: 5.98
- Mean tokens per case: 6311.9
- Mean latency: 49.32 s
- Wall time: 838.0 s; throughput 0.153 cases/s
- Runtime failures: 0

## Route slices

| Complexity | n | Reference accuracy | Health-Copilot accuracy | Answer agreement |
|---|---:|---:|---:|---:|
| basic | 82 | 68.29% | 60.98% | 58.54% |
| intermediate | 33 | 48.48% | 54.55% | 72.73% |
| advanced | 13 | 38.46% | 61.54% | 30.77% |


## Engineering parity targets

The prior handoff set sample-level targets of absolute accuracy delta ≤ 2 percentage points, parsed-answer agreement ≥ 85%, and complexity-route agreement ≥ 90%.

- Accuracy delta target: **PASS**
- Answer agreement target: **MISS**
- Route agreement target: **MISS**
- Combined: **MISS**

A pass is evidence of parity on these sampled cases only. It does not prove all 1,273 cases would match; this run deliberately stops at the sample and does not rerun full MedQA in Health-Copilot.

## Artifacts

- `health_copilot.jsonl`: one checkpoint row per completed question, including raw answer and parsed option.
- `paired_results.json`: case-level comparison against frozen reference rows.
- `metrics.json`: denominators, metric definitions, and run configuration.
- `manifest.json`: model, data, and code identities.
