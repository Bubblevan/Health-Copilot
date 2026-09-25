# E1.2 R2MED runtime-tail audit

## Decision

**P0 classification: `RUNTIME_NUMERICAL_SENSITIVITY_CONFIRMED`.** Batch size and attention backend produced reproducible low-order ranking changes on PMC-Clinical DEV, including four relevant-document rank changes. None changed top-10 membership, nDCG@10, or MRR@10. The observed changes are therefore real but do not explain the aggregate nDCG tail: every tested condition remained at nDCG@10 `0.1425234328` and MRR@10 `0.1649470899`.

Stop here; do not create the optional old PyTorch reference environment or continue chasing the `0.003677` difference.

## Scope and runtime

- Source: frozen R2MED source manifest; PMC-Clinical **DEV only** (114 queries, 60,406 documents).
- Wiring: `ARTICLE_ARTICLE`, CLS pooling, FP32, 512-token maximum, BEIR-style cosine ranking, top 100.
- No TEST data was loaded. No query or corpus text is present in the result artifact.
- Python 3.11.15; PyTorch 2.14.0+cu132; CUDA runtime 13.2; driver 616.92; Transformers 5.17.0; sentence-transformers 6.1.0; tokenizers 0.23.2; safetensors 0.8.0.
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU. Default attention resolved to `sdpa`; eager probe resolved to `eager`. TF32 matmul off, cuDNN TF32 on; deterministic algorithms and cuDNN deterministic off.
- All four runs completed; no OOM.

## Results against batch 32 / default attention

`changed_query_count` means a changed top-100 ordering. The detailed `changed_queries` array also includes queries whose top-100 order stayed fixed but a qrel-relevant document's full-corpus rank changed.

| Variant | nDCG@10 | MRR@10 | Top-100 order changed | Top-10 changed | Relevant doc rank pairs changed | Max embedding abs delta (query / document) | Max score abs delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| batch 16 / default | 0.1425234328 | 0.1649470899 | 6 | 0 | 1 | 1.27e-6 / 3.46e-6 | 5.36e-7 |
| batch 64 / default | 0.1425234328 | 0.1649470899 | 3 | 0 | 2 | 1.03e-6 / 3.46e-6 | 4.77e-7 |
| batch 32 / eager | 0.1425234328 | 0.1649470899 | 4 | 0 | 2 | 1.91e-6 / 6.41e-6 | 8.94e-7 |

Affected DEV query IDs, without query text:

- Batch 16: top-100 order — `q_PMC7884024`, `q_PMC7091520`, `q_PMC7980323`, `q_PMC2924564`, `q_PMC3047687`, `q_PMC3564101`; relevant-rank-only — `q_PMC5390630`.
- Batch 64: top-100 order — `q_PMC7884024`, `q_PMC7091520`, `q_PMC3047687`; relevant-rank-only — `q_PMC5467287`, `q_PMC5390630`.
- Eager: top-100 order — `q_PMC7884024`, `q_PMC7980323`, `q_PMC3047687`, `q_PMC3564101`; relevant-rank-only — `q_PMC5390630`, `q_PMC7829610`.

Relevant-document rank changes and score movements:

| Variant | Query ID | Relevant document ID | Batch-32 rank → variant rank | Batch-32 score → variant score |
| --- | --- | --- | ---: | ---: |
| batch 16 | `q_PMC5390630` | `PMC6931112_1` | 3152 → 3151 | 0.7801719308 → 0.7801720500 |
| batch 64 | `q_PMC5467287` | `PMC8639695_1` | 11132 → 11131 | 0.7435770631 → 0.7435771227 |
| batch 64 | `q_PMC5390630` | `PMC6931112_1` | 3152 → 3151 | 0.7801719308 → 0.7801720500 |
| eager | `q_PMC5390630` | `PMC3813753_1` | 9779 → 9780 | 0.7613946795 → 0.7613945603 |
| eager | `q_PMC7829610` | `PMC2852755_1` | 21678 → 21677 | 0.7516431808 → 0.7516431808 (float32 tie) |

The largest score movement was below `9e-7`; all listed relevant-rank shifts occurred far below the top 10. Thus the ranking implementation has small runtime numerical sensitivity, while the reported PMC-Clinical DEV aggregate metrics are invariant across these settings.

## Separate MIRAGE provenance gate

E1.3 **must stop before QA**. The provenance audit found six completed E1.2 answer-run manifests for partition `TEST`, each covering all 5,235 clean-test cases under the frozen E1.2 config. Only manifest metadata and output-file existence were inspected; answer contents, TEST questions, and gold labels were not read during this audit. The E1.2 clean holdout is no longer untouched for E1.3, and the sprint explicitly forbids making a replacement split and calling it a clean TEST.

Therefore no E1.3 generator sanity/benchmark calls, DEV answer arms, frozen E1.3 config, or DEV selection lock were created. MIRAGE is not eligible to proceed to a confirmatory result under this split. See [`mirage_holdout_audit.json`](../../runs/e1_3/mirage_holdout_audit.json).

## Reproducibility artifacts

- Probe implementation: [`audit_r2med_runtime_tail.py`](../../tools/audit_r2med_runtime_tail.py).
- Machine-readable output: `E:\Health-Copilot-E1.2\parity\runtime-tail\run-20260925T100243Z\runtime_tail_audit.json` (SHA-256 `bf1e5192881e651810f3519be85f9dfa5c056c0ae934ab75999a55987b694e35`).
- Frozen R2MED source manifest: `runs/e1_2/r2med_source_manifest.json`.
- MIRAGE holdout audit: [`runs/e1_3/mirage_holdout_audit.json`](../../runs/e1_3/mirage_holdout_audit.json).
