# E1.3 — MIRAGE exploratory recovery

> Evaluation status: **EXPOSED EXPLORATORY BENCHMARK**
> These results are not treated as an untouched confirmatory holdout.

## History decision

The E1.2 artifacts contain all six requested arms on the same frozen 5,235-case clean MIRAGE TEST split. The manifests bind the frozen E1.2 configuration and QWEN3_LOCAL; completed rows report the configured `Qwen3-8B-Q4_K_M.gguf` served path. The 90-case DEV pilot and frozen configuration bind the pinned artifact SHA-256 `d98cdcbd…5745785`. The model file itself is no longer mounted at its recorded F: path, so it was not rehashed during this audit; model identity is verified through the committed configuration, pilot artifact, and per-completion served-model field.

**REUSE_HISTORY = YES.** No equivalent arm was rerun. Each arm has one row for every frozen case ID with no duplicate IDs. BM25 has one provider failure (5,234/5,235 completed); that case remains incorrect in the fixed denominator. All other arms completed 5,235/5,235. The historical MedCPT configuration binds the query encoder, article encoder, and cross-encoder; it is not the R2MED ARTICLE_ARTICLE method.

The reproducibility identity is the frozen config SHA-256 `c950faa4…07320f`, split SHA-256 `1427c479…9e7009d9`, and E1.2 code commit `d8f38f04…efe3484e575d1c28`. Per-arm manifests and result hashes are in `runs/e1_3/mirage_history_audit.json`.

## Aggregate outcomes

Accuracy is exact-answer accuracy over all 5,235 cases. Provider failures and invalid answers count incorrect. Token sums below are observed sums; token measurement coverage is shown because the BM25 failure has no answer-token measurement. Latency is the runner's component-summed proxy, not production end-to-end latency.

| Arm | Accuracy | Completion | Invalid (count / rate) | Provider failures | Answer input tokens (observed) | Retrieval calls | Retrieved chunks | Context chars | Proxy p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Closed book | 61.99% | 100.00% | 60 / 1.15% | 0 | 1,127,913 | 0 | 0 | 183,225 | 233 / 409 |
| Random context | 60.52% | 100.00% | 40 / 0.76% | 0 | 6,667,133 | 5,235 | 26,175 | 22,856,233 | 1,311 / 1,549 |
| BM25 | 62.02% | 99.98% | 39 / 0.75% | 1 | 6,642,084 (99.98% measured) | 5,235 | 26,175 | 23,137,961 | 885 / 2,328 |
| MedCPT canonical | **62.45%** | 100.00% | 40 / 0.76% | 0 | 6,167,268 | 5,235 | 26,175 | 22,102,925 | 1,268 / 1,834 |
| Cheap capability router | 62.23% | 100.00% | 49 / 0.94% | 0 | 3,640,366 | 2,311 | 11,555 | 10,567,541 | 404 / 2,266 |
| Jev capability router | 62.35% | 100.00% | 43 / 0.82% | 0 | 5,876,822 | 4,908 | 24,540 | 20,724,018 | 2,428 / 3,044 |

Random context matched BM25's document count on 100% of cases, but character-budget match was 94.65%; treat this as an imperfect context-length control. Retrieved chunks are the configured five per retrieval call. Router actions used only the question at inference time; oracle labels were constructed post hoc from exposed outcomes.

## Clean subdataset accuracy

| Arm | MedQA (1,018) | MedMCQA (3,346) | MMLU (871) |
|---|---:|---:|---:|
| Closed book | 60.12% | 57.83% | 80.14% |
| Random context | 59.73% | 55.74% | 79.79% |
| BM25 | 60.31% | 57.68% | 80.71% |
| MedCPT canonical | **60.61%** | **58.16%** | **81.06%** |
| Cheap capability router | 60.31% | 57.95% | 80.94% |
| Jev capability router | **60.61%** | 58.04% | 80.94% |

PubMedQA (500) and BioASQ (618) are historical exposed subdatasets and are not part of this six-arm clean TEST result. The repository contains earlier DeepSeek-Flash runs over these 1,118 cases. They used a different model, a 512-token output cap, and local MedRAG/Textbooks evidence rather than official MIRAGE snippets, so they are listed only as `HISTORICAL_EXPOSED` and are not comparable to or blended into the Qwen headline.

| Historical arm (DeepSeek-Flash) | PubMedQA accuracy (n=500) | BioASQ accuracy (n=618) | Overall answer coverage | Overall accuracy |
|---|---:|---:|---:|---:|
| Closed book | 58.20% | 87.86% | 100.00% | 74.60% |
| BM25 textbook RAG | 54.60% | 87.54% | 100.00% | 72.81% |
| MedCPT textbook RAG | 55.80% | 87.86% | 100.00% | 73.52% |
| Legacy Harness RAG (already-run; not repeated) | 12.60% | 25.24% | 26.74% | 19.59% |

The old Harness RAG arm abstained on 73.26% of cases. It is included only to account for an existing artifact, not as a candidate or headline. The MedCPT Harness directory has no aggregate metrics artifact and is marked incomplete in the JSON report.

## Oracle opportunity (post hoc only)

For each case, choose the cheapest correct fixed arm in the order Closed book < BM25 < MedCPT. If none is correct, assign the globally strongest fixed arm as fallback (MedCPT). This is an exposed-label ceiling, not an inference-time policy.

| Quantity | Result |
|---|---:|
| Closed-book sufficient / oracle chooses Closed book | 3,245 / 5,235 (61.99%) |
| Oracle chooses BM25 | 196 / 5,235 (3.74%) |
| Oracle chooses MedCPT as the cheapest successful action | 96 / 5,235 (1.83%) |
| No fixed arm succeeds | 1,698 / 5,235 (32.44%) |
| Fallback action when no fixed arm succeeds | MedCPT (1,698 cases) |
| Oracle union accuracy | 67.56% |
| Best fixed accuracy (MedCPT) | 62.45% |
| Oracle headroom over best fixed | 5.12 percentage points |
| Retrieval calls avoidable vs always-BM25 under oracle | 61.99% |

The oracle demonstrates potential complementarity but does not show that a learnable router can identify the winning arm. On these exposed cases the routers remain below the oracle quality ceiling; Jev has the highest component-summed p95 among these arms. Detailed router quality and action-cost regret definitions are recorded in the JSON report.

## Artifacts and limits

- `runs/e1_3/mirage_history_audit.json` — identity and six-arm completeness audit.
- `runs/e1_3/mirage_exploratory_report.json` — per-arm and per-subdataset metrics, latency components, oracle metrics.
- `runs/e1_3/mirage_oracle_labels.jsonl` — case IDs, Boolean outcomes, selected actions, and oracle action only; no question, answer text, prediction text, or evidence.

This is exploratory analysis only. Do not call the split unseen, untouched, held-out, or confirmatory. The reported accuracy differences are descriptive; no new confirmatory inference was run.
