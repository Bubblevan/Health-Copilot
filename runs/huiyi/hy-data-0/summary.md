# HY-DATA-0 Summary

**HY-DATA-0 SMOKE · NOT FROZEN BENCHMARK · NOT CLINICAL ACCURACY**

## Data quality

- URLs discovered / fetched / failed: 151 / 151 / 0.
- Raw snapshots: 151; raw-only dynamic pages: 68.
- Canonical documents / rejected: 67 / 84.
- Document types: {"department": 10, "doctor_profile": 20, "faq": 27, "hospital_info": 5, "patient_education": 3, "perioperative_instruction": 2}.
- Freshness classes: {"POTENTIALLY_STALE": 50, "STATIC_EDUCATION": 3, "STATIC_PROFILE": 14}.
- Exact duplicate documents / near-duplicate flags: 2 / 0.
- Chunks: 181; median chars/tokens 45 / 21; p95 chars/tokens 236 / 107.
- Embeddings successful / failed: 181 / 0; dimension 1024.
- Milvus collection / entities: huiyi_knowledge_v0 / 181.

## Retrieval smoke

| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR@5 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.870 | 0.944 | 0.963 | 0.908 | 1.3 | 1.8 |
| Qwen3 Dense | 0.907 | 0.926 | 0.944 | 0.921 | 150.1 | 172.8 |
| Hybrid RRF | 0.907 | 0.926 | 0.963 | 0.925 | 151.5 | 174.0 |

- Smoke queries: 54.
- Metadata filter smoke: passed.
- The evaluation set is a retrieval sanity check, not a frozen benchmark, clinical accuracy measure, or production metric.
