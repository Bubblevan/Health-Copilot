# HY-DATA-0 Summary

**HY-DATA-0 SMOKE · NOT FROZEN BENCHMARK · NOT CLINICAL ACCURACY**

- Corpus identity: `f257e09f6e6ac5d708ec953d5f1f78d196fd13a3eda669b5e681c316aee74a7e`; index identity: `af1a5fe749f91971250476d8c731a08d205519cc1636cacd168a9dc641897b8b`.

## Data quality

- URLs discovered / fetched / failed: 151 / 151 / 0.
- Raw snapshots: 151; raw-only dynamic pages: 68.
- Canonical documents / rejected: 67 / 84.
- Document types: {"department": 10, "doctor_profile": 20, "faq": 27, "hospital_info": 5, "patient_education": 3, "perioperative_instruction": 2}.
- Freshness classes: {"POTENTIALLY_STALE": 50, "STATIC_EDUCATION": 3, "STATIC_PROFILE": 14}.
- Exact duplicate documents / near-duplicate flags: 2 / 0.
- Chunks: 138; median chars/tokens 77.0 / 37.0; p95 chars/tokens 256 / 132.
- Embeddings successful / failed: 138 / 0; dimension 1024.
- Milvus collection / entities: huiyi_knowledge_v0 / 138.

## Retrieval smoke

| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR@5 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.889 | 0.944 | 0.963 | 0.920 | 0.9 | 1.4 |
| Qwen3 Dense | 0.907 | 0.926 | 0.944 | 0.920 | 139.8 | 169.7 |
| Hybrid RRF | 0.926 | 0.963 | 0.963 | 0.941 | 141.3 | 170.9 |

- Smoke queries: 54.
- Metadata filter smoke: passed.
- The evaluation set is a retrieval sanity check, not a frozen benchmark, clinical accuracy measure, or production metric.
