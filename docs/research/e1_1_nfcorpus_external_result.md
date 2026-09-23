# E1.1 NFCorpus external retrieval ablation

## Outcome

E1.1 completed on the frozen E0 NFCorpus artifact. The run evaluated all four
M5 retrieval families on the same 323-query `test` split, 3,633-document corpus,
graded qrels, and `top_k=10`. No source was downloaded during the run, and the
learned arms did not fall back to hashing or lexical scoring.

The result is intentionally reported as retrieval evidence, not as clinical
performance or patient-care validation.

| Arm | Recall@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: |
| BM25 | 0.152033 | 0.518152 | **0.311041** |
| learned Dense | 0.102525 | 0.408797 | 0.236020 |
| Hybrid RRF | **0.152613** | **0.524585** | 0.309421 |
| Hybrid + CrossEncoder | 0.152613 | 0.501643 | 0.309199 |

The main finding is not “dense wins”: on this pinned NFCorpus protocol, learned
Dense is below BM25 on all three metrics. Hybrid RRF gives a small Recall/MRR
improvement over BM25 (+0.000580 Recall@10 and +0.006433 MRR), while BM25 keeps
the best nDCG@10. The CrossEncoder reranker does not improve the hybrid result
here; its Recall is unchanged and both MRR and nDCG@10 are lower. This is a
useful external negative result and does not change the frozen product default,
which remains BM25.

## Reproducibility and fairness

- Dataset: `nfcorpus-v1`, BEIR-v2.2.0, upstream revision
  `beir-v2.2.0@6ef8c9097ebfb203ad360bd64e0cfb93e64f4a44`.
- Pinned manifest SHA-256: `36f32a5721dc10976128a9a8b3578b37053fa6557b7a4408d87d2d4a5e574128`.
- Raw artifact SHA-256: `efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b`.
- Normalized dataset SHA-256: `7334540bf8d5c035d0ffe36d942d2773fff138fa925d89c9146360a4b165b300`.
- Test split manifest SHA-256: `8512deaceb875c0edd91ed4d3e40e48fa4a54e9490e17d72dde0ec3ddd34069b`.
- RRF `k=60`; reranking candidate pool and final result were both fixed at 10.
- Embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`,
  revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, dimension 384.
- Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, revision
  `1427fd652930e4ba29e8149678df786c240d8825`.
- Both learned components used `local_files_only=true`; model construction
  failure is fatal in the runner and cannot silently select a lexical fallback.
- Metrics use the existing `evaluate_nfcorpus` contract: positive graded qrels
  for Recall, first relevant rank for MRR, and graded gain for nDCG@10.

## Evidence artifacts

- [structured result](../../runs/e1_1/nfcorpus_external_v1/result.json)
- [metrics](../../runs/e1_1/nfcorpus_external_v1/metrics.json)
- [per-query results and failure evidence](../../runs/e1_1/nfcorpus_external_v1/case_results.jsonl)
- [exact run configuration](../../runs/e1_1/nfcorpus_external_v1/run_config.json)
- [reproducible runner](../../tools/run_e1_1_nfcorpus_eval.py)

The run was executed from repository baseline `main@33a46865a1106c4158ad51593dd21412a3328e55`.
The runner source SHA-256 recorded with the artifact is
`BEC0D3F3AFE5F4E8189782EE811D4798AEED38B87CB207AD94DBB87B5A31528B`.

## Resume-ready evidence

> Executed a reproducible four-arm external retrieval ablation on BEIR NFCorpus
> (323 test queries, 3,633 biomedical documents) using a pinned normalized
> dataset identity and identical Recall@10/MRR/nDCG@10 protocol across BM25,
> learned Dense, Hybrid RRF, and Hybrid + CrossEncoder arms.

> Preserved learned-model provenance with explicit revisions, embedding
> dimension, and offline-only loading; found that learned Dense underperformed
> BM25, Hybrid RRF delivered only a small MRR/Recall gain, and CrossEncoder
> reranking did not improve the fixed external protocol.

## Scope boundary

This closeout starts only E1.1. M0–M10.1, E0/E0.1, and H-Ref remain frozen.
E1 MIRAGE, the HealthBench judge, E2/E3/E4/E5, and M11 training were not
started. Existing gold, metric definitions, old run artifacts, profile hashes,
and the product default were not changed.
