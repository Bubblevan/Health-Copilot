# MEM-2C - Lossless Raw-Span Frozen-10 Diagnostic

Gate: `MEM2C_RAWSPAN_FROZEN_10_DIAGNOSTIC=YES`

## Protocol

- Sole intervention: MemoryRecord granularity, one exact raw turn vs deterministic exact-text spans.
- Segmenter: `raw-span-segmenter-v1`; representation: `raw-span-v1`.
- No semantic extraction, embeddings, dense retrieval, reranking, revisions, UPDATE/DELETE, judge, or hosted calls.
- Native M10 lexical ranking and exact-key behavior are unchanged; `top_k=8`.
- Frozen rank-aware projection: `m10-rank-aware-projection-v1`, 1,024 estimated-token budget.
- Final reader contract SHA256: `57d3df897a1cf20a6ab0277e4dca3b6ad58cc0348e2057aacfffb1a8184535e3`; reader calls: 10/10 local Qwen3-8B.
- Gold/category/session labels were joined only after prediction and call-ledger SHA freeze.
- TEST access: false; 102-case DEV: not run.

## Integrity And Efficiency

- Original turns reconstructed exactly: 4854/4854.
- Raw turns: 4854; raw spans / memory records: 52703 (expansion x10.86).
- Estimated tokens per unit, median/p95: RawTurn 198.5 / 813.0; RawSpan 118 / 142.0.
- Mean selected records: 7.2; mean M10 estimated memory tokens: 955.6; mean reader-tokenized context: 1768.6.
- Mean ingestion / native retrieval / projection / reader latency (ms): 1975.3067000000003 / 181.69279999999998 / 14.326699999999999 / 759.6829.
- SQLite bytes: 57880576; reader prompt/completion tokens mean: 1872.0 / 7.4.

## Paired MEM-2B Comparison

All figures below are frozen-ten diagnostic evidence, not a general performance ranking.

| Metric | MEM-2B RawTurn | MEM-2C RawSpan | Delta |
|---|---:|---:|---:|

## Finalization Audit

The first post-reader gate check rejected required zero-valued call counters and expected-false access flags due to a validator type/expectation bug. The validator was corrected after prediction and call-ledger freeze; the frozen cache was reused and no additional reader request was issued.
- Reader-execution runner SHA256: `177dcc90f243ad953a1de94b264aa34238a3dd126dea8dbf8b51064d3aa5ebe7`.
- Corrected finalizer runner SHA256: `e18b294b9df14c6d9c156cb8a1acb26d8b21665d7539478a59c98ba4693db2e7`.
- Additional reader requests: 0.
| Native answer-session Recall@5 | 0.4600 | 0.4900 | 0.0300 |
| Native answer-session Recall@8 | 0.4800 | 0.5600 | 0.0800 |
| Native MRR | 0.4500 | 0.5667 | 0.1167 |
| Projected answer-session Recall@5 | 0.3400 | 0.4900 | 0.1500 |
| Projected answer-session Recall@8 | 0.3400 | 0.5400 | 0.2000 |
| Projected MRR | 0.5000 | 0.5667 | 0.0667 |
| Gold-token context coverage | 0.3951 | 0.3831 | -0.0120 |
| Token precision | 0.1051 | 0.0745 | -0.0306 |
| Token recall | 0.2087 | 0.2600 | 0.0513 |
| Token F1 | 0.1245 | 0.1156 | -0.0089 |
| Normalized EM | 0.0000 | 0.0000 | 0.0000 |
| Reader-visible context tokens | 990.5000 | 1768.6000 | 778.1000 |
| Estimated memory tokens | 828.1000 | 955.6000 | 127.5000 |
| Selected-worse/better-dropped pairs | 1.5000 | 0.5000 | -1.0000 |
| Top-1 survival | 0.8000 | 1.0000 | 0.2000 |
| Top-3 survival | 0.4667 | 0.9667 | 0.5000 |
| Top-5 survival | 0.3800 | 0.9600 | 0.5800 |

Session recall is provenance-level coverage, not answer-bearing-turn recall. LongMemEval provides answer-session labels here, so no parent-turn answer labels are inferred. `Exact normalized gold sequence` is separately available per case in `deterministic_metrics.json`; answer scores remain downstream diagnostics only.

## Category Slices

| Question type | n | Native R@8 RawTurn to RawSpan | Projected R@8 RawTurn to RawSpan | Gold coverage RawTurn to RawSpan |
|---|---:|---:|---:|---:|
| knowledge-update | 2 | 0.5 to 0.75 | 0.25 to 0.75 | 0.0 to 0.0 |
| multi-session | 1 | 1.0 to 0.5 | 0.5 to 0.5 | 0.0 to 0.0 |
| single-session-assistant | 1 | 0.0 to 0.0 | 0.0 to 0.0 | 0.2 to 0.2 |
| single-session-preference | 2 | 0.0 to 0.5 | 0.0 to 0.5 | 0.3324972129319955 to 0.3010033444816054 |
| single-session-user | 2 | 1.0 to 1.0 | 1.0 to 1.0 | 1.0 to 1.0 |
| temporal-reasoning | 2 | 0.4 to 0.3 | 0.2 to 0.2 | 0.5428571428571429 to 0.5142857142857142 |

## Interpretation Boundary

A native-recall gain supports only that finer textual units improve lexical retrieval discrimination. A projected-coverage gain supports only that smaller records use the fixed Memory budget more efficiently. This ablation does not validate propositions, semantic extraction, revision memory, or RevMem. No quality/cost winner is claimed.

Case-level causal reflection remains intentionally unassigned in `docs/research/memory/mem_2c_case_review.json`.

## Frozen Artifacts

- `call_ledger.jsonl`: `d2e9fb8803ec7d05be81bcf5cc0357dfad0999b8a306e3e4bdd55b65f30147bb`
- `comparison_mem2b_vs_mem2c.json`: `39cdbca573a63b6ddfe750602d2b1786a06534727f38acb8ce408d339bce018d`
- `context_bundles.jsonl`: `2acac1991764da853b61c9b1b32a93cb2ef0cef5181c6ce27f897539493afb9e`
- `context_plans.jsonl`: `a033b8e00c82261b30760e4c0c33d6df93793097d5aefe61d3b8a68a83a641a3`
- `deterministic_metrics.json`: `15a03b8e21768aceb4a5999827321552f2a961eae9dd56d921a0fc22ecf8968c`
- `efficiency.json`: `94647368390ddc4c2c2572429f99c5c2072823f2f325cdd814790a9a8db3c548`
- `mem_2c_case_review.json`: `1afe5443ff3a4048431c86155536cf874a3069d34461eaf3714bf26dd95dc653`
- `memory_inventory.jsonl`: `93414251555c003e3acaa51968f5cf07a85ba3a189743a66301dd20ed7e09c26`
- `memory_operations.jsonl`: `3f9ecaea37539f4e8fee1cd6b4f02b2858d4b99b7ac61f50bf15560b9de1777f`
- `pre_reader_diagnostics.json`: `3d240ea61ea45e385a5c7996f7313e281dce7f0ed150776e7fdf21e368170e2d`
- `predictions.jsonl`: `6af8ecc6eb1f0fe38c0f891cb539df1724f5cd9033242eef1ce37304a9b77acb`
- `retrieval_results.jsonl`: `da101fa8d98672ac887facc40f4b54b921055fce3e5aff49fe790fac2a62fdc8`
- `segmentation_audit.jsonl`: `57472a35528a4d7eda69f899e1f74b4615d8382b1f1d1d2192f614581b49e2e2`

The run stops at this frozen-ten diagnostic. No 102 DEV, TEST, dense retrieval, proposition extraction, RevMem, or RL was run.
