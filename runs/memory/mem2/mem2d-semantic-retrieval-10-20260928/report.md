# MEM-2D - Frozen RawSpan Semantic Retrieval Ablation

Completion gate: `MEM2D_SEMANTIC_RETRIEVAL_FROZEN_10_DIAGNOSTIC=YES`.

## Scope

This is a retrieval-only semantic ablation over the immutable MEM-2C RawSpan inventory. The ten preselected DEV questions are diagnostic only; no 102-case DEV, TEST, tuning, proposition extraction, revision semantics, RevMem, or RL was run.

- Inventory SHA256: `93414251555c003e3acaa51968f5cf07a85ba3a189743a66301dd20ed7e09c26`; records: 52,703; unique retrieval documents: 52,703.
- Lexical parity: 80/80 exact top-8 records across six frozen fields per record.
- New reader calls: 20 (10 Dense + 10 Hybrid); judge, hosted, and memory-internal LLM calls: 0.
- Embedding: local `Qwen/Qwen3-Embedding-0.6B` on `NVIDIA GeForce RTX 4090 Laptop GPU`, no hosted fallback, document/query truncations: 0/0.
- Reader contract, projection, M10 RawSpan representation, and 1024 estimated-token Memory budget were held fixed.

## Descriptive Evidence

These ten cases are not a performance ranking or a public benchmark claim. Answer-session retrieval does not establish that the answer-bearing span was retrieved.

| Arm | Native Recall@5 | Native Recall@8 | Native MRR | Projected Recall@8 | Gold token coverage | Token F1 | Reader context tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lexical (frozen MEM-2C) | 0.49000000000000005 | 0.5599999999999999 | 0.5666666666666667 | 0.54 | 0.38305781175346393 | 0.11563025210084033 | 1768.6 |
| Dense | 0.9266666666666665 | 0.9466666666666667 | 1.0 | 0.9466666666666667 | 0.6886845039018953 | 0.18201877934272298 | 1875.5 |
| Hybrid RRF | 0.8633333333333333 | 0.9133333333333333 | 0.8833333333333332 | 0.9133333333333333 | 0.6096321070234113 | 0.16635586635586636 | 1831.1 |

## Efficiency and Crowding

- Unique-document embedding input tokens: 4,781,476; query tokens: 391; cached-vector verification: 0.5s for 52,703 hits; initial full-corpus embedding duration was not retained.
- Query repeatability was exact for identical frozen batch shapes; a singleton-versus-batched FP16 spot check was diagnostic only and did not determine retrieval results.
- Embedding cache hits/misses: 52,703/0; dense similarity wall time: 217.3ms.
- Mean context tokens (Qwen3-8B tokenizer): Lexical 1768.6, Dense 1875.5, Hybrid 1831.1.
- Per-question context token/estimated-token ratios, content-only diagnostic, top-8 session/turn crowding, and latencies are in `retrieval_diagnostics_pre_reader.json`, `efficiency.json`, and `deterministic_metrics.json`.

## Interpretation Boundary

Dense retrieval can support a semantic-versus-lexical retrieval observation only on these frozen cases. Hybrid evidence can indicate complementary rankings under the inherited RawSpan retrieval view. No statistical inference or quality/cost winner is claimed. The current semantic document view deliberately includes M10's structural metadata and opaque RawSpan key; a content-only view is a separate future ablation.

## Reproduction and Integrity

- Pre-reader aggregate freeze: `79aae73d7f396ab7c5992f63696270a5d7124b8f2aaa1633b4067134a053be47`.
- Context projection: `ee16902373d695307db42797f33a1f4a531484096b36c29639b98abfd58d52ed`; shared reader: `57d3df897a1cf20a6ab0277e4dca3b6ad58cc0348e2057aacfffb1a8184535e3`.
- Raw vectors stay in the local cache outside Git; only top-50 lexical/dense and top-8 RRF result rows are retained.
- TEST access: false; 102-case DEV: false; no tuning was performed.

Stop after this frozen-ten diagnostic and review the case-level evidence before proposing another retrieval component.
