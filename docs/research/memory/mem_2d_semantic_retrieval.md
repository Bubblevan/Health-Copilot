# MEM-2D - Frozen RawSpan Semantic Retrieval Ablation

Completion gate: `MEM2D_SEMANTIC_RETRIEVAL_FROZEN_10_DIAGNOSTIC=YES`.

## Research Question

Given the immutable MEM-2C RawSpan inventory, test whether local dense semantic retrieval recovers sessions/spans missed by M10 lexical overlap, and whether untuned equal-weight RRF retains complementary ranked evidence.

## Frozen Method

- Inventory: `runs/memory/mem2/mem2c-rawspan-10-20260928/memory_inventory.jsonl` SHA256 `93414251555c003e3acaa51968f5cf07a85ba3a189743a66301dd20ed7e09c26`; no re-ingestion or second inventory copy.
- Dense embedding: `Qwen/Qwen3-Embedding-0.6B` at `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, tree SHA `9d2d790d6448ef2c0911ffeb03f959d035c71ac3d2b14b7d586f2d2b39fb0efa`, weights SHA `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`, 1024-d, CUDA FP16, L2-normalized float32, local-only; truncations 0.
- Left-padded pooling selects the maximum sequence index with `attention_mask=1`; `sum(mask)-1` is not an index under left padding. This implements the frozen final-nonpadding-token contract and is unit-tested; the same corrected adapter serves every dense arm.
- Dense ranks cosine dot products over M10 scope/time-valid records; ties use memory ID ascending. No temporal boost, diversity, reranking, or lexical bonus.
- Hybrid: `rawspan-lexical-dense-rrf-v1`, lexical/dense source depth 50, k=60, equal weights, final top 8; k was protocol-selected before results and was not swept.
- Lexical top-8 parity: 80/80 across memory ID, rank, score, source session, source turn, and source span.
- Projection: frozen `m10-rank-aware-projection-v1`, SHA `ee16902373d695307db42797f33a1f4a531484096b36c29639b98abfd58d52ed`; estimated Memory budget 1024; reader contract `57d3df897a1cf20a6ab0277e4dca3b6ad58cc0348e2057aacfffb1a8184535e3`.
- Ten frozen DEV cases only. Lexical reader outputs are reused from MEM-2C; new calls are exactly 10 Dense and 10 Hybrid using the same local Qwen3-8B reader.
- No judge, hosted API, memory-internal LLM, 102-case DEV, TEST, proposition extraction, or revision semantics.

## Retrieval-View Limitation

Dense and lexical retrieval consume the exact M10 lexical retrieval string: `record.key + space + record.value-or-json.dumps(value, ensure_ascii=False)`. It includes the opaque RawSpan key, role, session date, source turn/span indices, and span content. This inherited metadata may influence semantic similarity; MEM-2D deliberately does not switch to a content-only view.

## Evidence and Efficiency

- Exact records: lexical parity 80/80; answer calls 20/20; judge/hosted/memory-internal calls 0/0/0.
- Deterministic token metrics and retrieval/crowding breakdowns are in `deterministic_metrics.json`; cache, embedding, context, and latency accounting are in `efficiency.json`.
- The ten-case outcomes are diagnostic and descriptive only. No performance ranking, statistical generalization, or quality/cost winner is claimed.
- Answer-session hit is not equivalent to answer-bearing span hit; session provenance cannot identify the answer turn.
- Raw vectors stay outside the repository; top-50/top-8 retrieval rows are retained with SHA sidecars.

## Stop Boundary

Stop for human Reflection. Do not run 102 DEV or TEST, tune fusion, change the retrieval view, implement proposition/revision memory, start RevMem, or train RL from this stage.
