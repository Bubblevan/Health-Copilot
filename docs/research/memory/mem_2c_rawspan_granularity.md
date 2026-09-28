# MEM-2C - Lossless Raw-Span Memory Granularity Ablation

Completion gate: `MEM2C_RAWSPAN_FROZEN_10_DIAGNOSTIC=YES`.

## Research Question

Measure whether exact deterministic textual spans, rather than whole dialogue turns, improve native lexical retrieval discrimination or fixed-budget context coverage. The sole intended intervention is MemoryRecord granularity.

## Frozen Method

- Segmenter: `raw-span-segmenter-v1`; identity and boundaries are in `raw_span_segmenter_contract.json`.
- Every span is an exact substring; concatenating spans reconstructs its source turn exactly. No text is stripped or normalized.
- Each non-empty span becomes one M10 `ADD / SESSION_NOTE / SESSION_DERIVED / NON_SENSITIVE`, version 1, ACTIVE record.
- The M10 SQLite store, native lexical overlap/exact-key scoring, `top_k=8`, temporal timestamp, ContextManager budgets and frozen rank-aware projection remain unchanged.
- Reader: frozen local Qwen3-8B Q4_K_M, final reader contract SHA `57d3df897a1cf20a6ab0277e4dca3b6ad58cc0348e2057aacfffb1a8184535e3`, 10 calls; memory-internal LLM, embedding, judge, and hosted calls are zero.
- Benchmark labels are joined only after prediction and call-ledger artifacts are hash-frozen.
- Scope is exactly the ten frozen DEV diagnostic questions. TEST access is false; 102 DEV is not run.

## Frozen Evidence

- Turn reconstructions: 4854/4854 exact.
- Raw turns: 4854; raw spans: 52703; expansion factor: 10.858.
- Mean native answer-session Recall@8: RawTurn 0.48 to RawSpan 0.5599999999999999.
- Mean projected answer-session Recall@8: RawTurn 0.33999999999999997 to RawSpan 0.54.
- Mean projected gold-token coverage: RawTurn 0.39507087115782774 to RawSpan 0.38305781175346393.
- Mean reader-visible context tokens: RawTurn 990.5 to RawSpan 1768.6.
- Full per-question, per-category, answer, retrieval, and efficiency diagnostics are in the frozen run artifacts linked from `runs/memory/mem2/mem2c-rawspan-10-20260928/report.md`.

## Interpretation Boundary

A native-recall gain supports only finer textual units improving lexical retrieval discrimination. A projected coverage gain supports only smaller units using the fixed Memory budget more efficiently. Answer metrics are downstream deterministic diagnostics. This experiment does not validate proposition memory, semantic extraction, revision memory, or RevMem. No quality/cost winner is claimed.

Answer-session provenance is measured at session level only. The public labels do not identify the answer-bearing turn, so parent-turn recall is not fabricated. Case-level outcome, failure locus, causal attribution, and notes remain null in `mem_2c_case_review.json` for subsequent human Reflection.

Stop after this frozen-ten diagnostic. Do not automatically run 102 DEV, TEST, dense retrieval, proposition extraction, RevMem, or RL.
