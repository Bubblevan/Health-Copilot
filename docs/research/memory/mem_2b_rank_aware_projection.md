# MEM-2B — Retrieval-Rank-Aware Context Projection Ablation

## Protocol

MEM-2B is a single-component counterfactual over the frozen MEM-2A M10-Base ten-case DEV run. The research question is whether preserving native M10 retrieval order during budgeted projection retains more ranked evidence under the same 1,024 estimated-token Memory budget.

The MEM-2A diagnostic recorded native answer-session Recall@8 of `0.48` and projected answer-session Recall@8 of `0.24`. In `9/10` frozen cases, the legacy projection selected at least one worse-ranked item while dropping a better-ranked candidate. Examples include `1cea1afa` (selected ranks `1, 6, 7`; dropped `2, 3, 4, 5, 8`), `1c549ce4` (selected `3, 8`; dropped `1, 2, 4, 5, 6, 7`), and `a82c026e` (native answer-session recall `1`, projected recall `0`). This is an independent projection failure surface, not a claim that it explains every M10-Base failure.

The only intervention is memory candidate order inside ContextManager. The raw-turn ADD-only inventory, native lexical retrieval top-8, scores, provenance, question/date, reader, prompt contract, and all budgets remain fixed. The runner reconstructs MemoryRecord objects from the frozen inventory and uses the frozen retrieval artifact directly. It does not re-ingest turns or call `MemoryStore.matches()`.

## Selection Semantics

Priority classes remain authoritative (`PROTECTED`, `HIGH`, `NORMAL`, `LOW`). With explicit rank hints, only eligible MEMORY items within their existing priority slots are reordered by ascending unique retrieval rank. Non-memory units retain legacy deterministic positions. If `selection_rank_hints` is omitted, the frozen priority-then-`item_id` behavior and resulting ContextPlan hash remain unchanged.

The experiment uses the frozen M10 budgets unchanged: 4,096 estimated input tokens, 256 reserved system tokens, 512 reserved current-turn tokens, 1,024 memory tokens, 2,048 history tokens, and 1,024 tool-observation tokens. Selection remains greedy: if a candidate does not fit, skip it and continue evaluating later candidates; no knapsack optimization, token-ratio scoring, truncation, or summarization is introduced.

The benchmark adapter validates complete contiguous ranks `1..N` for each frozen native candidate list. ContextManager independently rejects non-positive/non-integer ranks, duplicate ranks, missing or unknown item IDs, and hints targeting non-memory, protected, system, or current-user items. Rank data is diagnostic metadata only and is not added to reader-visible MemoryRecord serialization.

## Reader And Metrics

Projection artifacts are hash-frozen before any answer call. Then exactly one local reader-answer call is made per frozen question ID using the final Qwen3-8B Q4_K_M reader contract (`57d3df897a1cf20a6ab0277e4dca3b6ad58cc0348e2057aacfffb1a8184535e3`), temperature `0`, seed `42`, thinking disabled, and 256 max output tokens. Memory-internal LLM, embedding, judge, and hosted calls are all zero.

Gold answer, category, `has_answer`, and `answer_session_ids` are joined only after the MEM-2B predictions JSONL has been hash-frozen. Native retrieval metrics are copied from the frozen MEM-2A retrieval artifact and are not recomputed by retrieval. Projected-context MRR uses the item's one-based position in the projected context. Answer-session recall is provenance-level coverage, not answer-bearing evidence recall. Normalized gold-token coverage and exact normalized gold-sequence presence are separately reported. Reader answer scores are descriptive only; projection policy is selected by protocol semantics, not DEV answer score.

Rank diagnostics distinguish (a) selected-worse-while-better-dropped pairs, which can still arise from the required greedy skip behavior, from (b) projected-order inversions among selected items. Rank-aware ordering is expected to eliminate the latter, not necessarily the former.

## Frozen Scope

Only the ten IDs in the frozen MEM-1D1 diagnostic selection are used. `TEST` is not accessed, the 102-case DEV run is not run, and no external baseline is rerun. No dense retrieval, embedding, proposition extraction, rewriting, revision semantics, memory representation change, M10-Flat, or RevMem component is introduced. Results are diagnostic evidence for Reflection; this stage does not choose the next failure-surface track.

The MEM-2A source artifacts are immutable historical evidence. MEM-2B outputs are written under a new run directory and include retrieval parity, ContextPlans, ContextBundles, per-case projection diagnostics, predictions, call ledger, deterministic metrics, paired comparison, and a report. The separate case-review file leaves human outcome, failure-locus, and notes fields null.

## Execution Gates

1. Verify all seven required MEM-2A SHA sidecars.
2. Verify exact top-8 parity against frozen inventory and legacy projection; stop on any mismatch.
3. Run `--prepare-only` to freeze parity, ContextPlans, ContextBundles, and pre-reader diagnostics.
4. Only after those hashes verify, execute exactly ten local reader calls.
5. Freeze predictions and call ledger before loading benchmark labels.
6. Score descriptively and record the closeout gate.

Completion marker: `MEM2B_RANK_AWARE_PROJECTION_FROZEN_10_DIAGNOSTIC=YES` only when all structural, provenance, determinism, local-reader, and no-TEST gates pass.
