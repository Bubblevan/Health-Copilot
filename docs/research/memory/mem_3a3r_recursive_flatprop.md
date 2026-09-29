# MEM-3A.3R - Recursive Overflow Recovery and FlatProp Completion

## Stage boundary

This stage resumes the unchanged MEM-3A.3 frozen 785-initial-chunk plan and completes the 477-session frozen-ten FlatProp diagnostic. The historical MEM-3A.3 gate remains `MEM3A3_CHUNKED_FLATPROP_FROZEN_10_DIAGNOSTIC=NO`; this stage does not rewrite that record. The previous `sharegpt_vbNrVtS_151` response is retained for forensic inspection only and contributes zero propositions.

## Frozen method

Writer method remains `flat-proposition-extractor-v4-chunked`, with the v3 minimal proposition semantics (`proposition_text`, `evidence_refs`), the same prompt, schema, Qwen3-8B Q4_K_M artifact, temperature 0, seed 42, thinking disabled, 8,192 completion cap, and existing RawSpan source catalog. No revisions, entity/attribute fields, maxItems limit, retry, repetition penalty, hosted provider, judge, DEV, TEST, or MedMemoryBench run is in scope.

The frozen MEM-3A.3 chunk manifest remains the root plan. Only a local `finish_reason=length` triggers deterministic recursive subdivision. Split complete turns first; split only between existing RawSpans when the primary region is confined to a single turn. Token mass comes from the frozen Qwen tokenizer over each RawSpan content, with complete-turn mass equal to the sum of its RawSpan masses. A contiguous split minimizes left/right token-mass imbalance; ties choose the earliest boundary. Child overlap follows the frozen MEM-3A.3 policy and is context-only. Any child above 6,144 rendered prompt-plus-schema tokens is subdivided before inference. A one-RawSpan request that reaches the 8,192 completion cap is irreducible. All other transport, envelope, schema, provenance, or validation failures stop the stage.

## Cache and lineage

Each of the 147 verified successful MEM-3A.3 responses is imported from its hash-valid local cache with zero provider calls in this stage. If a cache is wholly absent, re-execute that chunk once and disclose it; a present but corrupt cache fails closed. The failed root response is never parsed into semantic memory. Every recursive child records its parent, root initial chunk, depth, split contract hash, primary/overlap ranges, request hash, and measured prompt tokens in `overflow_lineage.jsonl`.

Terminal leaf packets alone enter session aggregation. Exact identity is the SHA-256 of stripped proposition text and catalog-ordered unique evidence references. Same-leaf, cross-leaf, and overlap duplicates collapse; paraphrases, changed references, and near duplicates remain separate. Near-duplicate similarity is diagnostic only.

## Downstream boundary

After 477/477 sessions have exact primary RawSpan ownership and the complete FlatProp inventory is frozen, materialize only `ADD / SESSION_NOTE / SESSION_DERIVED / ACTIVE / version=1 / supersedes_id=null`. Reuse frozen MEM-2D Qwen3-Embedding-0.6B local CUDA, dense proposition-text retrieval top-8, unchanged rank-aware projection SHA `ee16902373d695307db42797f33a1f4a531484096b36c29639b98abfd58d52ed`, 1,024-token memory budget, and frozen shared-reader SHA `57d3df897a1cf20a6ab0277e4dca3b6ad58cc0348e2057aacfffb1a8184535e3` for exactly ten reader calls. Predictions and reader ledger freeze before labels load. No judge is used.

## Closeout gate

`MEM3A3R_RECURSIVE_FLATPROP_FROZEN_10_DIAGNOSTIC=YES` requires complete source coverage, auditable initial and recursive nodes, no unresolved/irreducible failures, zero retries or hosted calls, exact proposition identity canonicalization, no revision semantics, dense top-8 for ten questions, unchanged projection and budget, ten successful reader calls, zero judge calls, no DEV/TEST/MedMemoryBench access, and valid SHA sidecars. The result is a frozen-ten mechanism diagnostic only, not a public benchmark or performance-ranking claim.
