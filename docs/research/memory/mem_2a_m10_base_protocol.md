# MEM-2A M10-Base Protocol

## Scope

MEM-2A measures the existing deterministic M10 Memory substrate on the frozen ten-question LongMemEval-S DEV diagnostic. The canonical research name is `M10-Base-RawTurn`; the display name is `M10-Base`. This is a public-benchmark diagnostic, not a performance claim or exact MemEval numerical reproduction.

The pipeline is limited to lossless raw user/assistant turn ingestion as `ADD` records, existing M10 policy/materialization, native M10 lexical-overlap plus exact-key ranking, unchanged `ContextManager` (`m10-context-v2`), and the frozen shared reader v3. The `bm25` string in the runtime profile name does not describe MemoryStore's ranking algorithm.

## Frozen Inputs

- Base commit: `c2db56c0096d28e3dff432d3ff5aac4ac4e8c7d5`.
- Dataset: LongMemEval-S `data/longmemeval/longmemeval_s_cleaned.json`, SHA256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
- Selection: exact ordered IDs in `docs/research/memory/main_smoke_10_manifest.json`, SHA256 `5a38ff79d79be6a9db531227d63d6dc22dc619d11d2c701b2b4cc0295f04c911`; selection is DEV-only and `test_access=false`.
- Reader contract: `docs/research/memory/final_reader_contract.json`, SHA256 `57d3df897a1cf20a6ab0277e4dca3b6ad58cc0348e2057aacfffb1a8184535e3`.
- Reader: frozen local Qwen3-8B Q4_K_M, temperature 0, seed 42, thinking disabled, 256 output-token cap.
- M10 profile: `m10-memory-bm25-v1`; ContextManager budget `4096 / 256 / 512 / 1024 / 2048 / 1024`, history window 24, estimator `chars4-cjk1-v1`.

The generation/retrieval decoder strips `answer`, `answer_session_ids`, `question_type`, and nested `has_answer`; the adapter then reads only `question_id`, `question`, `question_date`, `haystack_session_ids`, `haystack_dates`, and `haystack_sessions`. Gold labels are joined only after the complete prediction JSONL and its SHA256 sidecar have been frozen.

## Method Contract

Every source user or assistant turn becomes one `ADD` with `SESSION_NOTE`, `SESSION_DERIVED`, and `NON_SENSITIVE`. The value has exactly `role`, the exact source `session_date`, and the exact source `content`. The key is `raw_turn:<source_session_id>:<zero_based_turn_index>`. There is no extraction, proposition or entity logic, semantic key, revision inference, query rewriting, temporal mode classification, embedding, reranking, RRF, or memory-internal LLM call.

Timestamp parsing is numeric and locale-independent. It validates the date, time, and English weekday, then converts to UTC ISO-8601. The original date string remains in the record value. Each question uses an isolated scope `longmemeval:<question_id>` and SQLite state below `runs/memory/mem2/<run-id>/state/<question_id>/memory.sqlite`.

Memory IDs and source event IDs derive from dataset SHA256, question ID, source session ID, zero-based turn index, role, and content SHA256. Reopening a question's SQLite store and replaying the same snapshot verifies every deterministic record and adds no duplicate operation.

The only query is `MemoryQuery(scope_id, text=original_question, now=parsed_question_date, top_k=8)`. The untouched ContextManager receives that native top-8, the question as `current_user`, empty history, and the original question as `retrieval_query`. The ContextBundle contains only projected MEMORY items; the reader's separate question field is not duplicated as a context item. Reader-visible text is compact canonical JSON serialization of the existing M10 memory-item content. Provenance and retrieval diagnostics remain outside that text.

ContextBundle token accounting has no embedding-token value (`null`, not zero); `context_reader_tokens` is measured with the frozen llama.cpp Qwen3-8B tokenizer on the serialized memory context. The ContextManager estimate remains a separate value. Full reader prompts are independently rendered/tokenized by frozen llama.cpp `/apply-template` and `/tokenize`; each reserves 256 output tokens and must fit the 131072-token slot without truncation.

## Reproducibility Repair

Caller-provided `MemoryOperation.ADD.memory_id` is now honored; an omitted ID still follows M10's existing generated UUID behavior. An explicit duplicate ID is rejected without replacing stored state, including when separate SQLite connections race on the primary key. This is classified only as `REPRODUCIBILITY_API_FIDELITY_FIX`; UPDATE and DELETE semantics are unchanged.

## Synthetic Fidelity Gate

`MEM2A_M10_BASE_FIDELITY_READY=YES` after 31 focused tests passed, including the complete existing M10 and M10.1 test sets. Covered checks include strict locale-independent timestamp/weekday validation, exact raw-text preservation, gold and `has_answer` stripping, deterministic ADD identities and logical inventory, same-store SQLite replay, stable retrieval order and ContextPlan/ContextBundle identities, future-turn exclusion, native top-8, 1024 estimated-memory-token cap, question scope isolation, provenance/order alignment, and zero-call restore from frozen global prediction/call artifacts.

Verification command: `.venv/Scripts/python.exe -m pytest tests/test_mem2a_m10_base.py tests/test_m10.py tests/test_m10_add_memory_id.py tests/test_m101_context_projection.py -q` (31 passed). Ruff passed for the changed M10 materializer and MEM-2A adapter/tests. The only warning was an existing `jieba` / `pkg_resources` deprecation warning.

## Interpretation Boundary

M10 core already supports ADD, UPDATE, DELETE, NOOP, versions, SUPERSEDED, `supersedes_id`, `valid_from`, and `valid_until`. M10-Base deliberately emits unique-key ADD only. It therefore does not evaluate semantic revision resolution, stale-fact detection, or CURRENT/AS_OF/CHANGE reasoning. Any stale/update miss is not evidence that the unused M10 revision primitives failed.

The local reader is one `reader_answer` call per question. Memory-internal LLM calls, embeddings, judge calls, and hosted calls are all zero. Deterministic token metrics are lexical diagnostics only; no LLM-as-judge or leaderboard claim is used. Existing MEM-1D4 five-system predictions may be read for a descriptive six-system comparison only after M10-Base predictions are hash-frozen. Those baselines are never rerun.

No M10-Flat, RevMem, proposition extraction, revision materializer, temporal query routing, TEST, 102-case DEV, or RL is in scope.

## Frozen Ten-Case Diagnostic

`MEM2A_M10_BASE_FROZEN_10_DIAGNOSTIC=YES`.

- Run: `runs/memory/mem2/mem2a-m10-base-10-20260928`.
- Selection: the exact ten IDs above; all ten predictions succeeded under the frozen shared reader; TEST access remained false.
- Calls: 10 local loopback `reader_answer`; memory-internal LLM 0; embedding 0; judge 0; hosted 0.
- All 4,854 memory operations were ADD. The five required JSONL sidecars, prediction/call-ledger sidecars, metric/efficiency/comparison artifacts, and case-review artifacts verified.
- Reader prompts fit the 131072-token slot with 256 completion tokens reserved; every answer reported `truncated=false` and matched preflight prompt-token accounting.
- Diagnostic summary: mean deterministic token F1 `0.0595`; normalized EM `0.0000`; native answer-session Recall@5 `0.46`, Recall@8 `0.48`, MRR `0.45`; projected answer-session Recall@8 `0.24`.
- Mean serialized memory context was `1012.6` Qwen reader tokens; mean ContextManager memory estimate was `891.4` tokens. These are separate measures.
- The six-system comparison is descriptive only and uses hash-verified frozen MEM-1D4 v3 predictions for the five external systems; none were rerun.

See `runs/memory/mem2/mem2a-m10-base-10-20260928/report.md` and `docs/research/memory/mem_2a_m10_base_case_review.json`. The case-review packet leaves `outcome`, `failure_loci`, and `notes` null for human Reflection. No failure locus was automatically assigned.
