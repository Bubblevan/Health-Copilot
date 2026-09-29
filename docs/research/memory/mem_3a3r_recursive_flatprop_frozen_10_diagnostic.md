# MEM-3A.3R - Recursive Overflow Recovery and FlatProp Completion

Completion gate: `MEM3A3R_RECURSIVE_FLATPROP_FROZEN_10_DIAGNOSTIC=YES`.

This is a frozen-ten mechanism diagnostic only. It is not a public benchmark result, performance ranking, or broad superiority claim. The historical MEM-3A.3 gate remains `MEM3A3_CHUNKED_FLATPROP_FROZEN_10_DIAGNOSTIC=NO` and its 785-root plan is unchanged.

## Writer Ledger Recovery

The first writer assembly stopped before downstream after detecting that the cache-session key had overwritten source-session attribution in attempt rows. The original partial attempt and lineage ledgers remain SHA-frozen and unchanged. All cached request/response artifacts and parent-child identities were revalidated, then source session IDs were restored from the frozen root manifest; no writer request was repeated during recovery.
- Replayed attempts/leaves: 791 / 788; writer calls during assembly recovery: 0; prior local writer calls retained from the ledger: 643.
- Repair audit SHA-256: `e5d8ebb6c89387e838504511d77111d9a92d54c268fe8848c39d0774d6ce49fd`; generation identity SHA-256: `8dab651697fd29ef67e44dbd41d3150af7a41f355f7fda4b83014d8323a4b849`.

## Writer Accounting

- Initial plan: 785 chunks; historical successful cache imports: 147; historical chunks re-executed after cache miss: 0.
- New initial-chunk calls: 637; overflow-parent calls this stage: 2 (all-time overflow parents: 3); recursive child calls: 6.
- Overflow roots: 3; maximum recursion depth: 1; generated child nodes: 6; irreducible overflows: 0.
- Local provider calls attributable to this stage: 643; retries: 0; hosted calls: 0.
- Sessions/terminal leaves: 477 / 788; source RawSpans: 52703; primary ownership exactly once: True.
- Completion tokens, all writer attempts: {'p50': 555.0, 'p90': 1066.0, 'p95': 1340.5, 'p99': 2396.6000000000004, 'max': 8192}; terminal leaves: {'p50': 552.5, 'p90': 1059.3000000000002, 'p95': 1323.3499999999995, 'p99': 2205.33, 'max': 7207}.
- The historical truncated parent response was verified as forensic-only (`semantic_output_used=false`); no partial parent proposition was aggregated.

## Exact Identity

- Raw generated propositions: 8285; final logical propositions: 8112.
- Exact duplicates removed: same leaf 146, across leaves 27, overlap-attributed 27.
- Near-duplicate pairs retained for diagnosis: 970; fuzzy/embedding merge: none.
- Logical identity is SHA-256 over stripped proposition text and canonical evidence refs. Only terminal successful leaves contribute.

## Frozen-Ten Diagnostic

- Dense top-8 rows: 80; shared-reader calls: 10/10; judge calls: 0.
- Mean answer-session Recall@5 / @8 / MRR: 0.9266666666666665 / 0.9466666666666667 / 0.95; projected Recall@8: 0.9466666666666667.
- Mean context reader tokens: 2099.7; token precision / recall / F1 / normalized EM: 0.14721306471306472 / 0.37105908584169456 / 0.20096768726256026 / 0.0.
- Comparison is MEM-2D RawSpan + Dense versus recursive chunked FlatProp + Dense. Per-question comparison is descriptive only.
- Frozen update cases `1cea1afa` and `c4ea545c` were inspected diagnostically; Instagram 500/600 state is not resolved or suppressed.

## Scope and Gate

- Materialization remains ADD / SESSION_NOTE / SESSION_DERIVED / ACTIVE / version 1; no revision semantics.
- Embedding is the frozen local Qwen3-Embedding-0.6B CUDA FP16; dense top-8; unchanged rank-aware projection SHA and 1,024-token budget; frozen shared local Qwen3-8B reader; no judge.
- No LongMemEval 102 DEV, TEST, MedMemoryBench, hosted provider, retry, M10-Flat, RevMem, or MEM-3B0 run.
- `historical_mem3a3_gate_preserved_no`: `True`
- `frozen_785_initial_chunk_plan_sha_valid`: `True`
- `historical_successes_imported_or_reexecuted`: `True`
- `writer_session_identity_repair_auditable`: `True`
- `all_785_initial_roots_attempted`: `True`
- `477_sessions_complete`: `True`
- `every_rawspan_has_one_terminal_primary_owner`: `True`
- `initial_and_recursive_nodes_auditable`: `True`
- `all_overflow_parents_discarded`: `True`
- `all_provider_requests_within_6144_and_non_length_statuses_valid`: `True`
- `zero_irreducible_overflows`: `True`
- `zero_unresolved_structural_failures`: `True`
- `zero_retries`: `True`
- `zero_hosted_calls`: `True`
- `flatprop_exact_identity_canonicalization_complete`: `True`
- `materialization_add_active_version1_no_revision`: `True`
- `frozen_embedding_local_cuda_fp16_no_truncation`: `True`
- `dense_top8_complete_for_ten`: `True`
- `projection_unchanged_and_budget_1024`: `True`
- `reader_10_of_10`: `True`
- `judge_zero`: `True`
- `prediction_and_reader_ledger_frozen_before_labels`: `True`
- `no_dev_test_or_medmemorybench`: `True`
- `all_required_artifact_sha_sidecars_valid`: `True`

`MEM3A3R_RECURSIVE_FLATPROP_FROZEN_10_DIAGNOSTIC=YES`

## Frozen Artifact Closeout Recovery

The completed reader predictions and call ledger were reused only after verifying all ten reader prompts, request hashes, ContextBundle hashes, model identity, and loopback runtime. The re-closeout made zero writer, embedding, or reader model calls; timing telemetry was excluded from the reuse decision.
