# MEM-3A.3R Session Identity Repair Addendum

## Incident

The first full recursive writer pass completed all 785 frozen initial roots, but closeout stopped before downstream because `execute_or_resume` returned its chunk-scoped cache key in the generic `session_identity_sha256` field. The writer response itself, prompt, model, extraction contract, evidence references, recursive split identities, and response cache keys were valid; the returned field shadowed the source-session identity when attempt rows were assembled.

The original `writer_attempt_ledger.partial.jsonl`, `overflow_lineage.partial.jsonl`, their SHA sidecars, the `run_failure.json` (`...=NO`), and the exact runner source used for generation are preserved unchanged. No labels were loaded and downstream had not started.

## Repair

Before downstream, the recovery path derives each attempt's source session from `root_initial_chunk_id` in the frozen initial chunk manifest. Recursive attempts must also carry the same source session in their hash-validated `chunk_identity`; the child chunk ID is recomputed and compared. The original cache key is retained in a separate field.

The recovery validates the root and child sets, parent-child lineage, all request hashes, all 791 local response/cache records, normalized packets against their visible evidence catalogs, and all three overflow parents as forensic-only. It reconstructs the exact writer method identity from the captured generation source hash and requires it to match the recovery cache namespace. Any mismatch fails closed.

This is a metadata attribution repair only. It does not change generation prompts, extraction semantics, chunk plan, split policy, reader behavior, embedding, or benchmark scoring. It issues zero writer calls; after successful replay validation, the already-approved local embedding and ten-question shared-reader diagnostic may proceed. No DEV, TEST, Mem3B, or public benchmark evaluation is started.
