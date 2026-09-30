# MEM-3B0R - Keyed Identity Map Review

Completion gate: `MEM3B0R_REVISION_IDENTITY_OVERLAY_COMPLETE=YES`.

This stage changes only the identity wire format and batch recovery. It does not change the v1 semantic prompt or decide currentness, supersession, or MemoryStore operations.

## Coverage and Recovery

- Terminal identities: 8112 / 8,112; model-validated: 8108; conservative UNKNOWN fallbacks: 4 (0.0493%).
- Model-returned UNKNOWN identities: 2 (0.0247% of model-validated identities).
- Fallback reasons: `{"INVALID_ATTRIBUTE_KEY": 2, "INVALID_SUBJECT_KEY": 2}`; UNKNOWN among all terminal records: 0.0740%.
- Initial batches: 254; unique local model requests: 448; successful initial batches: 205.
- Recovered parent batches: 97; subdivision calls: 194; maximum recovery depth: 5.
- Retries: 0; hosted calls: 0; elapsed seconds: 22175.976.
- Embedding, retrieval, reader-answer, and judge calls: 0 each.
- No LongMemEval 102 DEV, TEST, or MedMemoryBench run was performed.

## Identity Statistics

- Revision kinds: `{"EVENT": 704, "NON_REVISIONAL": 4014, "SET_STATE": 335, "SINGLETON_STATE": 3053, "UNKNOWN": 6}`.
- Unique subject / attribute keys: 1449 / 6543.
- Candidate singleton-state revision groups: 173; groups with multiple values: 105; cross-session groups: 116.
- Singleton orphan rate: 0.8402; fallback fraction: 0.0005.

## Critical Cases

`INSTAGRAM_REVISION_SLOT_IDENTIFIED=YES`

`GYM_REVISION_SLOT_IDENTIFIED=PARTIAL`

Case history and timestamps were joined only after identity records, call ledger, recovery lineage, and core identity manifest were frozen. No gold answer or reader output was loaded.

## Interpretation

Structural completion and semantic quality are separate. UNKNOWN fallbacks cannot enter candidate singleton groups. Fragmentation/collision flags are deterministic lexical heuristics only; no keys were merged, no groups split, and no MemoryStore side effects occurred.

Human Reflection decides whether slot quality supports MEM-3B1, whether fragmentation warrants MEM-3B0.1, or whether collision warrants revising the semantic identity layer. No benchmark performance metric was computed.

`MEM3B0R_REVISION_IDENTITY_OVERLAY_COMPLETE=YES`
