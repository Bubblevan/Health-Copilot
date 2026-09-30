# MEM-3B0Q Factorized Revision Admission Protocol

Status: frozen candidate protocol for the MEM-3B0Q run. This is a Harness-owned admission experiment, not a benchmark run and not a MemoryStore migration.

## Research question

Can Harness-owned candidate generation, factorized pair semantics, deterministic graph closure, and maximal-clique admission identify conservative revision slots without conflating a mutable state dimension with the coexistence of historical observations?

MEM-3B0P remains historical and immutable with `MEM3B0P_PAIRWISE_REVISION_ADMISSION_COMPLETE=NO` and `MEM3B0P_MEM3B1_READY=NO`. B0Q does not rewrite FlatProp, B0R, or that outcome.

## Frozen inputs and source boundary

The only raw FlatProp reader is the hardened top-level allowlist scanner from `revision_pairwise_admission.jsonl_projection`. Before candidate construction, the runner writes and SHA-freezes `revision_semantic_source_projection.jsonl` containing exactly `memory_id`, `scope_id`, `proposition_text`, and `source_authority`. It skips forbidden values before decoding them. Once frozen, all B0Q candidate and verifier work reads this projection, never raw FlatProp.

B0R identity overlay SHA-256 is `49af354bb535909a572df580cf02483ab56022dcf31873266154caf0086b4f75`. Its `revision_kind`, `subject_key`, `attribute_key`, `value_text`, and `identity_origin` are diagnostic hints only. Harness derives eligibility using B0S authority policy and B0P deterministic grounding. Assistant-origin, fallback-identity, non-`SINGLETON_STATE`, or ungrounded hints cannot be admitted. `value_text` is never sent to the model or materialized as state.

## Candidate generation

Eligible records produce the union of all unordered exact-hint pairs sharing `(scope_id, subject_key, attribute_key)` and semantic-neighbor pairs. Semantic neighbors use the frozen local `Qwen/Qwen3-Embedding-0.6B` artifact over `proposition_text` only, top-4 within each same-scope eligible set, excluding self. Similarity-descending / memory-ID-ascending ordering breaks ties. There is no similarity threshold and no label-driven tuning. Embeddings propose candidates only; they do not admit slots.

Each pair stores its IDs and candidate-source labels (`EXACT_HINT`, `SEMANTIC_NEIGHBOR`, or both), but the local Qwen3-8B request contains exactly `proposition_a` and `proposition_b`. No timestamps, scope, IDs, B0R keys, value hints, authority, question, or gold are sent.

## Factorized pair semantics

Contract ID: `mem3b0q-factorized-pair-semantics-v1`. The model emits only:

- `same_state_dimension`: `YES | NO | UNKNOWN`.
- `state_cardinality`: `SINGLE_VALUE_AT_A_TIME | MULTI_VALUE_OR_SET | NOT_A_STATE | UNKNOWN`.
- `value_relation`: `SAME | DIFFERENT | UNKNOWN`.

The calibration example fixes 500 and 600 Instagram followers as `YES / SINGLE_VALUE_AT_A_TIME / DIFFERENT`, regardless of whether both dated observations can be true. Completed events, one-off requests, advice, and procedures use `NOT_A_STATE`, including duplicate descriptions of the same purchase. Multiple simultaneous preference values use `MULTI_VALUE_OR_SET`.

Only Harness creates a positive edge, exactly when dimension is `YES` and cardinality is `SINGLE_VALUE_AT_A_TIME`; value relation does not encode chronology. Malformed or invalid local content becomes an all-`UNKNOWN` `HARNESS_UNKNOWN_FALLBACK` terminal result. There are no same-request retries. Infrastructure, model-hash, or leakage failures are global fatal.

## Graph closure and slot admission

Eligible memory IDs are vertices and positive singleton edges form a deterministic preliminary graph. For each connected component of size 2 through 16, Harness requests each missing intra-component pair once to complete its matrix. Components larger than 16 are blocked as `MATERIALIZATION_BLOCKED_AMBIGUOUS_LARGE_COMPONENT`, with no quadratic expansion and no slot.

Harness computes deterministic maximal cliques from the complete positive graph. A memory appearing in multiple maximal cliques marks each such membership `AMBIGUOUS_OVERLAPPING_CLIQUE`; none of those overlapping cliques is admitted. Remaining disjoint cliques of size at least two become candidate slots. Opaque IDs are `SHA256(scope_id + sorted(member_memory_ids) + admission_contract_sha256)` and do not use B0R keys. Same-value observations establish membership only; B0Q does not label transitions.

## Runtime and non-goals

All pair semantics use local Qwen3-8B Q4_K_M through the pinned loopback llama.cpp runtime, temperature 0, seed 42, thinking disabled, and 64-token completion cap. Embeddings use the pinned local Qwen3-Embedding-0.6B CUDA adapter. No hosted API, judge, retrieval QA, LongMemEval, MedMemoryBench, TEST, 102 DEV, or state mutation is allowed. `ADD/UPDATE/DELETE/SUPERSEDED` remain zero; no CURRENT/AS_OF/CHANGE behavior is implemented.

## Timestamp freeze barrier

Before any timestamp field is opened, freeze with SHA sidecars: sanitized source projection, candidate-pair manifest, all seed verdicts, closure manifest and verdicts, positive graph, clique membership, and opaque revision-slot IDs. The B0Q semantic-freeze marker is written only after these artifacts verify. Only then may diagnostic source history be joined. Human-review packets contain no timestamps.

## Gates

`MEM3B0Q_FACTORIZED_REVISION_ADMISSION_COMPLETE=YES` requires source projection and forbidden-field sentinel test success, deterministic candidate generation, terminal seed/closure coverage, deterministic graph/clique construction, Harness-owned slot IDs, no pre-freeze timestamp access, zero hosted calls and retries, unchanged MemoryStore, DEV/TEST/MedMemoryBench unopened, and valid artifact hashes.

`MEM3B0Q_MEM3B1_READY=YES/NO` is a separate semantic gate. It requires the Instagram positive control to be admitted; the B0P completed-purchase false admission to be blocked as `NOT_A_STATE`; historical broad false-safe examples to remain blocked; and exhaustive frozen-ten review with no clear false revision merge. Gym candidate discovery is reported but may remain partial. Revision precision takes precedence over recall.

The first B0Q execution freezes protocol/code and source projection before embedding or LLM calls. No B0Q result authorizes MEM-3B1 unless the readiness gate passes. A structurally complete experiment may still be integrated with readiness `NO`, after semantic closeout is recorded.
