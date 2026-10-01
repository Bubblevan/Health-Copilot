# MEM-3B0Q-R3 Span-Grounded Slot Identity Pilot

Status: corrected pre-inference protocol freeze for a small, exploratory local-model qualification on the already-reviewed LongMemEval frozen-10 diagnostic material. This is not a public benchmark run, a hidden test, or a final method freeze.

## R2 audit disposition

The earlier R2 protocol remains unchanged as an audit record but is an `INVALID_CANDIDATE / NOT USED FOR INFERENCE OR ADMISSION`. Static review found that unresolved identities could satisfy its negative `NOT_SAME_SLOT` and `NO_SLOT` predicates. R2 performed only a blocked runtime preflight; it generated no model output and made no admission decision. See `mem_3b0q_span_identity_gate_audit.md`.

## Research question

Can source-validated principal/object/attribute/value spans produce safer candidate slot identities than free-form identity keys and proposition-pair similarity, while preserving explicit abstention when a proposition does not identify its state dimension?

This stage addresses identity only. It does not decide whether a fact is current, obsolete, updated, or deleted. It does not implement `ADD`, `UPDATE`, `DELETE`, `SUPERSEDED`, `CURRENT`, `AS_OF`, or `CHANGE`.

## Frozen source and inputs

- Source artifact: `runs/memory/mem3/mem3b0q-factorized-admission-20260930/eligible_records.jsonl`.
- Required source SHA-256: `1ed014923a7213106e4da384929c6d303ee60a4a607e8b2942231dc03718da7c`.
- Case selection lineage: post-freeze B0Q control audit SHA-256 `0f2bb3c86f4666b12032395f7fbfbb96d06fb1e2c50d54f253de69dbea694a64` and user-authorized review decisions SHA-256 `766f602146776e114e6696fa3e404a7954a3ceb3e72c0f3516e51288a8bd5564`.
- The 19 proposition texts are unchanged from R2. Expected labels remain in a separate frozen case file and never enter the model request.
- This is a development challenge set assembled after observing B0Q failures. It is not independent evidence.
- No timestamp, source-session date, question, answer, B0Q human decision, candidate source, B0R key/value, or benchmark label enters a proposer request.

## Local-only model protocol

- Proposer: frozen local Qwen3-8B Q4_K_M, SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
- Runtime: pinned loopback llama.cpp `10068 (571d0d540)`, executable SHA-256 `3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb`.
- Endpoint: `127.0.0.1:8081`; no hosted provider, API key, embedding, judge, reader-answer call, or network fallback.
- Runtime arguments: context `131072`; YaRN scale `4`, original context `32768`; `99` GPU layers; Flash Attention; Q4_0 GPU K/V cache; parallelism `1`.
- Temperature `0`, seed `42`, thinking disabled. Dynamic JSON schema; response budget `8192` tokens for one batch. Exactly one inference request: no retry, batch subdivision, or fallback.
- The R3 prompt and schema are unchanged from R2. Only run identity, code identity, and the deterministic post-generation gate differ.

## Proposal and validation contract

For every proposition the model returns the R2 `span_grounded_memory_identity_v1` schema: normalized principal/object/attribute/value proposals, exact source witness spans, `property_kind`, and optional explicit `change_cue_span`. The model does not compare memories, group records, choose chronology, or decide whether an update should apply.

Every non-null witness must occur exactly once in that proposition. Harness checks normalized key syntax, token support, value grounding, source scope, and exact witness spans. It never repairs or guesses. A candidate key is the exact tuple:

```text
scope_id + principal_key + object_key-or-null + attribute_key
```

Only grounded `SINGLE_VALUE_STATE` records become slot candidates. Equal candidate keys mean only “same candidate slot”; they never prove a revision. The model's classification of property kind is a frozen proposal subject to the controls below, not authoritative state.

## Corrected deterministic gate

Every input record must appear exactly once in proposals, and every one of the 19 input IDs must be covered by the frozen diagnostic cases. There are ten frozen cases.

- `SAME_SLOT`: all records must be `GROUNDED`, all must be candidate slots, and all exact tuple keys must match.
- `NOT_SAME_SLOT`: every record must be `GROUNDED`; pass only when all are candidate slots with distinct exact keys, or all are explicitly vetoed as `MULTI_VALUE_STATE`, `EVENT`, or `NON_STATE`. `UNKNOWN` and unresolved identities do not pass.
- `completed_purchase_event`: every record must be `GROUNDED`, classified `EVENT`, and vetoed from candidate slots.
- `macrame_interest_not_singleton_revision`: every record must be `GROUNDED`; no exact candidate key may recur across two records. Individual distinct single-valued states may coexist.
- `generic_numeric_attribute`: both records must be unresolved solely because their normalized attribute key contains only frozen generic attribute tokens. The exact validation reason must be `attribute_key:UNSUPPORTED_KEY_TOKENS:`; missing/invalid witnesses or any additional failure do not pass.

No positive or negative semantic control may pass by abstention. Any failed control yields `MEM3B0Q_SPAN_IDENTITY_PILOT=NO`; infrastructure failure before generation is recorded as `NOT_RUN`, not as a method result. The gate implementation identity is frozen alongside the runner, span validator, and runtime verifier.

## Controls and interpretation

The cases cover Instagram follower count 500/600, repeated Node version, wallet color versus material, workout versus gaming PC, completed purchases, Japan versus India trip plans, generic numeric `reported_value`, and repeated macrame statements. These check identity/control behavior only. Even a successful result is a qualification signal on selected development examples, not a generalization result.

## Scope and stop conditions

- No MemoryStore mutation, retrieval, answer generation, benchmark scoring, LongMemEval 102 DEV/TEST access, MedMemoryBench, or medical evidence/policy interaction.
- Freeze inputs, R3 protocol, prompt, schema, validators, gate, code identity, runtime, and response journal before generation.
- Freeze model proposals before joining diagnostic labels.
- Report every case outcome, unresolved record, local call/token/latency details, and failure reason.
- Stop after this pilot. Do not start B1, the public scorecard, or medical transfer. `MEM3B1_READY` remains `NO` pending a separate human review and a deterministic revision materializer design.
