# MEM-3B0P Pairwise Revision Admission Closeout

This is a safety-admission experiment, not a QA benchmark or performance comparison. A protocol deviation means the structural completion gate does not pass.

## Frozen Inputs

- Base commit: `96fb49a05febfb65438d5c63ff115745ad8ddb79`.
- FlatProp rows/hash: 8112 / `250a24a10b532cb01bd841b965511df019fc0e597ec1e85ee12d7830c166c5fe`.
- B0R identity rows/hash: 8112 / `49af354bb535909a572df580cf02483ab56022dcf31873266154caf0086b4f75`; hints only.
- Scorecard contract commit: `b2938211fe2c6ebea10f1f771132b62d82a2bbc1`; committed before all B0P model requests.

## Grounding And Admission

- Grounding outcomes: `{"IDENTITY_HINT_BLOCKED_VALUE_NUMERIC_MISMATCH": 17, "IDENTITY_HINT_BLOCKED_VALUE_UNGROUNDED": 127, "NOT_EVALUATED": 6602, "VALUE_LOCALLY_GROUNDED": 1366}`.
- Repeated candidate groups: 48; unordered pair requests: 256.
- Pair verdicts: `{"COEXISTING_FACTS": 167, "SAME_MUTABLE_SLOT": 59, "UNKNOWN": 10, "UNRELATED": 20}`.
- Admitted / blocked / no-history groups: 7 / 41 / 1222.
- Opaque Harness revision slots: 7.
- Group admission required every unordered pair to be `SAME_MUTABLE_SLOT`; no transitive clustering was used.

## Critical Controls

- Instagram 500/600 positive control: `NO`.
- Historical B0S false-safe examples all blocked/not admitted: `True`.
- Gym cross-key recall diagnostic: `PARTIAL`; no cross-key merge was introduced.
- Exhaustively human-reviewed admitted groups: {'true_singleton_slot': 5, 'false_revision_merge': 1, 'uncertain': 1}.
- Human-review decision input SHA-256: `85266dd7d3c043303ba6da5a024cb8a0734044b776172e116627c75fa7b043ca`.

## Runtime And Safety

- Runtime model/server: `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785` / `10068 (571d0d540)`.
- Pairwise local provider calls: 256; hosted calls: 0; same-request retries: 0.
- Prompt tokens total / p50: 44660 / 177.0; latency p50 / p95 ms: 301.575 / 343.355.
- Timestamp-boundary deviation: the pre-freeze JSON decoder transiently decoded non-allowlisted values into its object-pairs hook before filtering. The count is unquantified; no timestamp value was retained in projected rows or used for candidates/prompts. This violates the literal freeze boundary.
- Post-freeze streaming projection skips non-allowlisted values before decoding and reproduces the frozen candidate/pair artifacts byte-for-byte: `True`. Frozen pairwise requests were not repeated.
- MemoryStore mutations `ADD/UPDATE/DELETE/SUPERSEDED`: `0/0/0/0`; embeddings, retrieval, answer-reader, judge, and benchmark calls: `0`.

## Outcome

`MEM3B0P_PAIRWISE_REVISION_ADMISSION_COMPLETE=NO` (timestamp-boundary protocol deviation).
`MEM3B0P_MEM3B1_READY=NO`.

No LongMemEval or MedMemoryBench performance claim is made from this stage.
