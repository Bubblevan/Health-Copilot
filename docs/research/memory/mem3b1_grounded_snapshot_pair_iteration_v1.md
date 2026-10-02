# MEM-3B1 Grounded Scalar Snapshot Pair Iteration

## Purpose

This source-only diagnostic follows the B0Q failure where the factorized verifier rejected the Instagram follower-count pair. It tests whether that rejection is attributable to weak source evidence or to model judgment, while adding same-key hard negatives. It is not a benchmark result and does not change the frozen B0Q method or readiness gate.

## Frozen Evidence

The audit uses LongMemEval-S DEV histories `1cea1afa`, `a82c026e`, and `06878be2`, plus the hash-verified B0Q eligible-record, identity-hint, candidate-pair, and verifier artifacts. Only user-authored source turns are projected. Question, gold answer, answer-session IDs, and TEST rows are not decoded. No model or hosted call is made.

For DEV case `1cea1afa`, the user reports `500` Instagram followers at source position 35 on 2023-05-27 and `600` at source position 217 on 2023-05-28. The frozen memories have the same scope, subject (`user`), and exact `instagram_followers` attribute key. Both propositions explicitly contain the metric terms and a scalar. B0Q's verifier nevertheless returned:

```text
same_state_dimension = NO
state_cardinality = SINGLE_VALUE_AT_A_TIME
value_relation = DIFFERENT
```

The pair was a discovered candidate but was not admitted to one revision slot. This is a clean source-anchored cross-turn snapshot pair, and the failure is in the pair verifier's dimension judgment, not missing chronology or missing proposition identity.

This finding narrows the earlier `0/5` exact-predecessor result: that result applied only to five hand-selected same-turn `instead of` mentions. It was never an exhaustive scan of state updates in DEV. The 500-to-600 pair demonstrates why contrast-cue mining alone misses valid cross-turn changes.

## Hard Negatives

The audit also checks two same-key candidate traps:

- `workout_preference` joins a bodyweight-workout proposition and a gaming-PC proposition. The key is not grounded in the second proposition; the candidate must not override the verifier.
- `trip_planning` joins plans for Japan and India. The key terms are grounded in both propositions, but the source describes distinct trip instances; they should coexist rather than supersede one another.

Across all 256 frozen B0Q exact-hint pairs, 127 received `same_state_dimension=NO`. A literal-token audit found all key components in both propositions for 94 pairs; among those, eight were still `NO`: one `instagram_followers`, six generic `reported_value`, and one `trip_planning`. These are disagreement counts, not labels or accuracy estimates. The cases show both sides of the design constraint: model review can reject a clear typed metric match, while exact key equality and even literal support do not prove a revision.

## Next Method Hypothesis

Do not replace the verifier with unconditional key equality. Prospectively test a narrow scalar-snapshot path that requires:

- exact scope, subject, and typed metric key;
- metric type and key terms independently grounded in both source-backed propositions;
- one distinct scalar observation per source record;
- unique source-turn provenance and strictly ordered observation times.

For event-like or multi-instance attributes such as travel plans, require an object/event qualifier or retain coexistence. The semantic verifier may still propose cardinality and value relation, but should not veto a same-dimension match when deterministic typed evidence establishes it. This is a hypothesis only; the current audit contains one clear positive and two selected hard negatives and does not validate a general resolver.

## Reproduction

- Runner: `tools/research/memory/audit_mem3b1_grounded_snapshot_pairs_v1.py`
- Frozen diagnostic output: `runs/memory/mem3/mem3b1-grounded-snapshot-pairs-audit-v1/`
- `snapshot_pair_audit.json` SHA256: `edbd04029185c694d3aee1504b3dea65ed987404b74bb3ea729f5ff645266362`.
- Source and B0Q input hashes are also recorded in `snapshot_pair_audit.json`.
- Runtime: zero model calls, zero hosted calls, no question/gold decoding.
- Gate: `MEM3B1_SLOT_ADMISSION_READY=NO` (unchanged).
- Not run: 102-case DEV, TEST, MedMemoryBench, M10-Flat, or RevMem.
