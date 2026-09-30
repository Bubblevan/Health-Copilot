# MEM-3B0R - Semantic-Quality Closeout

Status: structural overlay complete; do not proceed to MEM-3B1.

```text
MEM3B0R_REVISION_IDENTITY_OVERLAY_COMPLETE=YES
MEM3B0R_MEM3B1_READY=NO
MEM3B0R_NEXT_STEP=IDENTITY_SEMANTIC_CONTRACT_REVISION_REQUIRED
```

## Structural Evidence

The keyed-map and local-recovery implementation produced one terminal identity for each of the 8,112 frozen FlatProp propositions. There are 8,108 `MODEL_VALIDATED` records and four `HARNESS_UNKNOWN_FALLBACK` records (0.0493%). Two additional records are model-returned `UNKNOWN`, for six UNKNOWN records total. The four fallbacks are split evenly between `INVALID_ATTRIBUTE_KEY` and `INVALID_SUBJECT_KEY`.

The run used 254 initial batches and 448 unique local Qwen requests. Ninety-seven parent batches were subdivided, producing 194 subdivision calls; maximum recovery depth was five. There were zero same-request retries, hosted calls, embedding/retrieval/reader/judge calls, or MemoryStore writes. All 8,112 input IDs are unique in the terminal overlay; the 448 call-ledger rows match 448 lineage rows. The input inventory SHA and all required run sidecars verify. The historical MEM-3B0 `NO` run remains unchanged.

These results validate wire coverage and recovery behavior. They do not validate the semantic identity quality needed to materialize revisions.

## Semantic Findings

The Instagram diagnostic is positive: the 500- and 600-follower propositions both map to `SINGLETON_STATE`, subject `user`, attribute `instagram_followers`, with distinct values. This is a useful positive control for one explicit, same-attribute change.

The gym diagnostic is only `PARTIAL`. “Tuesday, Thursday, and Saturday” is emitted as `SET_STATE / workout_days`, while “four times a week” is emitted as `SINGLETON_STATE / gym_frequency_per_week`. The propositions remain understandable, but the identity layer does not align these two representations into one revision slot. This limits confidence for mixed-form revisions.

The candidate overlay contains 173 repeated singleton-state groups; 105 have multiple values and 116 span multiple source sessions. The singleton orphan rate is 84.02%. That orphan rate is not itself a measured error: many propositions may describe genuinely one-off facts, so it cannot establish fragmentation without labeled adjudication.

The deterministic lexical fragmentation diagnostic flagged 183 near-key pairs from 166,023 same-scope key-pair comparisons (about 0.11%). Examples include singular/plural variants such as `music_preference` / `music_preferences` and `interest` / `interests`. These are candidates for future review, not permission to merge keys automatically.

The content-Jaccard collision heuristic flagged 56 of 173 repeated singleton groups (about 32.4%). Manual inspection found materially broad slots, including unrelated propositions grouped under `user / software_preference` and unrelated assistant advice grouped under `assistant / recommendation`. The metric is a heuristic and can also flag legitimate diverse values, but these examples are strong enough to make collision risk the dominant materialization concern.

## Decision

Do not start MEM-3B1 and do not canonicalize or split this frozen overlay. Revise the identity semantic contract first, with particular attention to attribute specificity and distinguishing persistent user state from assistant-provided topics or recommendations. Any revised extractor/contract must be a new, versioned run; this overlay and its case findings remain immutable evidence.

The evidence favors semantic-contract revision over a standalone key-canonicalization pass: near-key flags are sparse relative to comparisons, while the inspected collisions expose over-broad slot meaning. This is a qualitative decision from the listed diagnostics, not a benchmark performance claim. No public benchmark score or causal claim was produced in MEM-3B0R.
