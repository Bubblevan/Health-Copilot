# Grounded Scalar Snapshot Pair Audit

This is a source-only DEV diagnostic, not a benchmark result or a revision-admission quality estimate. It decoded no question/gold fields and made no model or hosted calls.

## Positive Control

In frozen DEV case `1cea1afa`, the user reports reaching 500 Instagram followers at source position 35 (2023-05-27), then reports being at 600 at source position 217 (2023-05-28). The two frozen B0Q records share scope, subject, and the exact `instagram_followers` key; both propositions explicitly ground the key. The local pair verifier nevertheless returned `same_state_dimension=NO`, while also returning `state_cardinality=SINGLE_VALUE_AT_A_TIME` and `value_relation=DIFFERENT`. The pair was not admitted as a slot.

This is a source-anchored cross-turn scalar snapshot pair, unlike the five earlier `instead of` change mentions. The earlier 0/5 exact-predecessor finding applies only to those selected cue mentions; it does not imply that the DEV histories contain no independently anchored updates. The verifier failure is model judgment drift, not a missing timestamp or missing proposition identity.

## Hard Negatives

- The `workout_preference` key collides across a bodyweight-workout statement and a gaming-PC statement. The key is not grounded in the second proposition, so exact-key equality must not force same-slot admission.
- Two `trip_planning` propositions describe separate Japan and India trips. Both share the broad key and lexical terms, but no common trip instance is established; preserve them as separate events.

## What This Changes

A deterministic resolver should not replace pairwise LLM judgment with blind key equality. The next testable hypothesis is a narrow scalar-snapshot path: require a proposition-grounded metric type, source-backed distinct scalar values, and independently ordered observation times; retain event-instance qualification for plans and other multi-instance domains. The pair verifier may propose cardinality/value relation, but cannot veto a type-certified dimension match. This remains a hypothesis: one positive and two selected hard negatives do not validate the rule or support a performance claim.

`MEM3B1_SLOT_ADMISSION_READY=NO` remains unchanged. No 102-case DEV, TEST, MedMemoryBench, or performance ranking was run.
