# MEM-3B0Q R4 Transition Reducer v1

Status: offline synthetic-control prototype only. This is not candidate-extractor qualification, B1 admission, or a MemoryStore materializer.

## Why This Exists

The frozen B0Q human review labels 45 of 99 admitted groups as false revision merges, 41 as true singleton slots, and 13 as uncertain. All 45 false merges span multiple B0R attribute keys, but so do 39 of 41 true slots; exact-key-only admission would therefore discard substantial recall. The read-only reflection also found that a literal attribute-key token heuristic still joins 14 `reported_value` reports.

The R4 candidate builder improves source-span provenance, but it does not yet prove that independently selected fields form one fact. For example, its validator accepts source-backed `color` plus the unique source value `leather` in R4-P15. Alias canonical IDs also describe object types, not necessarily individual object instances, and the proposal's cardinality remains model-supplied. These are upstream admission gaps; this reducer does not repair them.

## Reducer Contract

`reduce_transition(previous, current, witness)` is a pure, pairwise classifier. It does not mutate state or infer entity links.

- Missing owner, object-instance, attribute, source, or value grounding fails closed as `UNRESOLVED`.
- A value text that is absent or repeated in its source fails closed; occurrence-level value identity is not inferred.
- Slot identity uses explicit scope, owner entity, object entity instance, and attribute IDs. An alias/type match alone is not object identity.
- Different resolved owner/object/attribute IDs remain `DISTINCT_SLOT`.
- Same resolved slot and same normalized value is `DUPLICATE_SUPPORT_NOOP`.
- Concurrent multi-value states remain `COEXISTING_VALUES`.
- Different single-valued states remain `UNRESOLVED` unless the full source statement is a narrowly supported first-person replacement form, contains an exact ISO event date, binds both values, and that event date is after the prior observation but no later than the current observation.
- A deletion produces only `TOMBSTONE_CANDIDATE` when the full source statement matches a supported first-person deletion form, carries an exact ISO event date, has no successor value, and event time is ordered between the prior and current observations.
- Every result has `store_mutation=NONE`; candidate outputs are not materialized.

The narrow English witness patterns are deliberate synthetic controls, not a general update-language parser. They trade recall for inspectable behavior and reject quoted, attributed, or multi-clause source statements rather than trying to interpret them. Date precision is represented at UTC midnight; same-day ordering that cannot be proven remains unresolved. The reducer trusts its caller's entity IDs and cardinality classifications. It does not establish that those inputs were correctly extracted.

## Offline Controls

The fixture in `mem3b0q_r4_reducer_controls_v1.json` covers 16 synthetic cases: duplicate support, missing object-instance resolution, distinct objects/owners/attributes, concurrent values, unresolved value changes, explicit dated replacement, a negated replacement, quoted attribution, backdated/non-monotonic time, explicit dated deletion, invalid delete payload, unknown cardinality, and repeated-value ambiguity. Unit tests assert every result is non-mutating.

This only verifies reducer behavior on hand-authored controls. It is not a measured model result, does not validate the candidate builder's semantic bindings, and is not evidence of public benchmark improvement.

## Remaining Gate

Before B1, the candidate path still needs independent semantic controls for cross-field binding, object-instance identity, and cardinality, plus separately reviewed local runtime witnesses and a frozen bounded-call manifest. The existing candidate validator's R4-P15 color/value false-accept remains open. No model request, MemoryStore mutation, LongMemEval DEV/TEST access, or MedMemoryBench run is authorized by this prototype.

`MEM3B0Q_R4_TRANSITION_REDUCER=OFFLINE_CONTROL_PASS`
`MEM3B0Q_MEM3B1_READY=NO`
`MEMORY_PUBLIC_CLOSEOUT=NO`
