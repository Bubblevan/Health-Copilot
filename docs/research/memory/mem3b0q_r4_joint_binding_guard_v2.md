# MEM-3B0Q R4 Joint Binding Guard v2

Status: `OFFLINE_PROPOSAL_REVIEW_APPROVED` for this offline proposal only.
No inference, protocol freeze, or B1 is authorized.

## Motivation

The post-hoc R4-P01 repeat returned the same unsupported owner phrase twice,
while changing the attribute boundary from `is` to `is running`. Against the
frozen proposition `My workout plan's activity is running.`, the expected typed
anchors are `My / workout plan / activity`. This is a field-boundary extraction
failure, not just an alias spelling issue. Candidate-builder v1 bounds each
field independently but still allows a model to combine fields from different
clauses.

This wrapper adds a conservative offline check around the unchanged
candidate-builder v1 and attribute/value locality guard. It changes neither
the frozen R4 v1 code/results nor its control pack.

## Method

- Model-facing typed fields are case-bound candidate IDs, not free-form spans.
  For P01, candidates expose only `My`, `workout plan`, and `activity` for
  their respective roles.
- For mutable-state slots, the selected object must be the nearest preceding
  object before the selected attribute. The owner must be the nearest
  preceding owner before that object, with no new owner intervening before the
  attribute.
- Event objects may follow the event attribute only where a frozen relation
  pattern explicitly links the typed pair. The sole current fixture pattern
  is `purchase of my bicycle`; proximity or coordination alone never links
  an event to an object.
- This event pair accepts only the frozen value phrase
  `completed the purchase`; an unrelated same-clause value such as `blue` is
  rejected. This exact phrase rule is fixture-scoped, not a general extractor.
- If the same canonical object type is mentioned more than once in one
  proposition, the guard rejects it as `object_instance_ambiguous`.
  Occurrence IDs alone do not prove physical-instance identity.
- The closed cardinality table validates the R4 fixture's known typed pairs.
  Unknown object/attribute pairs fail closed; the model's enum choice cannot
  independently authorize a state slot.
- For mutable-state slots, a value must follow the selected object and cannot
  cross an unselected owner/object/attribute anchor between the object,
  attribute, and value. A value span cannot swallow another typed anchor.
- The prior attribute/value clause-locality guard remains active.

The guard is a conservative Harness rule for a small, project-authored
synthetic vocabulary. It is not a general semantic parser, does not establish
cross-session object identity, does not decide whether a different value is an
update, and does not implement ADD/UPDATE/DELETE or temporal materialization.
It may reject valid but unrepresented combinations; that is preferable to
silently admitting an unknown slot in this qualification fixture.

Scope limitation: lexical typed-anchor checks cannot detect an intervening
unregistered noun or entity. This proposal is not open-vocabulary semantic
binding and must not be generalized beyond the closed R4 vocabulary. Any
separate inference-control pack must include unknown-object distractors and
value-binding negatives for every supported field type.

## Offline Controls

The targeted suite exercises all 20 frozen oracle-shaped controls plus
explicit counterexamples:

- R4-P01 exact alias candidate construction, so the free-form
  `My workout plan` owner output cannot be represented as an owner candidate.
- Cross-laptop owner/object/attribute/value mixing: local Linux and Windows
  tuples pass; combining `work laptop` with the second operating-system
  attribute and `Windows` is rejected.
- Same-clause value cross-binding is rejected in both directions for
  `color black with material leather`; value spans cannot cross or contain the
  other typed attribute.
- In a same-clause two-laptop proposition, the first laptop's operating
  system cannot take the second laptop's `Windows` value.
- Repeated `wallet` mentions fail closed instead of assuming one physical
  wallet from a shared type label.
- A `FAVORITE_FRUIT_SET / MEMBERSHIP` proposal labelled single-valued is rejected by the frozen policy.
- Event/object pairs separated by a sentence boundary or coordination are
  rejected: `I completed the purchase. My bicycle is blue.` and
  `I completed the purchase and my bicycle is red.` Frozen P14's
  `purchase of my bicycle` relation remains accepted.
- In `I completed the purchase of my bicycle, which is blue.`, the unrelated
  but source-grounded `blue` span cannot be used as the event value.
- An unregistered object/attribute pair is rejected rather than assigned a
  default cardinality.

These are deterministic unit controls, not model accuracy or benchmark
evidence. The oracle-shaped 20-control check uses the frozen fixture's expected
atoms to test resolver behavior only. A passing validator result is not
semantic admission or revision authority.

## Review and Verification

The independent reviewer approved the nearest-anchor rules, closed typed-pair
policy, repeated-object veto, and event relation/value controls for this
offline proposal. The reviewer also confirmed this does not authorize
inference or B1, and noted the open-vocabulary limitation above.

Verification completed without model calls:

- Joint-guard and related candidate/contract suite: `64 passed`.
- All `test_mem3b0q_r4*.py` files plus the factorized-revision-admission and
  pairwise-admission suites: `150 passed`.
- The full suite was rerun with an isolated writable workspace temp directory after the default Windows temp root returned permission errors.
- Ruff was unavailable in the current Python environment (`No module named ruff`); pytest import/execution validated syntax.

## Next Gate Before Inference

The next inference protocol must use a separately authored, non-overlapping
control pack and include unknown-object distractors and value-binding negatives
for every supported field type. This fixture's exact event value phrase does
not establish general value extraction. Freeze its prompt/schema, control pack,
oracle, exact runtime, offline tests, and acceptance predicates before
considering a bounded model request. The model must remain proposal-only, and
validation must not be described as semantic admission or revision authority.

Current results remain `R4_V1=NO`,
`MEM3B0Q_CANDIDATE_BOUNDED_EXTRACTOR_READY=NO`, and
`MEM3B0Q_MEM3B1_READY=NO`. No benchmark, memory mutation, or performance claim
is authorized by this proposal.
