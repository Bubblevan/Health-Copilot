# MEM-3B0Q Slot-Identity Failure Reflection

Status: read-only post-freeze counterfactual. This report does not alter B0Q, admit slots, or authorize B1.

## Frozen review evidence

- Reviewed slots: 99; labels: `{'false_revision_merge': 45, 'true_singleton_slot': 41, 'uncertain': 13}`.
- Slot shape by attribute-key cardinality: `{'false_revision_merge|multiple_attribute_keys': 45, 'true_singleton_slot|multiple_attribute_keys': 39, 'true_singleton_slot|same_attribute_key': 2, 'uncertain|multiple_attribute_keys': 13}`.
- Intra-slot pair source by label: `{'false_revision_merge|NO_EXACT_HINT': 60, 'true_singleton_slot|EXACT_HINT': 3, 'true_singleton_slot|NO_EXACT_HINT': 46, 'uncertain|EXACT_HINT': 10, 'uncertain|NO_EXACT_HINT': 17}`.
- Every false-merge slot spans multiple B0R attribute keys; most true slots also span multiple keys. Exact-key-only therefore trades away substantial recall and is not a sufficient method.

## Counterfactuals

- Exact-hint candidates: 256; existing B0Q-positive rule selects 125.
- A literal attribute-key token heuristic plus same subject/key and single-value cardinality selects 101 edges in components with size histogram `{'14': 1, '2': 4, '4': 1}`.
- Control outcomes under that diagnostic heuristic: `{'instagram': True, 'gym': False, 'completed_purchase': False}`.
- It admits the Instagram positive, blocks the gym and completed-purchase controls, but still joins 14 `reported_value` reports; a token-overlap gate does not establish a well-typed state slot.
- The heuristic is deliberately labelled non-causal and is not recommended for adoption: it also misses reviewed exact-key true/uncertain pairs when their attribute label is paraphrastic.

## Root cause and next question

B0Q tests whether two propositions describe one single-valued dimension, but admission needs a stronger relation: same subject, same object/target, same attribute, and actual update evidence. A semantically related pair or shared broad key is not enough. Repeated mentions may be duplicate support, concurrently valid facts, distinct objects, or revisions; temporal order alone cannot decide which.

Next development target: evidence-span-grounded subject/object/attribute/value proposals with deterministic validation and exact slot keys, followed by a separate update-evidence gate. If a slot cannot be grounded unambiguously, retain coexisting history and do not materialize a revision. This is a design hypothesis, not an evaluated result.

## Scope and guardrails

- Frozen B0Q inputs were only read and SHA-verified; no B0Q artifact was changed.
- No model, hosted API, MemoryStore mutation, DEV/TEST question, or MedMemoryBench run was used.
- `MEM3B1_READY=NO`; no materializer, CURRENT/AS_OF/CHANGE, or public scorecard is authorized by this diagnostic.
- Full machine-readable counts and input hashes are in `reflection_counterfactual.json`.
