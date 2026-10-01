# MEM-3B0Q-R3 Span-Grounded Identity Pilot

Status: development control pilot only; strict gate R3; not benchmark evidence or a final mechanism freeze.

- Records: 19; grounded: 9; unresolved: 10.
- Control cases: 3/10 passed.
- Local Qwen calls: 1; hosted calls: 0; MemoryStore mutations: 0.

## Case outcomes

| Case | Expected | Passed | Gate reason | Slot keys |
|---|---|---:|---|---|
| instagram_follower_revision_candidate | SAME_SLOT | True | grounded_same_candidate_slot | ['longmemeval:1cea1afa', 'user', 'instagram_followers', 'follower_count']; ['longmemeval:1cea1afa', 'user', 'instagram_followers', 'follower_count'] |
| node_version_same_slot | SAME_SLOT | True | grounded_same_candidate_slot | ['longmemeval:06878be2', 'user', 'node_version', 'version']; ['longmemeval:06878be2', 'user', 'node_version', 'version'] |
| wallet_color_same_slot | SAME_SLOT | False | requires_grounded_equal_candidate_slots | UNRESOLVED; ['longmemeval:1cea1afa', 'user', 'wallet', 'color_preference'] |
| wallet_color_vs_material | NOT_SAME_SLOT | False | negative_control_contains_unresolved_identity | UNRESOLVED; ['longmemeval:1cea1afa', 'user', 'wallet', 'material_preference'] |
| wallet_neutral_vs_material | NOT_SAME_SLOT | True | distinct_grounded_candidate_slots | ['longmemeval:1cea1afa', 'user', 'wallet', 'color_preference']; ['longmemeval:1cea1afa', 'user', 'wallet', 'material_preference'] |
| gym_false_key_collision | NOT_SAME_SLOT | False | negative_control_contains_unresolved_identity | UNRESOLVED; UNRESOLVED |
| completed_purchase_event | NO_SLOT | False | completed_event_must_be_grounded_and_vetoed | UNRESOLVED; UNRESOLVED |
| distinct_trip_destinations | NOT_SAME_SLOT | False | negative_control_contains_unresolved_identity | UNRESOLVED; UNRESOLVED |
| generic_numeric_attribute | BOTH_UNRESOLVED | False | unresolved_not_limited_to_generic_attribute | UNRESOLVED; UNRESOLVED |
| macrame_interest_not_singleton_revision | NO_SLOT | False | negative_control_contains_unresolved_identity | ['longmemeval:a82c026e', 'user', 'macrame_kit', 'kit_status']; ['longmemeval:a82c026e', 'user', 'macrame_kit', 'kit_status']; ['longmemeval:a82c026e', 'user', 'macrame_journey', 'journey_status']; UNRESOLVED |

`MEM3B0Q_SPAN_IDENTITY_PILOT=NO`.
`MEM3B1_READY=NO`. This pilot never decides UPDATE/DELETE, current validity, or historical validity. No public LongMemEval or MedMemoryBench claim is made.
