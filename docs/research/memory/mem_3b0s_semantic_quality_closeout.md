# MEM-3B0S - Harness-Governed Revision Safety Overlay

`MEM3B0S_REVISION_SAFETY_OVERLAY_COMPLETE=YES`
`MEM3B0S_MEM3B1_READY=NO`

## Scope
The overlay treats MEM-3B0R identities as semantic proposals. Harness provenance determines authority, fixed gates determine eligibility, and exact `(scope_id, subject_key, attribute_key)` grouping precedes a narrow local semantic verifier. No revision is materialized and MemoryStore is untouched.

## Frozen Inputs and Calls
- FlatProp propositions: 8,112; identity calls in this stage: 0.
- Historical B0R repeated groups: 173; post-authority exact repeated groups: 73; eligible records: 1510; singleton groups skipped: 1300.
- Local verifier calls: 73; hosted calls: 0; same-request retries: 0.
- Verifier outcomes: coherent 25, incoherent 48, unknown 0; verifier failures 0, oversized 0.
- Authority gate: assistant-origin excluded 4946; user-asserted retained 1362; mixed-origin retained 148.
- MemoryStore operations: `{'ADD': 0, 'DELETE': 0, 'SUPERSEDED': 0, 'UPDATE': 0}`.
- Embedding, retrieval, reader-answer, judge, benchmark runs: 0, 0, 0, 0, 0.
- Runtime: local Qwen3-8B Q4_K_M at loopback; 99 GPU layers, Flash Attention, GPU Q4_0 KV cache.
- Model SHA-256: `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`; llama.cpp build: `10068 (571d0d540)`; binary SHA-256: `3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb`; context: 131,072.
- Prompt tokens/group: mean 437.342, max 1588; verifier latency p50 453.847 ms / p95 705.395 ms.

## Terminal Statuses
- `MATERIALIZATION_BLOCKED_SEMANTIC_COLLISION`: 48
- `MATERIALIZATION_SAFE`: 25

## Safety Diagnostics
- Instagram positive control: `NO`.
- Gym control: `PARTIAL`; no cross-key merge was performed.
- Random safe groups reviewed: 10; random semantic-collision blocks reviewed: 10.
- Detailed proposition-level examples and historical collision mapping are in `critical_case_review.json`.
- Clear false-safe examples in the reviewed sample: retro_game_night_interest, value_reported_on_5_05_2021, interest_in_documentary_series, q_and_a_session_preparation, current_planner_search.

## Post-freeze Qualitative Inspection
The seeded sample is diagnostic, not a population estimate or LLM-as-judge score.
- Five sampled safe groups contain unrelated-fact collisions: `retro_game_night_interest` (05ba91f36c0c): Netflix viewing history is unrelated to the retro-game-night interest.; `value_reported_on_5_05_2021` (e3265cc2420c): Router/device obstructions and a separately reported numeric value are not observations of one mutable singleton slot.; `interest_in_documentary_series` (4236d5511792): An Ethiopian coffee exploration fact is unrelated to documentary-series interest.; `q_and_a_session_preparation` (7d6336bb6d01): Following an artist and entering contests is not the same state as choosing the Q&A topic.; `current_planner_search` (ab0d821712f5): Checking ArtistWorks is unrelated to searching for a planner.
- Broad/multi-valued false-safe groups: aunt_and_uncle_relationship_support, work_experience.
- Instagram 500/600 positive control: both propositions share the exact intended slot but are blocked as `INCOHERENT_GROUP`; group `1350b5cafe9611d6ae88c4827d09a41cf42cbc92d54f0bb0ff2fd5cc5c400128`.
- Likely false-block example: `pest_control_method`; proposition meanings agree while the group was blocked.
- Historical flagged groups: assistant/recommendation removed by Harness authority; software_preference and user/current_interest were reviewed member-by-member in `critical_case_review.json`.
- Decision: keep MEM-3B1 readiness NO; revise only the narrow slot-coherence/identity contract before any materializer.

## Protocol Deviation
An exploratory display of one historical B0R candidate-group diagnostic exposed observed_at values before the safety artifacts were frozen. Timestamp fields/values were not projected to the candidate grouping result or verifier request and were not used by eligibility/grouping. Protocol deviation retained for review.
No timestamp was sent to the verifier or used by eligibility/grouping, but observed_at values were displayed to the operator before freeze. The process therefore has a logged freeze-order deviation; readiness remains NO and the report is not presented as an unqualified research closeout.

## Interpretation
This is an engineering safety overlay, not a new memory algorithm claim. `MATERIALIZATION_SAFE` means only that the exact group passed the frozen authority/kind gates and the local slot-coherence verifier; it does not establish which observation is current. A blocked group remains available as historical FlatProp and is not deleted.
