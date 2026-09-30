# MEM-3B0S - Harness-Governed Revision Safety Overlay

## Research Question

Can a harness-owned authority and slot-coherence overlay turn the frozen MEM-3B0R semantic identity proposals into a conservative set of revision groups without allowing model output to control memory state transitions?

This is an engineering/research adaptation. It does not claim a new memory algorithm and does not perform materialization.

## Frozen Inputs

- FlatProp inventory: `runs/memory/mem3/mem3a3r-recursive-flatprop-frozen-10-20260929/flatprop_inventory.jsonl`, 8,112 rows, SHA-256 `250a24a10b532cb01bd841b965511df019fc0e597ec1e85ee12d7830c166c5fe`.
- MEM-3B0R identity proposals: `runs/memory/mem3/mem3b0r-revision-identity-keyed-map-20260930/revision_identity_records.jsonl`, SHA-256 `49af354bb535909a572df580cf02483ab56022dcf31873266154caf0086b4f75`.
- The B0R identity overlay is `SEMANTIC_IDENTITY_PROPOSAL`, not an authoritative state schema. Identity keys and values are never rewritten.

The runner verifies both file hashes and sidecars, a one-to-one 8,112-ID join, and no duplicate IDs before producing a candidate artifact.

## Harness Authority and Grouping

Authority comes only from FlatProp `source_authority`: `user -> USER_ASSERTED`, `mixed -> USER_MIXED`, `assistant -> ASSISTANT_ORIGIN`; any other or absent value maps to `HARNESS_UNKNOWN_FALLBACK`. Only user and mixed namespaces may be eligible. Assistant-origin memories remain historical/retrievable data but cannot participate in personal user-state supersession.

Only `SINGLETON_STATE` can enter mutable-slot grouping. Fallback identities, unknown kinds, other kinds, assistant-owned subjects (`assistant`, `assistant:`, `assistant/`, `assistant_`, `assistant.`), and unknown authority are ineligible. There is no key canonicalization, fuzzy merge, embedding clustering, or special-case exception.

Eligible records group by exact `(scope_id, subject_key, attribute_key)`. Groups of one are frozen as `SINGLETON_WITHOUT_REVISION_HISTORY` and never sent to the verifier. Only groups of two or more enter the semantic check.

## Freeze and Time Boundary

Candidate groups, the eligibility contract, and verifier contract are frozen with SHA-256 sidecars before verifier inference. Eligibility and grouping never read `observed_at` or any timestamp. Verifier requests contain only `subject_key`, `attribute_key`, and propositions with `memory_id`, `proposition_text`, `value_text`, and `source_authority`; timestamp/session/question/gold/reader/rank fields are forbidden.

The verifier's narrow task is to classify whether every proposition describes the same mutable, single-valued attribute of the same real-world subject. It cannot create keys or values, order observations, pick current state, or choose a transition.

## Local Verifier

- Local Qwen3-8B Q4_K_M, model SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
- Frozen llama.cpp 10068 / `571d0d540`, pinned binary SHA `3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb`, loopback only at `127.0.0.1:8081`.
- 131,072 context, 99 GPU layers, Flash Attention, Q4_0 GPU KV cache; temperature 0, seed 42, thinking disabled, max completion 64.
- No hosted fallback, embedding, retrieval, reader-answer, judge, benchmark run, or OpenAI credential.
- One request per repeated group at most. A durable `PENDING` journal row is written before each request. An interrupted request is conservatively blocked and never resent. No same-request retry.

Model verdicts are limited to `COHERENT_SINGLETON_SLOT`, `INCOHERENT_GROUP`, or `UNKNOWN`. Harness maps these to `MATERIALIZATION_SAFE`, `MATERIALIZATION_BLOCKED_SEMANTIC_COLLISION`, or `MATERIALIZATION_BLOCKED_UNKNOWN`. Malformed/illegal output is `MATERIALIZATION_BLOCKED_VERIFIER_FAILURE`; context overflow is `MATERIALIZATION_BLOCKED_OVERSIZED_GROUP`.

## No Side Effects

This stage writes only research artifacts. MemoryStore operations are fixed at `ADD=0`, `UPDATE=0`, `DELETE=0`, `SUPERSEDED=0`. Historical FlatProp remains available regardless of safety status. No benchmark labels or QA outputs are used.

## Closeout

The stage completion gate is structural and separately reported from `MEM3B0S_MEM3B1_READY`. Human semantic review covers the Instagram positive control, gym partial-recall control, historical software-preference/current-interest/assistant-recommendation collisions, and deterministic samples of safe and semantic-collision-blocked groups. A safe label is not a temporal/current-state decision. If any protocol deviation affects the freeze boundary, it must be explicitly recorded and readiness remains NO.
