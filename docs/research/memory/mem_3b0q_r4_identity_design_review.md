# MEM-3B0Q-R4 Identity Admission Design Review

Status: `DRAFT FOR HUMAN REVIEW`; not a frozen protocol. No R4 model call or benchmark run is authorized by this document.

## Current gate

- MEM-3B0Q-R2 remains immutable and is marked invalid candidate; only its blocked runtime preflight ran.
- MEM-3B0Q-R3 is the authoritative identity-pilot result: `3/10` controls passed, `9/19` identities grounded, `10/19` unresolved. `MEM3B0Q_SPAN_IDENTITY_PILOT=NO`; `MEM3B1_READY=NO`.
- R3 source, response, proposals, labels, result, and failure taxonomy remain frozen. Do not adjust its prompt/gate and do not rerun its 19 records as a qualification set.
- The pre-registered scorecard remains unchanged: Markdown SHA-256 `5492dfa093351c66f591c59f20c8df9cdb1eef848ca254a444a1f7f2f21540c1`; JSON SHA-256 `457efc4d3bb58766fafa4d2486b34a881326eef1983687b9fbd113857ab36568`.

## What R3 established

This was a development control pilot, not benchmark evidence. It gives a useful failure boundary:

- Obvious same-slot cases worked for Instagram follower count and Node version; the wallet color/material distinction worked when both records grounded cleanly.
- Free-form `attribute_key` labels frequently failed the literal support contract (`wallet_type`, `exercise_preference`, `pc_capabilities`, `purchase_status`, `trip_destination`, `experience_level`). Broadening the alias list after seeing these outputs would be post-result tuning.
- `property_kind` was `SINGLE_VALUE_STATE` for all 19 records, including completed purchases and generic numeric values. It is not reliable enough to grant admission authority.
- All non-null identity witness fields used the full proposition, despite the “shortest exact substring” instruction. The validator accepted this because it checked uniqueness/containment but not field locality or length.
- Some source propositions were not atomic. The black leather wallet sentence combines color and material; the macrame control includes two beginner-kit arrival statements, a journey state, and a skill-level statement. Its frozen global `NO_SLOT` label is not a clean test of “repeated interests.”
- The generic numeric records were conservatively left without a slot, but R3's exact single-error rule was narrower than the semantic objective: unsupported object and attribute anchors occurred together.

These observations do not show that revision-aware memory improves stale-state behavior. They show that an LLM's free-form labels and type proposal are not yet a reliable admission boundary.

## Recommended R4 direction

Keep the research question narrow: can the Harness materialize auditable candidate state slots from atomic, source-grounded proposals without allowing model-generated labels to create revision identity?

1. **Atomize before slot comparison.** A proposition that contains more than one independently mutable attribute must yield multiple atom proposals or an explicit `COMPOSITE_UNRESOLVED`. Do not force color and material, or plan and associated item state, into one attribute/value pair.
2. **Separate raw anchors from canonical identity.** The proposer returns exact source anchors plus a candidate code from a small, versioned closed vocabulary. The Harness owns canonical `attribute_id`/`object_id` resolution using a pre-registered exact-lexeme and inflection map. Unknown or ambiguous anchors remain unresolved; do not add semantic aliases after observing R4 outputs.
3. **Make span locality enforceable.** Keep the raw source offsets/text and reject full-sentence witnesses when they exceed pre-frozen field-specific span bounds or include unrelated anchors. The Harness must not silently shorten or reinterpret model text.
4. **Remove type-proposal authority.** `SINGLE_VALUE_STATE`, `MULTI_VALUE_STATE`, and `EVENT` are not free-form admission decisions. Define explicit Harness-owned admission rules from the frozen atom type and source evidence; uncertain records receive no candidate slot. Same slot still does not imply a revision.
5. **Keep revision admission separate.** First prove stable candidate slot identity and conservative event/multi-value vetoes. Only a later B1 may materialize `ADD/UPDATE/DELETE/SUPERSEDED`; chronology and `CURRENT/AS_OF/CHANGE` remain outside R4.

Do not add a graph database, learned retriever, multi-agent workflow, or RL to solve this identity problem. The memory story is about state correctness over time, not the number of components.

## R4 control-set requirements

If approved, create a new run ID and controls before any model call:

- Select new proposition IDs not present in the R3 19-record input; record an overlap audit.
- Human-review the target atoms, canonical slot codes, event/multi-value labels, and expected abstentions before prompting the model. Keep those labels outside the proposer request.
- Use unambiguous atomic facts: same-slot positive pairs, same-object/different-attribute negatives, different-object/same-attribute negatives, explicit completed events, explicit multi-value facts, and genuinely unscoped values that must abstain.
- Exclude the ambiguous macrame set. Include composite facts only in a separate atomization control with a reviewed expected split.
- Freeze source/input IDs, vocabulary and alias-map hashes, prompt/schema, field span limits, validators, exact acceptance predicates, local runtime, and one-shot policy before inference.
- Require every positive control to produce grounded matching canonical slots; every negative control to be either grounded in distinct slots or explicitly vetoed by the frozen Harness rule; no unresolved identity may pass a semantic negative control; generic unscoped values may abstain without requiring one particular validation error.
- Any false candidate merge, wrong event admission, invalid source anchor, missing positive pair, or coverage/hash defect yields `NO` or `INFRA_FAILURE` as appropriate. No post-result gate relaxation.

R4 remains a small development qualification only. Passing it would still require a separate human review before B1; it would not be LongMemEval or MedMemoryBench evidence.

## Narrative guardrail

The defensible story is not “we added memory, agents, and training.” It is:

```text
retrieval finds relevant history
    -> flat propositions expose stale/current conflicts
    -> revision admission must distinguish same-slot state from coexistence
    -> R3 shows free-form identity/type proposals are not trustworthy enough
    -> Harness-owned atomization and canonical state identity are the next test
    -> only then evaluate logical forgetting and temporal state behavior
```

Do not write a resume claim that RevMem reduces stale-memory reuse until the frozen LongMemEval revision-sensitive scorecard and medical-transfer `state_update` endpoint support it. R3 is evidence of a research decision and a negative qualification result, not a performance win.

## Review decision requested

Review the proposed closed-vocabulary, Harness-owned identity boundary and the new-control requirements. Only after approval should a separately named R4 protocol and non-overlapping inputs be frozen. No R4 inference is included in this draft.
