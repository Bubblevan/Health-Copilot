# MEM-3B0Q-R4 Identity Admission Design Review v2

Status: `DRAFT FOR INDEPENDENT REVIEW`; design only, not a frozen protocol. No R4 inference, benchmark run, or B1 implementation is authorized by this document.

## Decision context

This revision preserves the earlier R4 design draft and incorporates the user-authorized independent review of the frozen B0Q material. It does not change B0P/B0Q, their scorecard, or their results.

Frozen evidence:

- B0Q reviewed 99 admitted slots: 41 `true_singleton_slot`, 45 `false_revision_merge`, and 13 `uncertain`.
- Every false-merge slot spans multiple B0R attribute keys; 39 of 41 true slots do too. Attribute-key equality is neither a safe identity test nor a sufficient recall strategy.
- The read-only literal-token counterfactual selected 101 edges, including a 14-record generic `reported_value` component and two separate trip plans. It missed paraphrastic positives. It is diagnostic only, not a candidate method.
- The Instagram 500/600 pair illustrates two distinct questions: different values can share a canonical slot, but shared slot identity alone does not establish an update.
- The independent reviewer decision was `REVISE BEFORE APPROVAL AS A PROTOCOL CANDIDATE`.

Evidence identities:

| Artifact | SHA-256 |
|---|---|
| `human_review_decisions_user.json` | `766f602146776e114e6696fa3e404a7954a3ceb3e72c0f3516e51288a8bd5564` |
| `admitted_slot_human_review_packet.json` | `bb35d9467dcc86bb1202bffc21804296298a27729ecf9587501319597ad2687e` |
| `reflection_counterfactual.json` | `18515382b1bd6724a9ea96da2e051798e6d6a0fba2fd7bc26241bd38059274a1` |

## R4 question

Can a frozen, locally generated set of atom proposals be converted by deterministic Harness rules into stable state-slot identities without collapsing distinct owners, objects, attributes, events, or concurrent values? R4 qualifies proposal and identity semantics only. It does not decide that one fact supersedes another.

The reviewed layers are:

```text
raw proposition
  -> source-grounded atom(s) with exact spans
  -> owner / object / attribute identity
  -> state cardinality or event semantics
  -> relation: duplicate, coexisting value, revision candidate, or unresolved
```

No layer may borrow authority from a later one. In particular, matching identity does not imply a revision, and a different value does not define a different identity.

## Frozen candidate contract requirements

### Atom coverage and provenance

- Retain the exact raw source string and stable source ID. Span offsets are half-open Unicode code-point offsets into that unchanged string; the validator reconstructs each span and requires byte-for-byte-equivalent text after UTF-8 encoding.
- Proposals identify owner/subject, object/target when present, attribute, value, and atom span separately. No free-form canonical key or `property_kind` may directly admit a slot.
- The control oracle records the independently mutable attributes expected from each input. Every expected atom must appear exactly once; no missing or extra atom is allowed. A genuinely inseparable proposition must use `COMPOSITE_UNRESOLVED`, not silently collapse multiple state dimensions.
- Field-specific span-length bounds, vocabulary, aliases, normalization, and type rules must be frozen and hashed before inference. The Harness does not silently trim, paraphrase, infer missing anchors, or expand aliases after seeing model output.
- Missing, broad, cross-clause, or ambiguous evidence spans remain unresolved. Whole-proposition witnesses are rejected when they exceed the frozen field bound or contain unrelated fields.

### Identity and semantic type

- Canonical slot identity is the tuple `(scope_id, owner_id, object_id, attribute_id)`. `owner_id` distinguishes the user from another person, including a family member. An explicit `SELF` object is allowed only when the source unambiguously describes an owner-level attribute; it cannot stand in for an omitted object in an object-specific claim.
- The value is evidence attached to the slot, never part of the slot key. Canonical IDs come only from a small, versioned, Harness-owned vocabulary and pre-frozen exact-lexeme/inflection map. Unknown or multiply mapped anchors abstain; no post-output semantic alias is added.
- Cardinality is a separate closed-vocabulary judgment: `SINGLE_VALUE_AT_A_TIME`, `MULTI_VALUE_CONCURRENT`, `EVENT_OR_NOT_STATE`, or `UNKNOWN`. Model output is proposal-only. A Harness rule or frozen control oracle must validate it; the enum alone cannot authorize mutation.
- `MULTI_VALUE_CONCURRENT` means values may coexist under the same canonical slot. It is not a different slot and is never an implicit revision. Generic or unscoped numeric reports are not identities merely because their values or lexical templates match.

### Relation and revision boundary

- Same slot + same value may be duplicate/corroborating evidence and must not create a new revision.
- Same slot + different value establishes only a slot match. Without a separately qualified update-evidence rule it remains `RELATION_UNRESOLVED`; chronology, semantic similarity, or value difference alone cannot emit `UPDATE`/`SUPERSEDED`.
- Same slot with concurrent values is `COEXISTING_VALUE`, with no revision edge.
- Completed purchases, completed courses, advice requests, procedures, and one-off events are `EVENT_OR_NOT_STATE`; they cannot enter a mutable state slot.
- R4 emits no `MemoryOperation`, writes no `MemoryStore`, and does not create `ADD`, `UPDATE`, `DELETE`, or `SUPERSEDED` effects. The update-evidence gate is a separate reviewed prerequisite for B1.

## Qualification controls

Before inference, freeze a new control pack and its human-reviewed atom oracle. It must include:

| Control family | Required behavior |
|---|---|
| Same owner/object/attribute, paraphrased, same value | Same canonical slot; duplicate/corroboration, no revision |
| Same owner/object/attribute, different values | Same canonical slot; relation unresolved absent explicit update authority |
| Same owner/object, different attributes | Distinct canonical slots |
| Same attribute, different objects | Distinct canonical slots |
| Same object/attribute, different owners (user vs another named person) | Distinct canonical slots |
| Concurrent multi-value preference or candidate set | Same slot where oracle says so; coexistence, no revision |
| Separate travel plans or named objects | Preserve distinct targets; do not infer a shared update slot |
| Generic `reported_value` / unscoped number | Abstain; no identity from shared template or numeric shape |
| Completed event / purchase | Event veto; no mutable state slot |
| Composite sentence with multiple mutable attributes | Emit each expected atom once, or explicit unresolved result; never merge dimensions |
| Ambiguous owner/object/attribute reference | Abstain and preserve unresolved source |

Do not reuse R3's 19 records, any B0Q reviewed slot member, or any B0Q counterfactual/control input as an R4 qualification item. Prefer a newly authored synthetic control pack. If prior natural-language records are used instead, the overlap audit must exclude the full B0Q eligible-record universe by stable memory ID and normalized proposition hash, and also exclude the R3 inputs. The chosen source pool, exclusion set, normalized-text rule, and overlap report must be frozen before inference.

## Acceptance predicates

All predicates are mandatory; an unresolved result is a pass only for a control whose oracle expects abstention.

1. Every positive identity control maps to exactly its oracle canonical tuple, including same-slot/different-value cases.
2. Every different-owner, different-object, and different-attribute negative remains distinct; zero false candidate merges are allowed.
3. Concurrent-value controls share identity only where specified by the oracle and receive no revision edge.
4. Event controls receive no mutable state slot; unscoped or ambiguous controls abstain.
5. Atom coverage is exact: no missing, duplicated, spurious, or span-invalid atom; composite unresolved is accepted only where pre-labelled.
6. Required positive controls cannot pass as `UNKNOWN`/abstain. Any wrong identity, event admission, false merge, missing atom, or hash/coverage defect yields `MEM3B0Q_R4_IDENTITY_QUALIFICATION=NO` (infrastructure failures remain separately classified).
7. Replay under the same frozen request, runtime, and source bytes reproduces the same canonical atoms, decisions, artifact hashes, and zero store mutations.

The exact case IDs, input and oracle hashes, local Qwen runtime, prompt/schema, field span bounds, lexeme map, thresholds, validators, overlap audit, replay policy, and acceptance code must be frozen as a separately named R4 protocol before any model call. No post-result gate relaxation or same-case tuning is allowed.

## Review and next gate

Requested independent decision: `APPROVE_AS_DESIGN_CANDIDATE` or `REVISE`, with any remaining missing predicates. Approval would authorize only a separately frozen R4 protocol and fresh qualification controls. It would not authorize B1.

After an R4 pass, a separate review must still define the update-evidence predicate. Only then may B1 implement deterministic materialization and `CURRENT` / `AS_OF` / `CHANGE` over the existing frozen M10 substrate without changing its schema. M10 currently exposes active-only retrieval: superseded/deleted records are excluded from `_active`, historical as-of querying is not implemented, and UPDATE does not close the old `valid_until` interval. Do not claim historical semantics until the research layer implements and tests them.

`MEM3B0Q_MEM3B1_READY=NO` remains authoritative. No LongMemEval DEV/TEST, MedMemoryBench, benchmark scorecard, medical evidence authority, or B1 operation is authorized here.

## Narrative guardrail

```text
retrieval surfaces relevant history
  -> flat propositions still coexist with stale state
  -> B0Q pairwise admission merged 45/99 reviewed slots incorrectly
  -> R3/R4 must qualify grounded slot identity without granting revision authority
  -> only a separately validated update predicate can materialize temporal state
  -> public and medical benchmarks test whether that state reduces stale reuse
```

This is failure-driven method development, not a claim of improved Memory quality. The 99-slot review is a negative admission result and an engineering/design constraint, not a benchmark win.
