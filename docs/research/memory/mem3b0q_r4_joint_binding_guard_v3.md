# MEM-3B0Q R4 Joint Binding Guard v3

Status: `OFFLINE_PROPOSAL_REVIEW_APPROVED` for this offline proposal only.
Not frozen and not authorization for inference or B1.

## Triggering Failure

After v2 review, an offline counterexample demonstrated a remaining false
accept. In `My work laptop's operating system is Linux. A tablet's operating
system is Windows.`, v2 accepted the second `Windows` span while reusing the
only registered object candidate, `work laptop`. The unknown noun `tablet`
was invisible to the closed alias registry, so nearest-known-anchor selection
did not establish the intended object.

## Proposal

V3 wraps the reviewed v2 proposal without editing its code or results. For
mutable-state records it additionally requires owner, object, and attribute
anchors to share punctuation-delimited clauses. It also rejects an
unregistered determiner-plus-noun phrase between owner/object or object/
attribute anchors. These checks reject the cross-sentence example above and a
second control where an unknown tablet follows a conjunction, without treating
every conjunction as a boundary. Event records retain their separate
fixture-scoped relation/value checks.

This remains a closed-English-vocabulary heuristic. An unknown noun phrase
without a recognized determiner may evade detection. Conversely, the simple
determiner+noun check can conservatively reject a modifier sequence such as
`a new operating system` even though `operating system` is a registered
attribute. V3 is not open-vocabulary entity resolution, a general parser, or
revision authority.

## Offline Controls

- The current v2 guard accepts an unknown-tablet cross-binding; v3 rejects the
  same proposal both across a sentence boundary and after a conjunction
  introducing an unregistered tablet noun phrase.
- A registered work-laptop operating-system fact with its local value remains
  accepted.
- The frozen R4 oracle atoms, including coordinated multi-attribute and
  multi-value examples, are replayed through v3 and must remain accepted.
- A determiner introducing the registered attribute itself is not treated as
  an unknown entity phrase.

These controls are hand-authored deterministic validator tests. They measure
neither model output quality nor memory/benchmark performance. Inference must
remain blocked until a new disjoint control pack, exact prompt/schema/runtime,
acceptance predicates, and review are complete. The next pack must include
unknown-object distractors and value-binding negatives for every supported
field type; this proposal's two unknown-object cases are development controls
and cannot serve as held-out evidence.

Independent review approved this offline proposal only, with the limitations
above. V3 tests: `5 passed`; combined R4, factorized-revision-admission, and
pairwise-admission offline suite: `155 passed`. Neither result authorizes
inference, B1, or a performance claim.

Current gates remain `MEM3B0Q_CANDIDATE_BOUNDED_EXTRACTOR_READY=NO`,
`MEM3B0Q_MEM3B1_READY=NO`, and `MEMORY_PUBLIC_CLOSEOUT=NO`.
