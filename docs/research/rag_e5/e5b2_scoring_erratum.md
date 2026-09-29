# E5-B1 Scoring Erratum — Content and Grounding Separation

Status: corrected before any counterfactual model outcomes existed.

The B1 scorer accidentally gated guideline-content credit on citing a supplied
chunk from the required source and recommendation. That coupled two distinct
constructs and made an OFF arm unable to receive content credit even when the
reader stated the frozen guideline anchors correctly.

Before E5-B2 execution, scoring was separated into:

- `guideline_content_score`: deterministic term/number anchor matches in
  `guidance_facts`, independent of retrieval and citations;
- `grounding_score`: citations to actually supplied chunk IDs whose source and
  recommendation match the frozen teacher requirements.

Primary E2E weights are unchanged: T0 = state; T1 =
`0.75 × guideline_content + 0.25 × grounding`; T2 =
`0.50 × state + 0.25 × guideline_content + 0.25 × grounding`. A second
`content_only_quality` diagnostic is also recorded: T0 = state, T1 = guideline
content, and T2 = `0.50 × state + 0.50 × guideline_content`.

No model outcomes existed when this correction was made. It is a prospective
scorer-semantics correction, not post-result tuning. The frozen task IDs and
questions, state packets, teacher source/recommendation targets, rubric anchor
definitions, reader prompt/schema, model, and retrieval profiles are verified
unchanged by the B2 protocol freezer. Only scoring semantics/version and the
derived scorer/lock hashes change.

Synthetic T2 maxima are now 1.00 with all inputs, 0.50 without required state,
and 0.75 without external evidence/citation. This tests the declared weights;
it is not an observed model outcome.
