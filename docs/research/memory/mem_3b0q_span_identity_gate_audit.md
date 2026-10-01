# MEM-3B0Q Span-Identity Gate Audit

Date: 2026-10-01

## Finding

Before any model inference, static review found that the R2 analyzer treated a
negative relation as passed whenever the records did not all share a candidate
slot. Consequently, records that were all `UNRESOLVED` could pass
`NOT_SAME_SLOT`. The R2 `NO_SLOT` check likewise accepted any set of records
whose `slot_candidate` values were all false, including unresolved records.
This allowed abstention to masquerade as semantic control success and did not
enforce the R2 protocol's requirement that proposal evidence validate.

## R2 disposition

R2 is retained as an immutable historical protocol and preflight record. It is
marked `INVALID_CANDIDATE / NOT USED FOR INFERENCE OR ADMISSION`; only its
runtime preflight ran, and that preflight was blocked before generation. There
was no model output, no benchmark scoring, no MemoryStore mutation, and no
admission decision. R2's protocol, selected inputs, case labels, preflight
attempt, and preflight report are not rewritten.

## R3 correction

R3 retains the same 19 proposal texts, case selection, proposer model, runtime,
prompt, response schema, and one-request policy. Its new run ID and protocol
make the gate correction auditable. The corrected gate requires:

- every positive and negative semantic control to have grounded identity;
- positive same-slot controls to share an exact Harness-owned candidate key;
- negative controls to prove distinct grounded keys or use an explicit,
  grounded `MULTI_VALUE_STATE` / `EVENT` / `NON_STATE` veto;
- completed purchases to be grounded `EVENT` records;
- repeated macrame records to be grounded and produce no repeated candidate
  slot;
- the generic numeric control to remain unresolved solely because its
  attribute is generic, with all other witnesses validated.

Proposal, case, and source ID coverage is checked exactly. An unresolved row in
a positive or negative semantic control cannot pass. No prompt, input, model
output, or benchmark result was used to derive this correction. The change is a
pre-inference gate repair, not evidence of method performance.

R3 remains a small development/control pilot only. Even a pass does not prove
that a candidate slot is a revision, does not establish deterministic temporal
materialization, and does not authorize B1 without human review.
