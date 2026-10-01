# MEM-3B0Q-R3 Pilot Postmortem

Status: `MEM3B0Q_SPAN_IDENTITY_PILOT=NO`; `MEM3B1_READY=NO`.

## Frozen outcome

- Strict-gate controls passed: `3/10`.
- Proposals: `9/19` grounded, `10/19` unresolved.
- Passed: Instagram follower same-slot, Node version same-slot, and wallet neutral-color versus material distinct-slot.
- Failed: wallet color same-slot, wallet color/material, workout versus PC, completed-purchase event, Japan versus India destinations, generic numeric attribute, and macrame control.
- Exactly one local Qwen request; zero hosted calls, answer-reader calls, benchmark scoring, or MemoryStore mutations.
- Prompt tokens `2163`; dynamic-schema tokens `6661`; completion tokens `4417`; proposal request latency `118241.414 ms`; `truncated=false`.
- The configured 8081 and pre-existing 8092 services both remained healthy after inference. GPU snapshot was `15945 MiB / 16376 MiB` used. The additional 8081 service loaded and served the one local proposal request with the 8092 service still running.
- After the smoke, only the 8081 process started for this run (PID `60420`) was stopped; 8092 (PID `62696`) remained healthy. GPU usage returned to `7691 MiB / 16376 MiB`.

## Failure interpretation

The result is a real negative qualification outcome, not a runtime failure. It does not show that revision-aware Memory improves stale-state behavior, nor does it justify B1.

The model found two clean same-slot relations (Instagram and Node) and one clean cross-attribute distinction (wallet color/material when the source was unambiguous). Most unresolved records reveal a mismatch between semantic normalization and the frozen literal token-support rule: `wallet_type`, `exercise_preference`, `pc_capabilities`, `purchase_status`, `trip_destination`, and `experience_level` were reasonable-looking labels but lacked required lexical anchors under the no-stemming/no-synonym policy. This is a combined proposal/contract limitation; it is not evidence that the records were safely or correctly admitted.

The completed-purchase records expose a separate semantic issue: both were classified `SINGLE_VALUE_STATE`, not `EVENT`. All 19 records received that property kind, including generic numeric values. The output therefore does not support trusting the model's type proposal as an admission authority.

Two audit issues also matter for the next protocol. First, the generic numeric control did abstain from a slot, but R3 required exactly one generic-attribute failure; an additional unsupported generic object label caused the case to fail. That rule was intentionally strict, but it was narrower than the semantic objective and must not be relaxed post hoc in R3. Second, the macrame case is not a clean interest-only control: its first two propositions include the same beginner-kit arrival state and were assigned the same `macrame_kit/kit_status` key. The frozen `NO_SLOT` expectation needs independent review.

For all 19 propositions, the non-null principal/object/attribute/value witness fields used the full proposition as their span, despite the instruction to return the shortest exact substring. This is a deterministic validator gap: it proves uniqueness and exact source containment, but not minimality or field-specific support locality. The frozen R3 report is not rewritten; the discrepancy is recorded here.

## Research consequence

This pilot narrows the next research question rather than adding another Memory feature: can state-slot identity be made auditable without asking a small local model to invent brittle normalized labels? The next version should consider harness-owned canonical attribute/object vocabularies, multi-attribute proposition splitting, explicit event/state controls, and validator-enforced local witness spans. Any such change requires a new protocol and non-reused controls; the R3 cases, response, labels, and score remain frozen and must not be retuned or rerun as a qualification set.

The failure-driven narrative remains coherent but incomplete: retrieval made relevant evidence available; flat propositions still coexisted as competing states; the first harness-grounded identity pilot now shows that free-form normalized identity proposals are not yet reliable enough for revision admission. Deterministic temporal materialization and public benchmark claims remain future work, behind a fresh identity qualification and review gate.
