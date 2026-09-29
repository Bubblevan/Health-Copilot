# E5-A2 capability and temporal leakage hardening

## Capability metadata boundary

`E5IntegrationCase` no longer carries `allowed_external_source_families`. It
contains only the runtime case identity, question, decision timestamp, and
longitudinal-state reference. Capability availability is represented
separately by `E5RuntimeCapabilityContext`, constructed only from environment
source families, active corpus identity, permission actions, and budget state.
Its provenance is fixed to `capability_context_source="environment"`; task
family, required evidence group, teacher action, oracle label, and outcomes are
not constructor inputs or policy fields.

For v1, T0/T1/T2 synthetic evaluator records share the same environment
capability snapshot. Source-family availability cannot be inferred from the
task's required evidence group, and available actions cannot be inferred from
an oracle action. A guideline family without an active corpus SHA fails closed.

## Decision-time state packet

`LongitudinalStatePacket` is a deterministic bounded projection of profile,
timeline, and exam records. Timeline events after the decision timestamp are
excluded. Date-only exams on the decision date are conservatively excluded
because their time is unknown. The packet emits coarse condition/exam
categories, age bucket, trend labels, history span, recent-event count, hashes,
and bounded record IDs; it does not emit patient prose or raw measurements.
Unknown top-level payload keys and nested teacher/evaluation/evidence keys are
rejected. No retrieved guideline text or LLM-generated summary is used.

## Tests and limits

Focused synthetic tests cover task-independent capability snapshots,
capability provenance, teacher/evidence-field rejection, future timeline/exam
exclusion, deterministic summaries, source review gates, raw-source hash
verification, chunk provenance/limits, deterministic chunk IDs, family-scoped
views, and exact index-to-corpus binding.

This proves a code-level contract, not clinical correctness. Profile fields are
the pinned source snapshot and do not carry independent per-field effective
timestamps; before creating historical decision tasks, the task contract must
confirm the profile snapshot is valid at the decision time. Future integration
work must preserve the 202608 holdout and must not treat these synthetic tests
as outcome evidence.
