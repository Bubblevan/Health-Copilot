# U1 Longitudinal Environment Contract

Status: research-only deterministic prototype. The contract does not bind the
production Memory implementation, production retrieval, or a medical dataset.

## Frozen namespaces

| Namespace | Authority |
| --- | --- |
| `MEMORY_READ` | One subject's timestamped personal/session state, scoped by subject and decision time |
| `EXTERNAL_RETRIEVAL` | Independent external evidence records in a versioned evidence world |
| `TEAM_ORCHESTRATION` | Execution topology and delegation only; workers add no data authority |

`IntegrationEpisode` is a frozen runtime contract containing the query,
observable metadata, capability references, budget, and evaluator identity.
It contains no gold answer, required evidence labels, counterfactual outcome,
or privileged teacher analysis. Nested values use tuples or frozen dataclasses.

## Three planes

1. **Runtime**: `IntegrationEpisode` and `ObservableState`; visible inputs,
   budget, capability manifest, and allowed tools only.
2. **Evaluation**: `EvaluationPlane`; gold answer, required facts/evidence,
   success predicate, and evaluation-side failure labels. Only the evaluator
   receives this object.
3. **Privileged training**: `PrivilegedTrainingPlane` and
   `PrivilegedTeacherPacket`; sibling outcomes, oracle actions, hidden state,
   and failure attribution. Runtime and `StudentPacket` do not accept or refer
   to these types.

The run writer serializes runtime episodes, SFT candidates, GRPO groups, OPD
student packets, and OPD teacher packets through separate serializers/files.
`episodes.jsonl` cannot serialize an `EvaluationPlane` or teacher packet.

## ObservableState

The v1 fields are history existence/length/time-span, available personal-state
types, external source families, tool IDs, worker capabilities, budget class,
deadline class, and allowlisted task-intent metadata. Each field is explicitly
marked `PRE_DECISION_OBSERVABLE=YES` by `observability_flags()`.

The runtime schema has no slots for gold family, required evidence group,
oracle capability, retrieval hit, future tool result, counterfactual reward, or
answer label. Task-intent metadata keys are allowlisted.

## Time and evidence boundaries

`PatientStateStore.snapshot(subject_id, as_of_time)` requires timezone-aware
times and includes records only at or before the decision time. Patient records
have subject, timestamp, one of PROFILE/MEASUREMENT/EXAM/EVENT/CONVERSATION,
provenance, and content. External records live in a separate
`ExternalEvidenceWorld` with source family, publication/effective time,
authority metadata, and content. U1 fixtures for PUBLIC_HEALTH, GUIDELINE, and
LITERATURE are synthetic namespace checks only; production eligibility is not
claimed.

See `u1_capability_action.md` for masks and the Single/Team gate, and
`u1_counterfactual_protocol.md` for deterministic replay identity.
