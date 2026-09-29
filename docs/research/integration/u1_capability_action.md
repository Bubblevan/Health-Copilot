# U1 Capability Actions and Masks

`CapabilityAction` has exactly three independent fields:

```text
memory_read: bool
external_retrieval: OFF | STANDARD
architecture: SINGLE | TEAM
```

HETERO_SEQ/PARALLEL and STRONG retrieval are not exposed to policy. The eight
counterfactual IDs are NONE, MEMORY, RAG, TEAM, MEMORY+RAG, MEMORY+TEAM,
RAG+TEAM, and ALL.

## Namespace enforcement

- `MEMORY_READ` reads only the `PatientStateStore` subject snapshot at
  `episode.decision_time` and within `PatientStateRef.record_types`.
- `EXTERNAL_RETRIEVAL` queries only the referenced external-world version and
  source families.
- `TEAM_ORCHESTRATION` can partition those same grants across fixed workers;
  it cannot read state or evidence when the corresponding capability is off.

## Valid action masks

`ActionAvailability` rejects memory when no subject history or memory tool is
available, retrieval when there is no external world or retrieval tool, and
team when no worker pool is eligible. Capability-specific and global abstract
budgets are checked before any state read or tool side effect. A direct attempt
at an invalid action emits an `ACTION_REJECTED` trace entry and returns a
contract violation without accessing either state store.

## Single/Team equivalence hard gate

The Single envelope is compared with the union of worker grants over personal
state record types, external source families, tool IDs, model identity
placeholder, global budget regime, and evaluator identity. Team execution
fails closed before state/evidence reads if any union differs. The synthetic
test suite both verifies exact equality and deliberately removes one worker
grant to confirm rejection.

U1 does not change frozen E2-A semantics or adapt `WorkerCapabilitySpec` in
place. Its `WorkerManifest` is a separate integration adapter boundary.
