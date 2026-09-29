# U1 Counterfactual and Replay Protocol

For each frozen synthetic episode, `CounterfactualRunner` computes the valid
subset of the eight fixed action arms and reuses one runtime episode, observable
state, patient snapshot, external-world version, tool/worker manifest, budget
regime, and evaluator identity. Invalid arms are recorded in the availability
mask and are not executed.

## Outcome and cost

The deterministic executor makes zero provider calls; token counts and latency
are zero rather than fabricated. Outcomes keep independent typed quality
metrics. No combined `OverallScore` joins retrieval, memory, and team metrics.
Cost uses versioned abstract memory-read, retrieval, worker, and tool units
(`u1-abstract-cost-v1`), never USD.

`MinimalSuccessfulActionSet` keeps every arm within configured epsilon of the
lowest-cost arm satisfying task success, safety, and grounding. If there is no
successful arm, the set is empty. Safe abstention can count as task success for
the explicit insufficient-evidence contract.

## Replay identity

Every arm records:

```text
episode_hash
observable_state_hash
patient_snapshot_hash
external_world_hash
capability_manifest_hash
action_hash
executor_version
evaluator_version
```

All identity fields other than `action_hash` must match across arms of one
episode. Re-running the same episode and action produces identical serialized
result/trace hashes. The current implementation fixes latency to zero; a future
provider backend may add seed/model/provider identity.

## Synthetic case matrix

The fixture pack includes U1-NONE, U1-MEM, U1-RAG, U1-MEM-RAG, U1-TEAM,
U1-MEM-TEAM, U1-ALL, U1-OOD-INSUFFICIENT, and an additional
U1-TEMPORAL-LEAKAGE case. They are synthetic contract cases, not medical
benchmark results.
