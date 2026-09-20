# M8 empirical v1 failure audit

This audit preserves the original live M8 comparison before M8.3 diagnosis.
The frozen dataset SHA-256 is
`78a417bef691892fb0b248911044f0325849ae4a4ae1065ec7ad009968180549` and the
code SHA is `ab89f0d195cb8f0e7afe8de9c578232f3d2d9c32`. The L2 profile is
`m8-team-bm25-v1`; all 36 trials used the metadata-only trace policy.

The original L2 run exposed a Team Lead integration/contract failure and is
retained as a failure artifact. It is not used as evidence that Agent Teams
are intrinsically worse than the single-agent baseline.

## Exact L2 audit counts

| Observable | Count | Evidence / limitation |
| --- | ---: | --- |
| Trials | 36 | 12 cases × 3 trials |
| `route=abstain` | 33 | Harness result after fail-closed lead error |
| `route=human_review` | 3 | Safety short-circuit; no team activity |
| `team_stop_reason=lead_error` | 33 | Original generic stop reason |
| Accepted `DELEGATE` | 1 | One `task_created` event: `m8-cross-source-001/t2` |
| Lead `FINAL` action | unavailable | Old trace did not record the parsed action |
| Lead `ABSTAIN` action | unavailable | Do not infer from the final route |
| Provider error event | 0 | 34 lead calls ended with `provider_end:stop`; no `provider_error` |
| Team budget error | 0 | `m8.budget_exhaustion_rate=0` |
| Invalid delegation | 0 observable | No `invalid_delegation` stop/event |
| Second accepted delegation | 0 | One second lead call occurred after the failed worker, then `lead_error` |
| Worker error | 1 | Worker report failed with `max_tool_calls` |
| Final verification failure | 0 observable | No final verification was reached after the lead failures |

The old metadata-only boundary did not preserve `TeamLeadFailureKind`, provider
failure kind, response hash/length, JSON parse outcome, or contract-validation
outcome. Therefore the actual cause of the 33 `lead_error` cases cannot be
reconstructed from v1 artifacts. The data supports only the boundary-level
statement above; it does not support guessing whether the cause was empty
content, JSON syntax, contract validation, or an internal implementation error.

## Per-trial analysis table

`lead:stop` and `worker:tool_calls` are provider trace outcomes, not parsed
model actions. The table intentionally keeps `FINAL` and lead `ABSTAIN`
unresolved where v1 did not record them.

| case_id | trial | route | harness disposition | team stop | lead calls | provider calls | tasks | workers | delegated | provider trace outcome | failure stage/code |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| m8-simple-001 | 1 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-simple-002 | 1 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-simple-003 | 1 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-simple-004 | 1 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-decomposable-001 | 1 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-decomposable-002 | 1 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-decomposable-003 | 1 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-decomposable-004 | 1 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-cross-source-001 | 1 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-cross-source-002 | 1 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-ood-001 | 1 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | — |
| m8-ood-002 | 1 | human_review | — | — | — | 0 | — | — | no | — | — |
| m8-simple-001 | 2 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-simple-002 | 2 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-simple-003 | 2 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-simple-004 | 2 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-decomposable-001 | 2 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-decomposable-002 | 2 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-decomposable-003 | 2 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-decomposable-004 | 2 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-cross-source-001 | 2 | abstain | lead_error | lead_error | 2 | 3 | 1 | 1 | yes | lead:stop → worker:tool_calls → lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-cross-source-002 | 2 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-ood-001 | 2 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | — |
| m8-ood-002 | 2 | human_review | — | — | — | 0 | — | — | no | — | — |
| m8-simple-001 | 3 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-simple-002 | 3 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-simple-003 | 3 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-simple-004 | 3 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-decomposable-001 | 3 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-decomposable-002 | 3 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-decomposable-003 | 3 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-decomposable-004 | 3 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-cross-source-001 | 3 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-cross-source-002 | 3 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | safety/unexpected_route; orchestration/evidence_group_missing |
| m8-ood-001 | 3 | abstain | lead_error | lead_error | 1 | 1 | 0 | 0 | no | lead:stop | — |
| m8-ood-002 | 3 | human_review | — | — | — | 0 | — | — | no | — | — |
