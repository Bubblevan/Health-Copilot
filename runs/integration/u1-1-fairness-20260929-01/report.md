# U1.1 Execution Semantics Fairness Run

Run: `20260929-01`. Fixture: `SYNTHETIC_CONTRACT_FIXTURE; NOT_MEDICAL_BENCHMARK`.

This bundle uses synthetic deterministic resources only. It is not a medical benchmark, calls no provider, performs no training, and reads no external benchmark rows.

## Counterfactual results

| Episode | Successful arms | A* | Executable parity |
| --- | --- | --- | --- |
| U1-NONE | NONE, TEAM | NONE | PASS |
| U1-MEM | MEMORY, MEMORY+TEAM | MEMORY | PASS |
| U1-RAG | RAG, RAG+TEAM | RAG | PASS |
| U1-MEM-RAG | MEMORY+RAG, ALL | MEMORY+RAG | PASS |
| U1-TEAM | NONE, TEAM | NONE | PASS |
| U1-MEM-TEAM | MEMORY, MEMORY+TEAM | MEMORY | PASS |
| U1-ALL | MEMORY+RAG, ALL | MEMORY+RAG | PASS |
| U1-OOD-INSUFFICIENT | NONE, TEAM | NONE | PASS |
| U1-TEMPORAL-LEAKAGE | ∅ | ∅ | PASS |

## Fairness interpretation

For `U1-TEAM`, Single and Team both execute `query_part_a` and `query_part_b` through the same registered implementations and return `7`. Single cost is 2 abstract units; Team cost is 5. The corrected A* is `NONE` (Single), because the old Team-only success came from evaluator/fixture semantics and a Team-only execution path. This change repairs the contract; it is not a regression and it does not demonstrate a real Team gain.

## Provenance

- Tool calls record implementation hashes, input hashes, resource versions, and output hashes.
- Total abstract cost is activation cost plus observed workers/tools/reads/retrievals; no USD is inferred.
- One global budget is shared across architectures; Team worker shares partition that same cap.
- Student actions are `SCRIPTED_PROBE`; OPD status is `SCHEMA_PROBE`; GRPO source is `COUNTERFACTUAL_ENUMERATION`.
- SFT labels come from the deterministic counterfactual oracle and list backend, evaluator, cost model, epsilon, and arms.
- U0 commit `df8cac1` is `LOCAL_ONLY`; it was not included in this run or branch.
