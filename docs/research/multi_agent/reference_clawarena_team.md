# ClawArena-Team reference audit for E2-A

Repository inspected: `D:\MyLab\Jianli\external\multi-agent\ClawArena\ClawArena-Team`

Pinned local checkout: `630efd8a0d1dc8189718226c7da158cbe4c2fe64`.

## Relevant files and mechanisms

| Upstream path | Relevant mechanism | E2-A interpretation |
| --- | --- | --- |
| `src/clawarena_team/agent/subagent_manager.py` (`create`, `_build_harness`, `invoke`) | A manager owns subagent records and sessions; creation validates a tool allowlist and constrains accessible paths to the creator's scope; a subagent harness is built without subagent-management tools. | Adopt manager-owned, per-worker construction and least-privilege tool surfaces. E2 narrows a retriever and accepts only fixed roles; workers cannot recursively delegate. |
| `src/clawarena_team/types.py` (`SubagentSpec`, `SessionMeta`, `SubagentLifecycleStat`) | Separates the subagent spec, invocation/session lifecycle, and accumulated usage/permission statistics. | Adopt distinct capability contract, fresh execution state, and report accounting. E2's transcript/session is per task and is not shared. |
| `src/clawarena_team/tools/workflow/runtime.py` (`defineAgent`, `agent`, `_parallel`, `_pipeline`) | Workflow definitions bind named agent specs to manager-owned subagents; `parallel` and `pipeline` execute callbacks with `asyncio.gather`. | Use only the factorization: worker definitions can stay fixed while an orchestration strategy changes. E2-A's scheduler remains sequential. |
| `src/clawarena_team/config.py` and `src/clawarena_team/agent/harness.py` | Configuration bounds subagent creation/recursion and each subagent's token/context budget. | Adopt explicit per-worker and parent hard limits; do not import the larger arbitrary workflow or multimodal model pool. |
| `src/clawarena_team/scoring/metrics.py` | Computes task success and execution-derived permission/use statistics from actual tool grants and usage. | Adopt execution-derived worker metrics and explicit denominators. Add evidence provenance and unique source contribution because tool/path use is not medical evidence validity. |
| `src/clawarena_team/scoring/report.py` | Publishes named metrics with definitions and aggregation. | Keep E2 metrics separately named/versioned; do not create a single leaderboard score. |

ClawArena-Team's manager owns a reusable **dynamic subagent registry**. It is not a predeclared fixed medical worker roster. E2 adopts manager ownership and separate worker state, while deliberately replacing dynamic definitions with three closed roles and corpus-backed eligibility.

## “Same workers, only orchestration changes”

Before the later L3/L4 comparison, freeze a worker-set identity over each worker's role, capability-contract hash, source-catalog hash, tool IDs, retriever profile, objective-contract ID, model identity, and worker budget. L3 runs those assignments sequentially. L4 must reuse the identical worker-set and assignment hashes and change only the scheduler/concurrency setting. Both arms keep the same parent budget/deadline, provider configuration, verifier, and evidence semantics. The run manifest records the worker-set hash and scheduler ID, and the evaluator rejects a pair if any worker-set hash differs.

E2-A implements only the L3 sequential substrate. It has no parallel callback, fan-out scheduler, architecture router, or hidden `task_profiles.jsonl` input.

## Ideas rejected

- Dynamic arbitrary subagents, background tasks, nested workflows, recursive creation, and arbitrary tool/path grants. They prevent a controlled worker-set comparison and exceed the fixed three-role boundary.
- Parallel and pipeline runtime code in this phase. It would change scheduler and specialization at once.
- ClawArena's broad workspace tools, multimodal worker pool, Gemma-specific model/runtime requirements, full benchmark runtime, and SMS leaderboard implementation. None is needed for closed-corpus patient education or this capability experiment.
- ClawArena's permission/task scores as evidence quality measures. They do not establish source-family isolation, citation provenance, or complementary evidence.
