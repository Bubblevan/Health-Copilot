# Harness V1 migration inventory

Statuses use exactly `PRODUCTION_V1`, `ADAPTER_TO_V1`, `RESEARCH_LEGACY_READONLY`, or `DEPRECATED`. Historical result artifacts remain unchanged; MA-MVP implementation files were retired from the active tree by the follow-up pruning decision.

| Old path/component | Old responsibility | V1 replacement | Migration status | Safe to delete later |
|---|---|---|---|---|
| `src/health_ai_copilot/harness/*` | Shared request, profile, runtime, budget, trace, verification | Harness V1 | PRODUCTION_V1 | No |
| `src/health_ai_copilot/providers/model.py` | Common model provider boundary and vLLM/legacy adapters | `ModelProvider` | PRODUCTION_V1 | No |
| `src/health_ai_copilot/providers/retrieval.py` | Frozen RAG provider and KB qualification manifest | `RetrievalProvider` | PRODUCTION_V1 | No |
| `src/health_ai_copilot/providers/memory.py` | Read-only memory provider contract | `MemoryProvider` | PRODUCTION_V1 | No |
| `src/health_ai_copilot/reasoning/single.py` | Static Single control for paired evaluation | Common Eval baseline only; product default is adaptive | ADAPTER_TO_V1 | No; retain the control for attribution |
| `src/health_ai_copilot/reasoning/adaptive_mdt.py` | Unified provider adapter for adaptive MDAgents-style reasoning | `AdaptiveMDTReasoner` | PRODUCTION_V1 | No |
| `src/health_ai_copilot/multi_agent/mdagents_style.py` | MDAgents complexity triage, dynamic recruitment, specialist/team reasoning, synthesis | `AdaptiveMDTReasoner`; `run_context()` consumes preassembled context | PRODUCTION_V1 | No; this is the selected adaptive reasoning trunk |
| `src/health_ai_copilot/multi_agent/{runtime,orchestration,routing,evaluation,data,providers}.py` | MA-MVP1/2 fixed routing, Jev branch, worker artifacts, task/coverage ledgers, runtime and evaluators | AdaptiveMDTReasoner through Harness; frozen reports/artifacts remain | DEPRECATED; removed from active tree | Already removed; history and result artifacts retained |
| `src/health_ai_copilot/multi_agent/contracts.py`, `skills.py` | Scoped memory/evidence adapters used by the existing owned-universe tool surface | Preserved unchanged for Memory/RAG continuity | RESEARCH_LEGACY_READONLY | No; preserve the current Memory integration |
| `src/health_ai_copilot/multi_agent/api.py` | Former legacy runtime API plus current answer contract | One Harness-backed `/medical/answer` and `/medical/metrics` entry | PRODUCTION_V1 | No |
| `src/health_ai_copilot/pipeline.py` | Earlier safety/RAG/single product pipeline | Harness providers and `SingleReasoner` | DEPRECATED | Only after product callers and parity checks migrate |
| `src/health_ai_copilot/execution_policy.py` | Frozen policy/profile routing for earlier experiments | `SystemProfile` plus Harness configuration | RESEARCH_LEGACY_READONLY | No; retain historical profile evidence |
| `src/health_ai_copilot/e2_team.py` | Typed evidence/team execution experiment | Not selected for V1 runtime; preserve as research reference | RESEARCH_LEGACY_READONLY | No |
| `src/health_ai_copilot/research/integration/executor.py` | Owned longitudinal deterministic episode executor | `evaluation/integration.py` request/result adapters; scoped provider binding remains explicit | ADAPTER_TO_V1 | No; keep research evaluator |
| `src/health_ai_copilot/research/integration/*` | Longitudinal episodes, replay, counterfactual tools, evaluators | Integration adapters over Harness contracts | ADAPTER_TO_V1 | No |
| `src/health_ai_copilot/runtime/*` (M10) | Opt-in session/context/memory runtime | Future provider/strategy adapters; not silently made product default | ADAPTER_TO_V1 | No; frozen M10 hashes must remain intact |
| `src/health_ai_copilot/eval/*`, `eval/*`, `evals/*` | Subsystem and historical experiment evaluation | `src/health_ai_copilot/evaluation/*` for new system-level work | RESEARCH_LEGACY_READONLY | No; preserve data, code, and result hashes |
| `tools/research/multi_agent/{run_ma_mvp*.py,ma_mvp2_*.py}` | MA-MVP1/2 experiment and reserved-test entry points | None; frozen manifests/results remain under `runs/multi_agent/` | DEPRECATED; removed from active tree | Already removed; history and result artifacts retained |
| `tools/research/multi_agent/run_mdagents_health_copilot_parity.py` | Separate MDAgents graft runner | `tools/eval/run_common_eval.py --matrix core` and the shared profile registry | DEPRECATED; removed from active tree | Already removed; frozen parity output/report remain |
| `tools/eval/run_common_eval.py` | Common Eval checkpointing and Harness invocation; B0–B3 2×2 batch mode | One system runner for single arms and the core RAG×Adaptive matrix | PRODUCTION_V1 | No |

## Migration notes

- Adaptive MDAgents accepts only Harness `ReasoningContext`; its former direct request/skill pipeline was removed so safety, retrieval, memory, budgets, and verification have one owner.
- MA-MVP1/2 runtime, task/coverage-ledger types, Jev triage branch, legacy API contract, and separate runners were removed from active source. Historical protocols, results, and artifacts are retained unchanged.
- Product API defaults to one adaptive MDAgents path; Common Eval may invoke static Single only as a comparison control.
- The MDAgents-style path owns variable complexity and team size. Do not add a parallel fixed three-worker product runtime without new, positive paired evidence.
- `pipeline.py`, old evaluation trees, and experiment outputs are retained. No historical result or frozen profile hash was edited as part of this inventory.
- Retrieval profile availability is gated by a separate Common Medical KB qualification object; current product knowledge cards alone do not qualify the broad public-exam matrix. Existing RAG code, data and results were left intact.
