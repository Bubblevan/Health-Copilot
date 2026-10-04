# MDAgents-style Health-Copilot graft

## Design

The integration is an opt-in sub-runtime behind MedicalAgentRuntime. It does not replace PatientContext, Evidence, Care, RAG, Memory, or HospitalKnowledgeProvider, and the default runtime remains unchanged unless a ClinicalReasoningSkill is injected.

The independently implemented flow is:

1. Run the existing deterministic safety route before provider or harness work.
2. Ask the same model to classify the question as basic, intermediate, or advanced.
3. Basic uses one strong medical agent.
4. Intermediate recruits domain specialists and synthesizes their results.
5. Advanced uses multiple small teams followed by moderation.
6. Enforce provider-call, tool, token, and deadline budgets, recording trace and shared context through Health-Copilot contracts.

This implements the algorithmic pattern independently and does not import or copy upstream MDAgents implementation. The local provider only accepts HTTP loopback endpoints, disables inherited proxy environment settings, uses the OpenAI-compatible API, temperature 0, and Qwen no-think mode.

## Opt-in construction

```python
from health_ai_copilot.multi_agent import (
    ClinicalReasoningSkill,
    LocalVllmProvider,
    MedicalAgentRuntime,
)

provider = LocalVllmProvider(
    base_url="http://127.0.0.1:8000/v1",
    model="qwen3-8b-local",
)
reasoning = ClinicalReasoningSkill(provider)
runtime = MedicalAgentRuntime(
    provider=provider,
    clinical_reasoning_skill=reasoning,
)
response = await runtime.respond(request)
```

Call await provider.close() when the provider is no longer needed. Applications that do not inject clinical_reasoning_skill continue to use the existing routing path. The FastAPI response contract remains MedicalAgentResponse.

## 128-case parity sample

The deterministic sample uses random.Random(20261003).sample(range(1273), 128) and compares the same shuffled model_question and labels against the already frozen local reference Adaptive rows. It runs only the Health-Copilot arm; it does not rerun the reference or all 1,273 cases.

| Metric | Frozen reference | Health-Copilot graft | Delta |
|---|---:|---:|---:|
| Accuracy | 77/128 = 60.16% | 76/128 = 59.38% | -0.78 pp |
| Parse success | 87.50% | 99.22% | +11.72 pp |
| Exact answer agreement | — | 76/128 = 59.38% | — |
| Answer agreement when both parsed | — | 76/111 = 68.47% | — |
| Exact route agreement | — | 75/128 = 58.59% | — |

Health-Copilot used 5.98 provider calls and 6,311.9 total tokens per sampled case on average, with mean latency 49.32 seconds. There were no runtime failures. Route counts were basic 90, intermediate 21, and advanced 17 (reference counts: 82, 33, and 13).

The sample meets the predeclared absolute accuracy-delta target of at most 2 percentage points. It misses the answer-agreement target of 85% and route-agreement target of 90%. The measured sample accuracy is close; individual answers and route decisions are not highly concordant. This sample does not prove that all 1,273 cases would match, and no full Health-Copilot rerun was performed.

## Artifacts and identity

The checkpoint JSONL, paired case results, metrics, report, and manifest are preserved under:

runs/multi_agent/mdagents-graft-parity-final-20261004/

The manifest records the sample indices, seed, reference hash, model identity, source hashes, and result hashes. The parity runner is tools/research/multi_agent/run_mdagents_health_copilot_parity.py.
