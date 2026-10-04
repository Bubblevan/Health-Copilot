# Product runtime flow

There is one product request path:

```text
POST /medical/answer
        ↓
HarnessRequest → HealthCopilotHarness → safety → Memory/RAG providers
                                           ↓
                                  AdaptiveMDTReasoner
                             basic / intermediate / advanced
                                           ↓
                         verification → HarnessResponse + trace
```

The API accepts only runtime-visible fields. Gold answers, evaluator truth, expected routes, and benchmark labels cannot enter the request.

AdaptiveMDTReasoner is the only product reasoning strategy. It chooses a one-agent fast path for basic questions, dynamically recruits specialists for intermediate questions, and recruits variable teams for advanced questions. Safety, provider calls, memory, retrieval, evidence identity, hard budgets, and response verification belong to HealthCopilotHarness; the reasoner consumes only the supplied ReasoningContext.

Memory and retrieval are provider capabilities on the shared Harness path. The product-adaptive-v1 profile keeps both off until the production memory result and Common Medical KB qualification are ready. Evaluation profiles can enable the same providers for paired ablations, with identical retrieval evidence supplied to Single and Adaptive arms.

The stable FastAPI entry is create_fastapi_app(harness). Static Single is available only to the Common Evaluation runner as a baseline, and cannot be selected through the product API. See [Common Medical Eval V1](../evaluation/common_medical_eval_v1.md) for the one runner and factorial profile matrix.

Each response includes validated citations, safety flags, a trace ID, provider/tool counts, token totals, latency, selected profile, and reasoning route in Harness metrics. Traces contain request metadata and execution decisions but no evaluator truth.
