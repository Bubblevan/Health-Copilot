# Adaptive MDAgents integration and frozen parity result

## Active integration

There is one system entry point: HealthCopilotHarness. It owns request parsing, safety, Memory/RAG provider execution, budgets, evidence identity, verification, and traces. AdaptiveMDTReasoner receives the resulting ReasoningContext and runs the MDAgents-style complexity route:

1. Basic: one agent.
2. Intermediate: dynamically recruited specialists.
3. Advanced: dynamically recruited teams, with team and specialist limits set by configuration.

The former standalone MDAgents request/skill pipeline and the MA-MVP fixed-team runtime are no longer active entry points. Static Single remains only as a common-evaluation baseline. The product API defaults to the AdaptiveMDT profile; RAG and Memory are composed by Harness profiles without parallel retrieval or memory pipelines.

Use the single Common Evaluation runner and profile registry for paired model/RAG/reasoning comparisons. The core matrix is B0-B3; P0-P3 and R0-R3 add the SFT and GSPO model variants. Memory is off for stateless medical MCQ. RAG-enabled profiles remain blocked until a Common Medical KB is qualified.

## Frozen 128-case parity sample

This artifact predates the unified runtime and remains immutable. It compares the earlier Health-Copilot graft with frozen local MDAgents Adaptive outputs on a deterministic MedQA sample; it is a parity check, not a benefit estimate.

| Metric | Frozen reference | Earlier graft | Delta |
|---|---:|---:|---:|
| Accuracy | 77/128 = 60.16% | 76/128 = 59.38% | -0.78 pp |
| Parse success | 87.50% | 99.22% | +11.72 pp |
| Exact answer agreement | — | 76/128 = 59.38% | — |
| Agreement when both parsed | — | 76/111 = 68.47% | — |
| Exact route agreement | — | 75/128 = 58.59% | — |

The earlier graft averaged 5.98 provider calls, 6,311.9 tokens, and 49.32 seconds per case. This sample does not demonstrate an accuracy gain.

## Related frozen result

The full local Qwen3-8B MedQA reproduction is documented in [MDAgents local reproduction](mdagents_local_reproduction.md): Adaptive scored 56.09% (714/1,273) versus Single at 61.27% (780/1,273), a -5.18 pp difference, with higher calls, token use, and latency. These local results are preserved as historical evidence. The planned remote factorial run will use the unified Harness and the user's current server/model setup; it must produce a new paired result before any positive claim.

Frozen earlier output remains under runs/multi_agent/mdagents-graft-parity-final-20261004/. The standalone parity runner was removed from the active tree; the unified Common Evaluation runner is the sole ongoing evaluation entry.
