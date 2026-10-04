# Architecture pruning decision — 2026-10-04

## Decision

Health-Copilot has one product request trunk:

```text
API / Eval → HealthCopilotHarness → safety → shared context/providers
          → AdaptiveMDTReasoner → verification → response + trace
```

The product API defaults to `product-adaptive-v1`. The MDAgents-style reasoner owns complexity-based execution: basic requests use its single-agent path, intermediate requests recruit a variable specialist set, and advanced requests recruit variable teams. Static `SingleReasoner` remains an evaluation control, not a second product architecture.

RAG and Memory stay behind the Harness provider contracts. Retrieval is OFF until a common corpus/index is qualified. The existing Memory implementation is left untouched and is not changed by this cleanup. The provider profile remains selectable for future research evaluation. Historical RAG and Memory runners, datasets, and results remain intact.

MA-MVP1/2's fixed three-worker/Lead topology and Task/Coverage Ledger runtime are removed from active source and product entry points. Their frozen protocols, result tables, and run outputs remain for provenance. No other team runtime is connected to the product API or Common Eval path; older M8/E2 research code remains outside this selected trunk.

## Evidence used for the cut

| Line | Frozen evidence | Architecture decision |
|---|---|---|
| MA-MVP1 routed team | DEV n=1,024: success 57.52% vs Single 57.23% (+0.29 pp; paired McNemar p=0.824); required-fact coverage 69.12% vs 72.65% (-3.53 pp); complex success +2.82 pp but complex required-fact coverage -11.14 pp. Mean latency 2.44× and tokens 2.25× Single. | Remove the MA-MVP runtime, task/coverage-ledger machinery, and runners from active source. Keep frozen protocols, metrics, and run artifacts as historical evidence. |
| Local MDAgents reproduction | Frozen Qwen3-8B MedQA n=1,273: Adaptive 56.09% vs Single 61.27% (-5.18 pp), 17.25 vs 7.00 calls/question, 33,681 vs 9,857 tokens/question, and 161.95 s vs 35.68 s mean latency. | Keep MDAgents Adaptive as the requested L40 candidate architecture, but treat its benefit as unconfirmed until the planned unified remote ablation. |
| MDAgents graft parity | Qwen3-8B local, n=128: Health-Copilot 59.38% vs reference 60.16% (-0.78 pp); complexity-route agreement 58.59%; mean latency 49.32 s and 6,312 tokens. | This is a limited parity check, not a positive system delta. Keep adaptive MDAgents as the single selected architecture; do not claim it improves accuracy yet. |
| RAG-E6B RSEL | Synthetic structured-evidence IID RAG slice n=102: grounded success 70.59% → 100.00% (+29.41 pp; subject-cluster 95% CI [+19.05, +40.78]); identical retrieved evidence and zero extra method calls. | Preserve the frozen result and method as a positive, narrowly scoped research component. The RSEL transform is not yet wired into the V1 retrieval provider; it is not proof of public/natural-clinical RAG improvement and does not qualify the product KB. |
| General RAG closeout | MIRAGE exploratory n=5,235: best fixed MedCPT 62.45% vs closed-book 61.99% (+0.46 pp) at 5,235 retrieval calls; learned router had no supported accuracy/cost win. | Keep retrieval as an optional provider capability; do not make unconditional retrieval the product default. |
| Memory | MEM-3B0R/B0S are structural/safety diagnostics; materialization remains blocked and the scorecard's one-shot LongMemEval/controlled transfer efficacy result is not present. MEM-1D2 metrics are explicitly diagnostic-only. | Keep the read-only provider boundary and frozen research evidence. Do not enable memory writes or claim an end-to-end performance gain. |

## What “pruned” means here

- The active product default is a single adaptive strategy. There is no MA-MVP2 adapter or fixed-team fallback in that path.
- Single remains only because paired evaluation needs a control arm; adaptive MDAgents itself still uses a one-agent fast path for basic requests.
- The shared Harness owns request contracts, safety, budgets, context assembly, citations, trace, and response metrics. Providers and reasoners do not create their own top-level pipelines.
- Historical run artifacts and result documents are preserved. MA-MVP implementation and runner source is removed from the active tree; Git history retains the prior source revision.
- Common public RAG profiles stay blocked until one qualified KB is bound. Current 4090 work is smoke/contract validation only; it produces no fresh benchmark claim.

## Claim boundary

This is an evidence-based architecture selection, not a claim that the new Harness or adaptive orchestration has already improved clinical accuracy. The repository's full local MDAgents reproduction is below Single; the 128-case graft parity sample is also slightly below its reference. The RSEL gain is specific to project-owned synthetic structured relations. The planned remote factorial ablation is needed before claiming system-level improvement.
