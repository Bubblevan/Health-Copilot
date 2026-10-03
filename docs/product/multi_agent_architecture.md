# Routed Medical Team

Health-Copilot uses a Medical Router to send straightforward questions through a Single-Agent fast path and multi-source questions to a Lead with three specialist roles: Patient Context, Evidence, and Care. The Lead plans bounded work, receives explicit worker statuses, and synthesizes only Harness-observed context.

```text
User query → Medical Router
                 ├─ Single fast path → Strong Single Agent
                 └─ Lead → Patient Context / Evidence / Care
                           → Shared Context + Evidence Ledger
                           → Lead synthesis → Harness verification
```

The runtime keeps model choice replaceable through `ModelProvider`. The same underlying model is used for the Strong Single baseline, Lead, and workers. Single is given the union of the workers' registered skills, so the product comparison does not deliberately weaken the baseline.

## Execution limits

- At most three worker calls per request, with each role assigned once.
- Independent tasks start concurrently; Care can run in a second wave after patient or evidence reports.
- Each worker uses at most three model turns and two tool calls. The initial implementation performs a bounded skill-observe pass and one answer turn.
- Worker timeout, provider error, and tool error become explicit report statuses. The Lead sees those statuses and can return a partial answer; zero successful workers trigger the Single fallback.
- There is no recursive delegation.

## Harness boundary

The model makes semantic decisions: task objectives, worker selection, interpretation, and synthesis. The Harness creates identifiers, accounts for calls and tokens, enforces skill permissions and budgets, records evidence provenance, owns Shared Context mutations, and performs final safety checks. Model output cannot supply a source ID, evidence ID, task ID, worker ID, or budget value.

Patient state and external evidence are provided through swappable `MemoryProvider` and `ExternalEvidenceProvider` interfaces. The current owned-universe adapters are deterministic and synthetic. `HospitalKnowledgeSearchSkill` has an adapter interface and a disabled provider; no hospital corpus is loaded.

See [Skills and agents](skills_agents.md) and [Runtime flow](runtime_flow.md) for the call surfaces and response contract.
