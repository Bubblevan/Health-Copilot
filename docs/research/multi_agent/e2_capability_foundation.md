# E2-A Capability Foundation

Base checkout: `main@165e0b123552e19e6aa2469a36a89dde944e9524`.

Status: **synthetic capability foundation implemented; production heterogeneous worker set is not yet eligible.** No formal 48-case comparison was run.

## M8 audit and frozen-control boundary

The M8 v2 closeout reports 12/12 worker attempts unproductive. Eleven stopped at `max_tool_calls` and one at `model_error`, all before tool execution. This diagnoses the observed worker failures; it does not establish that homogeneous teams are intrinsically unhelpful.

In the audited implementation, `RuntimeBuilder.build()` constructs one retriever and one `ToolRegistry` per runtime profile. `_build_orchestrator()` makes role-specific model adapters/prompts but passes the same registry/retriever surface to `AgentTeamOrchestrator`. `_run_worker()` constructs a fresh `AgentLoop` per task, so its `AgentState` and default `AgentSession` are already execution-local. It reuses the same parent `RunContext`, global `RunBudgetState`, `ToolRegistry`, and `ToolRunner`; it has role/task provenance, but no child budget accounting or capability hash.

The M8 ledger deduplicates by source ID with first-write-wins. Its provenance stores initial/worker origin, worker ID, and task ID; it does not retain role, capability, retrieval/tool origin, first-seen sequence, or duplicate observations from later workers. M8 `WorkerReport.productive` is based on successful status plus claims/recovery source IDs, without capability identity.

The M8 profile remains `m8-team-bm25-v1`, `star-supervisor-v1`, and `sequential-v1`. Its historical interpretation, profiles, run bundles, and metrics are untouched. E2 uses separate role/report/ledger types and a separate runner.

## Corpus and benchmark-family gap audit

The active reviewed card set has 30 entries, all from WHO, CDC, or China's NHC, under `data/knowledge_scope.json` version 1 and knowledge pack `m0.2-2026-09-15`. The audited public-health capability maps only these reviewed patient-education sources to `public_health`. The active scope has ten evidence domains, and the catalog maps each source ID to those domains.

There are no reviewed guideline/recommendation cards in the active retriever. Raw and processed MedRAG textbook files exist locally, but they are neither wired into the active reviewed retriever nor an approved scholarly-paper corpus; they do not qualify the Literature worker. No live web search or new corpus acquisition was used.

The benchmark manifest describes 24 DEV and 24 TEST rows and still says `REVIEW_REQUIRED`; the task brief calls the benchmark frozen. This status discrepancy is preserved as architecture debt. For the read-only DEV audit, the 24 DEV cases have three examples in each of eight task families; required group counts are 0:3, 1:9, 2:9, and 3:3. DEV task-profile source-family annotations include `PUBLIC_HEALTH`, `GUIDELINE`, and `INTERNAL_REVIEWED_PATIENT_EDUCATION`. The `GUIDELINE` label does not resolve to a separate active guideline source family from the reviewed card catalog, and one NHC ID used by six DEV profile references (`nhc-2025-national-hypertension-day-04-monitoring`) is absent; the active card ID is `nhc-hypertension-day-04-adult-screening`. Those evaluation labels were not passed into runtime. No TEST row was loaded, parsed, or executed.

## Frozen capability set

| Role | Contract | Harness surface | Eligibility |
| --- | --- | --- | --- |
| `PUBLIC_HEALTH` | `e2-public-health-v1` | `public_health` family; the ten reviewed knowledge-scope domains; `search_knowledge`; BM25 v1; no guideline synthesis or open literature search. | `ELIGIBLE` for the current closed reviewed card catalog. |
| `GUIDELINE` | `e2-guideline-v1` | `reviewed_guideline` family; recommendation scope, population, and jurisdiction domains; `search_knowledge`; BM25 v1. | `NOT_YET_ELIGIBLE`: no reviewed guideline corpus in the active retriever. |
| `LITERATURE` | `e2-literature-v1` | `scholarly_literature` family; study design/population/findings domains; `search_knowledge`; BM25 v1. | `NOT_YET_ELIGIBLE`: no approved scholarly-paper corpus connected to the active retriever. |

All contracts cap a worker at two model turns and one tool call. The currently eligible roles are not different evidence surfaces: only `PUBLIC_HEALTH` can run against production cards. Synthetic fixtures use synthetic public-health and guideline families strictly to prove isolation/complementarity mechanics. This is not evidence of actual heterogeneous medical-source capability.

Full canonical contracts, hashes, and the reviewed source catalog are in `e2_worker_capabilities.json`.

## Harness enforcement and state isolation

`CapabilityScopedRetriever.search()` filters every candidate against the contract's source-family and domain allowlists, drops unknown IDs, and returns only eligible evidence. `CapabilityScopedSearchTool` is the only tool in the worker registry and checks the returned evidence again. Initial evidence is filtered by the same allowlist. The objective text and worker prompt are descriptive; they cannot expand the retriever or tool surface.

Every task gets a fresh `AgentLoop`, `AgentState`, `AgentSession`, `ToolRegistry`, and child budget counter. The child `RunContext` holds the same parent `RunIdentity` and trace sink while forwarding every side-effect guard to the parent deadline/global hard budget. Worker and parent denials stop the task before evidence commit. No Memory/session/context manager is supplied; the child metadata explicitly records `memory_policy=OFF`.

The runner only accepts the three fixed roles and no more than one task per role, with at most three tasks. Assignments are explicit `E2Task` values; there is no dynamic role generation, nested delegation, architecture routing, or task-profile lookup.

## EvidenceLedger v2 and report trust

`EvidenceLedgerV2` deduplicates evidence by `source_id` in deterministic first-seen order while retaining every distinct observation. Each observation carries source ID, origin, worker ID, role, capability ID, task ID, retrieval/tool origin, and first-seen order. The first evidence payload remains canonical for duplicates; later observations are retained for overlap accounting.

An `E2WorkerReport` is verified only after execution completion and harness validation that every claim citation is in that worker's observed evidence and within the worker's capability. An invalid citation clears the claim/citation payload and makes the task fail. A failed task cannot add partial observations to the final ledger. The Lead-facing property exposes the verified run's ledger evidence and `verified_worker_reports`, not arbitrary worker prose.

## Synthetic A–G smoke

Run bundle: `runs/multi_agent/e2a-<run-id>/`.

| Case | Check | Result |
| --- | --- | --- |
| A | Public-health retrieval under an instruction to fetch outside its allowed family. | PASS: only the public-health fixture source is returned. |
| B | Guideline worker is queried for literature evidence. | PASS: literature source is filtered; only the guideline fixture is visible. |
| C | Two eligible synthetic workers retrieve different groups. | PASS: reports are verified and ledger coverage increases to both groups. |
| D | Two workers observe one source. | PASS: one ledger source record, two provenance observations, deterministic overlap 1.0 and unique contribution 0.0. |
| E | One worker raises a provider error and the other succeeds. | PASS: failed worker contributes no ledger evidence; successful worker completes. |
| F | Parent provider budget is exhausted during a worker run. | PASS: no further call is allowed; incomplete worker evidence is not committed. Local tool limit also blocks search when set to zero. |
| G | Worker cites an unobserved source ID. | PASS: report fails provenance validation; fabricated citation is absent from claims and ledger. |

These fixtures use no provider network, clinical corpus, benchmark gold, or TEST data. The artifacts serialize only synthetic source IDs/content and deterministic contract metadata.

## Verification status and remaining debt

The required focused E2 tests cover canonical hash, manifest binding, source isolation, worker state isolation, provenance, deterministic dedup, parent/worker budget rejection, failure containment, report verification, and the zero-denominator `NA` rule. Full repository regression and static checks are reported with the final task results.

Architecture debt:

- RuntimeBuilder/RuntimeProfile and `HealthCopilotPipeline` still construct the frozen M8 orchestrator. E2-A exposes an explicit runner and manifest-binding helper but does not add a selectable product/runtime profile.
- Only one production worker is currently eligible, so a production L3 comparison would be invalid until a reviewed guideline or scholarly corpus is connected and source-family mappings are reviewed.
- The benchmark manifest status and the task brief's “frozen” status disagree. DEV source-family labels also need a reviewed mapping repair before comparison; no frozen file was edited.
- Current `Retriever.search()` has no native source-family filter. The E2 wrapper enforces the worker-visible result boundary after local candidate retrieval. If a future retriever exposes native filtered search, bind and audit that implementation before claiming storage/index-level isolation.

E2-A does **not** authorize L3 DEV comparison or L4 parallel work. The next stage can be considered only after corpus eligibility and DEV mapping are resolved; a formal `E2-B Heterogeneous Sequential DEV Diagnostic` is not started here.
