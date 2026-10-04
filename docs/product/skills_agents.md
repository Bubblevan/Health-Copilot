# Skills and agents

The runtime separates reusable capabilities from agent responsibilities. A Harness `SkillRegistry` checks the worker role before every invocation.

| Agent | Registered skills | Responsibility |
|---|---|---|
| PatientContextAgent | `PatientStateLookupSkill`, `TimelineCompareSkill` | Patient-specific records, time order, and changes |
| EvidenceAgent | `MedicalKnowledgeSearchSkill`, `ExternalEvidenceSearchSkill`, `HospitalKnowledgeSearchSkill` | Knowledge retrieval, evidence sources, and citations |
| CareAgent | `RiskAssessmentSkill`, `AnswerabilitySkill` | Risk cues, answerability, and actionability from observed context |
| Strong Single Agent | Union of all registered skills | Baseline with the same model and capability surface |

`MemoryProvider.lookup(context)` and `ExternalEvidenceProvider.search(context)` keep the agent API independent of a particular storage or retriever. The current implementation adapts the owned longitudinal world and its deterministic external-evidence namespace. Production Memory and RAG implementations can replace those providers without changing agent prompts or routing contracts.

`HospitalKnowledgeProvider.search(query, top_k)` is an interface only. `DisabledHospitalKnowledgeProvider` returns no results; the runtime does not crawl documents or load hospital content.

## Provenance

Skills return source material and implementation observations. The Harness generates the evidence ID and tool-call ID, then records the worker, skill/tool, input and output hashes, resource versions, source ID, and excerpt in the Evidence Ledger. Responses cite only ledger entries. Patient facts and evidence snippets remain untrusted data in model input.
