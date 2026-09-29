# U2-D Decision

Date: 2026-09-29

Outcome: `COMPLETE_WITH_OWNER_ACCEPTED_SCOPE`

Primary next stage: `U2-E — Project-Owned TRAIN/DEV Universe Specification`

## Decisions

1. Do not train on third-party benchmark or external evidence content. No such source is fully qualified for TRAIN.
2. Use a future project-authored synthetic longitudinal universe for TRAIN and a separate authored pool for DEV. Do not generate rows in U2-D.
3. Freeze split and OOD policy, but assign no records. u2d_split_registry.json is policy-only.
4. Use MedMemoryBench for non-commercial external evaluation only, with attribution and no redistribution or training.
5. Use MedAgentBoard as a partial external MAS benchmark: non-MIMIC medical QA, lay summary, and tjh EHR prediction; exclude mimic-iv and the separate workflow task.
6. Use the local ESL-Bench manifest-v16 202608 batch as external transfer only.
7. Use WHO/CDC cards for non-commercial retrieval only, with attribution/source links, no document redistribution, and no training; leave NHC cards unbound.
8. Keep MedMASLab REFERENCE_ONLY; do not download the multimodal package.
9. Keep GUIDELINE and LITERATURE production evidence unqualified. U1 synthetic namespace fixtures do not establish their eligibility.
10. Keep Memory/RAG implementation and worker eligibility unchanged.

## Qualification summary

| Decision area | Result |
| --- | --- |
| U1.1 published exact commit | VERIFIED — f0c741f3fcd4b37e1e54ee1a77c770000085c8e3 |
| U0 published exact commit | VERIFIED — df8cac1fafbbb8851aa2481c85c311c045c01230 |
| Source/license registry | Complete; owner-scoped use decisions recorded |
| Split/OOD policy | Frozen; no rows assigned |
| TRAIN source | Project-owned synthetic only, future stage |
| DEV source | Separate project-owned synthetic pool, future stage |
| Selected dataset availability | Owner reports complete remote downloads; no full-file scan was run in U2-D |
| PUBLIC_HEALTH retrieval | WHO/CDC qualified for the stated non-commercial retrieval scope; NHC reference only |
| Selected external benchmarks | MedMemoryBench, local ESL 202608, and partial non-MIMIC MedAgentBoard |
| GUIDELINE | BLOCKED |
| LITERATURE | BLOCKED |
| U2-D locked/test/gold row access | NO |
| U2-D gated data access | NO |
| Restricted content committed | NO |
| Benchmark execution / hosted model calls | NO |
| Memory/RAG modifications | NO |

## Explicitly excluded from the selected scope

- MedAgentBoard mimic-iv EHR variants and the separate clinical-workflow task.
- MedMASLab data package and underlying multimodal datasets.
- NHC retrieval cards.
- R2MED, NFCorpus, Medical MIRAGE, MedRAG, AgentClinic, and HealthAgentBench remain outside the selected next-stage data scope; their prior exposure or source-specific constraints remain documented.

## Next stage gate

Proceed to U2-E to write a generation specification, source-independence controls, and synthetic subject/template separation for the project-owned TRAIN/DEV universe. The owner-selected external evaluation and retrieval scopes are recorded; U2-D itself did not run them or generate data.
