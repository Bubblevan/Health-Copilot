# U2-D Decision

Date: 2026-09-29

Outcome: `COMPLETE_WITH_CONDITIONS`

Primary next stage: `U2-E — Project-Owned TRAIN/DEV Universe Specification`

## Decisions

1. Do not train on third-party benchmark or external evidence content. No such source is fully qualified for TRAIN.
2. Use a future project-authored synthetic longitudinal universe for TRAIN and a separate authored pool for DEV. Do not generate rows in U2-D.
3. Freeze split and OOD policy, but assign no records. u2d_split_registry.json is policy-only.
4. Treat public benchmark rows as EXTERNAL_TRANSFER by default. Retired answer-bearing or previously evaluated content is exposed and cannot support an unseen confirmatory claim.
5. Keep PUBLIC_HEALTH, GUIDELINE, and LITERATURE production evidence unqualified. U1 synthetic namespace fixtures do not establish production corpus eligibility.
6. Keep Memory/RAG implementation and worker eligibility unchanged.

## Qualification summary

| Decision area | Result |
| --- | --- |
| U1.1 published exact commit | VERIFIED — f0c741f3fcd4b37e1e54ee1a77c770000085c8e3 |
| U0 published exact commit | VERIFIED — df8cac1fafbbb8851aa2481c85c311c045c01230 |
| Source/license registry | Complete; conflicts explicit |
| Split/OOD policy | Frozen; no rows assigned |
| TRAIN source | Project-owned synthetic only, future stage |
| DEV source | Separate project-owned synthetic pool, future stage |
| Production PUBLIC_HEALTH evidence | Conditional per source; none fully qualified |
| GUIDELINE | BLOCKED |
| LITERATURE | BLOCKED |
| U2-D locked/test/gold row access | NO |
| U2-D gated data access | NO |
| Restricted content committed | NO |
| Benchmark execution / hosted model calls | NO |
| Memory/RAG modifications | NO |

## Open qualification items

- Resolve MedMemoryBench GitHub/Hugging Face data-license disagreement.
- Obtain item-level Zenodo and task-family terms for MedAgentBoard.
- Review MedMASLab and MIRAGE underlying sources independently.
- Resolve NFCorpus license review inconsistency before further content reuse.
- Resolve each WHO/CDC/NHC card's source notice, license, and derivative history.
- Pin ESL-Bench's current active batch to a full immutable commit before any future run.
- Qualify HealthAgentBench data source and label rights independently for each task family.

## Next stage gate

Proceed to U2-E to write a generation specification, source-independence controls, and synthetic subject/template separation for the project-owned TRAIN/DEV universe. U2-E should not begin external benchmark execution or generate data unless its own scope explicitly authorizes those actions.
