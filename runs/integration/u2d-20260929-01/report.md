# U2-D Run Report

- Run: u2d-20260929-01
- Audit date: 2026-09-29
- Status: COMPLETE_WITH_CONDITIONS
- Branch: integration-u2d-dataset-split-license-20260929
- Base commit: f0c741f3fcd4b37e1e54ee1a77c770000085c8e3
- U1.1 remote SHA verified: f0c741f3fcd4b37e1e54ee1a77c770000085c8e3
- U0 remote SHA verified: df8cac1fafbbb8851aa2481c85c311c045c01230

## Audit result

No benchmark rows, evaluation queries, gold answers, hidden payloads, or gated records were opened during U2-D. No benchmark was executed and no hosted model was called.

The source inventory separates code licenses from dataset and upstream content terms. MedMemoryBench has a data-license conflict; MedAgentBoard Zenodo/task rights remain unresolved; MedMASLab and MIRAGE need per-source review; NFCorpus has inconsistent local approval/review states. Public health cards lack per-card license metadata. HealthAgentBench has seven independent task families and is not qualified as one pooled dataset.

The split policy is frozen but has no row assignments or membership. Third-party benchmark material defaults to EXTERNAL_TRANSFER. No third-party TRAIN or DEV source qualifies. The recommended future TRAIN and DEV universe is synthetic, project-authored, and disjoint by subject/persona, scenario template, and transformation family.

PUBLIC_HEALTH remains conditional for named existing cards; GUIDELINE and LITERATURE remain blocked. No production external evidence corpus is bound. Memory, RAG, and worker eligibility remain unchanged.

## Historical exposure disclosures

- U0 parsed one unassigned public AgentClinic JSONL row for schema only; its sample ID and gold status were not retained, and AgentClinic has no official split.
- Existing project RAG work has historical evaluation exposure on R2MED, NFCorpus, and Medical MIRAGE. Those results are non-confirmatory for a new blind holdout.
- Foundation-model pretraining exposure is unknown and is distinct from project-level non-contamination.

## Routing

The only primary next stage is U2-E: define the project-owned synthetic TRAIN/DEV universe and generation controls. U2-D did not generate data.
