# U2-D — Dataset, Split, and License Qualification

Audit date: 2026-09-29

Status: `COMPLETE_WITH_CONDITIONS`

Branch: `integration-u2d-dataset-split-license-20260929`

Base: U1.1 commit `f0c741f3fcd4b37e1e54ee1a77c770000085c8e3`

## Scope

This is a metadata-only data governance audit. It does not evaluate systems, assign dataset rows, access benchmark answers, train models, call hosted models, or bind Memory/RAG implementations.

The audit classifies source code, dataset packaging, underlying content, labels, and derived artifacts separately. Public availability and repository code licenses do not establish rights to underlying clinical data or benchmark gold. Public benchmark material defaults to `EXTERNAL_TRANSFER`; moving it to TRAIN, DEV, or internal evaluation requires explicit permission and relinquishing the external-transfer claim for that material.

## Published provenance

- U1.1 is preserved at `f0c741f3fcd4b37e1e54ee1a77c770000085c8e3` on `integration-u1-1-exec-fairness-20260929`; remote verification returned the same SHA.
- U0 is preserved without cherry-pick at `df8cac1fafbbb8851aa2481c85c311c045c01230` on `integration-u0-benchmark-audit-20260929`; remote verification returned the same SHA.
- U2-D branch starts from the exact U1.1 commit.

## Source decisions

| Source | Code/content finding | U2-D role |
| --- | --- | --- |
| ESL-Bench | Dataset card and README state Apache-2.0; no official TRAIN/DEV identified; active evaluation batches and retired answer-bearing batches are public. U0 pinned a historical revision; current page exposes only short ref `ef0fbb2` so the exact current SHA still needs resolution. | Conditional future `EXTERNAL_TRANSFER` on a fully pinned active batch; no TRAIN/DEV/internal evaluation. |
| MedMemoryBench | GitHub data statement says CC-BY-4.0; Hugging Face card says CC-BY-NC-SA-4.0. Code Apache-2.0 does not resolve data rights. | `SCHEMA_REFERENCE_ONLY`; data license `UNRESOLVED`. |
| MedAgentBoard | No root code LICENSE found at the pinned commit; Zenodo item terms and task/source rights were not independently resolved; MIMIC-derived assets remain restricted. | `REFERENCE_ONLY`; task families not qualified. |
| MedMASLab | No repository license found at the pinned commit; a dataset-card license would cover only its package, not the several underlying source datasets. | `REFERENCE_ONLY`; component review required. |
| AgentClinic | MIT code; public case rights and split are separate. U0 read one unassigned public JSONL row for schema parsing; no official split identifies whether it was a test case. MIMIC-derived cases need separate credentialed authority. | Public cases conditional for exploratory external transfer only; MIMIC cases blocked. |
| HealthAgentBench | MIT harness; seven task families use independent source data and access terms. Running the benchmark can fetch test labels; U2-D did not inspect task tests or gold. | Wrapper reference only; future execution requires per-family qualification. |
| WHO / CDC / NHC cards | Local metadata shows 21 CDC, 5 WHO, and 4 NHC derivative cards. Card schema has no license field. WHO and CDC terms require source-specific checks; NHC article reuse rights are unresolved. Card content was not inspected in this audit. | Schema reference only; no production evidence binding. |
| R2MED | MIT code and package-level CC-BY-4.0 coexist with subset-level CC-BY, CC-BY-SA, MIT, CC-BY-NC-SA, and unspecified source terms. Historical test evaluation exists. | Conditional, per-subset external transfer only; tested subsets are exposed. |
| NFCorpus | Existing manifest has null license fields while local review says REVIEW_REQUIRED; previous test evaluation is recorded. | Unresolved; exposed, non-confirmatory reference only. |
| Medical MIRAGE | Aggregated component rights incomplete; local approval record conflicts with review-required notes. Historical 5,235-case exploratory evaluation exists. | Unresolved; exposed, non-confirmatory reference only. |
| MedRAG corpora | No frozen document-level source/rights manifest is present. | Blocked. |

Detailed records and evidence URLs are in u2d_source_registry.json and u2d_license_matrix.json.

## Derived-artifact lineage contract

Every future derived dataset, example, annotation, index, or training packet must carry these fields: derived_id, parent_source_ids, transformation_id, transformation_version, generated_by, creation_time, license_inheritance_note, and split_role.

Also preserve the immutable parent revision or source URL, content hash where permitted, jurisdiction/effective date for evidence, and access-control classification. Derived artifacts inherit the most restrictive applicable source terms unless a documented review establishes otherwise. No raw benchmark row, hidden label, answer key, or gated source content may be committed to the repository.

## Split policy

u2d_split_registry.json freezes the policy but creates no membership or row assignments. Future TRAIN and DEV should come from separately authored synthetic pools, disjoint by subject/persona, task/scenario template, and transformation/counterfactual siblings. Reserve IID and OOD memberships before training or tuning.

The registry defines TRAIN, DEV, IID_TEST, OOD_PATIENT, OOD_TASK, OOD_TEMPORAL, OOD_SOURCE, OOD_COMPOSITION, and EXTERNAL_TRANSFER. OOD axes remain distinct and must not be collapsed into one “OOD” score. Patient history stays in MEMORY_READ; independent evidence stays in EXTERNAL_RETRIEVAL.

## External evidence qualification

- PUBLIC_HEALTH: U1 synthetic fixtures establish namespace behavior only. Existing WHO/CDC/NHC cards are not yet production-qualified; CDC/WHO remain conditional per-card, NHC unresolved.
- GUIDELINE: not eligible; no jurisdiction-specific, versioned, rights-cleared guideline corpus is frozen.
- LITERATURE: not eligible; existing retrieval benchmarks are not an evidence corpus, underlying terms are heterogeneous or unresolved, and project test exposure exists.
- Worker eligibility is unchanged.

See u2d_external_evidence_qualification.json.

## Access and execution controls

- No ESL, MedMemoryBench, MedAgentBoard, MedMASLab, AgentClinic, or HealthAgentBench benchmark evaluation was run for U2-D.
- No evaluation/test query, gold answer, qrels, hidden payload, or gated MIMIC record was read for U2-D.
- U0 previously read one unassigned public AgentClinic row solely for schema parsing; its identity and gold status are unknown.
- R2MED, NFCorpus, and MIRAGE have historical project-level test/evaluation exposure recorded in the independence registry.
- No API or hosted LLM was called. No SFT, GRPO, OPD, E2-B, L4, Memory, or RAG work was started or changed.

## Decision

The audit completes with explicit source conflicts and conditions recorded. No third-party source is approved for TRAIN. Recommend U2-E as the only next stage: specify a project-owned synthetic TRAIN/DEV universe and generation controls. Do not create rows until that stage.
