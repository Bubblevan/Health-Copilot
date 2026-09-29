# U2-D Owner Scope Amendment

- Run: u2d-20260929-02
- Date: 2026-09-29
- Status: COMPLETE_WITH_OWNER_ACCEPTED_SCOPE
- Branch: integration-u2d-dataset-split-license-20260929
- Base commit: f0c741f3fcd4b37e1e54ee1a77c770000085c8e3

## Accepted project scope

The owner confirmed the project is non-commercial research and selected:

- MedMemoryBench for external benchmark evaluation only, with attribution and no redistribution or training.
- MedAgentBoard for partial external MAS validation: non-MIMIC medical QA, lay-summary tasks, and the tjh EHR prediction variant. Exclude mimic-iv mortality/readmission and the separate workflow task.
- ESL-Bench local mirror manifest v16, batch 202608, file data/202608/sample320-20260830.jsonl. The manifest-provided batch checksum is recorded in the source registry; it was not recomputed.
- WHO/CDC knowledge cards for non-commercial research retrieval only, with attribution/source links and no source-document redistribution or training. No per-article audit.
- MedMASLab as REFERENCE_ONLY; no multimodal package download.

## Access and execution

This amendment changes qualification scope only. It opens no benchmark rows or answers, reads no gated MIMIC record, runs no benchmark, calls no hosted model, and modifies no Memory/RAG implementation. The ESL manifest was inspected for version/path/checksum metadata; the selected JSONL rows were not opened.

MedMemoryBench's two upstream license statements and MedAgentBoard's unverified Zenodo item metadata remain accurately recorded. The owner accepts those metadata uncertainties as non-blocking for the specifically restricted research uses above; no redistribution or training is approved.

NHC cards, MedAgentBoard's mimic-iv EHR variants and workflow task, MedMASLab data, and unrelated retrieval corpora remain outside the selected scope. These exclusions do not block U2-E.

## Routing

Continue to U2-E for a project-owned synthetic TRAIN/DEV universe specification. No data rows were generated in this amendment.
