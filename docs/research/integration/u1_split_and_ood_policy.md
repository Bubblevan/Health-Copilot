# U1 Dataset Split and OOD Policy

This protocol defines future assignment rules only. U1 does not create a real
dataset split or import ESL evaluation rows.

## Split roles

| Split | Intended use |
| --- | --- |
| TRAIN | Training examples whose source terms explicitly permit training |
| DEV | Development/tuning with documented provenance and frozen membership |
| IID_TEST | Patient-disjoint in-distribution evaluation |
| OOD_TEST | Predeclared held-out distribution shift |
| EXTERNAL_TRANSFER | External benchmark/evidence transfer claim |

Public benchmark evaluation content defaults to `EXTERNAL_TRANSFER`. Moving it
to another split requires explicit provenance permission and a project decision
to relinquish the external-transfer claim for that use. The prototype enforces
this default in `SplitAssignment` and assigns no real rows.

## OOD axes

- `OOD_PATIENT`: unseen patient/subject identities.
- `OOD_TASK`: unseen task formulation or task family.
- `OOD_TEMPORAL`: changed timeline spacing, event ordering, or decision horizon.
- `OOD_SOURCE`: held-out evidence source family/version.
- `OOD_COMPOSITION`: capability combinations withheld from training.

OOD_COMPOSITION must permit patterns such as training on individual MEMORY,
RAG, and TEAM arms, then evaluating MEMORY+RAG, MEMORY+TEAM, RAG+TEAM, and ALL.
Patient, source, query, and gold provenance must be separately versioned. Split
and license approval is a prerequisite to materializing any real episode.
