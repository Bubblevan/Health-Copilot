# U2-D TRAIN/DEV Strategy

## Decision

Use a future project-authored synthetic longitudinal universe for TRAIN and DEV. No third-party benchmark or external medical corpus is qualified for training in this audit. U2-D creates no data and assigns no rows.

The nine U1.1 fixtures remain contract examples only. They are too small and too purpose-built to represent a TRAIN or DEV distribution. Do not copy benchmark wording, patient narratives, question templates, labels, or gold answers into the owned universe.

## TRAIN

- Author scenario grammars, persona attributes, event timelines, and task intents from project-owned specifications.
- Generate only synthetic people and synthetic events. Do not seed content from benchmark rows, real records, or gated sources.
- Record each artifact with the lineage contract in u2d_data_governance.md.
- Permit TRAIN only after a separate stage freezes the scenario families and verifies source independence, safety, and allowed use.

## DEV

- Create DEV from a separately reserved persona, scenario-template, and seed pool before any prompt or hyperparameter tuning.
- Keep subjects/personas, template families, and transformation/counterfactual siblings disjoint from TRAIN.
- Use DEV for prompt/configuration selection; record every access and selection decision.
- Do not use public test answers or previously exposed benchmark outputs as a proxy for DEV.

## Future internal and OOD evaluation

Reserve IID_TEST, OOD_PATIENT, OOD_TASK, OOD_TEMPORAL, OOD_SOURCE, and OOD_COMPOSITION memberships before training. Keep membership under evaluator control. OOD_SOURCE remains unavailable until a source family has passed its specific rights and provenance review.

Third-party benchmarks retain EXTERNAL_TRANSFER identity. Previously evaluated R2MED, NFCorpus, and MIRAGE test material is exposed and cannot support a new blind confirmatory claim. AgentClinic has one unassigned public row exposure and no official split; it is not a clean holdout. ESL's active batch may be considered for a future external transfer only after exact immutable revision resolution.

## Stage routing

Proceed to U2-E as the single primary next stage: construct the project-owned TRAIN/DEV universe specification and generation controls. Do not generate or materialize episodes until U2-E is authorized and the lineage, split, and safety constraints are implemented.
