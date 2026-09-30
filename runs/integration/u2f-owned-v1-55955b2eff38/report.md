# Health-Copilot U2-F — Owned Longitudinal Universe

- Dataset ID: `health-copilot-owned-longitudinal-v1`
- Run ID: `55955b2eff38`
- Status: `FROZEN`
- Generator commit: `fdaf394be287c8523c394fc626c42cd86e602745`
- Branch: `integration-u2d-dataset-split-license-20260929`
- Base commit: `ce68eb7d37463d255a8abd3ad0a1b73cfbbf518e`
- Specification hash: `60e3a1807683efef1c7147f549c9acc14b2087035bcfdbe3a480d25774367d4a`
- Content origin: `PROJECT_OWNED_SYNTHETIC`
- Notice: `SYNTHETIC_RESEARCH_WORLD; NOT_CLINICAL_GUIDANCE`
- Provider calls: `0`; training started: `NO`

## Scale and split ownership

| Split | Episodes | Subjects | Episodes per subject (min / median / mean / max) |
|---|---:|---:|---|
| TRAIN | 4096 | 320 | {'min': 12, 'median': 12.0, 'mean': 12.8, 'max': 14} |
| DEV_IID | 512 | 64 | {'min': 8, 'median': 8.0, 'mean': 8, 'max': 8} |
| DEV_STRUCTURAL | 512 | 64 | {'min': 8, 'median': 8.0, 'mean': 8, 'max': 8} |
| Total | 5120 | 448 | {'min': 8, 'median': 12.0, 'mean': 11.429, 'max': 14} |

- Timeline records per subject: {'min': 80, 'median': 80.0, 'mean': 80, 'max': 80}
- Timeline span in days per subject: {'min': 729, 'median': 729.0, 'mean': 729, 'max': 729}
- Shared-timeline progression: `True`; subjects gaining later records: 448
- History regime counts: `LONG` 1536, `MEDIUM` 1536, `SATURATED` 896, `SHORT` 1152
- History record-count buckets: `1-4` 159, `24-63` 1536, `5-7` 993, `64+` 896, `8-23` 1536
- Available record types: `CONVERSATION` 25479, `EVENT` 35639, `EXAM` 26579, `MEASUREMENT` 27493, `PROFILE` 29340
- Required Memory records by type: `CONVERSATION` 675, `EVENT` 631, `EXAM` 620, `MEASUREMENT` 620, `PROFILE` 676
- Scenario families: `COMPOSITIONAL_MULTI_FACT` 302, `CURRENT_ONLY` 806, `DISTRACTOR_HEAVY` 274, `EXTERNAL_LOOKUP` 516, `EXTERNAL_MULTI_SOURCE` 260, `EXTERNAL_VERSIONED` 272, `INSUFFICIENT_EVIDENCE` 634, `MEMORY_EXTERNAL_CONFLICT` 260, `MEMORY_EXTERNAL_JOIN` 438, `MEMORY_LOOKUP` 624, `MEMORY_MULTI_RECORD` 154, `MEMORY_REVISION` 258, `MEMORY_TEMPORAL_COMPARE` 156, `TEMPORAL_BOUNDARY` 166
- Answer types: `ABSTAIN` 634, `BOOLEAN` 258, `EXACT_SET` 1800, `EXACT_TOKEN` 2208, `NUMERIC` 62, `ORDERED_SEQUENCE` 158
- Derived capability requirements: `INSUFFICIENT` 679, `MEMORY` 1313, `MEMORY+RAG` 1000, `NONE` 1080, `RAG` 1048
- Dependency depth: `1` 3355, `2` 1215, `3` 458, `4` 88, `5` 4
- Dependency width: `1` 4802, `2` 248, `3` 56, `4` 14
- Independent dependency groups: `1` 4244, `2` 568, `3` 218, `4` 82, `5` 8
- Distractor counts: {'min': 4, 'median': 12.0, 'mean': 27.545, 'max': 72}; regimes: `EXTREME` 1536, `HIGH` 896, `LOW` 742, `MEDIUM` 1946
- Evidence-world sizes: {'min': 6, 'median': 16.0, 'mean': 20.471, 'max': 40}; regimes: `LARGE` 1670, `MEDIUM` 1722, `SMALL` 1728
- Revision depths: `1` 80, `2` 82, `3` 96
- Memory-required temporal distances: `1-7 days` 144, `181-365 days` 382, `31-180 days` 1395, `8-30 days` 477, `<1 day` 293, `>365 days` 531
- Insufficient-evidence subtypes: `CONFLICT_UNRESOLVED` 126, `FUTURE_ONLY` 127, `MISSING_SET_MEMBER` 142, `STALE_PERSONAL_STATE` 121, `WRONG_SOURCE_FAMILY_ONLY` 118
- Same surface, different requirement: 0.1516 of episodes
- Different surface, same latent dependency: 0.6312 of episodes
- IID lexical template overlap is intentional and reported; DEV_STRUCTURAL uses its four reserved families.
- Per-class TRAIN distribution is compared with the reference ranges in `manifest.json`; deviations are documented rather than silently rebalanced.

## Shortcut and integrity audits

- Majority and best single-feature diagnostics: `PASS`; max accuracy 0.6531; max lift over majority 0.0852.
- Combined context-aware Naive Bayes diagnostics: `PASS`; TRAIN→DEV_IID macro-F1/balanced-accuracy max 0.8199; TRAIN→DEV_STRUCTURAL 0.7892.
- Query-only diagnostic is separately recorded for DEV_IID and DEV_STRUCTURAL in `cheap_classifier_audit.json`.
- Split/sibling isolation: `PASS`; subject, persona, scenario-seed, and counterfactual sibling leaks are zero.
- Exact duplicate audit: `PASS`; exact query duplicate rows 388 (388 are approved same-surface pairs); duplicate runtime rows 0.
- Temporal/revision: `PASS`; future personal rows checked 342, external publication/effectivity checks 71456, revision examples 258, exact boundaries 166.
- U1.1 contract: `PASS`; 40960 arms; mismatches 0.
- Source independence and generated identifier scan: `PASS`.
- Runtime/evaluator separation: `True`; lineage rows checked 5120.
- Deterministic spot sample: TRAIN 32, DEV_IID 16, DEV_STRUCTURAL 16; status `PASS`.
- Combined-probe review disposition: `PASS`; max score 0.8199 against review threshold 0.90.

## Data and training boundary

- Reserved TEST/OOD pools: `PASS`; materialized rows 0.
- Real patient data / PHI: `NO`.
- External benchmark rows: `NO`.
- WHO/CDC content or external medical documents: `NO`.
- Architecture supervision: `UNRESOLVED`; budget supervision: `UNRESOLVED`.
- Partial capability selection data eligible: `True` for Memory read, External retrieval, and answerability only.
- Full execution-policy training ready: `NO`; post-training ready: `NO`.
- Memory and RAG subsystem tracks modified: `NO`.
- E2-B / L4 started: `NO`.

## Gate result

`SCALE_GATE = PASS`. See `manifest.json` for every boolean gate and `distribution_report.json`, `shortcut_audit.json`, `cheap_classifier_audit.json`, `split_audit.json`, `duplicate_audit.json`, `temporal_revision_audit.json`, `counterfactual_contract_report.json`, `source_independence_audit.json`, `human_spot_audit.json`, and `lineage_audit.json` for full measurements.
