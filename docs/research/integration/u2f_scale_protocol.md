# U2-F Scale and Generation Protocol

Date: 2026-09-30

Dataset ID: `health-copilot-owned-longitudinal-v1`
Generator version: `u2f-generator-v1.0.0`

## Scope

U2-F materializes the project-owned synthetic TRAIN, DEV_IID, and DEV_STRUCTURAL universe specified after U2-E. It validates longitudinal snapshots, capability requirements, split isolation, answer truth, source independence, lineage, and deterministic counterfactual execution. The environment is a research harness; no training run or model performance evaluation is part of this stage.

Only allowlisted repository specifications and source files are copied into two independent temporary execution directories. Both copies receive the same frozen plans, profile, and seed. The wrapper compares generated file hashes before placing the selected result in its output directory. `generator_commit_sha`, spec hash, seed, and artifact hashes are recorded in the manifest.

## Scale and split plan

| Split | Episodes | Subjects | Episodes per subject |
|---|---:|---:|---:|
| TRAIN | 4,096 | 320 | 12–14 |
| DEV_IID | 512 | 64 | 8 |
| DEV_STRUCTURAL | 512 | 64 | 8 |
| Total | 5,120 | 448 | — |

The profile seed is `20260930`; the deterministic spot-review seed is `2026093001`. TRAIN, DEV_IID, and DEV_STRUCTURAL use disjoint scenario-seed and persona-seed ranges. DEV_STRUCTURAL uses four reserved template families that do not appear in TRAIN. Counterfactual siblings remain within one split and subject identity.

## Longitudinal environment

Each subject has one shared 80-record synthetic timeline spanning 729 days. The records cycle through PROFILE, EVENT, MEASUREMENT, EXAM, and CONVERSATION. Each later episode snapshot is a time-bounded view of that same timeline; all 448 subjects have later snapshots that add records. Four history regimes target distinct time and record-count ranges:

| Regime | Records | Timeline span |
|---|---:|---:|
| SHORT | 2–6 | 1–14 days |
| MEDIUM | 8–24 | 30–120 days |
| LONG | 24–64 | 120–365 days |
| SATURATED | 64–128 | 365–730 days |

TRAIN minima are 20% SHORT, 25% MEDIUM, 25% LONG, and 15% SATURATED. No more than 70% of TRAIN may have only 1–4 visible records. Evidence rows are independently generated synthetic worlds with SMALL (4–8), MEDIUM (9–24), and LARGE (25–64) regimes across PUBLIC_HEALTH, GUIDELINE, and LITERATURE namespaces. These labels describe synthetic fixture families only and do not assert production-source qualification.

## Scenario and counterfactual construction

TRAIN samples 14 frozen scenario families. Four additional task/template families are reserved for structural DEV. The grammar includes current-context controls, personal-state lookups and revisions, temporal comparisons, multi-source retrieval, cross-capability joins, conflicts, distractor-heavy cases, and five insufficient-evidence subtypes. Six answer types are represented: EXACT_TOKEN, EXACT_SET, ORDERED_SEQUENCE, NUMERIC, BOOLEAN, and ABSTAIN.

Matched groups hold the visible surface constant while changing the latent requirement; other groups hold the dependency graph constant while varying the query surface. Requirement-independent surface wording is balanced across cases. Every generated identifier and value is opaque and project-owned; neutral query context is generated deterministically from the case seed.

The runtime counterfactual suite evaluates all eight U1.1 action arms for every episode (5,120 × 8 = 40,960 arms). It uses deterministic local tools and zero provider calls. A mismatch between the dependency-derived expectation, the structured evaluator, and the runtime arm fails the stage.

## Artifact boundaries

For each active split, runtime `episodes.jsonl`, evaluator-only `evaluator_truth.jsonl`, `lineage.jsonl`, and `partial_policy_supervision.jsonl` are written separately. Root-level `latent_worlds.jsonl` and `natural_language_realizations.jsonl` preserve the owned generation and replay views. Runtime leakage audit verifies evaluator/split/architecture labels do not appear in runtime episodes.

The only candidate partial policy dimensions are memory_read, external_retrieval, and answerability. Architecture and budget labels remain `UNRESOLVED`. `partial_policy_supervision.jsonl` records eligibility and remains unauthorized for training unless all scale gates pass. No training, fine-tuning, or post-training run is launched here.

Six reserved roles cover IID_TEST and patient, task, temporal, source, and composition OOD plans. Their seed ranges are recorded in `u2f_reserved_test_plan.json`; zero test/OOD rows are generated or materialized in U2-F.

## Reproduction

Use the repository's pinned `uv` environment, the frozen U2-F plans/profile, and `tools/research/integration/generate_u2f_universe.py`. The wrapper requires separate isolated generation copies, byte-identical artifacts, explicit review status, and a unique output root. The generated manifest and run-specific closeout record the exact command inputs, hashes, gate measurements, and generator commit.
