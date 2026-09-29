# RAG-E5A Foundation Readout

Base commit: `9e023dbddc1f8025e3608bfae4c5390c1a7957ef`.

| Gate | Result | Evidence |
| --- | --- | --- |
| Frozen RAG closeout unchanged | YES | R2MED/MIRAGE closeout files and final R2MED lock hashes match the pre-work audit. |
| ESL longitudinal substrate pinned | YES | Dataset manifest v16; exact disjoint cohorts and state-file fingerprints in `runs/rag_e5/esl_source_manifest.json`. |
| DEV / future holdout boundary | YES | 20 users in 202607 and 20 disjoint users in 202608; split is by user. |
| Profile / timeline / exam schema audited | YES, with optional field differences | See `e5_esl_source_audit.md`; the loader must tolerate batch-version optional fields. |
| PUBLIC_HEALTH eligible | YES | Active reviewed catalog: 30 sources, all `public_health`. |
| GUIDELINE eligible | NO | No reviewed guideline/recommendation corpus is connected to the active retriever. |
| LITERATURE eligible | NO | Reserved and outside E5 v1. |
| OFF / STANDARD / STRONG profiles bound | YES | Exact configs, model hashes, R2MED lock provenance, and profile digests in `runs/rag_e5/retrieval_action_profiles.json`. |
| Integration case and policy observation schemas frozen | YES | Runtime input and teacher metadata are separate; policy accepts only `ExecutionPolicyObservation`. |
| Feature leakage audit | PASS | Unknown runtime fields and nested evaluator-only fields fail closed. |
| Synthetic E5 smoke tests | PASS | 13 RAG-E5-specific checks are included in the 68-test focused regression run. |
| E5-A ready to proceed to E5-B | NO | The required reviewed GUIDELINE capability is absent. |

## What was and was not run

The permitted 202607/202608 state-file schema/hash audit ran. Synthetic-only
contract tests ran. No policy was trained; no E5 task outcomes or action
counterfactuals were calculated; no oracle label was created; no model/API was
called; no 202608 native question or answer content was opened. The external
baseline repositories under `D:/MyLab/Jianli/external/rag` were read-only.

## Stop condition

Do not start E5-B or construct T2 cases until official public guidelines have
been reviewed, hashed, indexed, and admitted to the active source catalog. The
empty `guideline_source_manifest.json` records this as a real blocker, not a
placeholder capability claim.
