# RAG-E5-B1 — Frozen Longitudinal Integration Overlay

Status: task/evaluator protocol frozen; counterfactual execution has not begun.
This is an integration-transfer study on the 202607 ESL-Bench-derived cohort,
not an official ESL-Bench evaluation or public benchmark claim.

## Scope and data boundary

B1 uses only the 20 users in batch `202607`, their `profile.json`,
`timeline.json`, `exam_data.json`, and the already activated external corpus.
Native questions, answers, KG queries, and all 202608 state/question/answer
files remain unopened. There were no answer-model calls, retrieval calls,
generated bridges, counterfactual outcomes, or oracle labels in B1.

Patient-specific state packets, questions, and teacher records are kept outside
Git at `D:\MyLab\Jianli\external\rag_e5\e5b1\`. Git contains code, protocol,
hashes, and aggregate reports only.

## Decision-time state

Each boundary is the latest unique timestamp observed in that user's timeline.
The packet includes records strictly before that timestamp; the equal-time
group is excluded. `generated_at` is never used. Missing required state fails
coverage; no boundary or metric rescue is allowed.

The 202607 schema has no safe full birth-date field among `date_of_birth`,
`birth_date`, or `birthday`. All 20 age buckets are therefore `unknown`, and
`profile.demographics.age` is excluded. State packets were versioned to
`e5-longitudinal-state-v4`.

| Coverage check | Count |
|---|---:|
| Users | 20/20 |
| Body-weight trend selected | 20 |
| BMI fallback selected | 0 |
| Neither trend / failed user | 0 |
| Safe age / unknown age | 0 / 20 |

## Frozen task overlay

There are exactly 3 tasks per user, 60 total. Select `body_weight`, otherwise
`body_mass_index`; every user selected body weight. T0 asks only for the
recorded trend. T1 has no state packet or runtime boundary. T2 asks separately
for the recorded trend and general adult dietary-fat guidance, forbidding
causal attribution, diagnosis, and individualized advice.

T1 alternates by sorted user index: even selects the eligible adult
physical-activity recommendation; odd selects the eligible total-fat
recommendation. Hypertension pharmacological guidance is not task-authoring
eligible and is never a teacher source. Questions do not reveal family labels,
action names, publisher/source IDs, guideline titles, or retrieval intent.

Task identity hashes batch, user, the user's boundary, family, and template
version. For T1 the boundary participates in identity but is absent from the
runtime projection. Runtime records have only `case_id`, `user_id`, `question`,
`decision_boundary`, `state_packet_ref`, and `runtime_capability_context_ref`.
Expected trends, source/recommendation IDs, rubric refs, and family labels are
in a separate teacher artifact.

## Scoring freeze

The active corpus identity is
`9b19ad467f47641032277707cb1cfdb1d05c2fb39c558039b180efc7394692bd`. Teacher
keypoints are deterministic term/number rules derived from approved eligible
recommendation chunks; no model writes the rubric and no LLM judge is used.

Frozen content anchors are: adult moderate aerobic activity 150–300 minutes
per week; vigorous aerobic activity 75–150 minutes per week; muscle
strengthening on at least 2 days per week; and, for total-fat recommendation 1,
adults aged 20 or older at 30% of total energy or less, with no instruction to
increase intake for people already below the threshold.

- State score is the fraction of required trend fields exactly matched after
  whitespace/case normalization; valid enum values are `rising`, `falling`,
  and `stable`.
- Guideline keypoints receive credit only when a supplied chunk from the
  required source and recommendation is cited and the content anchors appear in
  a guidance fact. This evidence gate resolves the older *prospective* draft
  sentence that content could score independently of retrieval. No outcomes
  existed when this rule was frozen.
- Grounding is the fraction of required recommendations cited by a chunk ID
  actually supplied and mapped to the required source and recommendation.
- T0 quality = state. T1 = `0.75 × guideline + 0.25 × grounding`. T2 =
  `0.50 × state + 0.25 × guideline + 0.25 × grounding`.

Synthetic maximum-score checks pass on all 20 T2 rubric instances: full state
and evidence can score 1.0; removing either state or evidence/citation limits
the maximum to 0.5. This is a scorer test, not an observed model result.

## Runtime and B2 lock

All cases share source families `public_health` and `reviewed_guideline`,
actions `OFF`, `STANDARD`, `STRONG`, provider/tool/token budgets `2/1/8192`,
deadline `120000 ms`, and the same corpus. Frozen action-profile config hashes:

- STANDARD: `6ddb91bb0c31f5bc5b69372df6a3bdd3b4f71d70cdbcece8ef22c5bfd9a94330`
- STRONG: `d9e3de9bf986a1f08b1c217153422d6e86ad0a84bc1982d90bade897c9274a8b`

The reader and STRONG bridge generator use the same Qwen3-8B GGUF artifact,
revision `6a569868d07d3bd59e8b97fb001bf8c0b254bb20`, SHA-256
`d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`,
temperature 0, reasoning disabled, 256 output tokens, one call, no retry, and
loopback-only service. The recorded llama.cpp runtime is `10068`
(`571d0d540`). B2 is locked to case-ID order then `OFF → STANDARD → STRONG`,
with the same question/state/reader/evaluator across arms and retrieved
evidence as the only intended difference.

Machine-readable artifacts:

- [`e5b1_state_coverage_report.json`](../../../runs/rag_e5/e5b1_state_coverage_report.json)
- [`e5b1_task_manifest.json`](../../../runs/rag_e5/e5b1_task_manifest.json)
- [`e5b1_leakage_audit.json`](../../../runs/rag_e5/e5b1_leakage_audit.json)
- [`e5b_counterfactual_lock.json`](../../../runs/rag_e5/e5b_counterfactual_lock.json)

Task-ID set SHA-256:
`7ac2d95f98efc6fcced665f93977c76c4d219af0cfe7e16f05b1fdd67358979a`.
State-packet set SHA-256:
`860f5da64a6b94790944a5bd2d50aa4977a098bc51d1b6f822321bf4fbbd1d1c`.
Question leakage, runtime/teacher separation, capability invariance, and
synthetic dependency gates pass. B1 ends here; B2 has not started.
