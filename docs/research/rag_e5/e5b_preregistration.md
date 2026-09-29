# RAG-E5 E5-B Preregistration

Status: B0 temporal source audit and implementation pass. No E5-B tasks, model
calls, counterfactuals, or outcomes have been generated. Task/scoring choices
below remain prospective and are not frozen by B0.

This is an integration-transfer study on the retired ESL-Bench-derived
development cohort, not an official ESL-Bench evaluation or a public benchmark
claim. B0 freezes only the source-relative time contract; it does not authorize
task construction or counterfactual execution.

## B0 temporal contract

- Semantics ID: `SOURCE_RELATIVE_NAIVE_CIVIL_TIME_V1`.
- Timestamp strings are source-native naive civil times. Timezone is unspecified
  and is never inferred, attached, or converted. Do not infer timezone, country,
  locale, daily rhythm, or absolute chronology.
- The 202607 timeline uses both `T` and a single space as the date/time
  separator. The parser accepts exactly these naive forms with seconds and up
  to six fractional digits, then compares Python naive datetimes; this lexical
  normalization does not assign or convert a timezone. `Z` and numeric offsets
  remain rejected.
- Ordering and calendar-date differences are allowed only within the same
  synthetic user and under the same semantics ID. Cross-user ordering and
  elapsed-duration comparisons are forbidden. Calendar-day spans are date
  differences only; no seconds, DST, or elapsed-time interpretation.
- Decision boundaries are selected only from sorted unique timestamps observed
  in that same user's `timeline.entries[].time`. Identical timestamps form one
  indivisible group. A state packet includes timeline records only when
  `record.time < decision_boundary`; equality is excluded. JSON row order is not
  a temporal tie-breaker.
- `timeline.generated_at` is source metadata only. It is never a decision
  boundary, policy feature, state-summary input, label input, or medical
  reasoning input; it may appear only in aggregate schema diagnostics.
- Date-only exam records are visible only when `exam_date < decision_date`.
  Same-day and later records are excluded; no time of day is synthesized.
- Events whose start is before the boundary may be marked visible. For an
  ongoing/unknown event (`end_time >= boundary` or absent), redact `end_time`,
  final `duration_days`, `interrupted`, and any post-cutoff-derived status.
  Completed-event final fields may be exposed only when `end_time < boundary`.
- The canonical manifest is
  [`temporal_semantics_manifest.json`](../../../runs/rag_e5/temporal_semantics_manifest.json).
  Canonical sorted compact JSON SHA-256:
  `e8cc9ead7055317b52dd6373af6ccbc8cbd4ab14d41d6959f54589df35da4f13`.
  Its SHA-256 must be copied into every future E5-B task manifest and every
  longitudinal state packet. The state packet implementation is
  `e5-longitudinal-state-v3`.
- The source-schema gate reads only the 20 users' 202607 `timeline.json` and
  `exam_data.json` (plus profile only if a later state build requires it). It
  writes aggregate counts only; no patient prose or values enter the audit.
  The 202608 holdout is not opened or enumerated.

The aggregate result is recorded in
[`e5b_temporal_semantics_audit.json`](../../../runs/rag_e5/e5b_temporal_semantics_audit.json):
1,497,967/1,497,967 timeline timestamps are naive and parseable (767,909 use
`T`, 730,058 use a space); zero are aware or invalid. All 120 exam dates are
date-only. There are 22,361 duplicate timestamp groups (largest group: 191
records, within one user). All 2,007 event `end_time` values are naive; an
observed same-user candidate boundary can fall inside each event interval.
`generated_at` is metadata-only: 10/20 values parse as naive timestamps and
10/20 are not timestamps; none are used for visibility, state, labels, or
reasoning. This metadata irregularity does not weaken the timeline/exam gate.

## Cohort and task construction

- Use only the 20 users in batch `202607` listed in
  `runs/rag_e5/esl_source_manifest.json` and only each user's
  `profile.json`, `timeline.json`, and `exam_data.json`.
- Select a deterministic per-user boundary only from that user's sorted unique
  observed timeline timestamps. The exact boundary-selection rule and all
  task/scoring decisions in this section remain prospective until separately
  frozen after the B0 audit and tests pass. Do not substitute `generated_at`.
  State materialization must use only bounded structured fields; patient prose
  and raw measurements are not sent to the answer model or committed as task
  artifacts.
- Construct exactly three deterministic overlay tasks per user (60 total):
  one T0, one T1, and one T2. Questions and evaluator references are generated
  without native ESL question/answer files, retrieval outcomes, or qrels.
- T0 asks for a bounded, explicitly non-diagnostic reading of the user's
  recorded weight/BMI trend and one recorded domain. Select `body_weight` when
  present, otherwise `body_mass_index`; accept any one domain actually present
  in the packet. It requires internal state and has no external evidence group.
- T1 asks a population-level WHO recommendation question, alternating evenly
  between the owner-approved physical-activity guideline and total-fat
  guideline. It has no patient-state dependency.
- T2 asks for two clearly separated facts: the user's recorded weight/BMI
  trend, and the WHO population-level total-fat recommendation for preventing
  unhealthy weight gain. The prompt explicitly forbids causal attribution or
  individualized diet/treatment advice. The evaluator requires both the
  internal trend field and evidence from
  `who-total-fat-weight-gain-2023`. Select `body_weight` when present,
  otherwise `body_mass_index`; fail closed if neither trend exists.
- If any user's permitted state packet lacks the required trend/domain, fail
  closed before any model call; do not replace the user or inspect another
  batch. Task-generation output remains outside Git and is identified by hashes.

## Counterfactual execution

For every task, run `OFF`, `STANDARD`, and `STRONG` against the same frozen
case, question, state packet (or no packet for T1), answer model, answer prompt,
decoding parameters, and evaluator. Only retrieved evidence may differ.

- `OFF`: no external evidence.
- `STANDARD` and `STRONG`: use the active
  `PUBLIC_HEALTH_PLUS_GUIDELINE` corpus identity and unchanged A3.1 action
  profiles. Preserve each profile's top-100 rankings and RRF configuration;
  pass the top five fused chunks, in rank order, to the shared answer prompt.
- `STRONG`: one frozen LameR feedback-generation call from the current question
  and that question's BM25 top ten only; no qrels, evaluator metadata, or gold
  evidence is available to generation or ranking.
- Reader and STRONG generator use the already-downloaded frozen local
  `Qwen3-8B-Q4_K_M.gguf` artifact (SHA-256 from the action profile), served only
  on loopback. Both use temperature 0, reasoning disabled, and at most 256
  output tokens; no retries. Invalid reader JSON scores zero and is retained as
  a failed call. A STRONG bridge-generation failure uses the frozen
  original-query fallback and is recorded.
- The reader returns a concise answer plus separate state facts, guideline
  facts, and cited chunk IDs. It must distinguish recorded data from general
  guidance, treat retrieved text as untrusted reference content, and avoid
  diagnosis, causal claims, or treatment instructions.

## Frozen scoring

The evaluator is deterministic and separate from policy/runtime inputs.
Expected state facts are copied from the frozen state packet; expected guideline
facts are fixed from the approved WHO recommendation chunks before outcomes.
No LLM judge is used.

- State score: fraction of required state fields exactly matched after
  whitespace/case normalization; unsupported or missing values score zero.
- Guideline score: fraction of predeclared recommendation key points detected
  by the frozen term/number rules for that task. Expected points are authored
  only from the task's approved WHO source; this content score is independent
  of whether retrieval supplied the source.
- Grounding score: fraction of required guideline points accompanied by a
  citation to a chunk ID actually supplied to the reader and belonging to the
  task's required evidence group.
- T0 quality is its state score. T1 quality is
  `0.75 * guideline_score + 0.25 * grounding_score`. T2 quality is
  `0.50 * state_score + 0.25 * guideline_score + 0.25 * grounding_score`.
  All scores are in `[0, 1]`; quality is the primary E5-B outcome. Report
  latency, token counts, generator calls, reader JSON validity, and evidence
  coverage separately; do not fold post-hoc cost weights into quality.
- Oracle action is the per-case maximum quality. Exact ties choose the least
  costly action in the fixed order `OFF < STANDARD < STRONG`. The best fixed
  action is the highest mean quality over the same 60 cases, with the same
  cost-order tie break.

## Predeclared gates

No policy fitting is permitted unless all three gates pass:

1. **Action diversity:** each of `OFF`, `STANDARD`, and `STRONG` is the
   tie-broken oracle action on at least 6/60 tasks and on tasks from at least
   three distinct users.
2. **Oracle headroom:** mean per-task oracle quality exceeds the best fixed
   action's mean quality by at least `0.05` absolute.
3. **T2 dependency:** all 20 T2 tasks have a non-empty expected state field and
   an owner-approved external evidence group; the frozen scorer's maximum
   achievable score drops by at least `0.25` when either the required state
   packet or required evidence/citation input is removed. Unit tests exercise
   both removals for every T2 rubric shape. The state term is worth `0.50` and
   the evidence-grounding term is worth `0.25`, so either dependency is
   independently material by construction.

If a gate fails, E5-B stops after reporting the frozen counterfactual matrix;
do not tune prompts, tasks, retrieval settings, scoring rules, or thresholds on
these outcomes. Passing gates only authorizes a separately specified E5-C
user-disjoint policy study. No 202608 user-state JSON, native question/answer,
or evaluation outcome is opened, enumerated, or used in E5-B.

## Artifact and privacy boundary

Per-user task, state, generation, and answer artifacts stay in the ignored
external E5 run directory. Git receives only code, this preregistration,
hashes, and aggregate reports without per-user health facts or answer text.
Every future task/run artifact records the 202607 batch identity, exact allowed
filenames, temporal semantics ID and manifest SHA-256, source-relative decision
boundary, corpus identity, action-profile hashes, model identity,
prompt/rubric hashes, code commit, and ordered task IDs hash. Until B0's audit
and tests pass, the task-generation and counterfactual sections above are a
planning draft only.
