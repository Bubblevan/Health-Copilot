# E5-A2 WHO guideline owner-review packet

**Decision requested:** approve or reject each source for the future E5
`reviewed_guideline` evidence family. These are quarantined candidates; none is
owner-approved, active, indexed, or available to STANDARD/STRONG.

Review basis: the 202607-only aggregate audit covers 20 development users and
opens only `profile.json`, `timeline.json`, and `exam_data.json`. It finds
coarse BP/hypertension, physical-activity, nutrition, and weight/metabolic
signals in at least 5 users per selected domain. These are schema/indicator-name
signals, not adjudicated disease prevalence or proof that future tasks are
answerable. No native questions or answers were used to select sources.

All three PDFs identify WHO as publisher and CC BY-NC-SA 3.0 IGO as the
licence. Each PDF also contains a third-party-material warning: exclude such
content unless separately cleared, do not use the WHO logo, and do not imply
WHO endorsement. The proposed use is non-commercial research with attribution.

## 1. Hypertension pharmacological treatment (2021)

- **Source ID / status:** `who-hypertension-pharmacological-2021` —
  `READY_FOR_OWNER_REVIEW`; owner status `PENDING`.
- **Publisher / date / version:** WHO; 24 Aug 2021; ISBN 978-92-4-003398-6.
- **Population:** non-pregnant adults with appropriately diagnosed hypertension
  who have received lifestyle counselling.
- **Scope:** medication-initiation thresholds, treatment targets, first-line
  drug classes, combination therapy, laboratory checks, risk assessment,
  follow-up, and non-physician treatment.
- **202607 fit:** hypertension-labelled conditions occur in 6/20 profiles;
  BP-labelled longitudinal/exam signals occur in 20/20 under the coarse audit.
- **Recommendation-section inventory:** 8 sections (3.1–3.8); chunk count 0;
  retention audit pending.
- **Currency / supersession:** WHO publication page presents it as current
  global guidance; no replacement edition is linked there as of retrieval.
- **Caveat:** this is specifically pharmacological management, not a general
  lifestyle guideline. The state/task contract does not yet expose medication
  management fields; future tasks must respect that boundary.
- **Raw PDF:** 843,539 bytes; SHA-256
  `57f6376d5c9bc4ea6c44873625547c8a5443d5a6ad8c8c422733681563db05fb`.
- **Official record:** [WHO publication page](https://www.who.int/publications/i/item/9789240033986).
- **Owner decision:** `APPROVE` / `REJECT`.

## 2. Physical activity and sedentary behaviour (2020)

- **Source ID / status:** `who-physical-activity-sedentary-2020` —
  `READY_FOR_OWNER_REVIEW`; owner status `PENDING`.
- **Publisher / date / version:** WHO; 25 Nov 2020; ISBN 978-92-4-001512-8.
- **Population:** children aged 5–17, adults 18–64, older adults 65+, pregnant
  and postpartum women, adults/older adults with chronic conditions, and people
  living with disability.
- **Scope:** physical-activity and sedentary-behaviour recommendations across
  covered population groups.
- **202607 fit:** activity-labelled longitudinal signals occur in 20/20 users;
  examples in the aggregate inventory include daily steps and exercise duration.
- **Recommendation-section inventory:** 12 population/topic sections; chunk
  count 0; retention audit pending.
- **Currency / supersession:** updates the 2010 global recommendations for the
  populations covered; WHO continues to reference the 2020 guidance for people
  aged 5 and older. Under-five guidance is separate.
- **Caveat:** broad public-health guidance, not individualized exercise
  clearance; clinical contraindications and population scope still matter.
- **Raw PDF:** 4,041,257 bytes; SHA-256
  `50b66c44e7083cc3752849f301ae1d60d9360d511e34c0c948b39bdf6e13dc88`.
- **Official record:** [WHO publication page](https://www.who.int/publications/i/item/9789240015128).
- **Owner decision:** `APPROVE` / `REJECT`.

## 3. Total fat intake to prevent unhealthy weight gain (2023)

- **Source ID / status:** `who-total-fat-weight-gain-2023` —
  `READY_FOR_OWNER_REVIEW`; owner status `PENDING`.
- **Publisher / date / version:** WHO; 17 Jul 2023; ISBN 978-92-4-007365-4.
- **Population:** recommendation 1 applies to adults; recommendation 2 has its
  own remarks for adults and children aged 2+.
- **Scope:** total dietary fat intake for prevention of unhealthy weight gain.
- **202607 fit:** weight/metabolic and nutrition-labelled signals occur in
  20/20 users under the coarse indicator audit; weight/BMI are observable.
- **Recommendation-section inventory:** 2 recommendations; chunk count 0;
  retention audit pending.
- **Currency / supersession:** explicitly replaces WHO total-fat guidance from
  1989 and 2002; no newer total-fat guideline is linked on the current WHO page.
- **Caveat:** prevention guidance, not treatment of existing obesity; it must
  not be used to infer an individual's diet or prescribe a clinical target.
- **Raw PDF:** 741,784 bytes; SHA-256
  `a9f12218be0b1720c9be4f0dee79c2dbdcbd13fb44cd3e0d751bf04b5fa63381`.
- **Official record:** [WHO publication page](https://www.who.int/publications/i/item/9789240073654).
- **Owner decision:** `APPROVE` / `REJECT`.

## Considered but not selected

- WHO diabetes second-/third-line medicines and insulin guideline (2018):
  diabetes signals exist, but its medication-escalation scope is not currently
  matched by an E5 medication-management task/state contract.
- WHO risk reduction of cognitive decline and dementia, second edition (2026):
  no direct cognitive/dementia longitudinal signal was found in the permitted
  202607 aggregate audit.

## Promotion rule

Please provide an explicit decision for each `source_id`. A source is admitted
only when both automated and owner review statuses are `APPROVED`, with an
owner-review timestamp. Silence, general permission to proceed, or this packet
itself is not approval. Until then, guideline chunks and indexes remain absent;
the E5 external-corpus identity stays `null`, guideline capability is
ineligible, and E5-A2 stops here.
