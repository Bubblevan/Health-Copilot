# M3 expansion evaluation QA and annotation audit

## Scope and provenance

This audit reviews the completed expansion live artifacts without replacing
their metrics or altering their frozen inputs:

- capability run: `runs/m3/20260920T122325+0800/` (30 cases);
- claim-support run: `runs/m3/20260920T122507+0800/` (21 fixtures).

The approved release manifests bind the exact input SHA256 files. This audit
does **not** assert a new human approval, change a gold label, or retune M3
runtime/policy behavior. The two capability cases below remain an explicit
human-review decision point.

## Capability expansion annotation review

The preserved run reports 28/30 decision agreement (93.3% accuracy) and
macro-F1 0.9327. These remain the historical artifact values; they must not be
presented as a clean model-error count until the following two question/evidence
semantics are resolved.

| case | frozen input and observed output | audit finding | QA treatment |
| --- | --- | --- | --- |
| `m3cx-005` | Question asks why high blood pressure may have no warning symptoms. The cited CDC card says it usually has no warning signs/symptoms and that measurement is how to learn whether pressure is high. Gold `SUFFICIENT`; observed `RECOVERABLE`. | The card supports the observation, not a causal/mechanistic explanation of **why** it occurs. The proposed query requests that missing rationale. | Ambiguous fixture; do not count its observed `RECOVERABLE` as a confirmed model error. A future annotation decision must either narrow the question to the supported observation or add evidence supporting the requested rationale. |
| `m3cx-019` | Question asks why a care plan should be discussed with a medical team. The cited CDC card recommends co-producing and discussing a care plan. Gold `SUFFICIENT`; observed `RECOVERABLE`. | Recommendation existence does not establish the causal rationale for the recommendation. | Ambiguous fixture; do not count its observed `RECOVERABLE` as a confirmed model error. A future annotation decision must either ask whether this recommendation exists or supply evidence for its rationale. |

This ambiguity does not change the core capability-boundary observation: all
10 reviewed insufficient boundary cases were denied recovery (`INSUFFICIENT`),
including the 4 `insufficient_out_of_scope` cases that define the reported
out-of-scope false-recovery metric (`0/4`). No reviewed boundary case was
released as `RECOVERABLE` in this run.

## Claim-support metric QA

The existing category metrics are **disposition-level** metrics. For example,
`unsupported_reject_rate` only asks whether an unsupported fixture was rejected;
it does not require the verifier to choose exactly `UNSUPPORTED` rather than
`CONTRADICTED`.

For the completed 21-fixture expansion run, the semantic denominator excludes
the fabricated-citation fixture because deterministic citation integrity rejects
it before a verdict is requested. There are 23 semantic claim verdicts:

| metric | value |
| --- | ---: |
| disposition-level category outcomes | 100% for every reported category rate |
| fine-grained claim verdict accuracy | 21/23 = 91.3% |
| deterministic citation rejections | 1 |
| semantic verifier rejections | 11 |

The two fine-grained disagreements are both safe rejections but are not exact
three-class matches:

- `m3sx-012`, second claim: gold `UNSUPPORTED`, observed `CONTRADICTED`.
- `m3sx-021`: gold `UNSUPPORTED`, observed `CONTRADICTED`.

The frozen M3 result is retained separately: `m3s-004` expected
`SUPPORTED/SUPPORTED` but observed `UNSUPPORTED/SUPPORTED`, a multi-claim
false reject. The two independent expansion `multi_claim_all_supported`
fixtures both passed. Together this is evidence of improved verifier behavior,
not evidence that multi-claim verification is deterministic or perfect.

## Required reporting language

- Say “single-call batched cited-evidence binding,” not “per-claim isolation.”
- It excludes observed evidence cited by no claim and explicitly binds
  `claim_index` to `cited_evidence`.
- It is not cryptographic or physical isolation: one verifier context may still
  contain another claim's cited evidence. Physical isolation would require one
  verifier call per claim and extra round trips.
- The latest focused three-trial diagnostic observed M3 OOD tool execution
  `0/12`, expected answers `18/18`, unexpected abstains `0/18`, mean model
  turns `1.30`, and mean tool executions `0.30`.
- An earlier same-configuration small sample observed `15/18` expected answers.
  Neither small-sample focused result is a stable answer-rate or generalization
  claim.
