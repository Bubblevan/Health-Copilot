# M3 expansion evaluation QA and annotation audit

## Scope and provenance

This audit reviews the completed expansion live artifacts without replacing
their metrics or altering their frozen inputs:

- capability run: `runs/m3/20260920T122325+0800/` (30 cases);
- claim-support run: `runs/m3/20260920T122507+0800/` (21 fixtures).

The release manifest binds the exact input SHA256 file. This audit does not
retune M3 runtime/policy behavior. The M4.0 precondition below narrows only
the two question wordings; it does not change their expected decisions or
KnowledgeScope.

## Capability expansion annotation review

The superseded pre-repair run reports 28/30 decision agreement (93.3% accuracy)
and macro-F1 0.9327. It remains a historical artifact for its original input
hash, but it is not a clean model-error count for the repaired release.

| case | frozen input and observed output | audit finding | QA treatment |
| --- | --- | --- | --- |
| `m3cx-005` | Original question asked why high blood pressure may have no warning symptoms. The cited CDC card says it usually has no warning signs/symptoms and that measurement is how to learn whether pressure is high. Gold `SUFFICIENT`; observed `RECOVERABLE`. | The card supports the observation, not a causal/mechanistic explanation of **why** it occurs. | M4.0 annotation repair narrows the question to `高血压是否可能没有明显的预警症状？`; expected decision remains `SUFFICIENT`. |
| `m3cx-019` | Original question asked why a care plan should be discussed with a medical team. The cited CDC card recommends co-producing and discussing a care plan. Gold `SUFFICIENT`; observed `RECOVERABLE`. | Recommendation existence does not establish the causal rationale for the recommendation. | M4.0 annotation repair narrows the question to `现有 CDC 资料是否建议与医疗团队共同制定并讨论高血压管理计划？`; expected decision remains `SUFFICIENT`. |

The old ambiguity did not change the core capability-boundary observation: all
10 reviewed insufficient boundary cases were denied recovery (`INSUFFICIENT`),
including the 4 `insufficient_out_of_scope` cases that define the reported
out-of-scope false-recovery metric (`0/4`). No reviewed boundary case was
released as `RECOVERABLE` in the superseded run. A clean artifact will bind the
repaired input hash before final M3 freeze.

## M4.0 resolution and clean artifact

The M4.0 precondition narrowed `m3cx-005` and `m3cx-019` exactly as recorded
above, retained their `SUFFICIENT` decisions and topic mappings, and produced
release hash `b964d9c8e1dcfec03c907ed69a6437e0c9202fc50c1e9d15511cade259acd273`.
The clean live artifact is `runs/m3/20260920T125210+0800/`, recorded from
implementation/data commit `7e0c945964e040ca02def56218af4d3cd473caa0`.

It observed 30/30 decision agreement, macro-F1 `1.0`, no out-of-scope false
recovery, and topic exact-set accuracy `0.9`. This run has a different frozen
input hash and remains stochastic provider output; it is a clean artifact for
the repaired fixture semantics, **not** evidence of a model-performance
increase over the superseded 28/30 artifact.

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
