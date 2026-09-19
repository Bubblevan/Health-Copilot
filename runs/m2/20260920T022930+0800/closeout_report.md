# M2 empirical closeout

Source commit: `9043d316772d331df49420f3c5889d91d256cc05`.

This closeout references three independently timestamped, live artifacts:

- Policy standalone: `runs/m2/20260920T022022+0800/`
- Grounding standalone: `runs/m2/20260920T022231+0800/`
- Focused M1-vs-M2 A/B (3 trials): `runs/m2/20260920T022248+0800/`

Each standalone `config.json` records the commit, model, base URL, temperature,
dataset path and SHA-256, knowledge-pack version, case count, timeout, and retry
count. No API key is persisted.

## Policy standalone

- Accuracy: `0.8333333333333334` (20/24)
- Macro-F1: `0.8222222222222223`
- Sufficient recall: `1.0` (8/8)
- Recoverable recall: `1.0` (4/4)
- Insufficient recall: `0.6666666666666666` (8/12)
- Conflicting has no production denominator; its synthetic contract test remains offline.

Confusion matrix (expected rows, predicted columns):

| expected | sufficient | recoverable | insufficient | conflicting | error |
| --- | ---: | ---: | ---: | ---: | ---: |
| sufficient | 8 | 0 | 0 | 0 | 0 |
| recoverable | 0 | 4 | 0 | 0 | 0 |
| insufficient | 0 | 4 | 8 | 0 | 0 |
| conflicting | 0 | 0 | 0 | 0 | 0 |

All false decisions are false recoveries: `m2p-013`, `m2p-014`, `m2p-015`, and
`m2p-016` were expected `insufficient` but predicted `recoverable`, each with
`related_but_incomplete` and `missing_required_evidence`. Their exact frozen
evidence IDs and proposed queries are in `policy_decisions.jsonl`.

## Focused end-to-end M1-vs-M2 (3 trials)

- OOD tool execution: M1 `12/12` (`1.0`) to M2 `8/12` (`0.6666666666666666`).
- Expected-answer rate: M1 `17/18` (`0.9444444444444444`) and M2 `17/18`
  (`0.9444444444444444`): no observed decrease.
- Unexpected-abstain rate: M1 and M2 both `1/18` (`0.05555555555555555`).
- The three frozen paraphrase cases (`m1-001`, `m1-002`, `m1-003`) received
  `recoverable` on every trial. Their routes were answer for all except
  `trial-3:m1-003`, which abstained after recovery. Actual Agent-generated
  `proposed_search_query` values are retained per trajectory; production
  metadata-only events still do not persist query text.

## Grounding standalone

| fixture | expected category | observed verdict(s) | disposition |
| --- | --- | --- | --- |
| m2g-001 | supported | supported | accepted |
| m2g-002 | unsupported | unsupported | semantic rejection |
| m2g-003 | contradicted | unsupported | semantic rejection (correct reject, wrong fine-grained verdict) |
| m2g-004 | coverage_missing | supported | false accept |
| m2g-005 | fabricated_citation | none | deterministic citation rejection |

Rates: supported accept `1.0`; unsupported reject `1.0`; contradicted reject
`1.0`; coverage-failure reject `0.0`; fabricated-citation reject `1.0`.
There was one deterministic citation rejection and two semantic-verifier
rejections. The coverage-missing false accept is a semantic-verifier failure.

## Interpretation

Policy, generator, and grounding verifier all used the same configured model.
This is not an independent verification setup. The coverage-missing false accept
is consistent with possible same-model self-verification bias, but this small
evaluation does not establish causality. The M2 empirical hypothesis is
**partially supported**: M2 lowered OOD tool execution without reducing the
observed expected-answer rate, and correctly handled all recoverable standalone
cases, but it still wrongly authorized 4/12 OOD policy cases and did not reject
the coverage-missing grounding fixture.
