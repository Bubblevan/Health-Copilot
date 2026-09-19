# M3 empirical closeout

Source implementation commit: `b99f455`. This closeout references live runs:

- capability standalone: `runs/m3/20260920T031005+0800/`
- claim-support standalone: `runs/m3/20260920T031131+0800/`
- focused M2-vs-M3, three trials: `runs/m3/20260920T031217+0800/`

Every run config records commit, models, same-model status where applicable,
base URL, temperatures, dataset and scope hashes, pack/scope versions, case and
trial counts, timeout, retries, and bounded runtime limits. No API key is stored.

## Capability hypothesis

The reviewed `hypertension-patient-education-v1` scope has 10 topics and maps all
30 product cards explicitly. It is the closed-corpus source of truth, not a tag
inference or an unsupported-topic blacklist.

Capability standalone (16 frozen states):

- capability policy accuracy: `1.0` (16/16)
- macro-F1: `1.0`
- sufficient / recoverable / insufficient recall: `1.0`, `1.0`, `1.0`
- out-of-scope false-recovery rate: `0.0`
- in-scope false-reject rate: `0.0`
- policy-error rate: `0.0`
- matched-topic exact-set accuracy: `0.875` (14/16)

There were no false decisions. Topic-set differences were non-decision errors:
`m3c-001` returned no optional topic for a sufficient state; `m3c-002` added
`risk_factors` beside expected `lifestyle_prevention`. Gold was not changed.
The anesthesia, air-travel, hair-loss, and insurance hard cases were all
`INSUFFICIENT` with no recovery tool authorization.

## Claim-first contract result

M3 final wire output is claims plus `abstain`; runtime rejects malformed claims,
runs citation integrity before semantic support, requires every claim to be
SUPPORTED, then renders normalized/deduplicated claim text in order. It ignores
any internal free `answer` field. Thus free-text answer/claim coverage mismatch
is removed from the M3 user-visible path by construction; it is not reported as
a probabilistic "coverage accuracy".

Claim-support standalone (7 fixtures):

| fixture | result |
| --- | --- |
| supported | accepted |
| unsupported | semantic reject |
| contradicted | semantic reject |
| multi_claim_all_supported | false reject (first claim marked unsupported) |
| multi_claim_one_unsupported | semantic reject |
| fabricated_citation | deterministic citation reject |
| wrong_citation_binding | false accept (verifier marked it supported) |

Rates: supported accept `1.0`, unsupported reject `1.0`, contradicted reject
`1.0`, multi-claim-all-supported accept `0.0`, multi-claim-one-unsupported
reject `1.0`, fabricated-citation reject `1.0`, wrong-citation-binding reject
`0.0`. The last two semantic failures are verifier-quality limitations; runtime
contract tests still fail closed for an unobserved supporting source or a support
source outside a claim's cited IDs.

## Focused M2-vs-M3, three trials

The frozen M2 baseline was OOD tool execution `8/12` (`0.6667`) and expected
answer rate `17/18` (`0.9444`). This fresh paired live run observed:

| metric | M2 arm | M3 arm |
| --- | ---: | ---: |
| OOD tool execution | 11/12 (0.9167) | 0/12 (0.0) |
| expected answer | 16/18 (0.8889) | 15/18 (0.8333) |
| unexpected abstain | 2/18 (0.1111) | 3/18 (0.1667) |
| mean model turns | 1.6667 | 1.2667 |
| mean tool executions | 0.6667 | 0.2667 |
| mean policy calls | 0.7000 | 0.6333 |
| mean verifier calls | 0.6000 | 0.5667 |

M3 produced no OOD tool execution. Eleven OOD trajectories received explicit
`INSUFFICIENT` with empty matched topics; the remaining one abstained without a
proposal. No new Agent tool or extra model round trip was added. The observed
lower expected-answer rate means the end-to-end answer-preservation part of the
capability hypothesis is not supported in this small run.

A successful M3 trajectory is `trial-1:m1-001`: a recoverable query matched
`diagnostic_confirmation` and `home_monitoring`, executed the sole search tool,
verified its final claims, and materialized only those claims. A corpus-uncovered
veto is `trial-1:m1-007`: policy returned `INSUFFICIENT`, matched no topic, and
did not execute the tool. `m3s-002` demonstrates unsupported-claim rejection.

M3 false rejects were `trial-1:m1-002` and `trial-3:m1-002` (claim-support
failure after a recoverable search), plus `trial-1:m1-003` (model abstained
without a policy decision). The fine-grained verifier failures are the
multi-claim false reject and wrong-citation-binding false accept above.

## Same-model and conclusion

The focused generator, policy, and verifier all used `deepseek-flash`;
`same_model_verification=true`. Structural materialization must not be attributed
to model independence. M3 is **PARTIALLY_SUPPORTED**: explicit scope eliminated
observed standalone false recoveries and reduced OOD execution to zero in this
run, while claim-first removes free-answer coverage mismatch structurally; but
answerable-route degradation and semantic verifier false decisions remain.

M4 was not started.
