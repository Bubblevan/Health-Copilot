# M3.1 claim-support closeout

Source implementation commit: `4ecf06b`. This closeout records the completed
live claim-support run at `runs/m3/20260920T112419+0800/`.

## Scope of this repair

The capability policy and `evals/m3_capability.jsonl` gold decisions were not
retuned. `data/knowledge_scope.json` gained only explicit capability-mapping
review provenance (`reviewed_at`, `reviewer`), validated as non-empty by the
loader. This metadata is not a clinical-expert validation claim.

M3 ClaimSupportVerifier now receives one API payload containing per-claim
`cited_evidence` only. The runtime materializes this binding after deterministic
citation integrity and before the verifier call. It does not search for or show
uncited observed evidence to a claim. The verifier still uses one API call.

Both M2 and M3 result validators now require exactly one result per claim,
unique indices, and an index set exactly equal to `range(len(claims))`.

## Annotation repair: m3s-007

| field | old | new |
| --- | --- | --- |
| claim | 规律测量血压有助于了解血压是否偏高 | 家庭自测血压时应至少测量两次，每次间隔 1 到 2 分钟 |
| cited source | `who-hypertension-03-silent` | `who-hypertension-03-silent` |
| other observed source | `cdc-high-blood-pressure-managing-01-monitoring` | `cdc-high-blood-pressure-measuring-02-repeat` |
| gold | `UNSUPPORTED` | `UNSUPPORTED` |

The old cited WHO card materially supported its measurement claim, so its gold
was an annotation error. The new WHO citation does not support the count or
interval, while the observed CDC repeat-reading card does; that card is
deliberately not cited. This is an annotation repair, not a label change made
to fit a prediction. Full audit: `evals/m3_claim_support_audit.md`.

## Completed standalone live result

| category | result |
| --- | --- |
| supported | accepted (1/1) |
| unsupported | rejected (1/1) |
| contradicted | rejected (1/1) |
| multi_claim_all_supported | false reject; accepted 0/1 |
| multi_claim_one_unsupported | rejected (1/1) |
| fabricated_citation | deterministic rejection (1/1) |
| wrong_citation_binding | rejected (1/1) |

There was one deterministic citation rejection, five semantic verifier
rejections, and zero verifier errors. The multi-claim-all-supported false reject
is retained as observed; no gold was modified for it.

## Focused M2-vs-M3

The first focused attempts appeared stalled because the evaluator wrote only at
batch completion. TCP connectivity to `api.deepseek.com:443` and a foreground
M3 CLI request both succeeded; after adding progress records, the fresh complete
three-trial run finished at `runs/m3/20260920T114456+0800/`.

| metric | fresh M2 | fresh M3 | frozen M2 baseline |
| --- | ---: | ---: | ---: |
| OOD tool execution | 9/12 (0.75) | 0/12 (0.0) | 8/12 (0.6667) |
| expected answer | 15/18 (0.8333) | 18/18 (1.0) | 17/18 (0.9444) |
| unexpected abstain | 3/18 (0.1667) | 0/18 (0.0) | — |
| mean model turns | 1.6 | 1.3 | — |
| mean tool executions | 0.6 | 0.3 | — |
| mean policy calls | 0.7 | 0.6667 | — |
| mean verifier calls | 0.5667 | 0.6 | — |

M3 had no expected-answer false reject in this stochastic run. All eleven M3
OOD proposals with a policy decision were `INSUFFICIENT` with no matched topic
and no tool execution; the twelfth OOD trajectory abstained before policy. This
is an observed three-trial result, not a guarantee of future improvement.

## Verification

- `ruff check .`: pass
- `pytest -q --basetemp ...`: 102 passed
- `git diff --check`: pass
- M0 frozen baseline: unchanged, safety route 1.0, Hit@1 0.9032, Hit@3 0.9516,
  MRR 0.9274
- M1/M2/M3 mechanics: 52 passed

M4 was not started.
