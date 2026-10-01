# MEM-3B0Q Candidate-Bounded Gap Audit v1

Status: `OFFLINE_DIAGNOSTIC_ONLY`; no model request, protocol amendment, or B1 authorization.

## Scope

This audit checks what the existing candidate-bounded proposal validator actually guarantees after the consumed R4-P01 attempt. It does not change the frozen R4 v1 result, control pack, alias registry, candidate builder, or the separate P15 locality guard.

The archived P01 request remains one consumed request: HTTP 200, zero retries, `sampler_gate=INFRA_FAILURE_UNCORRELATED_EVIDENCE`, `quality_failure=R4ContractError:owner_span:unsupported_exact_alias`. The sampler prompt witness was absent, so initialization is `UNVERIFIED`, not proven failed. The guarded overlay later refused before network access because the attempt directory already existed. Do not retry R4-P01.

## Existing Offline Coverage

The candidate builder, P15 binding guard, and factorized-admission targeted suites pass `43/43` tests. On the frozen 20-case control pack, 14 cases have at least one candidate for all three typed fields; those cases contain 15 raw owner × object × attribute tuples across 15 expected atom rows. Six cases have a missing typed field and therefore cannot emit an atom under the generated schema. This is a small, closed fixture inventory; the count is not a generalization or accuracy metric.

Candidate-bounded v1 does establish that the model cannot invent an owner/object/attribute alias outside each case's generated field enum. Candidate IDs resolve deterministically to exact spans and typed IDs. The P15 guard adds a clause-local attribute/value heuristic. None of these facts proves that independently selected fields belong to the same semantic atom.

## Reproduced Residual Failures

### Cross-Clause Field Binding

Offline source:

```text
My work laptop's operating system is Linux, and my personal laptop's operating system is Windows.
```

The candidate builder exposes distinct owner/object/attribute occurrence IDs. A proposal selecting `My`, object `work laptop`, the second `operating system` occurrence, and value `Windows` is accepted by both the candidate validator and the current clause-local attribute/value guard. It returns a candidate type-level slot key `SELF / WORK_LAPTOP / OPERATING_SYSTEM`, although the selected attribute and value belong to the personal-laptop clause. The current guard checks attribute/value clause agreement only; it does not bind owner/object/attribute as one tuple.

### Cardinality Is Still Model-Asserted

For frozen `R4-P09`, the source-grounded value is `apples and pears` under the `FAVORITE_FRUIT_SET / MEMBERSHIP` slot. The oracle labels this `MULTI_VALUE_CONCURRENT`, but the current candidate validator accepts the same spans with `cardinality_proposal=SINGLE_VALUE_AT_A_TIME` and returns a slot candidate. Schema membership validates the enum, not its semantic correctness.

### Object Type Is Not Object Instance

Occurrence-specific candidate IDs preserve distinct offsets, but resolution uses the alias registry's canonical object type in `slot_key`. Two mentions of `wallet` for the same owner therefore resolve to the same `WALLET` slot. The current representation does not establish whether those mentions refer to the same physical wallet or two distinct wallets.

### Duplicate, Update, and Deletion Remain Downstream Questions

The candidate proposal schema has no `ADD`, `UPDATE`, `DELETE`, or `NOOP` action and does not provide temporal validity. It cannot distinguish repeated evidence of an unchanged value from a changed state or deleted fact. That must remain a separate revision-admission/materialization problem; it is not solved by typed span candidates.

## Research Disposition

Candidate-bounded extraction remains a useful authority-reduction technique for exact typed spans, but it is not a complete atom-binding or revision method. Do not describe it as semantic extraction accuracy, revision safety, or evidence that stale reuse is reduced.

Before a new, separately versioned inference protocol is considered, an offline method must address and test:

- complete owner/object/attribute/value binding, including cross-clause counterexamples;
- ambiguous same-type object mentions, with conservative abstention where instance identity is unresolved;
- cardinality constraints against multi-value/set examples;
- duplicate/no-op versus changed-value update and delete semantics at the revision layer;
- all 20 frozen controls plus explicit adversarial negative controls, with frozen gold and no post-result tuning.

Any next method needs a fresh review and separately authorized bounded inference. This audit authorizes none. Current gates remain `R4_V1=NO`, `MEM3B0Q_CANDIDATE_BOUNDED_EXTRACTOR_READY=NO`, and `MEM3B0Q_MEM3B1_READY=NO`.
