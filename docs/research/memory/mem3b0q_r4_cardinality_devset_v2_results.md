# MEM-3B0Q R4 Cardinality-Authority Devset v2 Results

Status: `COMPLETE`; six fresh one-shot local requests and all raw artifacts are
under `runs/memory/mem3/mem3b0q-r4-cardinality-devset-v2/`.

## Main Result

| Case | Matched / expected | Admitted | Outcome |
|---|---:|---:|---|
| R4C-01 | 2/2 | 2 | Exact |
| R4C-02 | 3/3 | 3 | Exact; completed below the same 256-token cap |
| R4C-03 | 2/2 | 2 | Exact; Harness-derived multi-value membership |
| R4C-04 | 1/2 | 1 | La Paz retained; purchase-event value rejected as unproven |
| R4C-05 | 1/1 | 1 | Exact; tablet-to-laptop cross-binding quarantined |
| R4C-06 | 1/1 | 1 | Exact; apples retained, cross-clause oranges quarantined |
| **Aggregate** | **10/11** | **10** | **100% precision; 90.9% recall; 5/6 exact cases** |

All six requests were fresh. There were six local posts, zero retries, zero
hosted calls, zero infrastructure failures, zero truncations, and zero
MemoryStore mutations. The unchanged validators retained 100% precision on
the admitted atoms. The sole missed gold atom was the bicycle purchase event in
R4C-04: the model selected `last month` as its value; the frozen event guard
requires the event predicate and quarantined it as
`event_value_unproven`. That is the next measured failure to address.

## Development Comparison

| Diagnostic | Exact cases | Matched atoms | Precision | Recall | Truncated cases |
|---|---:|---:|---:|---:|---:|
| v1 model-proposed cardinality | 2/6 | 5/11 | 100% | 45.5% | 1 |
| v2 Harness-derived cardinality | 5/6 | 10/11 | 100% | 90.9% | 0 |
| v1 raw-output offline counterfactual* | 4/6 | 7/11 | 100% | 63.6% | 1 |

`*` Counterfactual only: old v1 proposals had the cardinality field removed,
then were passed through the v2 deterministic policy adapter and unchanged
guards. It is not a fresh model run and is not counted as an independent
result.

The real v2 request run adds five matched atoms over v1's observed 5/11:
three facts in the formerly truncated R4C-02 response, the pears membership in
R4C-03, and the apples membership in R4C-06. This is a development-set result
on six authored prompted
controls, after inspecting v1; it must not be generalized to natural dialogue
or presented as public benchmark performance.

## Runtime Diagnostics

| Case | Reader prompt tokens | Completion tokens | Local latency |
|---|---:|---:|---:|
| R4C-01 | 378 | 166 | 107.441 s |
| R4C-02 | 393 | 231 | 145.536 s |
| R4C-03 | 388 | 175 | 112.158 s |
| R4C-04 | 378 | 167 | 107.249 s |
| R4C-05 | 341 | 166 | 107.110 s |
| R4C-06 | 340 | 171 | 110.487 s |

Median local latency was 108.964 s. V1's five new requests had median
129.323 s; these observations are descriptive only because local GPU load was
not experimentally controlled. For the five case IDs newly posted in v1, v2
used 910 completion tokens versus 1,062 in v1; C02 no longer hit the cap. Token
and latency differences are local diagnostic accounting, not a benchmark
efficiency claim.

## Interpretation and Next Iteration

The evidence supports a specific engineering conclusion: cardinality is
deterministic slot metadata, so requiring the language model to predict it
creates avoidable schema failures and consumes output capacity. Moving that
choice to a closed Harness-owned `(object, attribute)` policy repaired the
multi-valued fruit cases, removed the observed completion truncation, and
preserved the cross-scope/cross-clause quarantine behavior in this small set.

The remaining failure is narrower: event predicate vs. temporal expression.
Next iteration should explicitly separate *what happened* from *when it
happened*, without weakening `event_value_unproven`. The current experiment
does not evaluate revision materialization, CURRENT/AS_OF/CHANGE, stale-memory
suppression, public LongMemEval, Memora, MedMemoryBench, or medical safety.

## Integrity

The run manifest covers 32 artifacts; all hashes verified. The v1 raw run,
protocol, and results remain unchanged. Both versions are development
diagnostics and not independent evaluations.
