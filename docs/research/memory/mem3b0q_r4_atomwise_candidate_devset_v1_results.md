# MEM-3B0Q R4 Atomwise Candidate Devset v1 Results

Status: `COMPLETE`; immutable run artifacts are under
`runs/memory/mem3/mem3b0q-r4-atomwise-candidate-devset-v1/`.

## Result

This is a six-case, project-authored, prompted synthetic development
diagnostic. It is not a public benchmark, unbiased generalization estimate,
LongMemEval result, temporal-memory result, or clinical result. Five cases used
one local Qwen3-8B request each; R4C-05 replays its earlier one-shot response.
There were no hosted calls, retries, MemoryStore mutations, or infrastructure
failures.

| Case | Matched / expected | Admitted | Precision | Recall | Outcome / observed failure |
|---|---:|---:|---:|---:|---|
| R4C-01 | 2/2 | 2 | 100% | 100% | Exact match |
| R4C-02 | 0/3 | 0 | n/a | 0% | JSON truncated at the 256-token completion cap (`finish_reason=length`) |
| R4C-03 | 1/2 | 1 | 100% | 50% | Pears atom proposed as `SINGLE_VALUE_AT_A_TIME`; frozen slot policy requires multi-value membership |
| R4C-04 | 1/2 | 1 | 100% | 50% | La Paz retained; bicycle purchase event failed the frozen cardinality/value contract |
| R4C-05 | 1/1 | 1 | 100% | 100% | Prior response replay; tablet-to-laptop cross-binding quarantined |
| R4C-06 | 0/1 | 0 | n/a | 0% | Fruit membership proposed as single-valued; basket distractor also failed cross-clause binding |
| **Aggregate** | **5/11** | **5** | **100%** | **45.5%** | **2/6 exact cases; 4 quality failures** |

The strict document-level comparison admitted 2/11 expected atoms (18.2%).
Atomwise quarantine admitted 5/11 expected atoms (45.5%), a diagnostic
increase of 27.3 percentage points while retaining 100% precision on this
small prompted set. The raw runner field named
`valid_atom_retention_vs_strict` actually contains aggregate atom recall; this
report computes the strict-vs-atomwise comparison explicitly rather than
reusing that misleadingly named field.

## Failure Attribution

- R4C-02 is a generation-capacity/truncation failure, not evidence of a
  semantic extraction error. It remains a failed case in the denominator.
- R4C-03 and R4C-06 expose a contract-design issue: the model is asked to emit
  a cardinality value that is already deterministically defined by the frozen
  `(object_id, attribute_id)` slot policy. R4C-04 has the same field mismatch
  in addition to event-value grounding requirements.
- R4C-04's event span/value issue and R4C-06's basket cross-clause issue are
  separate from cardinality and must remain visible after any contract change.
- The tests are explicitly prompted with distractors and do not establish
  performance on naturally occurring conversation histories.

The five new requests took 127.989s, 160.461s, 130.983s, 126.429s, and
129.323s; median 129.323s. This local runtime is recorded as operational
diagnostic evidence, not a speed claim.

## Next Development Iteration

For a separately versioned development iteration, the model will emit only
candidate-bounded owner/object/attribute IDs and a source value span. A
deterministic adapter will derive cardinality from the frozen slot policy
before the unchanged v1/v2/v3 validators run. The completion cap remains 256
for this isolated change. R4C-02 truncation, event semantics, and cross-clause
binding remain independent failure modes. This is development-set adaptation;
no new generalization claim follows from reusing these six cases.

## Artifact Integrity

`run_manifest.json` hashes every run artifact except itself. The frozen
protocol/lock and original one-case result remain unchanged. The exact raw
responses and per-case results are retained with the run.
