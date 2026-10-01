# MEM-3B0Q R4 Candidate Schema Preflight v1.1

Status: `PASS_SCHEMA_FILE_PARSE_ONLY`. The candidate-bounded method remains `PROPOSAL_ONLY`; no inference was run.

## Evidence

- Frozen R4 control cases: 20/20 generated schemas pass Draft 2020-12 validation and proceed through the pinned llama.cpp CLI schema-file parsing stage to the expected missing-model failure.
- Negative control: deliberately malformed JSON schema is rejected by the pinned CLI before the missing-model failure.
- Six cases with at least one missing typed candidate use `atoms.maxItems=0`; no empty `enum` is emitted.
- Deterministic gold-response adapter round-trip: all 20 frozen control cases pass, including all abstention cases.
- Pinned CLI: llama.cpp `10068 (571d0d540)`, SHA-256 `48566cf6e2969464b799dbcac7393b3549f9efd6884688074dc803125dbafa85`.
- Preflight runner SHA-256: `a5570918782303345fc7547f390a7978c15ca7f90a720786d40aba55dd3257dc`.
- Candidate builder SHA-256: `361c0e8d1aae1020f8ad76f973c586bd17bf7c19a1d71482e6c1d36c42cf243d`.
- Detailed per-case evidence: [mem3b0q_r4_candidate_schema_preflight_v1_1.json](mem3b0q_r4_candidate_schema_preflight_v1_1.json), SHA-256 `98768cc9da38389c6eccd6f4278b02150b815c4ae4e763cedbd80bf600bd1e29`.

## Interpretation

The CLI was given an intentionally nonexistent GGUF path. It did not load a model, initialize a sampler, call local port 8081 or 8092, make a hosted request, or generate an answer. The malformed-schema negative control demonstrates that schema-file parsing occurred before the expected missing-model failure. This does **not** prove valid schemas were converted into grammar or that a sampler accepts them; both grammar conversion and sampler initialization remain unverified.

An initial run artifact, [mem3b0q_r4_candidate_schema_preflight_v1.json](mem3b0q_r4_candidate_schema_preflight_v1.json), records a detector false positive: the intentionally missing model filename contained `schema`, so the first classifier marked the otherwise expected model-load failure as a schema error. The classifier was corrected, a negative control was added, and the failed attempt was retained rather than overwritten.

## Gate State

- `R4_CANDIDATE_SCHEMA_FILE_PARSING=PASS`
- `R4_CANDIDATE_GRAMMAR_CONVERSION=NOT_VERIFIED`
- `R4_CANDIDATE_SAMPLER_INIT=NOT_VERIFIED`
- `R4_CANDIDATE_MODEL_QUALITY=NOT_RUN`
- `MEM3B0Q_MEM3B1_READY=NO`

This is not a benchmark result and does not authorize B1, a model request, revision materialization, or a public performance claim. Frozen R4 v1, B0P/B0Q artifacts, and the scorecard protocol remain unchanged.
