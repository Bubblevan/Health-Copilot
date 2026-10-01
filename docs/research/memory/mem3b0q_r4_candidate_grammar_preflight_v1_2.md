# MEM-3B0Q R4 Candidate Schema Grammar Preflight v1.2

Status: `PASS_GRAMMAR_CONVERSION_PRE_MODEL_LOAD`. Candidate-bounded admission remains `PROPOSAL_ONLY`; sampler initialization and model quality are not verified.

## Evidence

- The preserved v1.1 run contains 20/20 generated candidate schemas reaching the expected missing-model failure, with zero schema errors. This includes the six zero-candidate schemas using `atoms.maxItems=0`.
- The v1.1 JSON report sidecar matches. Report SHA-256: `98768cc9da38389c6eccd6f4278b02150b815c4ae4e763cedbd80bf600bd1e29`.
- Pinned CLI in that run: `10068 (571d0d540)`, SHA-256 `48566cf6e2969464b799dbcac7393b3549f9efd6884688074dc803125dbafa85`.
- This turn's CLI `--version` output also reports `10068 (571d0d540)`. Its output SHA-256 is `c2fab25eddd59f96c453c4228bbd9597996719537aa11c1893a2bff313ff6598`.
- Conversion negative control used valid JSON with an unresolved local `$ref`: `{"$ref":"#/definitions/missing","definitions":{}}`. It exited 1 with `JSON schema conversion failed: Error resolving ref #/definitions/missing` before any missing-model diagnostic. Output SHA-256: `c03414b68cb6d8a7f1081375563d3597030a0accac1c7e727994cda5094055d5`.
- The negative control used `--json-schema`; the 20 positive cases used `--json-schema-file`. Both pinned CLI handlers call the same `json_schema_to_grammar` converter.

## Source-Order Basis

At pinned upstream commit [`571d0d540df04f25298d0e159e520d9fc62ed121`](https://github.com/ggml-org/llama.cpp/tree/571d0d540df04f25298d0e159e520d9fc62ed121), the [`--json-schema-file` argument handler](https://github.com/ggml-org/llama.cpp/blob/571d0d540df04f25298d0e159e520d9fc62ed121/common/arg.cpp#L1978-L2000) reads JSON and synchronously calls `json_schema_to_grammar`. The [`llama_cli` entry point](https://github.com/ggml-org/llama.cpp/blob/571d0d540df04f25298d0e159e520d9fc62ed121/tools/cli/cli.cpp#L30-L59) completes `common_params_parse` before constructing and initializing `cli_context`.

Therefore, the 20 successful cases reaching the later missing-model error establish that their schemas passed grammar conversion before model loading. The unresolved-reference negative control confirms that a conversion error exits during that earlier stage. This is converter evidence, not sampler evidence.

## Audit Corrections And Limits

- The v1.1 Markdown narrative lists runner SHA-256 `a5570918782303345fc7547f390a7978c15ca7f90a720786d40aba55dd3257dc`. Its machine-readable JSON records `fd56a087d4e4340dc5c79c1ddea81c86e3890adff31429cc7502d11e5a0778a4`, which matches the current runner bytes. The original v1.1 files are preserved unchanged; this supplement treats the machine report and matching source bytes as authoritative for runner identity.
- The current process can launch the WinGet executable and verify its version, but cannot read the executable bytes due to Windows access denial. Thus the current run does not revalidate its SHA-256; the exact binary hash is inherited from v1.1's recorded execution, not independently re-established here.
- No model was loaded, no sampler was initialized, no generation ran, and no local endpoint or hosted API was called.

## Gate State

- `R4_CANDIDATE_SCHEMA_FILE_PARSING=PASS`
- `R4_CANDIDATE_GRAMMAR_CONVERSION=PASS_PRE_MODEL_LOAD`
- `R4_CANDIDATE_SAMPLER_INIT=NOT_VERIFIED`
- `R4_CANDIDATE_MODEL_QUALITY=NOT_RUN`
- `MEM3B0Q_MEM3B1_READY=NO`

This supplement does not authorize a model request, sampler qualification, B1, revision materialization, benchmark execution, or public performance claims. Frozen R4 v1, B0P/B0Q artifacts, and the scorecard protocol remain unchanged.
