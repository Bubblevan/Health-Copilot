# MA-MVP1 protocol

## Objective

Compare a Strong Single Agent with the final Routed Medical Team on the frozen project-owned longitudinal universe. This is a product closeout, not a multi-architecture ablation.

## Systems

- `STRONG_SINGLE`: one model call after Harness skill execution; its available capability surface is the union of all worker skills.
- `ROUTED_MEDICAL_TEAM`: query router, structured Lead plan, three specialist workers, Shared Context, Evidence Ledger, parallel execution, synthesis, and Harness verification.

The systems share the same local Qwen3-8B Q4_K_M model, deterministic owned-world tools, time snapshot, evaluator, and external world. llama.cpp runs CPU-only with zero GPU layers and reasoning disabled for bounded structured tasks. The provider adds Qwen3's `/no_think` marker. Retry policy is disabled.

## Data and isolation

- Dataset: `health-copilot-owned-longitudinal-v1`; root hash `e28ea9ef9ecae47d3f27f28c68042066e9af297fe808cafebf1d3c8c80fb2134`.
- Engineering sample: 256 deterministic SHA-256-selected episodes from TRAIN only.
- Final evaluation: all 512 DEV_IID and 512 DEV_STRUCTURAL episodes per system.
- Reserved TEST/OOD rows are not loaded or materialized.
- Runtime receives only the U2-F runtime episode and the corresponding patient/evidence providers. Evaluator truth, required IDs/facts, scenario family, and split labels stay in the scoring process.
- One engineering round is permitted by this frozen configuration. Any change after TRAIN_DEV requires a new recorded round and a new config hash; final DEV starts only after `MA_MVP1_CONFIG_V1` is frozen.

## Metrics

Primary metrics are Task Success, Grounded Task Success, and Required Fact Coverage, reported for ALL DEV, COMPLEX, and SIMPLE. COMPLEX is evaluator-side scenario metadata for `MEMORY_EXTERNAL_JOIN`, `MEMORY_EXTERNAL_CONFLICT`, `EXTERNAL_MULTI_SOURCE`, `MEMORY_MULTI_RECORD`, and `COMPOSITIONAL_MULTI_FACT`; SIMPLE is `CURRENT_ONLY`, `MEMORY_LOOKUP`, and `EXTERNAL_LOOKUP`.

Secondary metrics include answer accuracy, external evidence coverage, patient-state fact coverage, grounding pass, correct abstention, and citation validity. Routing metrics use the evaluator-side capability requirement and the runtime's predicted specialist set. Single fast-path rate means the query router selected Single directly. Router Team-candidate rate means the query router passed the request to the Lead. Team activation rate means the Lead actually selected Team and at least one Worker was run. Engineering metrics also include workers per Team request, completion/usefulness/incorporation, latency P50/P95, calls/tokens, parallel speedup, and partial-failure recovery.

Metric denominators and per-episode data are preserved in the final run artifacts. No résumé wording is supplied until the actual final run is complete.

## Commands

Use the repository's uv project and the dedicated loopback-only CPU llama.cpp server on port 8083:

```powershell
uv run python tools/research/multi_agent/run_ma_mvp1.py --stage train-dev --run-id 20261003-cpu-r2 --base-url http://127.0.0.1:8083/v1
uv run python tools/research/multi_agent/run_ma_mvp1.py --stage freeze --run-id 20261003-cpu-r2 --base-url http://127.0.0.1:8083/v1
uv run python tools/research/multi_agent/run_ma_mvp1.py --stage final-dev --run-id 20261003-cpu-r2 --base-url http://127.0.0.1:8083/v1
```

The final command validates the frozen config hash before it opens DEV rows.
