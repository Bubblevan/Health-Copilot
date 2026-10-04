# MA-MVP1 closeout

## Status

MA-MVP1 runtime, TRAIN_DEV selection, frozen configuration, and full 1,024-episode DEV evaluation are complete. The complete runnable artifacts are stored in `runs/multi_agent/ma-mvp1-20261003-cpu-r2/`.

## Version control

- Implementation and run-artifact commit: `24c97c090a0836d7a2255892d9159379ebca6757`.
- Branch: `multi-agent-ma-mvp1-20261002`.
- Push target: `origin/multi-agent-ma-mvp1-20261002`.

## Baseline identity

- Base stable public capability commit: `8938be0e8e21d030c2790b33cd6a31f8198baf4a` (`origin/rag-e6b-reserved-confirmation-20261002`).
- MA-MVP1 branch: `multi-agent-ma-mvp1-20261002`.
- RAG capability SHA: `8938be0e8e21d030c2790b33cd6a31f8198baf4a` (latest stable E6B result used as the branch base).
- Memory capability status: the public `origin/main` baseline contains the MEM-3B0Q closeout through `367d3e4d0fa592e01de23dd5b91b5e677d3fe516`; no active Memory topic branch was merged. The product runtime uses the owned deterministic longitudinal provider behind `MemoryProvider`.
- U2-F dataset root hash: `e28ea9ef9ecae47d3f27f28c68042066e9af297fe808cafebf1d3c8c80fb2134`.

## Delivered runtime

MA-MVP1 replaces the E2-B fixed-arm frontier with a product-shaped Single fast path and routed Lead/Worker team. Product architecture, skills/agents, and runtime flow are documented under `docs/product/`. `MedicalAgentResponse` is FastAPI-friendly and `create_fastapi_app` exposes answer and aggregate-metrics endpoints when the API extra is installed.

The runtime uses a query router; structured bounded Lead plan; PatientContextAgent, EvidenceAgent, and CareAgent; role-scoped Skills; Harness-generated request/task/worker/tool/evidence IDs; append-only Shared Context events; evidence provenance; concurrent independent work; two-wave dependencies; explicit timeouts/errors; partial synthesis; Single fallback; and final safety/citation checks.

## Evaluation protocol and artifacts

- Run ID: `20261003-cpu-r2`; frozen config `MA_MVP1_CONFIG_V1`; config SHA-256: `d0ae79ac5ad550dc255d55c7e65d0fb50448df46eb9d2534d9272c4e5b22a774`.
- TRAIN_DEV used 256 deterministic TRAIN-only episodes per system; the final evaluation used all 512 DEV_IID and 512 DEV_STRUCTURAL episodes per system.
- Reserved TEST/OOD rows were not accessed or materialized.
- Model artifact: `Qwen3-8B-Q4_K_M.gguf`, SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`; llama.cpp ran CPU-only with zero GPU layers, 24 threads, three server slots, reasoning disabled, temperature 0.
- The run contains 2,048 result rows, 2,048 traces, 1,024 routing records, 561 worker reports, train/dev metrics, full DEV and slice metrics, latency metrics, frozen system config, and a run manifest.

## Quantitative closeout

On all 1,024 paired DEV episodes, Strong Single succeeded on 586 (57.23%) and Routed Medical Team on 589 (57.52%), a net difference of only 3 episodes (+0.29 pp). Paired outcomes were 42 Team-only wins and 39 Single-only wins (943 ties); exact two-sided McNemar p=0.824.

On the 248 COMPLEX episodes, task success was 50/248 (20.16%) for Strong Single and 57/248 (22.98%) for Team (+2.82 pp; paired exact p=0.470). Required fact coverage moved in the opposite direction, from 71.14% to 60.00% (-11.14 pp). On the 400 SIMPLE episodes, task success was identical at 355/400 (88.75%) and required fact coverage was identical at 93.88%.

Routed Team's all-DEV mean latency was 23.64 s versus 9.68 s for Strong Single; P95 was 72.73 s versus 16.40 s. Team used 877,323 tokens versus 389,975 and 1,865 provider calls versus 1,024. The router selected the fast path on 744 cases and activated the Team on 280; exact specialist-set match was 36.23%.

**Conclusion:** This run does not support replacing Strong Single with Routed Medical Team as the default. The Team remains a product-shaped research prototype: its nominal COMPLEX-slice success gain is uncertain and accompanies lower required-fact coverage and substantially higher latency and token use. All results are from the project-owned synthetic universe and do not establish clinical safety or real-patient generalization.

## Verification

- Full repository suite through the repository's `uv` environment: 997 passed, 2 skipped.
- Ruff on the new MA-MVP1 source, runner, and tests: passed.
- Python compileall on the new runtime and runner: passed.
- `git diff --check`: passed.
- Final run manifest records 1,024 DEV episodes per system, frozen config and dataset hashes, CPU-only serving, and `reserved_test_ood_accessed: false`.
