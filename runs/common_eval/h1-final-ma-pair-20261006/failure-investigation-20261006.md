# H1 final B0/B2 recovery note — 2026-10-06

## Status at the nine-hour check

- DiagnosisArena B0 completed all 915 frozen IDs: 387 correct (42.30%), parser success 883/915 (96.50%).
- The first DiagnosisArena B2 process wrote 915 rows and its runner manifest says `COMPLETE`, but this is not a valid B2 result. It has 128/915 correct (13.99%) and 597 reasoning-failure rows: 182 `ValueError` classifier parse failures, 413 `APIConnectionError`, and 2 `APITimeoutError`.
- 318 B2 rows had no reasoning-failure flag and a valid MDT route. These can be reused. The remaining 597 rows must be replayed. The original cases, traces, and manifest are retained unchanged at `diagnosisarena-915/B2/`.
- The original supervisor status remained `RUNNING` after its parent process disappeared. It never reached CMB B2. Frozen CMB B0 remains 727/1,024 (71.00%).

## Failure evidence

At inspection, vLLM PID 146755 remained as a frontend process but its `VLLM::EngineCore` child was a zombie, port 8001 had no listener, and the GPU was idle. The run's vLLM log was empty and the frontend stdout was attached to a pipe, so the crash cause is not recoverable from this artifact. Current evidence does not establish OOM or Ninja as the cause; no Oct 6 OOM/Xid entry was found in the kernel log excerpt.

The traces also expose a deterministic parser gap. The classifier prompt lists routes as `1) basic`, `2) intermediate`, and `3) advanced`; Qwen sometimes returns a JSON scalar such as `{"difficulty": "2"}`. `_parse_complexity` previously accepted a bare `2` but rejected that JSON object as `invalid_complexity_output`, before any specialist calls. The parser now maps numeric or named scalar values under `difficulty`, `complexity`, `classification`, or `level` to the corresponding route. The focused `tests/test_adaptive_mdt_adapter.py` suite passed: 5 tests.

The project's isolated test environment was reused at `/root/gpufree-data/health-copilot-harness/.venv`. No package was installed into the shared vLLM environment, whose Python did not have pytest.

## Recovery protocol

`run_final_recovery.py` starts a logged loopback-only vLLM server with the frozen serving settings, reuses only the 318 technically clean DiagnosisArena B2 rows, and replays the other 597 under the corrected classifier parser. It then runs CMB B2 from the frozen 1,024-case manifest. Health checks interrupt a runner if the local vLLM endpoint disappears; retry outputs retain successful case rows and replay only transport-failed rows. B1/B3 remain cancelled. Port 8000 is not touched.

The recovery outputs use separate attempt directories. They do not overwrite the failed first attempt or the frozen B0 artifacts.
