# MEM-3B0Q-R3 Concurrent Runtime Capacity Smoke

Date: 2026-10-01

Status: `RUNTIME_LOAD_PASS`; no pilot proposal inference was sent.

- The frozen Windows llama.cpp server started on `127.0.0.1:8081` as PID `60420`, while the pre-existing service on `127.0.0.1:8092` (PID `62696`) was left running and unchanged.
- The 8081 process served the frozen Qwen3-8B Q4_K_M artifact (`d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`) using llama.cpp `10068 (571d0d540)`; binary SHA-256 `3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb`.
- Actual 8081 props reported context `131072`. Launch arguments included 99 GPU layers, Flash Attention, Q4_0 GPU K/V cache, YaRN scale 4, and one parallel slot.
- `/health` returned `ok` for both 8081 and 8092 after the second server loaded.
- NVIDIA RTX 4090 Laptop snapshot after both were loaded: `15,905 MiB / 16,376 MiB` used, leaving `471 MiB` reported free.
- Preflight attempt 001 was blocked because the sandboxed Python child could not query Windows process metadata. Attempt 002, run with local process-query permission, passed the frozen model/build/argument/context checks. Both attempts recorded zero model-generation calls.
- No chat completion, pilot proposal, hosted/API call, or MemoryStore mutation occurred in this capacity check.

Interpretation: the two configured services can both load and report healthy at this snapshot, so the 131K frozen 8081 model fits concurrently with 8092. The remaining VRAM headroom is small; this does not establish inference throughput or stability under concurrent generation. Keep other GPU workloads idle during the one-request pilot and record any allocation/runtime failure as infrastructure, not method quality.
