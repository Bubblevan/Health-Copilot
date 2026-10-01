# MEM-3B0Q-R2 Runtime Preflight

Status: `PREFLIGHT_BLOCKED`. The frozen identity pilot has not generated any model output.

- Expected endpoint: `127.0.0.1:8081`, pinned llama.cpp `10068 (571d0d540)`, 131,072 context, 99 GPU layers, Flash Attention, Q4_0 GPU KV.
- Observed: no reader listener at `127.0.0.1:8081`; the active local `llama-server` was on port `8092`, with 65,536 context, Vulkan device, and `--n-gpu-layers all`.
- GPU snapshot at preflight: NVIDIA GeForce RTX 4090 Laptop GPU, 7,691 MiB used of 16,376 MiB, 7% utilization.
- Preflight attempt: `runtime_preflight_attempt_001.json`; failure was `expected_exactly_one_loopback_reader`.
- Model generation calls: `0`; hosted calls: `0`; API key required: `false`; MemoryStore mutations: `0`.

The 8092 process was not stopped, reconfigured, or used for the pilot. Starting a second 131K-context server beside it was avoided to prevent resource contention and preserve the frozen runtime. This is an infrastructure precondition failure, not a negative identity-method result.

The correct next action is to make the pinned `8081` runtime available with the required arguments, then rerun `preflight`. No prompt, selection, model output, or admission rule may change. `MEM3B0Q_SPAN_IDENTITY_PILOT=NOT_RUN`; `MEM3B1_READY=NO`.
