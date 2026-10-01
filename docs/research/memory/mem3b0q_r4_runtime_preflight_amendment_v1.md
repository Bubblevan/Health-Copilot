# MEM-3B0Q R4 Runtime Preflight Source Amendment v1

Status: `REVIEW_REQUIRED`; applies only to runtime-preflight evidence for the frozen R4 qualification. The original protocol and freeze manifest remain unchanged.

## Finding

The frozen protocol's runtime table names `/slots` as the source for sampler defaults before and after each completion. The pinned llama.cpp `10068 / 571d0d540` server does not expose sampler defaults in its `/slots` payload. The idle slot response reports slot/runtime state such as slot id, processing state, and context length; it has no `params` object. A runner that reads sampler values from `/slots` therefore cannot verify the frozen defaults and must not treat missing values as a mismatch or silently substitute an unverified source.

## Authoritative Mapping

For this pinned server only, read global default sampler settings from `/props.default_generation_settings.params`. The pinned `get_props` implementation copies `params.sampling` into `task_params.sampling`, serializes that object into `default_generation_settings.params`, and publishes the effective slot context as `default_generation_settings.n_ctx` ([pinned `server-context.cpp`](https://github.com/ggml-org/llama.cpp/blob/571d0d540df04f25298d0e159e520d9fc62ed121/tools/server/server-context.cpp#L4112-L4157)). The frozen request builder sets temperature `0`, seed `42`, and `max_tokens=256`; it does not override `top_k`, `top_p`, `min_p`, or `repeat_penalty` ([`mem3b0q_r4.py`](../../../tools/research/memory/mem3b0q_r4.py)). Thus the four frozen global defaults remain applicable to this request.

The runner must check these fields from `/props.default_generation_settings.params`:

| Field | Frozen value |
| --- | ---: |
| `top_k` | `40` |
| `top_p` | `0.95` |
| `min_p` | `0.05` |
| `repeat_penalty` | `1.0` |

Continue using `/slots` for the separate facts it actually exposes: exactly one slot, idle before the request and after completion, and `n_ctx=131072`. Continue checking `/props.default_generation_settings.n_ctx=131072`, served model/build identity, loopback-only process identity, binary/model hashes, and frozen launch arguments.

## Preflight And Postflight

Immediately before the request, `/props` is authoritative for launch-time global sampler defaults and `/slots` is authoritative for slot count, idle state, and slot context. Immediately after the request, repeat both observations and require them to satisfy the same frozen values; additionally require the process snapshot to remain identical. Postflight does not claim that `/slots` exposes request-resolved sampler parameters or proves a slot-local parameter snapshot. The exact request body and its SHA-256 remain the evidence for per-request overrides; the correlated server trace remains the evidence for sampler initialization.

Any absent/malformed `/props` sampler field, default mismatch, `/slots` shape/state mismatch, process change, or endpoint error is a fail-closed preflight/runtime failure. Do not infer defaults from `/slots`, user-facing UI settings, or remembered launch configuration.

## Scope And Freeze

This is an endpoint-field/source mapping clarification only. It does not alter model weights, launch arguments, prompts, request body, JSON schema, call order/count, retry policy, quality criteria, or acceptance gates. This amendment does not authorize inference by itself. Its exact bytes are SHA-256-pinned in the separate R4 gate runner lock. The original `mem_3b0q_r4_protocol_v1.md`, freeze manifest, and manifest sidecar must remain byte-identical.
