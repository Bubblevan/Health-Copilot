# MEM-3B0Q R4 Read-Only Runtime Preflight v1

Status: `PROPOSAL_ONLY`; this entrypoint takes process and HTTP GET snapshots only. It does not authorize or issue inference.

## Purpose

The frozen R4 request runner is intentionally not used for readiness checks because its successful runtime preflight is followed by a POST to the completion endpoint. This separate entrypoint reuses the frozen runner's pinned manifest, process snapshot, service GET snapshot, and source mapping, while applying stricter fail-closed type/shape validation first.

## Strict Checks

- `/slots` must be a one-element JSON list. Its sole item must be an object with integer `n_ctx=131072` and boolean `is_processing=false`.
- `/props.default_generation_settings.n_ctx` and `total_slots` must be exact integers, not booleans.
- `top_k` must be integer `40`; `top_p`, `min_p`, and `repeat_penalty` must be finite JSON numbers (not booleans or strings) within the frozen tolerance.
- `/v1/models.data` must contain exactly the frozen model id.
- After strict checks, the entrypoint applies the frozen runner's remaining process/model/build/loopback/launch-argument checks.
- Endpoint observations are GET-only (`/health`, `/props`, `/v1/models`, `/slots`). No POST method or chat-completion code path is invoked.

Tests cover `NaN`, infinity, numeric strings, booleans masquerading as numbers, malformed slot shapes, null/non-boolean processing state, duplicate slots, and the GET-only entrypoint call path. The validator cannot prove live process identity without running the Windows process and GET snapshots; those checks are deferred until this offline review gate passes.

The frozen R4 protocol, freeze manifest, runner, and runtime-source amendment are unchanged. The candidate-binding guard remains a separate offline heuristic and is not part of this runtime gate.
