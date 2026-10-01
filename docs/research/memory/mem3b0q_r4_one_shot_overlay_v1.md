# MEM-3B0Q R4 One-Shot Strict Sampler-Gate Overlay v1

Status: `FROZEN_AFTER_OFFLINE_REVIEW`; invocation blocked by the pre-existing R4-P01 gate artifact.

## Boundary

The existing R4-P01 frozen runner remains byte-identical. This overlay wraps it in memory for one pre-registered `R4-P01` pass-1 sampler-init request. The command requires the exact, unabbreviated `--run-p01-sampler-init-gate` flag (`argparse` abbreviation is disabled); importing the module, omitting the flag, or passing an abbreviated flag does not invoke the runner.

The overlay temporarily replaces the runner's `_validate_preflight` callback so its existing preflight and postflight checkpoints first run the strict read-only shape/type validator and then the original frozen checks. It also wraps `HTTPConnection.request` for the duration of the call:

- only GETs to `/health`, `/props`, `/v1/models`, and `/slots` on `127.0.0.1:8081` are allowed;
- before the single permitted POST, it takes a fresh process snapshot and fresh GET service snapshot, validates them strictly, and requires the process snapshot to match the runner's preflight snapshot;
- the only permitted POST path is `/v1/chat/completions`, the body must byte-match the independently rebuilt frozen P01 request, and a second POST is rejected before forwarding;
- all patches are restored in `finally`.

The separate overlay lock pins the overlay source hash, frozen runner hash, frozen protocol manifest hash, runtime-amendment hash, `R4-P01`, and `max_completion_posts=1`. The CLI fails closed until that lock is frozen and valid. Offline reviewer approval is limited to one exact P01 sampler-init gate invocation after lock validation; it does not authorize retries, the 40-call pass, MemoryStore mutation, B1, benchmark execution, or performance claims.

## Offline Tests

Tests use a fake `HTTPConnection` and synthetic snapshots; no endpoint is called. They prove that malformed fresh pre-POST state prevents forwarding, malformed postflight state fails after at most one forwarded POST, a second POST/body/path/origin is blocked, successful simulated execution uses strict checks at preflight/boundary/postflight, frozen files remain byte-identical, and missing/abbreviated CLI flags cannot invoke the runner. Targeted offline suite: 33 passed.

This overlay only addresses runtime enforcement for the sampler-init gate. The reviewer approved exactly one invocation if the preconditions remained clear, with no retries or additional calls. The final invocation-time guard found that the frozen gate directory already existed and refused before loading the lock, taking snapshots, or issuing a request. The lock was validated separately offline. No request was made by this overlay.

## Existing Gate Artifact and Stop

The gate directory `runs/memory/mem3/mem3b0q-r4-identity-qualification-v1/pass1_r4-p01_sampler_init_gate/` predated the overlay invocation and contains a completed request record timestamped 2026-10-01 15:01:47 to 15:02:34. Its frozen record shows one request attempt, HTTP 200, zero retries, zero hosted calls, `sampler_gate=INFRA_FAILURE_UNCORRELATED_EVIDENCE`, and `quality_failure=R4ContractError:owner_span:unsupported_exact_alias`. There is no `strict_overlay_manifest.json`. The response SHA-256 matches the pre-existing failure analysis. The attempted overlay command exited with `gate_attempt_directory_exists_refusing_retry` (exit code `1`, forwarded completion POSTs `0`) before any endpoint access; the existing directory and evidence were not modified. The separately validated overlay-lock SHA-256 is `6c56fdc88897074b88b34d682ebb411f191e2caf3cfd34f5e16800a3cadb30eb`.

This means the pre-registered P01 request has already been consumed by the earlier unguarded attempt. Do not retry or reinterpret it as overlay-qualified. R4 v1 remains `NO`, sampler evidence remains `UNVERIFIED`, the model output remains a `QUALITY_FAILURE`, and `MEM3B0Q_MEM3B1_READY=NO`. Any next inference must belong to a separately versioned, reviewed, frozen method/protocol with new authorization; the candidate-bounded offline proposal is a possible direction, not approval to run it.
