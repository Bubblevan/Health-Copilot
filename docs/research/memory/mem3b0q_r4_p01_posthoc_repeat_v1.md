# MEM-3B0Q R4-P01 Post-Hoc Diagnostic Repeat v1

Status: `COMPLETED_NON_QUALIFYING_POSTHOC_DIAGNOSTIC`; user explicitly requested `去重试` on 2026-10-01. This is a separately recorded exploratory repeat, not a retry of or qualification evidence for the frozen R4 v1 gate.

## Purpose and Boundary

The original R4-P01 attempt is immutable and already consumed one request. Its result remains `R4_V1=NO`, `sampler_gate=INFRA_FAILURE_UNCORRELATED_EVIDENCE`, and `quality_status=QUALITY_FAILURE`. This repeat exists only to observe whether the exact same frozen request produces the same model response again after the user explicitly asked to retry; it cannot promote the original gate, satisfy sampler attribution, or authorize B1.

The repeat uses the exact frozen request body for `R4-P01`, verified against SHA-256 `f02e0f4960f7c71ab386a7b233334df24a3cd1290887e07e14273a7439b0ef7e`. It writes only to `runs/memory/mem3/mem3b0q-r4-identity-qualification-v1/posthoc_repeat_r4_p01_20261001/`; it does not overwrite or move the original gate directory. The original R4 protocol, runner, manifest, response, and qualification result remain byte-identical.

One completion POST maximum is enforced by the reviewed in-memory strict overlay. There are zero retries within the repeat. Exact loopback/runtime checks run before the request and after it; any preflight failure stops before POST. The repeat lock binds the repeat wrapper, original overlay and lock, frozen runner/manifest/amendment, original request/response/gate artifact hashes, exact repeated request hash, case, and maximum POST count.

## Runtime and Attribution

The repeat is local-only: Qwen3-8B Q4_K_M at `127.0.0.1:8081`; no hosted call and no service restart. The current GET-only preflight passed for the pinned llama.cpp build and model. Its idle `/slots` response does not expose a `prompt` field. Therefore, even if the repeat returns HTTP 200, sampler/task attribution can remain `UNVERIFIED`; the output comparison is exploratory only. Service `8092` is untouched.

## Reporting

Record the repeat response hash and compare its parsed proposal with the historical response. Do not edit gold, prompt, schema, aliases, parser, or the historical output based on the result. Report any difference as post-hoc response variability, not an accuracy estimate. Report an identical invalid output as evidence of repeatability for this exact local request only, not general model determinism.

The result has no effect on `R4_V1=NO`, `MEM3B0Q_CANDIDATE_BOUNDED_EXTRACTOR_READY=NO`, or `MEM3B0Q_MEM3B1_READY=NO`. It does not authorize the 40-call pass, candidate-bounded inference, benchmark runs, MemoryStore mutation, or performance claims.

## Completed Repeat Result

The one authorized completion POST finished on the pinned local Qwen3-8B endpoint:

| Field | Result |
|---|---|
| Request body SHA-256 | `f02e0f4960f7c71ab386a7b233334df24a3cd1290887e07e14273a7439b0ef7e` |
| HTTP / model | `200` / local Qwen3-8B Q4_K_M, `b10068-571d0d540` |
| Request attempts / retries | `1 / 0` |
| Hosted calls | `0` |
| Latency | `38843.3 ms` |
| Prompt / completion tokens | `360 / 91` |
| Sampler attribution | `UNVERIFIED`; active prompt witness did not contain the input proposition |
| Gate outcome | `INFRA_FAILURE_UNCORRELATED_EVIDENCE` plus `QUALITY_FAILURE` |
| Qualification effect | None; `R4_V1=NO` remains unchanged |

The repeat is not byte-identical to the historical response. Both proposals returned `owner_span="My workout plan"`, `object_span="activity"`, and `value_span="running"`; the historical `attribute_span` was `"is"`, while the repeat returned `"is running"`. Both were rejected at the same first deterministic check, `owner_span:unsupported_exact_alias`.

For the frozen proposition `My workout plan activity is running.`, the current alias vocabulary has `my → SELF`, `workout plan → EXERCISE_PLAN`, and `activity → ACTIVITY`. The proposal appears to shift the span boundaries across owner/object/attribute (and varies at the copula boundary), while the validator rejects the combined owner span before later aliases are checked. This is consistent with an extractor boundary error interacting with an exact-alias contract; this single repeated case cannot establish whether the prompt, schema, model behavior, or their interaction is the dominant cause. It is not evidence of general nondeterminism or a contract defect by itself.

The complete immutable repeat artifacts are in `runs/memory/mem3/mem3b0q-r4-identity-qualification-v1/posthoc_repeat_r4_p01_20261001/`, including the request/response, gate result, slot observations, sampler evidence, log delta, and SHA-256 manifest. The recorded status is `NON_QUALIFYING_POSTHOC_DIAGNOSTIC`.
