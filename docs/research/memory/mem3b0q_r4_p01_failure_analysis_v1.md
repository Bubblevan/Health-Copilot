# MEM-3B0Q R4-P01 Failure Analysis v1

Status: `OFFLINE_ANALYSIS_COMPLETE`; this addendum does not alter R4 v1, its frozen inputs, or the gate result. The proposed trace observer is not wired into the frozen runner and authorizes no inference.

## Gate Result

The first archived preflight attempt remains `PREFLIGHT_FAILURE`, `delivery_state=NOT_SENT`, and `request_attempts=0`. After independent approval to resume that one pre-registered request, exactly one local request was issued: HTTP 200, `request_attempts=1`, `retry_count=0`, `hosted_calls=0`, `memory_store_mutations=0`, and `revision_edges=0`.

The authoritative outcome remains:

| Dimension | Result | Reason |
|---|---|---|
| Sampler/task trace | `UNVERIFIED` | Required active-prompt witness absent from all 89 active `/slots` samples. |
| Model response contract | `QUALITY_FAILURE` | Frozen validator's first error: `owner_span:unsupported_exact_alias`. |
| R4 qualification | `NO` | The infrastructure trace gate did not pass. |
| B1 readiness | `MEM3B0Q_MEM3B1_READY=NO` | No subsequent request or materialization is authorized by this result. |

These classifications are independent. The exact response had `owner_span="My workout plan"`, `object_span="activity"`, and `attribute_span="is"`; the frozen P01 gold expects `owner_span="My"`, `object_span="workout plan"`, and `attribute_span="activity"`. The recorded validator short-circuited at the owner error, so this report does not claim separate validator outcomes for the object or attribute spans. No frozen prompt, gold, parser, or result is retrospectively adjusted.

This single sample establishes a method/contract failure, not output drift. The frozen prompt says to return the shortest complete exact substring and defines owner as possessor, but its example says to use `My sister` rather than `My` when the proposition says `My sister's ...`. P01 has the related construction `My workout plan's ...`; without an explicit rule separating the possessor (`My`) from the possessed object (`workout plan`), that example may invite an overextended owner span. The response also assigned `activity` to the object field and the copula `is` to the attribute field, so alias strictness alone does not explain the whole mismatch. With one output and no authorized repeat, we cannot distinguish prompt-induced generalization from model parsing instability. Any revised contract must resolve the semantic field boundary offline, before a separately reviewed and authorized run.

## Infrastructure Root Cause

The gate's active-slot observer read `/slots.prompt`; when absent, it normalized the value to an empty string and hashed that empty string. All 89 active samples were slot `0`, task `9`, with SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`; all lacked both the exact source ID and proposition witnesses. Seven grammar-prefill lines fell within the unique launch-to-processing interval and a matching `init sampler` line existed, but that evidence does not bind the request's exact proposition to the observed active task. The contract therefore correctly stays `UNVERIFIED`.

Pinned llama.cpp source explains why the witness was unavailable. `slots_debug` defaults to `0` and is read from `LLAMA_SERVER_SLOTS_DEBUG`; the `/slots` handler calls the serializer with `only_metrics=true` when the setting is zero, and that serializer omits `prompt`. See the pinned [`server-context.cpp` serializer](https://github.com/ggml-org/llama.cpp/blob/571d0d540df04f25298d0e159e520d9fc62ed121/tools/server/server-context.cpp#L585-L615), [debug setting initialization](https://github.com/ggml-org/llama.cpp/blob/571d0d540df04f25298d0e159e520d9fc62ed121/tools/server/server-context.cpp#L844-L849), [environment read](https://github.com/ggml-org/llama.cpp/blob/571d0d540df04f25298d0e159e520d9fc62ed121/tools/server/server-context.cpp#L1225-L1231), and [`/slots` serialization call](https://github.com/ggml-org/llama.cpp/blob/571d0d540df04f25298d0e159e520d9fc62ed121/tools/server/server-context.cpp#L2252-L2267).

This is an observer-configuration defect, not evidence that the server never had the prompt. But the missing witness cannot be reconstructed from request bytes, prompt-token count, server time ordering, or grammar logs. The original gate result is not promoted.

## Preserved Evidence

The raw gate directory is `runs/memory/mem3/mem3b0q-r4-identity-qualification-v1/pass1_r4-p01_sampler_init_gate/`. Its `gate_run_manifest.json` hashes the raw evidence files. Key digests:

| Artifact | SHA-256 |
|---|---|
| Frozen protocol manifest | `278dd2079d0196a5d7b58111b3604a5a11ebe92bfdc22bc5fe868e8d3b547e1f` |
| Request body | `f02e0f4960f7c71ab386a7b233334df24a3cd1290887e07e14273a7439b0ef7e` |
| Response body | `e693ecc7eba4e0b295ff0e45f9c53574b8075b0732a0fe6193093e23ad894523` |
| Slot observations | `881513b1a4567ccaceb9deda5ef9b8a5812835710fc2c0f35c3560087c0412bc` |
| Server log delta | `218cd4e7bf6cb300f87d08977907b1d7e88170b1ec43eedeeb8ab8dad6eaebea` |

The earlier no-request preflight artifact remains archived separately at `pass1_r4-p01_preflight_failure_20261001T065857Z`; its recovery record confirms zero requests before the separately approved single attempt. Neither record is rewritten.

## Offline Trace Observer v2 Proposal

The proposal in `tools/research/memory/mem3b0q_r4_trace_v2_proposal.py` is deliberately isolated from the frozen v1 runner. Its tests are fixture-only and perform no network or model calls. The proposal:

- requires an active slot's `prompt` field to be present and contain both the exact frozen `source_id` and full proposition;
- fails closed on absent/malformed prompt, multiple active slots, missing or inconsistent task identity, or a missing active witness;
- stores only a prompt SHA-256 and witness booleans, never the prompt text;
- treats an idle slot without `prompt` as valid idle state, not as a prompt witness.

For any separately reviewed future attempt, the pinned build should be started with `LLAMA_SERVER_SLOTS_DEBUG=1` on the isolated loopback service. The observer must validate the exact build/process and verify that `/slots` actually exposes the active prompt before relying on the witness; missing prompt must stop before an inference request. Raw prompt content must remain transient and local because debug `/slots` can expose the full request prompt. The exact runtime/environment setup, observer version, tests, hashes, call count, and any proposed reuse of R4-P01 require a new review and explicit authorization. This note does not authorize restarting either service or sending a request.

Independent review approved this offline proposal only. If it is ever integrated, the caller must obtain the expected source ID and proposition from the hash-verified frozen control pack, not from caller-supplied, response-derived, or mutable data.

Offline tests cover: missing prompt, empty prompt, exact positive witness, split witnesses across slots, duplicate active slots, idle-without-prompt, and malformed payload. The proposal is not yet a replacement gate or a pass condition.

## Next Gate

No R4 retry, new R4 request, B1, or benchmark run follows automatically. Before any inference: independently review the v2 proposal and its source-backed startup configuration; freeze a new instrumentation manifest; verify the debug-enabled loopback process and prompt visibility without sending an inference request; then obtain explicit authorization for one new bounded attempt. The existing model-quality failure remains visible and must be addressed only through a separately versioned, pre-reviewed method/protocol change, never by editing this response or its gold after observation.
