# MEM-3B0Q-R4 Sampler Trace Instrumentation v1

Status: `REVIEW_REQUIRED`; supplemental to, and does not modify, the frozen R4 protocol or freeze manifest.

## Purpose

The pinned llama.cpp build does not emit an HTTP request-completion trace by default. Its HTTP logger registration is disabled in the pinned source. A time-window log excerpt by itself is therefore not sufficient to attribute grammar/sampler messages to `R4-P01`.

This supplement defines a fail-closed request/task correlation check before the sampler-init gate may pass:

1. Capture the exact outgoing strict-JSON-Schema request body and SHA-256.
2. While that one request is in flight, poll the local `/slots` endpoint and retain normalized observations. At least one processing-slot observation must expose the exact frozen `source_id` and full proposition text in the active prompt. All observed active slots must identify one slot/task pair and must contain those same input witnesses.
3. Correlate the unscoped grammar-prefill line only through the pinned source's synchronous call order. Require exactly one `launching slot` record for the observed slot with the pre-task id `-1`; at least one `grammar accepted prefill token` line strictly after that launch and before exactly one `processing task` record for the same slot and observed task id; and exactly one matching `init sampler` record after `processing task`. Every positive grammar-prefill line in the captured delta must lie inside that one launch-to-processing interval. Any duplicate launch/task, out-of-interval prefill, reversed/missing record, or sampler/grammar error is unverified.
4. Bind the sampler evidence artifact to the outgoing request-body SHA-256. Any missing slot observation, observer error, second task, task-ID mismatch, absent correlated grammar acceptance line, uncorrelated prefill line, or error line leaves the sampler gate unverified and stops the run.
5. `sampler_gate=PASS` additionally requires HTTP 200, a valid response envelope, and the frozen structural/source-span validator to pass. Oracle disagreement is reported separately as `QUALITY_FAILURE` and does not convert infrastructure failures into quality zero.

## Pinned Source Basis

The instrumentation is specific to llama.cpp commit `571d0d540df04f25298d0e159e520d9fc62ed121`, matching the frozen server binary:

- [`tools/server/server-context.cpp`](https://github.com/ggml-org/llama.cpp/blob/571d0d540df04f25298d0e159e520d9fc62ed121/tools/server/server-context.cpp#L1551-L1659): within `launch_slot_with_task`, the server logs `launching slot`, synchronously calls `common_sampler_init`, assigns `slot.task`, then logs `processing task`. The subsequent task-scoped `init sampler` trace is emitted from `server_slot::init_sampler`. Slot JSON exposes the active task and prompt.
- [`common/sampling.cpp`](https://github.com/ggml-org/llama.cpp/blob/571d0d540df04f25298d0e159e520d9fc62ed121/common/sampling.cpp#L175-L278): output-format grammars are initialized in `common_sampler_init`; successful prefill emits `grammar accepted prefill token` at debug level, while initialization errors emit grammar/sampler error diagnostics.
- [`tools/server/server-http.cpp`](https://github.com/ggml-org/llama.cpp/blob/571d0d540df04f25298d0e159e520d9fc62ed121/tools/server/server-http.cpp): the generic HTTP request logger is not registered, so request/task correlation must use the active slot prompt and task-scoped sampler trace, not an assumed access-log line.

The request's frozen `response_format.type=json_schema` and canonical schema hash establish which grammar was supplied. The active slot prompt binds the unique `R4-P01` proposition to the observed task ID. Because llama.cpp's grammar-prefill log has no task ID, the parser accepts it only when the pinned synchronous source ordering and a unique log interval tie it to that slot's single sampler initialization; a prompt/task witness plus an arbitrary nearby positive log is insufficient. If the actual binary's log format does not satisfy these checks, do not loosen the parser after seeing the output. Preserve the attempt as an infrastructure failure and obtain a separately reviewed instrumentation change before any further R4 request.

## Scope

This is an observability supplement only. It does not change the pack, prompt, schema, runtime, call count, order, retry policy, metrics, expected labels, or acceptance criteria. It does not authorize B1, benchmark execution, MemoryStore writes, or public claims. The original frozen manifest remains byte-identical; the runner lock and post-attempt manifest separately identify the instrumentation and evidence hashes.
