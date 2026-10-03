# Runtime flow and API contract

1. The Harness creates a unique `trace_id` and records the raw query.
2. `MedicalRouter` evaluates only the query and explicit conversation context. Single-source questions use the Single fast path; multiple capability cues go to the Lead.
3. The Lead emits a JSON plan containing only `mode` and bounded `{worker, objective}` tasks. The Harness validates roles, uniqueness, and limits, then assigns task and worker IDs.
4. The Harness executes each worker's allowed skills. Independent workers use concurrent tasks; a dependent Care task runs in the second wave. Provider and tool calls, evidence, and failure status are appended to Shared Context.
5. The Lead receives successful and failed worker statuses plus the evidence ledger and synthesizes a direct answer. If no worker returns a usable result, the runtime falls back to Strong Single.
6. Harness verification filters citations against observed evidence, applies deterministic urgent-symptom flags, and returns a response.

The response object is a plain dataclass with `to_dict()`, suitable for a FastAPI response body:

```json
{
  "answer": "...",
  "route_mode": "SINGLE",
  "workers_used": [],
  "citations": [],
  "safety_flags": [],
  "trace_id": "trace-...",
  "latency_ms": 123
}
```

`create_fastapi_app(runtime)` exposes `POST /medical/answer` and `GET /medical/metrics` when the optional `api` extra is installed. The runtime accepts injected model, Memory, Evidence, and hospital providers; deployments must construct it with their approved providers and request identity policy.

Aggregate metrics include request count, Single/Team ratio, average workers, latency percentiles, provider/tool calls, token use, and worker error rate. Per-request traces preserve route decision, Lead plan, skill/tool calls, evidence acquisition, worker report status, synthesis input, verification, and final response.
