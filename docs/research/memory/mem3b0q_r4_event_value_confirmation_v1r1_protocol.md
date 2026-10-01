# Event-Value Confirmation v1r1 Runtime Retry

Status: same three-case dataset, prompt, model, request bodies, answer schema,
and scoring as confirmation v1. The prior v1 run is retained as
`INFRA_INCOMPLETE` with zero completion posts because the Windows listener
query was denied. This new run ID avoids overwriting its reservation and
preflight-failure artifacts.

## Runtime Snapshot Repair

The primary `Get-NetTCPConnection` process snapshot remains unchanged. Only if
it fails specifically with `PermissionDenied` while querying that cmdlet, use
`netstat -ano -p tcp` to enumerate TCP listeners at the pinned port, then
resolve the unique owning PID via `Get-CimInstance Win32_Process`. The fallback
must yield exactly one listener. The same frozen checks then require the exact
loopback address, executable path and SHA256, model path and SHA256, complete
llama-server command line, build, served model, context, sampler defaults, and
idle slot. Any ambiguity or mismatch fails closed before POST.

The three request body hashes and dataset SHA must be byte-identical to
confirmation v1. The lock also pins the v1 zero-POST failure artifacts. This
is a runtime-preflight repair only; no prompt or method change is allowed.
