# Event-Value Confirmation v1 Preflight Failure

Run `mem3b0q-r4-event-value-confirmation-v1` is preserved as
`INFRA_INCOMPLETE`. The first case failed before any completion request because
the Windows `Get-NetTCPConnection` query returned `PermissionDenied` while
enumerating the listener on port 8081. Accounting is explicit: zero posts,
zero retries, zero hosted calls, and no quality outcomes. The two later cases
were not run after the preflight failure.

Independent, read-only checks confirmed `/health`, `/props`, and `/v1/models`
were available on `127.0.0.1:8081`; `netstat` showed exactly one listener at
`127.0.0.1:8081` (PID 73952). `Get-CimInstance Win32_Process` returned the
expected pinned llama-server executable path and full command line. The
failed run's five-artifact manifest has no hash mismatches.

This is an infrastructure preflight failure, not a model result. Its output
directory and lock are retained unchanged. The next attempt uses a new run ID
and lock, with a narrowly scoped `netstat`-based listener enumeration fallback
only when the frozen query is denied. The fallback must still identify exactly
one listener, resolve its process through CIM, and pass the same executable
hash, model hash, loopback address, command-line, service build, and runtime
checks before any request is sent.
