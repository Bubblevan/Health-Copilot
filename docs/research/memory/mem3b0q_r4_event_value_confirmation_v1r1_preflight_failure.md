# Event-Value Confirmation v1r1 Preflight Failure

Run `mem3b0q-r4-event-value-confirmation-v1r1` is preserved as
`INFRA_INCOMPLETE`. The runtime-only retry retained the exact v1 dataset and
request hashes, but its `netstat` plus CIM process resolution was denied by
Windows before the first completion POST. Accounting: zero posts, zero retries,
zero hosted calls, and no quality outcomes; two cases were not run. All five
manifested artifacts match their recorded hashes.

This failure is separate from the v1 `Get-NetTCPConnection` denial. Read-only
manual checks of the same PID had succeeded in an interactive shell, but the
runner's non-interactive PowerShell process could not query `Win32_Process`.
The fail-closed behavior is retained. The next attempt requires process-query
permission for the runtime preflight; no listener, binary, model, prompt, data,
or scoring check will be skipped.
