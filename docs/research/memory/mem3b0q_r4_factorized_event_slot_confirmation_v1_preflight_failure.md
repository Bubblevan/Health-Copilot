# Factorized Event-Slot Confirmation v1 Preflight Failure

Status: invalid lock candidate; no inference request was sent.

The initial v1 lock was created, but the runner failed while validating its
locked payload with `lock_field_mismatch:runner_sha256`. The shared engine
recomputed its runner hash from the legacy engine module rather than the new
factorized adapter. The failure occurred before run reservation and before any
completion POST; no output run directory or model response was created.

The v1 protocol and lock are preserved as audit records and are not used for
evaluation. v1r1 adds a small, separately locked wrapper that binds the
factorized adapter's lock builder into the shared engine during the run.
