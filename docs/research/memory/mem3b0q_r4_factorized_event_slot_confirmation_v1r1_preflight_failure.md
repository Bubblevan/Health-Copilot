# Factorized Event-Slot Confirmation v1r1 Preflight Failure

Status: invalid runner attempt; zero completion POSTs.

After lock validation passed, the shared engine created a run reservation but
looked up case IDs from the legacy `EVTCONF-*` registry, while the frozen pack
contains `EVTPROJ-*`. It stopped on the first lookup with `KeyError` before
calling the model. The partial run reservation is preserved as an audit
artifact. No case request or response was produced.

v1r2 changes only the run-time case registry mapping and pins this failed
reservation. Dataset, request bytes, model schema, projection, and scorer stay
unchanged.
