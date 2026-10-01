# Factorized Event-Slot Confirmation v1r2

Status: case-registry dispatch repair only; v1 data and request bytes remain
unchanged.

The v1 attempt failed before lock validation because the engine hashed the
legacy runner. The v1r1 lock-dispatch wrapper fixed that; v1r1 then validated
the lock but failed before any completion POST because the engine's loop used
legacy `EVTCONF-*` case IDs. Both failed attempts and their records remain
preserved.

This version binds the engine's case registry to the three frozen `EVTPROJ-*`
cases for the duration of the run. It does not change prompts, schema, data,
projection, admission, scoring, or runtime settings. The model may select only
object/attribute IDs; the harness owns owner occurrence and event-value
projection.

Maximum three loopback POSTs, one per case, zero retries. No hosted API, key,
judge, clinical content, or MemoryStore mutation is used. This remains small
synthetic development evidence only.
