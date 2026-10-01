# Factorized Event-Slot Confirmation v1r1

Status: lock-dispatch repair only; v1 data and model request bytes are unchanged.

The v1 attempt failed before any POST because the shared engine recomputed
`runner_sha256` from its legacy module. The v1 lock remains an invalid candidate
and is preserved. This wrapper explicitly dispatches lock validation to the
factorized adapter's builder and pins both wrapper and adapter in the new lock.

The method remains fixed: the model may select only object and attribute
candidate IDs. Harness code deterministically selects nearest preceding owner
occurrence and the source-grounded event cue, after which the unchanged
scorer/guards validate the materialized proposal.

Execution is limited to three fresh project-authored synthetic cases, one local
POST per case, zero retries. No hosted API, key, judge, clinical content, or
MemoryStore mutation is used. Results are development evidence only and are not
public benchmark or generalization claims.
