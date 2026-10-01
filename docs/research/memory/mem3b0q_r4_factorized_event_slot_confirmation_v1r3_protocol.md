# Factorized Event-Slot Confirmation v1r3

Status: legacy overlap-audit global isolation only; v1 inputs remain unchanged.

The v1 and v1r1 attempts stopped before inference on shared-engine lock
dispatch and case-registry mismatches. v1r2 bound the new case registry but
exposed that the legacy prior-pack overlap audit also reads the shared case-ID
global. v1r3 temporarily restores the legacy IDs only during that read-only
audit and restores `EVTPROJ-*` before the case loop. All prior failed attempts
and locks remain preserved.

The frozen method remains: the model selects only object and attribute
candidates; deterministic harness code supplies nearest preceding owner and
the unique source-grounded event cue; the same scorer and admission guards then
evaluate the materialized proposal.

Three project-authored synthetic cases, one local POST each, zero retries.
No hosted API, judge, key, clinical content, or MemoryStore mutation. This is
development evidence only, not a benchmark/generalization claim.
