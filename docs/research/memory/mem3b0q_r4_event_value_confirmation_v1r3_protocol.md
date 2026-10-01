# Event-Value Confirmation v1r3

Status: append-never harness repair and continuation of the frozen v1r2 run.

The v1r2 lock, request map, model, prompts, data, and scoring are unchanged.
EVTCONF-01 already received one local POST in v1r2 and will not be sent again.
Its exact request and response bytes are copied into this run and scored offline;
the earlier v1r2 `INFRA_FAILURE` record remains untouched. EVTCONF-02 and
EVTCONF-03 are each eligible for at most one fresh local POST. Total new POSTs
are capped at two, with zero retries.

The scorer adapter wraps the v1 scoring request in the interface expected by
the v1r2 runner. This repairs only the harness call shape; it does not change
the proposal prompt, candidate schema, admission semantics, gold, or scorer.

The elevated permission is used only for the frozen read-only Windows process
and listener query. Inference remains direct to the validated loopback
llama-server at 127.0.0.1:8081. No hosted API, API key, judge, clinical content,
or MemoryStore mutation is used.

The dataset remains a small project-authored synthetic confirmation set. It is
diagnostic evidence only and is not a public benchmark or generalization claim.
