# MEM-3B0Q R4 Event-Value Prompt Tuning Smoke v3 Results

Status: `COMPLETE`; raw request/response and runtime evidence are under
`runs/memory/mem3/mem3b0q-r4-event-value-smoke-v3/`.

`R4C-04` was already observed in both v1 and v2. After the prompt explicitly
distinguished an event predicate from a phrase that only says when it happened,
the one fresh local response emitted `completed the purchase` for the
`PURCHASE_EVENT` value instead of `last month`.

| Case | Matched / expected | Precision | Recall | Completion tokens | Latency |
|---|---:|---:|---:|---:|---:|
| R4C-04 | 2/2 | 100% | 100% | 168 | 108.574 s |

La Paz and the bicycle purchase event both passed the unchanged v2 materializer
and v1/v2/v3 binding guards. The guard was not weakened. The response used one
local request, with zero retries, zero hosted calls, and zero MemoryStore
mutations. All five run-manifest artifact hashes verified.

This is a prompt-tuning smoke on an already observed synthetic case, not
independent evidence or a generalization result. The relative expression
`last month` is not resolved to a calendar date by this adapter. A fresh,
non-overlapping event diagnostic set must be evaluated with the prompt frozen
before making any broader statement.
