# R4 Event-Value Confirmation v1r3 Results

Status: complete diagnostic run; not public benchmark evidence.

## Protocol and execution

This run retained the v1/v1r1/v1r2 protocol, frozen v3 prompt, model, dataset,
and scorer. EVTCONF-01 had already received one POST in v1r2. Its exact
request/response bytes were replayed and rescored offline; they were not sent
again. EVTCONF-02 and EVTCONF-03 each received one fresh loopback POST. The
scoring-request wrapper defect was repaired in the v1r3 adapter only.

| Check | Result |
|---|---:|
| Cases scored | 3/3 |
| New local POSTs | 2 |
| Infra/preflight failures | 0 |
| Retries | 0 |
| Hosted calls | 0 |
| MemoryStore mutations | 0 |
| Exact atom matches | 0/3 |
| Admitted atoms | 0/3 |
| Generation truncations | 0 |

## Per-case diagnosis

| Case | Model value span | Frozen expected span | Admission result | Failure attribution hint |
|---|---|---|---|---|
| EVTCONF-01 | `purchase of my bicycle` | `completed the purchase` | quarantined | `event_value_unproven`; omitted the action cue |
| EVTCONF-02 | `completed the purchase of my bicycle two` | `completed the purchase` | quarantined | `event_value_unproven`; included object and a temporal fragment |
| EVTCONF-03 | `completed the purchase of my bicycle on` | `completed the purchase` | quarantined | `event_value_unproven`; included object and date introducer |

All responses passed JSON/schema validation, used the expected owner/object/
attribute candidate IDs, and completed without truncation. The evidence points
to unstable span selection under the extractor contract, not candidate binding
or runtime failure. The lexical/slot-policy diagnosis is still a local
failure-attribution hint, not a causal claim about model behavior.

The existing guard requires a full match to the frozen event predicate
`completed the purchase`. This is intentionally strict for the diagnostic, but
the model repeatedly emits an overlong or incomplete span despite a prompt
request for the shortest complete exact substring. Simply widening the regex
would admit object/time text as event value and would weaken provenance.

## Next method iteration

Keep the guard strict. Prototype a deterministic, slot-scoped event-value
projection: the model proposes typed anchors; a frozen harness cue policy
locates the event predicate in the same source clause and emits that exact
source span. Compare it offline against these frozen responses before deciding
whether a fresh confirmation pack is warranted. Any resulting pass rate must
be labeled as a harness-projection ablation, not raw model extraction accuracy.

## Reproducibility

- Run ID: `mem3b0q-r4-event-value-confirmation-v1r3`
- Lock SHA256: `7e5aa7e1b8965f418bb11cc0c4810211884f9130d409c109fff44ecd098ffe82`
- Dataset SHA256: `4b662914ded880c977cc33e7084ce95661db410a8af5683354b851b8c2afd115`
- Reader: frozen local Qwen3-8B Q4_K_M; llama-server build `b10068-571d0d540`
- Runtime report and raw outputs: `runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1r3/`
