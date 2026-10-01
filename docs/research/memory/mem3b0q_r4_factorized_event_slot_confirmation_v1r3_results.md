# Factorized Event-Slot Confirmation v1r3 Results

Status: complete synthetic development confirmation; not benchmark evidence.

## Experimental question

After earlier responses repeatedly misselected owner occurrences and event
value spans, does a narrower proposal contract work better: the model chooses
only object/attribute candidate IDs, while frozen deterministic code chooses
the nearest preceding owner occurrence and projects the licensed event cue?

## Result

| Measure | Result |
|---|---:|
| Cases | 3 |
| Exact atom matches | 3/3 |
| Atom precision / recall | 1.00 / 1.00 |
| Admitted atoms | 3/3 |
| Schema-valid model proposals | 3/3 |
| Infra/preflight failures | 0 |
| Generation truncations | 0 |
| New local POSTs / retries | 3 / 0 |
| Hosted calls / MemoryStore mutations | 0 / 0 |

| Case | Model-selected object/attribute | Harness owner projection | Harness event-value projection | Result |
|---|---|---|---|---|
| EVTPROJ-01 | BICYCLE / PURCHASE_EVENT | nearest `my` at 56:58 | `completed the purchase` | exact pass |
| EVTPROJ-02 | BICYCLE / PURCHASE_EVENT | nearest `my` at 54:56 | `completed the purchase` | exact pass |
| EVTPROJ-03 | BICYCLE / PURCHASE_EVENT | nearest `my` at 45:47 | `completed the purchase` | exact pass |

Reader prompt/completion tokens by case: 196/75, 197/75, 200/76. Retrieval is
not involved in this writer/admission diagnostic. Local query latency was
54.78s, 54.65s, and 54.62s (median 54.65s). The run used the pinned Qwen3-8B
Q4_K_M reader and llama.cpp build `b10068-571d0d540` through the validated
loopback endpoint.

## Interpretation

This is evidence that, for this one frozen synthetic event slot, a reduced
object/attribute proposal plus deterministic owner/value projection can
materialize the exact source-grounded atom without asking the model to choose
repeated owner mentions or compose an exact value span. The same frozen scorer
and downstream guards passed all three cases.

This does not establish broad extraction accuracy, memory quality, revision
resolution, stale-memory reduction, or public-benchmark improvement. The pack
is small, uses one event predicate and one object/attribute slot, and was
designed after inspecting an earlier development pack. The deterministic cue
registry is narrow and must fail closed for unsupported slots. The 3/3 result
is not a LongMemEval, Memora, or MedMemoryBench result and must not be used as a
resume headline.

The appropriate next validation is broader slot coverage with frozen
synthetic cases, followed by integrating only demonstrated deterministic
materialization rules into the public memory adapter. LongMemEval 102 DEV,
paired bootstrap, held-out TEST, and MedMemoryBench remain unrun.

## Reproducibility

- Run ID: `mem3b0q-r4-factorized-event-slot-confirmation-v1r3`
- Lock SHA256: `757625271f7938ab4c8a4cb76b522b1d2a569175b0a0529a312e5b66b15d03a6`
- Dataset SHA256: `601841f93db7a4732b4be5750dd513afe1515af21026b08cd19dc2144d40a22b`
- Raw requests, responses, projection audit, and admitted atoms:
  `runs/memory/mem3/mem3b0q-r4-factorized-event-slot-confirmation-v1r3/`
- Model request schema exposes only `object_candidate_id` and
  `attribute_candidate_id`; owner and value fields are harness-derived.
