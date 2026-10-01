# R4 Factorized Event Projection: Offline Ablation

Status: development-only counterfactual; no new inference was run.

## Question

The frozen v1r3 model responses all failed strict admission. Review revealed two
LLM-selected fields were unstable: the owner mention (which occurrence of
`my`/an actor) and the value span (event predicate vs. object/time tail). This
ablation tests whether the already-frozen local rules can deterministically
project those fields from the source while leaving model-selected object and
attribute candidates and the existing scorer/guards intact.

## Result

The exact v1r3 response bytes were replayed offline and scored with the same
frozen scorer. No model call, prompt change, gold change, or admission-rule
relaxation occurred.

| Measure | Raw v1r3 outputs | Deterministic projection |
|---|---:|---:|
| Exact atom matches | 0/3 | 3/3 |
| Admitted atoms | 0/3 | 3/3 |
| Infra failures | 0 | 0 |
| New model calls | 0 | 0 |

Projection selected the nearest preceding owner candidate before the selected
object and, for the frozen `BICYCLE/PURCHASE_EVENT` slot, extracted the unique
source cue `completed the purchase` from the same clause. The original model
responses had selected:

| Case | Model owner candidate | Projected owner candidate | Model value | Projected value |
|---|---|---|---|---|
| EVTCONF-01 | `owner:SISTER:0:9` | `owner:SELF:53:55` | `purchase of my bicycle` | `completed the purchase` |
| EVTCONF-02 | `owner:SELF:0:2` | `owner:SELF:48:50` | `completed the purchase of my bicycle two` | `completed the purchase` |
| EVTCONF-03 | `owner:SELF:0:2` | `owner:SELF:50:52` | `completed the purchase of my bicycle on` | `completed the purchase` |

## Interpretation

This supports a narrow harness-design hypothesis: for a closed slot policy,
the LLM need not control owner occurrence or event-value span if those can be
derived from source-bound candidates and explicit event cues. It does not show
raw extractor improvement. The 3/3 result is an offline counterfactual over
the same three examples that motivated the projection, so it is highly
optimistic and is not independent confirmation, benchmark evidence, or a
general-purpose event parser result.

The next diagnostic should freeze fresh synthetic texts before inference and
ask the model only for object/attribute candidates; deterministic harness code
will own the owner and value projections. Even that will remain small
development evidence until the public benchmark and broader slot coverage are
run.

## Reproducibility

- Offline output: `runs/memory/mem3/mem3b0q-r4-event-projection-offline-v1/`
- Source run: `mem3b0q-r4-event-value-confirmation-v1r3`
- Source dataset SHA256: `4b662914ded880c977cc33e7084ce95661db410a8af5683354b851b8c2afd115`
- Projection code: `tools/research/memory/mem3b0q_r4_event_value_projection_v1.py`
- Re-run: `python -m tools.research.memory.run_mem3b0q_r4_event_projection_offline_v1 --score-offline`
