# Factorized Event-Slot Confirmation v1

Status: frozen synthetic confirmation; one local model request per case.

## Hypothesis

For the closed `BICYCLE/PURCHASE_EVENT` slot, typed object and attribute
candidate selection can be delegated to the model while the harness
deterministically selects the nearest preceding owner candidate and projects
the unique source event cue. The model request schema therefore contains only
object and attribute candidates; it does not request owner occurrence or event
value spans.

## Frozen scope

- Dataset: `mem3b0q-r4-factorized-event-slot-confirmation-pack-v1`
- Three project-authored synthetic English cases, frozen before inference
- Reader and model: the pinned local Qwen3-8B Q4_K_M endpoint
- One direct loopback POST per case; no retries
- No hosted API, judge, key, clinical content, or MemoryStore mutation
- Existing v1r3 scorer and admission guards remain unchanged after deterministic
  proposal materialization

## Interpretation boundary

This is a small development confirmation of the factorization mechanism, not a
public benchmark or generalization estimate. The event cue policy covers one
synthetic slot and was designed after inspecting an earlier development pack.
Any result is harness-assisted admission, not raw model extraction accuracy.
No LongMemEval, Memora, MedMemoryBench, held-out public evaluation, or
performance-ranking claim is made.
