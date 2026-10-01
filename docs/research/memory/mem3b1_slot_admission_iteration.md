# MEM-3B1 Slot Admission Iteration

## Status

This is a development-stage reflection, not a public benchmark result. The current B0Q admission design is **not ready for B1**. No 102-case DEV, held-out TEST, or MedMemoryBench score is claimed here.

## What the Diagnostics Show

The first B0Q-reviewed 15-group sample produced 14/15 group decisions, admission precision 1.00, and recall 0.80. A second non-overlapping 15-group sample exposed poor transfer: the broad typed key scored 9/15 (0.600 accuracy, 0.429 precision, 0.600 recall); qualifier/equality variants traded recall for precision and none was strong. These are exploratory samples drawn from an already-reviewed B0Q pool, not independent estimates.

The relation diagnostic then returned zero automatic updates across 17 pairs. Its apparent 10/15 group accuracy is not a valid revision metric: the inherited `true_singleton_slot` label describes broad slot grouping, not a supersession relation. Inspection shows positive groups include exact/paraphrased repeats such as Node version 19.7.0, the same green-bean recipe, and the same car-cleaning plan. Those should consolidate provenance or coalesce an unchanged value; they are not old-value-to-new-value revisions. The model's `IDENTICAL` or `COMPLEMENTARY` outputs on those pairs are often semantically appropriate. Therefore the reported zero update recall is not interpretable against those slot labels.

The reverse error is equally important: sharing a topic or a broad slot is not sufficient evidence that a new proposition replaces an old value. The second sample includes task plans, accumulating details, multi-valued interests, and uncertain relations. A usable design must distinguish at least:

- same-slot / unchanged value: merge provenance, do not create a revision;
- same-slot / explicit mutually exclusive successor: admit a revision;
- complementary or multi-valued facts: retain concurrently;
- ambiguous or composite task state: abstain from materialization as a revision.

## Reader-Side Evidence

The frozen reader diagnostics independently show why correct retrieval alone is insufficient. On the Instagram current-count case, both 500 and 600 were retrieved; the frozen reader returned no answer from the raw context and also from a 600-only snapshot, while a targeted structured current-state rendering elicited the correct value. The gym frequency-change case likewise had both values available but the reader failed on the raw records. A concise diagnostic reader prompt helped those two hand-selected cases but collapsed to near-zero on the frozen ten-case diagnostic, so it is not promoted as the answer contract. All of these are development diagnostics, not benchmark wins.

## DEV Revision-Pair Probe

A separate focused probe used only frozen DEV `knowledge-update` histories. In the shortest of the 17 histories (`4d6b87c8`), V1 sent 257 user turns / 49,961 characters (17,643 tokenizer-measured prompt tokens) to local Qwen. The complete prompt fit and server token accounting matched; Qwen returned a valid empty `transitions` array. This is not a benchmark score, but it missed a source phrase later found by a narrow deterministic cue scan: `2:30 pm instead of 3 pm` in one user turn. The V1 contract required a later statement, so its boundary excluded an explicit same-turn correction.

The corrected V2 contract allowed same-turn self-correction, but its full-history call failed during local Vulkan decode with `vk::Device::waitForFences: ErrorDeviceLost`, after 417 generated tokens. The response is classified as `INFRA_FAILURE`; no prediction or quality score exists. A V3 candidate-only attempt reduced the model input to the one frozen cue and capped completion at 256 tokens, but 8081 reset the connection after the preceding device-loss event. 8092 was not used or modified, and 8081 was not restarted.

A deterministic zero-model-call scan over all 17 DEV knowledge-update histories found exactly one explicit numeric time-of-day `instead of` cue, in `4d6b87c8`: `3 pm` to `2:30 pm`. This `1/17` is only the coverage of that narrow cue rule on this development sample. It is not revision recall, because the full histories were not exhaustively relation-labeled and the rule intentionally ignores non-time and implicit changes.

The useful method change is to avoid asking one long-history generation to both find and adjudicate all possible updates. Next iteration should separate candidate retrieval / per-session source-grounded proposals from deterministic pair admission and temporal materialization. Same-turn events require an explicit temporal-order representation; until that is implemented and tested, same-time contradictory assertions must remain conflict/unresolved rather than being silently ordered.

## Broad Replacement-Cue Audit

A zero-model-call follow-up reviewed every literal `instead of` / `rather than` occurrence in the 17 frozen DEV knowledge-update user histories. The miner found 37 occurrences across 32 turns. The initial v1 labels called five occurrences personal-state revisions and one a task-plan revision. A terminology correction in v2 narrows those labels: the five are **explicit adopted change mentions**, not verified revision chains; the task example is an explicit plan-change mention. Their cue-mention rates are 5/37 (13.5%) for personal-state changes and 6/37 (16.2%) including the task-plan mention. The other 31 occurrences were not source-asserted changes: three were assertions without a prior value, seven were proposals not yet adopted, and 21 came from questions, imported text, instructions, or other non-self content.

This is not a recall estimate, benchmark gold, or system score. The histories are a small, non-random DEV slice, and the labels have one reviewer. It falsifies the broad rule "contrast marker => revision": most lexical hits are not even source-asserted state changes, and some occur inside pasted material. In the clearest tea-time example, a zero-model scan found no exact `3 pm` mention in any of the 30 earlier user turns; the old value appears only in the same correction sentence. The accepted extractor proposal therefore establishes a change mention, not an independently observed predecessor. The materializer correctly rejects a same-timestamp edge because it cannot assign a nonzero historical interval without temporal evidence. A safer gate must establish (1) the user is asserting their own state, (2) the successor is adopted rather than proposed, (3) both values map to the same typed single-valued slot, and (4) an independently grounded predecessor exists before closing its validity interval. Candidate detection can stay permissive; materialization cannot.

Reproduction artifacts:

- `tools/research/memory/audit_mem3b1_dev_alternative_cues_v1.py`
- `docs/research/memory/mem3b1_dev_alt_cue_manual_labels_v1.json`
- `runs/memory/mem3/mem3b1-dev-alternative-cue-audit-v1/`
- `tools/research/memory/audit_mem3b1_dev_alternative_cues_v2.py`
- `docs/research/memory/mem3b1_dev_alt_cue_annotation_amendment_v2.json`
- `runs/memory/mem3/mem3b1-dev-alternative-cue-audit-v2/`

## Method Decision

Stop treating slot-label accuracy as revision-admission quality. Keep model output as an auditable proposal only. A deterministic materializer should own temporal ordering and state transitions, while admission must have a separately validated relation contract. Exact/paraphrased repeats should be idempotent; only a source-grounded successor for the same scalar state can close the prior validity interval. Complementary, set-valued, composite, or uncertain facts must remain coexistent or unresolved rather than being forcibly collapsed.

The next evaluation set should come from frozen LongMemEval-S DEV `knowledge-update` conversations, not from B0Q's slot-group labels. It must contain manually checked positive transitions and negative controls, and the model input must omit question text, gold answer, answer-session IDs, review label, and benchmark-derived target value. The broad 102-case and held-out evaluations remain untouched until that contract has survived this focused diagnostic.

## Current Gate

`MEM3B1_SLOT_ADMISSION_READY=NO`

Current focused-run evidence: the v1 model proposed 0 transitions; the 17-history narrow cue audit found 1 explicit time-replacement candidate; the broader audit found 5/37 explicit adopted personal-state change mentions and one task-plan change mention, while the only checked predecessor anchor had no earlier exact old-value occurrence. This audit scores neither revision-chain quality nor answer accuracy. The full-history v2 and candidate-only v3 local reader calls produced infrastructure failures on the shared Vulkan reader. Regression tests are being extended around the revised evidence boundary.

Evidence reviewed:

- `runs/memory/mem3/mem3b1-b0q-reviewed-slot-sample-v1/`
- `runs/memory/mem3/mem3b1-b0q-reviewed-slot-sample-v2/`
- `runs/memory/mem3/mem3b1-pairwise-revision-relation-diagnostic-v1/`
- `runs/memory/mem3/mem3b1-factorized-slot-proposal-diagnostic-v1/`
- `runs/memory/mem3/mem3b1-instagram-structured-current-diagnostic-v1/`
- `runs/memory/mem3/mem3b1-gym-change-chain-diagnostic-v1/`
- `runs/memory/mem3/mem3b1-dev-revision-pairs-diagnostic-v1/`
- `runs/memory/mem3/mem3b1-dev-revision-pairs-diagnostic-v2/`
- `runs/memory/mem3/mem3b1-dev-revision-pairs-diagnostic-v3/`
- `runs/memory/mem3/mem3b1-dev-explicit-time-cue-audit-v1/`
- `runs/memory/mem3/mem3b1-dev-alternative-cue-audit-v1/`
- `runs/memory/mem3/mem3b1-dev-alternative-cue-audit-v2/`

The next gate is a focused, DEV-only revision-pair study with the target label defined as actual temporal supersession rather than topical/single-slot membership, run against a healthy local reader endpoint.
