# Memora LoCoMo Local-Qwen Judge Amendment

Date: 2026-10-04

## Trigger

The full run completed all ten memory ingestions, then the first semantic QA pass exposed a local judge output-contract mismatch. The pinned Memora prompt asks for a short rationale followed by a JSON label, while the original adapter accepted either a whole-response JSON object or a final standalone label. Qwen sometimes emitted a valid `{"label":"..."}` object followed by rationale, or exhausted the original 64-token allowance before emitting the label. Those cases were recorded as judge-format infrastructure failures; predictions were preserved and were not assigned a judge score of zero.

## Amendment

The local-only compatibility wrapper `tools/research/memory/run_memora_qwen_locomo_judge_compat.py` resumes the existing full run without changing its pinned runner source, run identity, memory stores, retrieval strategy, answer prompt, reader model, or deterministic F1/EM metrics.

- Judge model remains the same local Qwen3-8B Q4_K_M loopback service.
- The pinned Microsoft Memora `ACCURACY_PROMPT`, JSON response mode, temperature `0`, and seed `42` are unchanged.
- Every judge row in the final pass uses `max_tokens=128`, providing room for the requested rationale and label.
- Parsing accepts exactly one unambiguous `label` field (`CORRECT` or `WRONG`) in a JSON object anywhere in the response, or a strict final standalone label. Conflicting or missing labels remain failures; no label is inferred from answer wording.
- One additional same-prompt, same-model, same-budget local retry is allowed for an unparseable judge response. The retry is recorded in the judgment row.
- Existing predictions are resumed as-is. The initial 64-token judge file is retained under `judgments-initial64/` as audit evidence and is not mixed into final judge accuracy.

The wrapper writes `judge_format_amendment.json` inside the run artifact root with the parent run identity, both relevant source hashes, and the exact amended format settings. This is a local-model output-format compatibility amendment, not an attempt to improve answer quality or reproduce the paper's proprietary judge numerically.
