# Common Eval contamination audit

## Scope and status

The required comparison is the frozen DiagnosisArena-MCQ-915 and CMB-COMMON-1024 set against every planned SFT/GSPO training manifest. In this checkout, the evaluation snapshots and training manifests are absent. `eval/contamination/common_eval_v1_report.json` therefore records `BLOCKED_MISSING_EVALUATION_AND_TRAINING_SNAPSHOTS`; exact, lexical, semantic, and confirmed counts are unknown. This is not a zero-match result.

## Implemented audit levels

`src/health_ai_copilot/evaluation/contamination.py` provides deterministic candidate discovery:

1. Unicode NFKC, case-folded, punctuation/whitespace-normalized question exact hashes.
2. Token 3-gram Jaccard candidates at a configurable threshold.
3. Optional embedding cosine candidates using an injected embedding function; embeddings discover candidates only.
4. Human review fields remain unresolved until a reviewer marks each pair. Confirmed overlap handling removes the training row and requires a regenerated training manifest; it never removes evaluation rows.

When data become available, record counts per training dataset: exact matches, lexical candidates, semantic candidates, human-confirmed overlaps, and removed training rows. Preserve the frozen eval IDs and source hashes.

## Claim boundary

Only after the audit and row removal are complete may the project say: “No matching examples from our post-training datasets were retained after the frozen contamination audit.” The repository cannot establish whether public questions appeared in Qwen pretraining; no such claim is made.
