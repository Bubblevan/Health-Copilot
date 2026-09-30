# U2-F Shortcut and Cheap-Classifier Audit

Date: 2026-09-30

Scope: runtime-observable feature leakage diagnosis; no learned model performance claim.

## Acceptance thresholds

The frozen profile requires the best single-feature accuracy to be at most 0.80 and its lift over the majority-class baseline to be at most 0.15, for both memory_read and external_retrieval requirement targets. A combined cheap-feature classifier score of 0.90 or greater triggers human review; it does not establish architecture performance. The auditor excludes scenario family, gold labels, required fact IDs, latent dependency graphs, and counterfactual group IDs from runtime features.

## Single-feature results

| Target | Majority baseline | Best single feature | Accuracy | Lift over majority |
|---|---:|---|---:|---:|
| memory_read required | 0.5482 | query has a value token | 0.6334 | 0.0852 |
| external_retrieval required | 0.6000 | query contains “published” | 0.6531 | 0.0531 |
| Overall maximum | — | — | 0.6531 | 0.0852 |

Both maximums pass the 0.80 / 0.15 gate. The generator adds requirement-independent query wording and deterministic neutral lexical context to reduce accidental association between task family wording and capability labels. The residual values above are measured on the candidate's complete runtime/evaluator join; they are not tuned predictions.

## Combined and query-only diagnostics

Bernoulli Naive Bayes was fit on TRAIN using cheap runtime-visible features and evaluated separately on each development split. No held-out labels or task-family identifiers are included in its input.

| Diagnostic | DEV_IID max balanced accuracy / macro-F1 | DEV_STRUCTURAL max balanced accuracy / macro-F1 | Review threshold | Result |
|---|---:|---:|---:|---|
| Context-aware cheap features | 0.8199 | 0.7892 | 0.90 | PASS; below review trigger |
| Query-only features | 0.8135 | 0.7800 | 0.90 | PASS; below review trigger |

Per-target query-only scores are preserved in `cheap_classifier_audit.json`: on DEV_IID, memory 0.7786 balanced accuracy / 0.7694 macro-F1 and retrieval 0.8135 / 0.8078; on DEV_STRUCTURAL, memory 0.7800 / 0.7731 and retrieval 0.7773 / 0.7699. All scores remain below the review threshold.

## Interpretation and limits

Passing this audit means the declared univariate thresholds and the cheap-classifier review threshold were met on frozen run `55955b2eff38`. It does not prove absence of all shortcuts, establish robustness to a new grammar, or demonstrate a router's accuracy. Any later generator, query-template, answerability, or runtime-observable feature change requires a fresh audit. See `runs/integration/u2f-owned-v1-55955b2eff38/shortcut_audit.json` and `cheap_classifier_audit.json` for per-feature rules, thresholds, train/evaluation prevalence, feature exclusions, and attribution lists.
