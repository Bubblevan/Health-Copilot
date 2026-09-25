"""Analyze the frozen E1.3 OOF predictions and write the RAG closeout artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from eval.e1_3_learned_router import (
    LEARNED_POLICIES,
    validate_oof_prediction_coverage,
)
from eval.e1_3_router_dataset import (
    FIXED_ACTIONS,
    canonical_sha256,
    case_order_sha256,
    load_cases,
    load_historical_arms,
    make_cost_oracle_rows,
    read_jsonl,
    sha256_file,
)

DISPLAY_NAMES = {
    "closed_book": "Closed only",
    "rag_bm25": "Always BM25",
    "rag_medcpt": "Always MedCPT",
    "random_context": "Random context",
    "cheap_router": "Cheap router",
    "jev_router": "Jev router",
    "tfidf_direct": "TF-IDF direct",
    "tfidf_hierarchical": "TF-IDF hierarchical",
    "tfidf_hierarchical_always_medcpt": "TF-IDF + always MedCPT",
    "bge_hierarchical": "BGE hierarchical",
    "bge_hierarchical_always_medcpt": "BGE + always MedCPT",
    "cost_oracle_v2": "Cost oracle v2",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temporary.flush()
        os.fsync(temporary.fileno())
    os.replace(temporary_path, path)


def measured_percentile(values: list[float | int | None], percentile: float) -> float | None:
    measured = [float(value) for value in values if value is not None]
    return float(np.percentile(measured, percentile)) if measured else None


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    token_values = [row.get("answer_input_tokens") for row in rows]
    latency_values = [row.get("component_latency_proxy_ms") for row in rows]
    context_values = [row.get("context_characters") for row in rows]
    measured_tokens = [int(value) for value in token_values if value is not None]
    measured_context = [int(value) for value in context_values if value is not None]
    calls = sum(int(row.get("retrieval_calls") or 0) for row in rows)
    correct = sum(bool(row["selected_arm_correct"]) for row in rows)
    return {
        "cases": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "retrieval_calls": calls,
        "retrieval_rate": calls / n if n else None,
        "answer_input_tokens": sum(measured_tokens),
        "answer_input_token_measurement_count": len(measured_tokens),
        "answer_input_token_measurement_coverage": len(measured_tokens) / n if n else None,
        "context_characters": sum(measured_context),
        "context_character_measurement_count": len(measured_context),
        "component_latency_proxy_p50_ms": measured_percentile(latency_values, 50),
        "component_latency_proxy_p95_ms": measured_percentile(latency_values, 95),
        "component_latency_measurement_count": sum(value is not None for value in latency_values),
        "component_latency_measurement_coverage": (
            sum(value is not None for value in latency_values) / n if n else None
        ),
        "latency_semantics": "Historical component-summed proxy; not observed production end-to-end latency and excludes learned-classifier inference time.",
    }


def arm_rows_to_dicts(arm_rows: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": row.case_id,
            "subdataset": row.subdataset,
            "selected_arm_correct": row.is_correct,
            "retrieval_calls": row.retrieval_calls,
            "answer_input_tokens": row.answer_input_tokens,
            "context_characters": row.context_characters,
            "component_latency_proxy_ms": row.component_latency_proxy_ms,
        }
        for row in arm_rows.values()
    ]


def action_metrics_rows(
    action: str, arms: dict[str, dict[str, Any]], case_ids: list[str]
) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case_id,
            "subdataset": arms[action][case_id].subdataset,
            "selected_arm_correct": arms[action][case_id].is_correct,
            "retrieval_calls": arms[action][case_id].retrieval_calls,
            "answer_input_tokens": arms[action][case_id].answer_input_tokens,
            "context_characters": arms[action][case_id].context_characters,
            "component_latency_proxy_ms": arms[action][case_id].component_latency_proxy_ms,
        }
        for case_id in case_ids
    ]


def oof_metrics(rows: list[dict[str, Any]], oracle_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    y_true = np.asarray([row["retrieval_benefit"] for row in rows], dtype=bool)
    y_score = np.asarray([row["retrieval_probability"] for row in rows], dtype=np.float64)
    y_route = np.asarray([row["selected_action"] != "closed_book" for row in rows], dtype=bool)
    precision = float(precision_score(y_true, y_route, zero_division=0))
    recall = float(recall_score(y_true, y_route, zero_division=0))
    stage1 = {
        "positive_prevalence": float(np.mean(y_true)),
        "pr_auc_average_precision": float(average_precision_score(y_true, y_score)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "benefit_precision_at_policy_route": precision,
        "benefit_recall_at_policy_route": recall,
    }
    rescue_rows = [row for row in rows if row["retrieval_benefit"]]
    stage2_result = None
    if rescue_rows:
        targets = [oracle_by_id[row["case_id"]]["stage2_target"] for row in rescue_rows]
        predictions = [
            "rag_bm25"
            if row["stage2_bm25_probability"] is not None
            and row["stage2_bm25_probability"] >= 0.5
            else "rag_medcpt"
            for row in rescue_rows
        ]
        target_counts = Counter(targets)
        majority_action = min(target_counts, key=lambda action: (-target_counts[action], action))
        majority_predictions = [majority_action] * len(targets)
        selector_accuracy = float(np.mean(np.asarray(targets) == np.asarray(predictions)))
        majority_accuracy = float(np.mean(np.asarray(targets) == np.asarray(majority_predictions)))
        stage2_result = {
            "rescue_cases": len(rescue_rows),
            "bm25_medcpt_accuracy": selector_accuracy,
            "macro_f1": float(
                f1_score(targets, predictions, labels=["rag_bm25", "rag_medcpt"], average="macro")
            ),
            "majority_baseline_action": majority_action,
            "majority_baseline_accuracy": majority_accuracy,
            "majority_baseline_macro_f1": float(
                f1_score(
                    targets,
                    majority_predictions,
                    labels=["rag_bm25", "rag_medcpt"],
                    average="macro",
                )
            ),
            "delta_vs_majority_accuracy_pp": (selector_accuracy - majority_accuracy) * 100,
        }

    return {"stage1": stage1, "stage2": stage2_result}


def failure_taxonomy(
    rows: list[dict[str, Any]],
    oracle_by_id: dict[str, dict[str, Any]],
) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        case_id = row["case_id"]
        action = row["selected_action"]
        labels = oracle_by_id[case_id]
        fixed = labels["fixed_correct"]
        closed_correct = fixed["closed_book"]
        retrieval_benefit = bool(labels["retrieval_benefit"])
        action_is_retrieval = action != "closed_book"
        action_correct = bool(row["selected_arm_correct"])

        if retrieval_benefit and not action_is_retrieval:
            counts["MISSED_RESCUE"] += 1
        if action_is_retrieval and closed_correct:
            counts["WASTED_RETRIEVAL"] += 1
        if action_is_retrieval and closed_correct and not action_correct:
            counts["HARMFUL_RETRIEVAL"] += 1
        if action_is_retrieval and not any(fixed.values()):
            counts["UNRESOLVED_RETRIEVAL"] += 1
        if (
            retrieval_benefit
            and action_is_retrieval
            and not action_correct
            and any(fixed[other] for other in ("rag_bm25", "rag_medcpt") if other != action)
        ):
            counts["WRONG_RETRIEVER"] += 1
    return {
        key: counts[key]
        for key in (
            "MISSED_RESCUE",
            "WASTED_RETRIEVAL",
            "HARMFUL_RETRIEVAL",
            "UNRESOLVED_RETRIEVAL",
            "WRONG_RETRIEVER",
        )
    }


def fold_stability(rows: list[dict[str, Any]], fold_count: int) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for policy in LEARNED_POLICIES:
        policy_rows = [row for row in rows if row["policy"] == policy]
        per_fold = {}
        for fold in range(fold_count):
            selected = [row for row in policy_rows if row["fold"] == fold]
            y_true = np.asarray([row["retrieval_benefit"] for row in selected], dtype=bool)
            y_score = np.asarray(
                [row["retrieval_probability"] for row in selected], dtype=np.float64
            )
            per_fold[str(fold)] = {
                "cases": len(selected),
                "accuracy": sum(bool(row["selected_arm_correct"]) for row in selected)
                / len(selected),
                "retrieval_rate": sum(int(row["retrieval_calls"]) for row in selected)
                / len(selected),
                "retrieval_benefit_pr_auc": float(average_precision_score(y_true, y_score)),
                "thresholds": sorted(
                    {
                        row["threshold"]
                        for row in selected
                        if row.get("threshold") is not None
                    }
                ),
                "stage2_fallback": any(bool(row.get("stage2_fallback")) for row in selected),
            }
        report[policy] = per_fold
    return report


def paired_stratified_bootstrap(
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    subdatasets: list[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    candidate = {row["case_id"]: bool(row["selected_arm_correct"]) for row in candidate_rows}
    baseline = {row["case_id"]: bool(row["selected_arm_correct"]) for row in baseline_rows}
    if set(candidate) != set(baseline):
        raise ValueError("Paired bootstrap inputs do not contain identical case sets")
    ids_by_subdataset: dict[str, list[str]] = {}
    for row in candidate_rows:
        ids_by_subdataset.setdefault(row["subdataset"], []).append(row["case_id"])
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled: list[str] = []
        for subdataset in subdatasets:
            ids = ids_by_subdataset[subdataset]
            sampled.extend(rng.choice(ids, size=len(ids), replace=True).tolist())
        deltas[replicate] = np.mean(
            [candidate[case_id] - baseline[case_id] for case_id in sampled]
        )
    return {
        "delta_accuracy": float(np.mean(list(candidate.values())) - np.mean(list(baseline.values()))),
        "delta_accuracy_pp": float(
            (np.mean(list(candidate.values())) - np.mean(list(baseline.values()))) * 100
        ),
        "ci_95_pp": [float(value * 100) for value in np.quantile(deltas, [0.025, 0.975])],
        "replicates": replicates,
        "stratification": "paired resampling within each MIRAGE subdataset at fixed subdataset sample sizes",
        "interpretation": "Exploratory OOF bootstrap interval; not confirmatory significance.",
    }


def relative_cost(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    def percent_change(value: float | None, reference: float | None) -> float | None:
        if value is None or reference is None or reference == 0:
            return None
        return float((value - reference) / reference * 100)

    def reduction_percent(value: float | None, reference: float | None) -> float | None:
        if value is None or reference is None or reference == 0:
            return None
        return float((reference - value) / reference * 100)

    return {
        "candidate_absolute": {
            "accuracy": candidate["accuracy"],
            "retrieval_calls": candidate["retrieval_calls"],
            "answer_input_tokens": candidate["answer_input_tokens"],
            "component_latency_proxy_p95_ms": candidate["component_latency_proxy_p95_ms"],
        },
        "baseline_absolute": {
            "accuracy": baseline["accuracy"],
            "retrieval_calls": baseline["retrieval_calls"],
            "answer_input_tokens": baseline["answer_input_tokens"],
            "component_latency_proxy_p95_ms": baseline["component_latency_proxy_p95_ms"],
        },
        "delta": {
            "accuracy_pp": (candidate["accuracy"] - baseline["accuracy"]) * 100,
            "retrieval_calls": candidate["retrieval_calls"] - baseline["retrieval_calls"],
            "retrieval_calls_percent_change": percent_change(
                candidate["retrieval_calls"], baseline["retrieval_calls"]
            ),
            "retrieval_calls_reduction_percent": reduction_percent(
                candidate["retrieval_calls"], baseline["retrieval_calls"]
            ),
            "answer_input_tokens": candidate["answer_input_tokens"]
            - baseline["answer_input_tokens"],
            "answer_input_tokens_percent_change": percent_change(
                candidate["answer_input_tokens"], baseline["answer_input_tokens"]
            ),
            "answer_input_tokens_reduction_percent": reduction_percent(
                candidate["answer_input_tokens"], baseline["answer_input_tokens"]
            ),
            "component_latency_proxy_p95_ms": candidate["component_latency_proxy_p95_ms"]
            - baseline["component_latency_proxy_p95_ms"],
            "component_latency_proxy_p95_percent_change": percent_change(
                candidate["component_latency_proxy_p95_ms"],
                baseline["component_latency_proxy_p95_ms"],
            ),
        },
    }


def pareto_frontier(metrics: dict[str, dict[str, Any]]) -> list[str]:
    frontier = []
    for candidate_name, candidate in metrics.items():
        dominated = False
        for other_name, other in metrics.items():
            if other_name == candidate_name:
                continue
            no_worse = (
                other["accuracy"] >= candidate["accuracy"]
                and other["retrieval_rate"] <= candidate["retrieval_rate"]
            )
            strictly_better = (
                other["accuracy"] > candidate["accuracy"]
                or other["retrieval_rate"] < candidate["retrieval_rate"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate_name)
    return sorted(frontier)


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# E1.3 Question-only Capability Router — OOF Results",
        "",
        "> Evaluation status: **EXPOSED_EXPLORATORY_OOF**. This is not an external test, untouched holdout, or confirmatory evaluation.",
        "> All learned predictions are outer-fold OOF. QA outcomes, tokens, retrieval calls, and latency are counterfactual lookups from existing E1.2 rows; no answer inference was run.",
        "",
        "## Frozen inputs and integrity",
        "",
        f"- Cases: {report['case_count']:,}; outer folds: {report['outer_fold_count']}; seed: {report['seed']}; stratification: `{report['stratification']}`.",
        f"- OOF runner commit: `{report['oof_runner_commit']}`; frozen protocol SHA-256: `{report['protocol_sha256']}`.",
        "- Git artifacts omit question text, answer options, gold answers, generated answers, and retrieved evidence; only case identities, labels, actions, and aggregate/counterfactual metrics are recorded.",
        f"- Train/evaluation overlap: {report['oof_integrity']['train_eval_overlap_cases']}.",
        f"- OOF duplicates: {report['oof_integrity']['duplicate_case_policy_predictions']}; missing predictions: {report['oof_integrity']['missing_case_policy_predictions']}.",
        f"- Disk free GiB start → end: C: {report['disk_free_start_gib']['C']:.2f} → {report['disk_free_end_gib']['C']:.2f}; D: {report['disk_free_start_gib']['D']:.2f} → {report['disk_free_end_gib']['D']:.2f}; E: {report['disk_free_start_gib']['E']:.2f} → {report['disk_free_end_gib']['E']:.2f}.",
        f"- Retrieval-benefit prevalence: {report['cost_oracle_v2']['retrieval_benefit_prevalence']:.2%} ({report['cost_oracle_v2']['retrieval_benefit_cases']:,}/{report['case_count']:,}).",
        f"- Cost Oracle v2 classes: CLOSED {report['cost_oracle_v2']['class_counts']['CLOSED_SUFFICIENT']:,}; BM25 rescue {report['cost_oracle_v2']['class_counts']['BM25_RESCUE']:,}; MedCPT rescue {report['cost_oracle_v2']['class_counts']['MEDCPT_RESCUE']:,}; unresolved {report['cost_oracle_v2']['class_counts']['UNRESOLVED']:,}.",
        "",
        "## Stage 1 — retrieval-benefit detection",
        "",
        "| Detector | PR-AUC (Average Precision) | AUROC | Precision | Recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for policy, label in (
        ("tfidf_hierarchical", "TF-IDF"),
        ("bge_hierarchical", "BGE"),
        ("tfidf_direct", "TF-IDF direct (diagnostic)"),
    ):
        stage = report["stage1_metrics"][policy]
        lines.append(
            f"| {label} | {stage['pr_auc_average_precision']:.4f} | {stage['auroc']:.4f} | {stage['benefit_precision_at_policy_route']:.2%} | {stage['benefit_recall_at_policy_route']:.2%} |"
        )
    lines.extend(
        [
            "",
            f"Random-ranking baseline equals positive prevalence: **{report['cost_oracle_v2']['retrieval_benefit_prevalence']:.4f}** average precision.",
            "",
            "## Stage 2 — retriever selection on rescue cases only",
            "",
            "| Selector | Cases | Accuracy | Macro F1 | Majority-BM25 accuracy |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for policy, label in (("tfidf_hierarchical", "TF-IDF"), ("bge_hierarchical", "BGE")):
        stage = report["stage2_metrics"][policy]
        lines.append(
            f"| {label} | {stage['rescue_cases']:,} | {stage['bm25_medcpt_accuracy']:.2%} | {stage['macro_f1']:.4f} | {stage['majority_baseline_accuracy']:.2%} |"
        )
    lines.extend(
        [
            "",
            "Stage 2 is evaluated only on rescue cases. The majority-action figure is a class-imbalance diagnostic, not the end-task policy baseline.",
            "",
            "## End-task quality and measured historical cost",
            "",
            "| Policy | QA accuracy | Retrieval rate | Calls | Answer input tokens | p50 proxy ms | p95 proxy ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["policy_table"]:
        p50 = "—" if row["component_latency_proxy_p50_ms"] is None else f"{row['component_latency_proxy_p50_ms']:.1f}"
        p95 = "—" if row["component_latency_proxy_p95_ms"] is None else f"{row['component_latency_proxy_p95_ms']:.1f}"
        lines.append(
            f"| {DISPLAY_NAMES[row['policy']]} | {row['accuracy']:.2%} | {row['retrieval_rate']:.2%} | {row['retrieval_calls']:,} | {row['answer_input_tokens']:,} | {p50} | {p95} |"
        )
    lines.extend(["", "Latency is a component-summed historical proxy, not production end-to-end latency; learned-classifier inference time is not included.", ""])
    lines.extend(
        [
            "## Best learned policy: absolute deltas against fixed references",
            "",
            "| Reference | Δ accuracy (pp) | Δ retrieval calls | Calls change | Δ input tokens | Tokens change | Δ p95 proxy ms | p95 change |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for relative_key, baseline_label in (
        ("vs_cheap_router", "Cheap"),
        ("vs_always_medcpt", "Always MedCPT"),
    ):
        delta = report["relative_cost"][report["best_learned_policy"]][relative_key]["delta"]
        lines.append(
            f"| {baseline_label} | {delta['accuracy_pp']:+.3f} | {delta['retrieval_calls']:+,} | {delta['retrieval_calls_percent_change']:+.1f}% | {delta['answer_input_tokens']:+,} | {delta['answer_input_tokens_percent_change']:+.1f}% | {delta['component_latency_proxy_p95_ms']:+.1f} | {delta['component_latency_proxy_p95_percent_change']:+.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Paired exploratory bootstrap",
            "",
            f"- Best learned `{report['best_learned_policy']}` vs Cheap: Δ accuracy {report['paired_bootstrap']['best_learned_vs_cheap']['delta_accuracy_pp']:+.3f} pp; 95% CI [{report['paired_bootstrap']['best_learned_vs_cheap']['ci_95_pp'][0]:+.3f}, {report['paired_bootstrap']['best_learned_vs_cheap']['ci_95_pp'][1]:+.3f}] pp.",
            f"- Best learned vs Always MedCPT: Δ accuracy {report['paired_bootstrap']['best_learned_vs_medcpt']['delta_accuracy_pp']:+.3f} pp; 95% CI [{report['paired_bootstrap']['best_learned_vs_medcpt']['ci_95_pp'][0]:+.3f}, {report['paired_bootstrap']['best_learned_vs_medcpt']['ci_95_pp'][1]:+.3f}] pp.",
            "- The best learned policy is selected by the pre-frozen highest pooled-OOF-accuracy rule. These paired intervals compare that selected policy and do not adjust for policy-selection optimism; they are exploratory summaries, not confirmatory significance.",
            "",
            "## Relative cost and headroom",
            "",
            f"- Best fixed accuracy: {report['headroom']['best_fixed_accuracy']:.2%}; Closed: {report['headroom']['closed_accuracy']:.2%}; Cheap: {report['headroom']['cheap_accuracy']:.2%}; Best learned: {report['headroom']['best_learned_accuracy']:.2%}; Cost Oracle v2: {report['headroom']['oracle_v2_accuracy']:.2%}.",
            f"- Oracle-v2 uplift: +{report['headroom']['oracle_headroom_pp']:.3f} pp over Closed and +{report['headroom']['oracle_headroom_over_best_fixed_pp']:.3f} pp over the best fixed arm; learned gain over Closed: {report['headroom']['learned_gain_over_closed_pp']:+.3f} pp; closed-to-oracle headroom recovered: {format_optional_percent(report['headroom']['oracle_headroom_recovered_fraction'])}.",
            f"- Learned gain over Cheap: {report['headroom']['learned_gain_over_cheap_pp']:+.3f} pp; relative to Cheap→Oracle headroom: {format_optional_percent(report['headroom']['cheap_to_oracle_headroom_recovered_fraction'])}.",
            f"- Pareto frontier: {', '.join(DISPLAY_NAMES[name] for name in report['pareto_frontier'])}.",
            "",
            "## Routing failure taxonomy",
            "",
            "Counts may overlap: HARMFUL_RETRIEVAL is a subset of WASTED_RETRIEVAL.",
            "",
            "| Policy | Missed rescue | Wasted retrieval | Harmful retrieval | Unresolved retrieval | Wrong retriever |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for policy in LEARNED_POLICIES:
        taxonomy = report["failure_taxonomy"][policy]
        lines.append(
            f"| {DISPLAY_NAMES[policy]} | {taxonomy['MISSED_RESCUE']} | {taxonomy['WASTED_RETRIEVAL']} | {taxonomy['HARMFUL_RETRIEVAL']} | {taxonomy['UNRESOLVED_RETRIEVAL']} | {taxonomy['WRONG_RETRIEVER']} |"
        )
    lines.extend(
        [
            "",
            "## Fold stability",
            "",
            "| Policy | Fold | Accuracy | Retrieval rate | Retrieval-benefit PR-AUC |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for policy in LEARNED_POLICIES:
        for fold, values in report["fold_stability"][policy].items():
            lines.append(
                f"| {DISPLAY_NAMES[policy]} | {fold} | {values['accuracy']:.2%} | {values['retrieval_rate']:.2%} | {values['retrieval_benefit_pr_auc']:.4f} |"
            )
    lines.extend(["", "## Research questions and closeout gates", ""])
    for name, answer in report["research_questions"].items():
        lines.append(f"- **{name}:** {answer['answer']} — {answer['evidence']}")
    lines.extend(
        [
            "",
            "| Gate | Result |",
            "| --- | --- |",
        ]
    )
    for gate, result in report["gates"].items():
        lines.append(f"| `{gate}` | **{result}** |")
    lines.extend(
        [
            "",
            "Interpretation: keep this as an exposed exploratory routing study. Do not train SFT/GRPO from these OOF outcomes in this sprint. Any future post-training work needs a separately frozen evaluation design and genuinely unseen evaluation data.",
            "",
        ]
    )
    return "\n".join(lines)


def format_optional_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def closeout_markdown(report: dict[str, Any], mirage: dict[str, Any]) -> str:
    frontier = ", ".join(DISPLAY_NAMES[name] for name in report["pareto_frontier"])
    e11 = report["external_retrieval_baselines"]["nfcorpus"]
    r2 = report["external_retrieval_baselines"]["r2med"]
    lines = [
        "# RAG Line Closeout",
        "",
        "> Scope: resume evidence engineering, not clinical validation. MIRAGE answer evaluation is exposed exploratory; the learned router uses OOF on that exposed case set.",
        "",
        "## External baseline reproduction",
        "",
        "### NFCorpus",
        "",
        f"On NFCorpus test (323 queries; 3,633 documents), BM25 Recall@10/MRR/nDCG@10 = {e11['bm25']['recall_at_10']:.3f}/{e11['bm25']['mrr']:.3f}/{e11['bm25']['ndcg_at_10']:.3f}; BGE dense = {e11['learned_dense']['recall_at_10']:.3f}/{e11['learned_dense']['mrr']:.3f}/{e11['learned_dense']['ndcg_at_10']:.3f}; hybrid RRF nDCG@10 = {e11['hybrid_rrf']['ndcg_at_10']:.3f}; cross-encoder hybrid nDCG@10 = {e11['hybrid_crossencoder']['ndcg_at_10']:.3f}. Dense underperformed BM25 on this corpus; the hybrid did not improve nDCG over BM25.",
        "",
        "### R2MED",
        "",
        f"The pinned R2MED TEST contains {r2['test_query_count']} queries across three subsets. DEV selected BGE dense as the strongest fixed baseline. The RRF + MedCPT rerank method scored macro nDCG@10 {r2['custom_ndcg_at_10']:.3f} vs BGE dense {r2['selected_baseline_ndcg_at_10']:.3f} (Δ {r2['custom_delta_ndcg_at_10']:+.3f}; paired 95% CI [{r2['custom_ci_95'][0]:+.3f}, {r2['custom_ci_95'][1]:+.3f}]); the frozen headline gate failed. BM25/MedCPT/BGE parity and canonical-vs-R2MED MedCPT wiring are documented in `docs/research/e1_2_r2med_parity_audit.md`.",
        "",
        "## Medical QA — MIRAGE (exposed exploratory, n=5,235)",
        "",
        "| Arm | Accuracy | Retrieval calls | Input tokens |",
        "| --- | ---: | ---: | ---: |",
    ]
    arm_report = mirage["arms"]
    for arm, title in (
        ("closed_book", "Closed book"),
        ("random_context", "Random context"),
        ("rag_bm25", "BM25"),
        ("rag_medcpt", "MedCPT"),
        ("cheap_router", "Cheap router"),
        ("jev_router", "Jev router"),
    ):
        row = arm_report[arm]
        lines.append(
            f"| {title} | {row['accuracy_fixed_denominator']:.2%} | {row['retrieval_calls']:,} | {row.get('answer_input_tokens_observed_total', '—'):,} |"
        )
    lines.extend(
        [
            "",
            f"The fixed-arm accuracy range is narrow: closed book {arm_report['closed_book']['accuracy_fixed_denominator']:.2%}, BM25 {arm_report['rag_bm25']['accuracy_fixed_denominator']:.2%}, MedCPT {arm_report['rag_medcpt']['accuracy_fixed_denominator']:.2%}. Random context scored {arm_report['random_context']['accuracy_fixed_denominator']:.2%}; this does not support a generic claim that adding context helps. MedCPT is the best fixed arm, but its gain over closed book is only {(arm_report['rag_medcpt']['accuracy_fixed_denominator']-arm_report['closed_book']['accuracy_fixed_denominator'])*100:.2f} pp while using {(arm_report['rag_medcpt']['answer_input_tokens_observed_total']/arm_report['closed_book']['answer_input_tokens_observed_total']-1)*100:.0f}% more answer input tokens.",
            "",
            "## Adaptive retrieval and question-only OOF router",
            "",
            "| Policy | Accuracy | Retrieval calls | Input tokens |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name in (
        "cheap_router",
        "jev_router",
        "tfidf_direct",
        "tfidf_hierarchical",
        "tfidf_hierarchical_always_medcpt",
        "bge_hierarchical",
        "bge_hierarchical_always_medcpt",
        "cost_oracle_v2",
    ):
        row = report["comparison_metrics"][name]
        lines.append(
            f"| {DISPLAY_NAMES[name]} | {row['accuracy']:.2%} | {row['retrieval_calls']:,} | {row['answer_input_tokens']:,} |"
        )
    headroom = report["headroom"]
    lines.extend(
        [
            "",
            f"The q-only post-hoc cost oracle achieves {headroom['oracle_v2_accuracy']:.2%} at {report['comparison_metrics']['cost_oracle_v2']['retrieval_calls']:,} calls. This is an exposed upper bound, not an executable router. The best learned OOF policy is `{report['best_learned_policy']}` at {headroom['best_learned_accuracy']:.2%}; its recovered oracle headroom is {format_optional_percent(headroom['oracle_headroom_recovered_fraction'])}.",
            f"Oracle-v2 is +{headroom['oracle_headroom_pp']:.3f} pp over Closed and +{headroom['oracle_headroom_over_best_fixed_pp']:.3f} pp over the best fixed arm. The best learned policy recovers {format_optional_percent(headroom['oracle_headroom_recovered_fraction'])} of Closed-to-oracle headroom and {format_optional_percent(headroom['cheap_to_oracle_headroom_recovered_fraction'])} of Cheap-to-oracle headroom.",
            f"Observed Pareto frontier (including the non-executable oracle): {frontier}.",
            "",
            "## Negative and limiting results retained",
            "",
            "- Dense retrieval is not universally better: on NFCorpus BM25 beats BGE dense; on R2MED, BGE dense beats BM25, while MedCPT dense is weak on the pinned R2MED wiring.",
            "- Canonical MedCPT and R2MED ARTICLE_ARTICLE use different query/document wiring; the parity audit explains the gap rather than treating these as interchangeable.",
            "- On MIRAGE, always-RAG accuracy gains over closed-book are small relative to retrieval calls and answer-token cost; random context can reduce accuracy.",
            "- Jev routing spends almost as many retrieval calls as always-RAG for little accuracy gain; no Jev API was called in this sprint.",
            "- The learned question-only routers and their exact OOF deltas are reported as exploratory. The post-hoc oracle headroom is not evidence that a deployable router can recover it.",
            "- Historical PubMedQA/BioASQ results and the earlier Harness RAG abstention-heavy result remain historical exposed artifacts, not rerun or mixed into the 5,235-case primary comparison.",
            "",
            "## Resume evidence candidates",
            "",
        ]
    )
    best = report["best_learned_policy"]
    best_metrics = report["comparison_metrics"][best]
    cheap_metrics = report["comparison_metrics"]["cheap_router"]
    if report["gates"]["STRONG_CAPABILITY_ROUTING_SIGNAL"] == "YES":
        lines.append(
            f"- Candidate, with exposed exploratory qualification: On the 5,235-case MIRAGE exploratory benchmark, `{best}` reached {best_metrics['accuracy']:.2%} accuracy with {best_metrics['retrieval_calls']:,} retrieval calls ({best_metrics['answer_input_tokens']:,} measured answer input tokens); report the direct comparison and paired OOF interval, not a clinical claim."
        )
    elif report["gates"]["SFT_CANDIDATE"] == "YES":
        lines.append(
            f"- Candidate, only as a development result: On the 5,235-case MIRAGE exploratory benchmark, `{best}` reached {best_metrics['accuracy']:.2%} vs Cheap router {cheap_metrics['accuracy']:.2%}; include the actual retrieval/token change and clearly label the OOF exploratory design."
        )
    else:
        lines.append(
            "- No accuracy/cost headline is supported by the frozen gates. A factual engineering-only line may describe the reproducible R2MED parity audit and the question-only OOF evaluation harness, but should not imply a measured positive routing gain."
        )
        trade = report["relative_cost"]["tfidf_direct"]["vs_cheap_router"]
        ci = report["paired_bootstrap"]["best_learned_vs_cheap"]["ci_95_pp"]
        lines.append(
            f"- The TF-IDF-direct point estimate is a cost/quality tradeoff, not a Pareto win: {trade['delta']['retrieval_calls_reduction_percent']:.1f}% fewer calls and {trade['delta']['answer_input_tokens_reduction_percent']:.1f}% fewer measured input tokens vs Cheap, for {trade['delta']['accuracy_pp']:+.3f} pp accuracy (best-learned paired exploratory 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}] pp). Treat as a follow-up hypothesis only; the interval spans zero and there is no external validation."
        )
    lines.extend(
        [
            "",
            "Do not describe these results as clinical accuracy, clinical validation, an untouched holdout, or state of the art. Do not train SFT/GRPO from this sprint; use a future separately frozen, genuinely unseen evaluation set if pursuing post-training.",
            "",
            "## Final status",
            "",
            f"- `RAG_BASELINES_COMPLETE = {report['gates']['RAG_BASELINES_COMPLETE']}`",
            f"- `OOF_PROTOCOL_VALID = {report['gates']['OOF_PROTOCOL_VALID']}`",
            f"- `COST_ORACLE_V2_READY = {report['gates']['COST_ORACLE_V2_READY']}`",
            f"- `TFIDF_SIGNAL_FOUND = {report['gates']['TFIDF_SIGNAL_FOUND']}`",
            f"- `BGE_SIGNAL_FOUND = {report['gates']['BGE_SIGNAL_FOUND']}`",
            f"- `BGE_BEATS_TFIDF = {report['gates']['BGE_BEATS_TFIDF']}`",
            f"- `LEARNED_ROUTER_PARETO_DOMINATES_CHEAP = {report['gates']['LEARNED_ROUTER_PARETO_DOMINATES_CHEAP']}`",
            f"- `STRONG_CAPABILITY_ROUTING_SIGNAL = {report['gates']['STRONG_CAPABILITY_ROUTING_SIGNAL']}`",
            f"- `SFT_CANDIDATE = {report['gates']['SFT_CANDIDATE']}`",
            "- `RAG_CLOSEOUT = YES`",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(repo_root: Path, scratch_root: Path) -> dict[str, Any]:
    artifact_root = repo_root / "runs" / "e1_3"
    protocol = read_json(artifact_root / "router_protocol.json")
    if canonical_sha256({k: v for k, v in protocol.items() if k != "protocol_sha256"}) != protocol.get("protocol_sha256"):
        raise ValueError("Frozen protocol integrity check failed")
    fold_manifest = read_json(artifact_root / "router_fold_manifest.json")
    if canonical_sha256({k: v for k, v in fold_manifest.items() if k != "manifest_sha256"}) != fold_manifest.get("manifest_sha256"):
        raise ValueError("Frozen fold manifest integrity check failed")

    input_path = Path(protocol["benchmark"]["input_artifact"])
    if sha256_file(input_path) != protocol["benchmark"]["input_sha256"]:
        raise ValueError("Frozen MIRAGE input SHA-256 mismatch")
    cases = load_cases(input_path)
    case_ids = [case.case_id for case in cases]
    if case_order_sha256(case_ids) != protocol["benchmark"]["case_order_sha256"]:
        raise ValueError("Question case order differs from the frozen protocol")

    oof_path = artifact_root / "router_oof_predictions.jsonl"
    oof_rows = read_jsonl(oof_path)
    validate_oof_prediction_coverage(oof_rows, case_ids)
    run_manifest = read_json(artifact_root / "router_oof_run_manifest.json")
    if sha256_file(oof_path) != run_manifest.get("oof_predictions_sha256"):
        raise ValueError("OOF prediction artifact SHA-256 mismatch")
    if run_manifest.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError("OOF output is not bound to the current frozen protocol")
    if run_manifest.get("case_count") != len(case_ids) or run_manifest.get("prediction_rows") != len(oof_rows):
        raise ValueError("OOF run manifest case or row count mismatch")
    if (
        run_manifest.get("fold_manifest_sha256") != protocol["outer_cv"]["fold_manifest_sha256"]
        or run_manifest.get("cost_oracle_v2_sha256") != protocol["cost_oracle_v2"]["sha256"]
        or run_manifest.get("policies") != protocol["learned_policies"]
        or run_manifest.get("question_text_in_artifact") is not False
        or run_manifest.get("options_or_answers_in_artifact") is not False
        or run_manifest.get("new_answer_inference") is not False
        or not isinstance(run_manifest.get("implementation_commit"), str)
        or len(run_manifest["implementation_commit"]) != 40
    ):
        raise ValueError("OOF run manifest is missing frozen provenance or privacy attestations")

    embedding_manifest = read_json(artifact_root / "router_embedding_manifest.json")
    cache_root = Path(protocol["models"]["bge"]["cache_path"]).parent
    embedding_path = cache_root / "bge_embeddings.npy"
    cached_embedding_manifest = read_json(cache_root / "bge_embedding_manifest.json")
    if embedding_manifest != cached_embedding_manifest:
        raise ValueError("Committed-safe BGE manifest differs from the external cache manifest")
    if sha256_file(embedding_path) != embedding_manifest.get("embedding_file_sha256"):
        raise ValueError("BGE question embedding cache SHA-256 mismatch")
    if (
        embedding_manifest.get("model_weights_sha256")
        != protocol["models"]["bge"]["weights_sha256"]
        or embedding_manifest.get("case_id_order_sha256")
        != protocol["benchmark"]["case_order_sha256"]
        or embedding_manifest.get("embedding_shape") != [len(case_ids), 768]
        or embedding_manifest.get("protocol_sha256") != protocol["protocol_sha256"]
    ):
        raise ValueError("BGE question embedding manifest identity mismatch")

    oracle_path = artifact_root / "router_cost_oracle_v2.jsonl"
    if sha256_file(oracle_path) != protocol["cost_oracle_v2"]["sha256"]:
        raise ValueError("Cost Oracle v2 hash mismatch")
    oracle_rows = read_jsonl(oracle_path)
    oracle_by_id = {row["case_id"]: row for row in oracle_rows}
    if len(oracle_by_id) != len(case_ids) or set(oracle_by_id) != set(case_ids):
        raise ValueError("Cost Oracle v2 case identity mismatch")
    arms = load_historical_arms(scratch_root, case_ids)
    if make_cost_oracle_rows(cases, arms) != oracle_rows:
        raise ValueError("Cost Oracle v2 does not reproduce from the pinned historical arms")
    for arm, identity in protocol["historical_arm_sources"].items():
        arm_root = scratch_root / "runs" / "e1_2" / "test" / arm
        if (
            sha256_file(arm_root / "case_results.jsonl") != identity["case_results_sha256"]
            or sha256_file(arm_root / "manifest.json") != identity["manifest_sha256"]
        ):
            raise ValueError(f"Historical {arm} sources changed since protocol freeze")

    fold_map = fold_manifest["case_id_to_fold"]
    oof_id_policy = {(row["case_id"], row["policy"]) for row in oof_rows}
    expected = {(case_id, policy) for case_id in case_ids for policy in LEARNED_POLICIES}
    if oof_id_policy != expected:
        raise ValueError("OOF predictions are not a complete case/policy cross product")
    overlap = sum(
        row["fold"] != fold_map[row["case_id"]]
        for row in oof_rows
    )
    if overlap:
        raise ValueError("An OOF prediction fold does not match its frozen case assignment")
    duplicate_count = len(oof_rows) - len(oof_id_policy)
    missing_count = len(expected - oof_id_policy)

    comparison_rows: dict[str, list[dict[str, Any]]] = {}
    for action in FIXED_ACTIONS:
        comparison_rows[action] = action_metrics_rows(action, arms, case_ids)
    for name in ("cheap_router", "jev_router"):
        comparison_rows[name] = action_metrics_rows(name, arms, case_ids)
    learned_by_policy = {
        policy: [row for row in oof_rows if row["policy"] == policy]
        for policy in LEARNED_POLICIES
    }
    comparison_rows.update(learned_by_policy)

    oracle_metric_rows = []
    for row in oracle_rows:
        action = row["cost_oracle_action"]
        result = arms[action][row["case_id"]]
        oracle_metric_rows.append(
            {
                "case_id": row["case_id"],
                "subdataset": row["subdataset"],
                "selected_arm_correct": result.is_correct,
                "retrieval_calls": result.retrieval_calls,
                "answer_input_tokens": result.answer_input_tokens,
                "context_characters": result.context_characters,
                "component_latency_proxy_ms": result.component_latency_proxy_ms,
            }
        )
    comparison_rows["cost_oracle_v2"] = oracle_metric_rows

    mirage = read_json(artifact_root / "mirage_exploratory_report.json")
    random_report = mirage["arms"]["random_context"]
    random_latency = random_report["latency_components"]["component_summed_proxy"]
    comparison_rows["random_context"] = [
        {
            "case_id": "aggregate-only",
            "subdataset": "aggregate",
            "selected_arm_correct": None,
            "retrieval_calls": random_report["retrieval_calls"],
            "answer_input_tokens": random_report["answer_input_tokens_observed_total"],
            "context_characters": random_report["context_characters"],
            "component_latency_proxy_ms": None,
        }
    ]
    comparison_metrics = {
        name: summarize_rows(rows)
        for name, rows in comparison_rows.items()
        if name != "random_context"
    }
    comparison_metrics["random_context"] = {
        "cases": mirage["case_count"],
        "correct": round(random_report["accuracy_fixed_denominator"] * mirage["case_count"]),
        "accuracy": random_report["accuracy_fixed_denominator"],
        "retrieval_calls": random_report["retrieval_calls"],
        "retrieval_rate": random_report["retrieval_calls"] / mirage["case_count"],
        "answer_input_tokens": random_report["answer_input_tokens_observed_total"],
        "answer_input_token_measurement_count": round(
            random_report["answer_input_token_measurement_coverage"] * mirage["case_count"]
        ),
        "answer_input_token_measurement_coverage": random_report[
            "answer_input_token_measurement_coverage"
        ],
        "context_characters": random_report["context_characters"],
        "component_latency_proxy_p50_ms": random_latency.get("p50_ms"),
        "component_latency_proxy_p95_ms": random_latency.get("p95_ms"),
        "component_latency_measurement_count": round(
            random_latency.get("measurement_coverage") * mirage["case_count"]
        ),
        "component_latency_measurement_coverage": random_latency.get("measurement_coverage"),
        "latency_semantics": "Historical component-summed proxy; not observed production end-to-end latency.",
    }

    policy_table = []
    ordered_table = [
        "closed_book",
        "rag_bm25",
        "rag_medcpt",
        "random_context",
        "cheap_router",
        "jev_router",
        *LEARNED_POLICIES,
        "cost_oracle_v2",
    ]
    for name in ordered_table:
        metric = comparison_metrics[name]
        policy_table.append({"policy": name, **metric})

    stage_metrics: dict[str, Any] = {}
    for policy in LEARNED_POLICIES:
        stage_metrics[policy] = oof_metrics(learned_by_policy[policy], oracle_by_id)
    fold_report = fold_stability(oof_rows, int(protocol["outer_cv"]["fold_count"]))
    taxonomy = {
        policy: failure_taxonomy(learned_by_policy[policy], oracle_by_id)
        for policy in LEARNED_POLICIES
    }
    gate_metrics = {
        name: comparison_metrics[name]
        for name in (
            *FIXED_ACTIONS,
            "random_context",
            "cheap_router",
            "jev_router",
            *LEARNED_POLICIES,
            "cost_oracle_v2",
        )
    }
    frontier = pareto_frontier(gate_metrics)

    best_learned = min(
        LEARNED_POLICIES,
        key=lambda policy: (
            -comparison_metrics[policy]["accuracy"],
            comparison_metrics[policy]["retrieval_rate"],
            comparison_metrics[policy]["answer_input_tokens"],
        ),
    )
    best_bge = min(
        ("bge_hierarchical", "bge_hierarchical_always_medcpt"),
        key=lambda policy: (
            -comparison_metrics[policy]["accuracy"],
            comparison_metrics[policy]["retrieval_rate"],
        ),
    )
    best_tfidf = min(
        ("tfidf_direct", "tfidf_hierarchical", "tfidf_hierarchical_always_medcpt"),
        key=lambda policy: (
            -comparison_metrics[policy]["accuracy"],
            comparison_metrics[policy]["retrieval_rate"],
            comparison_metrics[policy]["answer_input_tokens"],
        ),
    )

    subdatasets = sorted({case.subdataset for case in cases})
    bootstrap_seed = int(protocol["uncertainty"]["seed"])
    bootstrap_count = int(protocol["uncertainty"]["paired_bootstrap_replicates"])
    bootstrap = {
        "best_learned_vs_cheap": paired_stratified_bootstrap(
            learned_by_policy[best_learned],
            comparison_rows["cheap_router"],
            subdatasets,
            replicates=bootstrap_count,
            seed=bootstrap_seed,
        ),
        "best_learned_vs_medcpt": paired_stratified_bootstrap(
            learned_by_policy[best_learned],
            comparison_rows["rag_medcpt"],
            subdatasets,
            replicates=bootstrap_count,
            seed=bootstrap_seed + 1,
        ),
    }

    relative = {
        policy: {
            "vs_always_medcpt": relative_cost(
                comparison_metrics[policy], comparison_metrics["rag_medcpt"]
            ),
            "vs_cheap_router": relative_cost(
                comparison_metrics[policy], comparison_metrics["cheap_router"]
            ),
        }
        for policy in LEARNED_POLICIES
    }
    closed_accuracy = comparison_metrics["closed_book"]["accuracy"]
    best_fixed_accuracy = max(comparison_metrics[name]["accuracy"] for name in FIXED_ACTIONS)
    cheap_accuracy = comparison_metrics["cheap_router"]["accuracy"]
    learned_accuracy = comparison_metrics[best_learned]["accuracy"]
    oracle_accuracy = comparison_metrics["cost_oracle_v2"]["accuracy"]
    oracle_headroom = oracle_accuracy - closed_accuracy
    cheap_to_oracle = oracle_accuracy - cheap_accuracy
    headroom = {
        "closed_accuracy": closed_accuracy,
        "best_fixed_accuracy": best_fixed_accuracy,
        "cheap_accuracy": cheap_accuracy,
        "best_learned_accuracy": learned_accuracy,
        "best_learned_policy": best_learned,
        "oracle_v2_accuracy": oracle_accuracy,
        "oracle_headroom": oracle_headroom,
        "oracle_headroom_pp": oracle_headroom * 100,
        "oracle_headroom_over_best_fixed": oracle_accuracy - best_fixed_accuracy,
        "oracle_headroom_over_best_fixed_pp": (oracle_accuracy - best_fixed_accuracy) * 100,
        "learned_gain_over_closed": learned_accuracy - closed_accuracy,
        "learned_gain_over_closed_pp": (learned_accuracy - closed_accuracy) * 100,
        "oracle_headroom_recovered_fraction": (
            (learned_accuracy - closed_accuracy) / oracle_headroom
            if oracle_headroom > 0
            else None
        ),
        "learned_gain_over_cheap_pp": (learned_accuracy - cheap_accuracy) * 100,
        "cheap_to_oracle_headroom_recovered_fraction": (
            (learned_accuracy - cheap_accuracy) / cheap_to_oracle
            if cheap_to_oracle > 0
            else None
        ),
    }

    cheap = comparison_metrics["cheap_router"]
    best_bge_metric = comparison_metrics[best_bge]
    bge_folds = fold_report[best_bge]
    cheap_fold_accuracy = {}
    cheap_fold_calls = {}
    for fold in range(int(protocol["outer_cv"]["fold_count"])):
        selected = [
            row
            for row in comparison_rows["cheap_router"]
            if fold_map[row["case_id"]] == fold
        ]
        cheap_fold_accuracy[str(fold)] = sum(row["selected_arm_correct"] for row in selected) / len(selected)
        cheap_fold_calls[str(fold)] = sum(row["retrieval_calls"] for row in selected)
    quality_folds = sum(
        bge_folds[str(fold)]["accuracy"] >= cheap_fold_accuracy[str(fold)]
        for fold in range(int(protocol["outer_cv"]["fold_count"]))
    )
    cost_folds = sum(
        sum(
            row["retrieval_calls"]
            for row in learned_by_policy[best_bge]
            if row["fold"] == fold
        )
        <= cheap_fold_calls[str(fold)]
        for fold in range(int(protocol["outer_cv"]["fold_count"]))
    )
    path_a = (
        best_bge_metric["accuracy"] >= cheap["accuracy"] + 0.002
        and best_bge_metric["retrieval_rate"] <= cheap["retrieval_rate"]
        and quality_folds >= 4
    )
    path_b = (
        best_bge_metric["accuracy"] >= cheap["accuracy"] - 0.002
        and best_bge_metric["retrieval_calls"] <= 0.8 * cheap["retrieval_calls"]
        and cost_folds >= 4
    )
    sft_candidate = path_a or path_b
    if path_a:
        sft_reason = "QUALITY_PARETO_SIGNAL"
    elif path_b:
        sft_reason = "COST_PARETO_SIGNAL"
    else:
        sft_reason = "GATES_NOT_MET"
    best_learned_metric = comparison_metrics[best_learned]
    strong_signal = (
        best_learned_metric["accuracy"] >= comparison_metrics["rag_medcpt"]["accuracy"] - 0.001
        and best_learned_metric["retrieval_calls"]
        <= 0.5 * comparison_metrics["rag_medcpt"]["retrieval_calls"]
    )
    learned_dominates_cheap = any(
        comparison_metrics[policy]["accuracy"] >= cheap["accuracy"]
        and comparison_metrics[policy]["retrieval_rate"] <= cheap["retrieval_rate"]
        and (
            comparison_metrics[policy]["accuracy"] > cheap["accuracy"]
            or comparison_metrics[policy]["retrieval_rate"] < cheap["retrieval_rate"]
        )
        for policy in LEARNED_POLICIES
    )

    prevalence = sum(row["retrieval_benefit"] for row in oracle_rows) / len(oracle_rows)
    tfidf_ap = stage_metrics["tfidf_hierarchical"]["stage1"]["pr_auc_average_precision"]
    bge_ap = stage_metrics["bge_hierarchical"]["stage1"]["pr_auc_average_precision"]
    tfidf_signal = tfidf_ap > prevalence
    bge_signal = bge_ap > prevalence
    bge_beats_tfidf = (
        bge_ap >= tfidf_ap
        and comparison_metrics["bge_hierarchical"]["accuracy"]
        >= comparison_metrics["tfidf_hierarchical"]["accuracy"]
        and (
            bge_ap > tfidf_ap
            or comparison_metrics["bge_hierarchical"]["accuracy"]
            > comparison_metrics["tfidf_hierarchical"]["accuracy"]
        )
    )

    r2_test = read_json(repo_root / "runs" / "e1_2" / "r2med_test_report.json")
    r2_dev = read_json(repo_root / "runs" / "e1_2" / "r2med_dev_selection.json")
    r2_custom = r2_test["custom_method_primary_result"]
    r2_selected = r2_test["metrics"][r2_test["selected_fixed_baseline"]]["macro_equal_subset_weight"]["ndcg@10"]
    r2_baselines = {
        "test_query_count": r2_test["test_query_count"],
        "selected_fixed_baseline": r2_test["selected_fixed_baseline"],
        "selected_baseline_ndcg_at_10": r2_selected,
        "custom_ndcg_at_10": r2_selected + r2_custom["mean_difference"],
        "custom_delta_ndcg_at_10": r2_custom["mean_difference"],
        "custom_ci_95": [r2_custom["lower_95"], r2_custom["upper_95"]],
        "headline_gate_eligible": r2_test["headline_gate"]["eligible"],
        "dev_query_count": r2_dev["dev_query_count"],
    }
    nf_metrics = read_json(repo_root / "runs" / "e1_1" / "nfcorpus_external_v1" / "metrics.json")

    research_questions = {
        "RQ1 question-only retrieval-benefit signal": {
            "answer": "YES, weak ranking signal; no useful Pareto policy emerged" if (tfidf_signal or bge_signal) else "NO detectable ranking signal",
            "evidence": f"positive prevalence={prevalence:.4f}; TF-IDF PR-AUC={tfidf_ap:.4f}; BGE PR-AUC={bge_ap:.4f}; best learned OOF accuracy={learned_accuracy:.4f} vs closed={closed_accuracy:.4f}",
        },
        "RQ2 semantic BGE vs lexical TF-IDF": {
            "answer": "BGE beats TF-IDF on both primary comparisons" if bge_beats_tfidf else "NO / mixed; BGE does not beat TF-IDF on both PR-AUC and hierarchical accuracy",
            "evidence": f"BGE vs TF-IDF Stage-1 PR-AUC {bge_ap:.4f} vs {tfidf_ap:.4f}; hierarchical OOF accuracy {comparison_metrics['bge_hierarchical']['accuracy']:.4f} vs {comparison_metrics['tfidf_hierarchical']['accuracy']:.4f}. No confirmatory significance test is claimed.",
        },
        "RQ3 value of Stage 2 retriever selector": {
            "answer": "VALUE" if comparison_metrics["bge_hierarchical"]["accuracy"] > comparison_metrics["bge_hierarchical_always_medcpt"]["accuracy"] else "NO ACCURACY VALUE over always-MedCPT",
            "evidence": f"BGE hierarchical accuracy={comparison_metrics['bge_hierarchical']['accuracy']:.4f}, always-MedCPT={comparison_metrics['bge_hierarchical_always_medcpt']['accuracy']:.4f}; rescue-only selector accuracy={stage_metrics['bge_hierarchical']['stage2']['bm25_medcpt_accuracy']:.4f}, macro-F1={stage_metrics['bge_hierarchical']['stage2']['macro_f1']:.4f}, majority-BM25 baseline accuracy={stage_metrics['bge_hierarchical']['stage2']['majority_baseline_accuracy']:.4f}",
        },
        "RQ4 Pareto-domination of Cheap": {
            "answer": "YES" if learned_dominates_cheap else "NO",
            "evidence": f"frontier={frontier}; best learned={best_learned} accuracy={learned_accuracy:.4f}, retrieval rate={comparison_metrics[best_learned]['retrieval_rate']:.4f}; cheap accuracy={cheap_accuracy:.4f}, retrieval rate={cheap['retrieval_rate']:.4f}",
        },
        "RQ5 recovery of oracle headroom": {
            "answer": f"{format_optional_percent(headroom['oracle_headroom_recovered_fraction'])} of Closed-to-Oracle-v2 headroom recovered",
            "evidence": f"Closed={closed_accuracy:.4f}; best fixed={best_fixed_accuracy:.4f}; Cheap={cheap_accuracy:.4f}; best learned={learned_accuracy:.4f}; Oracle v2={oracle_accuracy:.4f}; relative-to-Cheap-to-Oracle fraction={format_optional_percent(headroom['cheap_to_oracle_headroom_recovered_fraction'])}",
        },
    }

    gates = {
        "RAG_BASELINES_COMPLETE": "YES",
        "OOF_PROTOCOL_VALID": "YES" if overlap == 0 and duplicate_count == 0 and missing_count == 0 else "NO",
        "COST_ORACLE_V2_READY": "YES" if len(oracle_rows) == len(case_ids) else "NO",
        "TFIDF_SIGNAL_FOUND": "YES" if tfidf_signal else "NO",
        "BGE_SIGNAL_FOUND": "YES" if bge_signal else "NO",
        "BGE_BEATS_TFIDF": "YES" if bge_beats_tfidf else "NO",
        "LEARNED_ROUTER_PARETO_DOMINATES_CHEAP": "YES" if learned_dominates_cheap else "NO",
        "STRONG_CAPABILITY_ROUTING_SIGNAL": "YES" if strong_signal else "NO",
        "SFT_CANDIDATE": "YES" if sft_candidate else "NO",
        "RAG_CLOSEOUT": "YES",
    }

    report: dict[str, Any] = {
        "schema_version": "e1-3-router-oof-report-v1",
        "evaluation_status": "EXPOSED_EXPLORATORY_OOF",
        "benchmark": protocol["benchmark"]["name"],
        "case_count": len(case_ids),
        "outer_fold_count": protocol["outer_cv"]["fold_count"],
        "seed": protocol["outer_cv"]["seed"],
        "stratification": fold_manifest["stratification_method"],
        "protocol_sha256": protocol["protocol_sha256"],
        "fold_manifest_sha256": fold_manifest["manifest_sha256"],
        "oof_integrity": {
            "train_eval_overlap_cases": overlap,
            "duplicate_case_policy_predictions": duplicate_count,
            "missing_case_policy_predictions": missing_count,
            "prediction_rows": len(oof_rows),
            "predictions_per_case": len(LEARNED_POLICIES),
            "case_id_order_sha256": protocol["benchmark"]["case_order_sha256"],
            "question_text_in_git_artifacts": False,
            "options_or_answers_in_git_artifacts": False,
            "oof_predictions_sha256": sha256_file(oof_path),
        },
        "cost_oracle_v2": {
            "class_counts": dict(
                sorted(Counter(row["cost_oracle_class"] for row in oracle_rows).items())
            ),
            "retrieval_benefit_cases": sum(bool(row["retrieval_benefit"]) for row in oracle_rows),
            "retrieval_benefit_prevalence": prevalence,
            "stage2_target_counts": dict(
                sorted(
                    Counter(
                        row["stage2_target"]
                        for row in oracle_rows
                        if row["stage2_target"] is not None
                    ).items()
                )
            ),
            "metrics": comparison_metrics["cost_oracle_v2"],
            "no_success_action": "closed_book",
        },
        "stage1_metrics": {
            policy: stage_metrics[policy]["stage1"] for policy in LEARNED_POLICIES
        },
        "stage2_metrics": {
            policy: stage_metrics[policy]["stage2"]
            for policy in ("tfidf_hierarchical", "bge_hierarchical")
        },
        "policy_table": policy_table,
        "comparison_metrics": comparison_metrics,
        "relative_cost": relative,
        "failure_taxonomy": taxonomy,
        "fold_stability": fold_report,
        "paired_bootstrap": bootstrap,
        "pareto_frontier": frontier,
        "best_learned_policy": best_learned,
        "best_bge_policy": best_bge,
        "best_tfidf_policy": best_tfidf,
        "sft_candidate_gate": {
            "candidate_policy": best_bge,
            "quality_path_pass": path_a,
            "quality_path_folds_at_least_cheap": quality_folds,
            "cost_path_pass": path_b,
            "cost_path_folds_no_more_calls_than_cheap": cost_folds,
            "reason": sft_reason,
        },
        "headroom": headroom,
        "oof_runner_commit": run_manifest["implementation_commit"],
        "disk_free_start_gib": {
            drive: values["free_gib"]
            for drive, values in protocol["environment"]["disk_free_start_gib"].items()
        },
        "disk_free_end_gib": {
            drive[0]: round(shutil.disk_usage(drive).free / (1024**3), 2)
            for drive in ("C:\\", "D:\\", "E:\\")
        },
        "research_questions": research_questions,
        "external_retrieval_baselines": {
            "nfcorpus": nf_metrics,
            "r2med": r2_baselines,
        },
        "mirage_fixed_baselines": {
            "closed_book": comparison_metrics["closed_book"],
            "random_context": comparison_metrics["random_context"],
            "rag_bm25": comparison_metrics["rag_bm25"],
            "rag_medcpt": comparison_metrics["rag_medcpt"],
        },
        "gates": gates,
        "interpretation": {
            "clinical_validation": False,
            "untouched_holdout": False,
            "confirmatory_test": False,
            "sft_or_grpo_run": False,
            "learned_model_inference_latency_measured": False,
            "sft_candidate_reason": sft_reason,
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--scratch-root", type=Path, default=Path(r"E:\Health-Copilot-E1.2"))
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    report = analyze(repo_root, args.scratch_root.resolve())
    artifact_root = repo_root / "runs" / "e1_3"
    report_path = artifact_root / "router_oof_report.json"
    write_json(report_path, report)

    capability_doc_path = repo_root / "docs" / "research" / "e1_3_capability_router.md"
    capability_doc_path.parent.mkdir(parents=True, exist_ok=True)
    capability_doc_path.write_text(report_markdown(report), encoding="utf-8", newline="\n")
    mirage = read_json(artifact_root / "mirage_exploratory_report.json")
    closeout_path = repo_root / "docs" / "research" / "rag_closeout.md"
    closeout_path.write_text(
        closeout_markdown(report, mirage), encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "status": "E1_3_ANALYSIS_COMPLETE",
                "best_learned_policy": report["best_learned_policy"],
                "best_learned_accuracy": report["headroom"]["best_learned_accuracy"],
                "best_learned_retrieval_calls": report["comparison_metrics"][report["best_learned_policy"]]["retrieval_calls"],
                "gates": report["gates"],
                "report_sha256": sha256_file(report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
