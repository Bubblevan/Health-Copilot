"""Freeze E1.3 MIRAGE cost labels, outer folds, and learned-router protocol."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from eval.e1_3_router_dataset import (
    HISTORICAL_ARMS,
    canonical_sha256,
    case_order_sha256,
    load_cases,
    load_historical_arms,
    make_cost_oracle_rows,
    sha256_file,
)

SEED = 20260925
FOLD_COUNT = 5
THRESHOLDS = [round(value / 100, 2) for value in range(5, 100, 5)]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def select_stratification(rows: list[dict[str, Any]]) -> tuple[list[str], str]:
    combined = [f"{row['subdataset']}::{row['cost_oracle_class']}" for row in rows]
    if min(Counter(combined).values()) >= FOLD_COUNT:
        return combined, "subdataset_x_cost_oracle_class"
    actions = [row["cost_oracle_class"] for row in rows]
    if min(Counter(actions).values()) >= FOLD_COUNT:
        return actions, "cost_oracle_class"
    benefits = ["benefit" if row["retrieval_benefit"] else "no_benefit" for row in rows]
    if min(Counter(benefits).values()) < FOLD_COUNT:
        raise ValueError("No valid stratification target has at least five examples per class")
    return benefits, "retrieval_benefit_binary"


def build_fold_manifest(
    rows: list[dict[str, Any]], case_ids: list[str], source_sha256: str
) -> dict[str, Any]:
    stratification_labels, method = select_stratification(rows)
    folds = np.full(len(rows), -1, dtype=np.int8)
    splitter = StratifiedKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=SEED)
    dummy = np.zeros((len(rows), 1), dtype=np.uint8)
    for fold, (_, heldout_indices) in enumerate(splitter.split(dummy, stratification_labels)):
        folds[heldout_indices] = fold
    if np.any(folds < 0):
        raise AssertionError("Every case must be assigned to exactly one fold")

    per_fold: dict[str, Any] = {}
    for fold in range(FOLD_COUNT):
        heldout = [rows[index] for index in np.flatnonzero(folds == fold)]
        per_fold[str(fold)] = {
            "case_count": len(heldout),
            "cost_oracle_class": dict(sorted(Counter(r["cost_oracle_class"] for r in heldout).items())),
            "subdataset": dict(sorted(Counter(r["subdataset"] for r in heldout).items())),
            "retrieval_benefit": {
                "true": sum(bool(r["retrieval_benefit"]) for r in heldout),
                "false": sum(not bool(r["retrieval_benefit"]) for r in heldout),
            },
        }
    manifest: dict[str, Any] = {
        "schema_version": "e1-3-router-fold-manifest-v1",
        "benchmark_status": "EXPOSED_EXPLORATORY",
        "case_count": len(rows),
        "case_order_sha256": case_order_sha256(case_ids),
        "cost_oracle_v2_sha256": source_sha256,
        "seed": SEED,
        "fold_count": FOLD_COUNT,
        "stratification_method": method,
        "stratification_rule": (
            "Try subdataset x cost-oracle class; if any stratum has fewer than 5 cases, "
            "fall back to cost-oracle class; if still insufficient, use retrieval-benefit binary."
        ),
        "case_id_to_fold": {case_id: int(fold) for case_id, fold in zip(case_ids, folds, strict=True)},
        "target_distribution": {
            "cost_oracle_class": dict(sorted(Counter(r["cost_oracle_class"] for r in rows).items())),
            "retrieval_benefit": {
                "true": sum(bool(r["retrieval_benefit"]) for r in rows),
                "false": sum(not bool(r["retrieval_benefit"]) for r in rows),
            },
        },
        "per_fold_distribution": per_fold,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def build_protocol(
    repo_root: Path,
    scratch_root: Path,
    input_path: Path,
    cases: list[Any],
    arms: dict[str, dict[str, Any]],
    cost_oracle_sha256: str,
    fold_manifest: dict[str, Any],
) -> dict[str, Any]:
    source_manifest_path = repo_root / "runs" / "e1_2" / "r2med_source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    bge_source = source_manifest["model_sources"]["bge_base_en_v1_5"]
    model_path = scratch_root / "models" / "bge-base-en-v1.5" / bge_source["file"]
    actual_model_sha = sha256_file(model_path)
    if actual_model_sha != bge_source["sha256"]:
        raise ValueError("Local BGE weights do not match the E1.2 pinned SHA-256")

    arm_sources: dict[str, Any] = {}
    for arm in HISTORICAL_ARMS:
        arm_root = scratch_root / "runs" / "e1_2" / "test" / arm
        manifest = json.loads((arm_root / "manifest.json").read_text(encoding="utf-8"))
        arm_sources[arm] = {
            "result_identity": manifest["result_identity"],
            "config_sha256": manifest["config_sha256"],
            "case_results_sha256": sha256_file(arm_root / "case_results.jsonl"),
            "manifest_sha256": sha256_file(arm_root / "manifest.json"),
            "metrics_sha256": sha256_file(arm_root / "metrics.json"),
            "rows": len(arms[arm]),
        }

    disks = {}
    for drive in ("C:\\", "D:\\", "E:\\"):
        usage = shutil.disk_usage(drive)
        disks[drive[0]] = {"free_gib": round(usage.free / (1024**3), 2)}

    protocol: dict[str, Any] = {
        "schema_version": "e1-3-oof-learned-capability-router-protocol-v1",
        "status": "FROZEN_BEFORE_OOF_METRICS",
        "evaluation_status": "EXPOSED_EXPLORATORY_OOF",
        "start_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip(),
        "scope": "Question-only learned routing over existing MIRAGE E1.2 outcomes; no new answer inference.",
        "benchmark": {
            "name": "Medical MIRAGE / MedRAG clean TEST subdatasets",
            "case_count": len(cases),
            "input_artifact": str(input_path),
            "input_sha256": sha256_file(input_path),
            "case_order_sha256": case_order_sha256([case.case_id for case in cases]),
            "subdataset_counts": dict(
                sorted(Counter(case.subdataset for case in cases).items())
            ),
            "question_feature_boundary": "Only record['question']; options, labels, IDs and outcomes are excluded from features.",
        },
        "historical_arm_sources": arm_sources,
        "cost_oracle_v2": {
            "artifact": "runs/e1_3/router_cost_oracle_v2.jsonl",
            "sha256": cost_oracle_sha256,
            "action_rule": [
                "closed_book_correct -> closed_book",
                "else rag_bm25_correct -> rag_bm25",
                "else rag_medcpt_correct -> rag_medcpt",
                "else -> closed_book",
            ],
            "classes": {
                "CLOSED_SUFFICIENT": "closed book correct",
                "BM25_RESCUE": "closed book wrong and BM25 correct",
                "MEDCPT_RESCUE": "closed book wrong, BM25 wrong, and MedCPT correct",
                "UNRESOLVED": "all fixed arms wrong; choose the zero-retrieval closed-book action",
            },
        },
        "outer_cv": {
            "fold_count": FOLD_COUNT,
            "seed": SEED,
            "stratification": fold_manifest["stratification_method"],
            "fold_manifest": "runs/e1_3/router_fold_manifest.json",
            "fold_manifest_sha256": fold_manifest["manifest_sha256"],
            "case_order_sha256": fold_manifest["case_order_sha256"],
        },
        "inner_cv": {
            "fold_count": 3,
            "seed_rule": "20260925 + outer_fold * 100 + inner_cv_instance_index",
            "stratification": "retrieval-benefit binary target, training partition only",
        },
        "models": {
            "direct_tfidf": {
                "features": {
                    "word": {"ngram_range": [1, 2], "min_df": 2, "max_features": 200000},
                    "char_wb": {"ngram_range": [3, 5], "min_df": 2, "max_features": 300000},
                    "sublinear_tf": True,
                    "norm": "l2",
                    "dtype": "float32",
                },
                "classifier": {
                    "name": "multinomial logistic regression",
                    "solver": "lbfgs",
                    "C": 1.0,
                    "class_weight": "balanced",
                    "max_iter": 2000,
                    "tol": 0.0001,
                },
                "fit_boundary": "vectorizers and classifier fit only on the current training partition",
            },
            "hierarchical": {
                "stage1": "balanced binary logistic regression for retrieval benefit",
                "stage2": "balanced binary logistic regression, trained only on rescue cases; target BM25 if correct else MedCPT",
                "stage2_small_class_fallback": "If either rescue retriever has fewer than 2 training examples, choose the majority rescue retriever and record fallback=true.",
                "threshold_grid": THRESHOLDS,
                "threshold_selection": {
                    "method": "3-fold inner OOF predictions within outer-train only",
                    "quality_constraint": "policy accuracy >= Always-MedCPT accuracy - 0.25 percentage points",
                    "primary_choice": "minimum retrieval-call rate; tie -> higher threshold",
                    "no_feasible_threshold": "highest accuracy, then lowest retrieval-call rate, then highest threshold",
                    "probability_comparison": "retrieve when p(retrieval benefit) >= threshold",
                },
            },
            "bge": {
                "repo_id": bge_source["repo_id"],
                "revision": bge_source["revision"],
                "weights_file": bge_source["file"],
                "weights_bytes": bge_source["bytes"],
                "weights_sha256": actual_model_sha,
                "model_path": str(model_path.parent),
                "encoding": "Raw question only; SentenceTransformer encode; L2-normalize embeddings; frozen model, no fine-tuning.",
                "cache_path": "E:\\Health-Copilot-E1.3\\router\\bge_embeddings.npy",
            },
        },
        "learned_policies": [
            "tfidf_direct",
            "tfidf_hierarchical",
            "tfidf_hierarchical_always_medcpt",
            "bge_hierarchical",
            "bge_hierarchical_always_medcpt",
        ],
        "class_imbalance": "class_weight=balanced; no oversampling",
        "counterfactual_costs": "Look up selected action's historical per-case answer tokens, context characters, retrieval calls, and component-summed latency proxy; provider failures count incorrect.",
        "uncertainty": {
            "paired_bootstrap_replicates": 10000,
            "seed": SEED,
            "sampling": "paired cases, stratified by MIRAGE subdataset, percentile 95% CI",
            "comparisons": ["best learned vs cheap_router", "best learned vs rag_medcpt"],
            "interpretation": "Exploratory inference only; not confirmatory significance.",
        },
        "selection_and_gates": {
            "best_learned": "Highest OOF accuracy among all learned policies; tie -> lower retrieval rate, then fewer input tokens.",
            "best_bge": "Highest OOF accuracy among the two BGE hierarchical policies; tie -> lower retrieval rate.",
            "sft_candidate_quality": "Best BGE accuracy >= cheap_router + 0.20 pp AND retrieval rate <= cheap_router AND >=4/5 folds have accuracy >= cheap_router.",
            "sft_candidate_cost": "Best BGE accuracy >= cheap_router - 0.20 pp AND calls <=80% of cheap_router AND >=4/5 folds have calls <= cheap_router.",
            "strong_signal": "Best learned accuracy >= Always-MedCPT - 0.10 pp AND retrieval calls <=50% of Always-MedCPT.",
            "pareto_dominance": "Any learned policy has accuracy >= cheap_router and retrieval rate <= cheap_router, with at least one strict improvement.",
            "no_metric_driven_tuning": True,
        },
        "bootstrap_best_policy_rule": "Among all learned policies, highest pooled OOF accuracy; ties resolved by retrieval rate then input-token sum.",
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "disk_free_start_gib": disks,
        },
        "storage_and_privacy": {
            "scratch_root": "E:\\Health-Copilot-E1.3\\router",
            "question_text_in_git": False,
            "options_or_answers_in_git": False,
            "model_files_in_git": False,
            "answer_inference_or_cloud_calls": False,
        },
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--scratch-root", type=Path, default=Path(r"E:\Health-Copilot-E1.2")
    )
    parser.add_argument(
        "--input", type=Path, default=Path(r"E:\Health-Copilot-E1.2\inputs\benchmark_test.json")
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    scratch_root = args.scratch_root.resolve()
    input_path = args.input.resolve()

    cases = load_cases(input_path)
    case_ids = [case.case_id for case in cases]
    arms = load_historical_arms(scratch_root, case_ids)
    oracle_rows = make_cost_oracle_rows(cases, arms)
    oracle_path = repo_root / "runs" / "e1_3" / "router_cost_oracle_v2.jsonl"
    write_jsonl(oracle_path, oracle_rows)
    oracle_sha = sha256_file(oracle_path)

    folds = build_fold_manifest(oracle_rows, case_ids, oracle_sha)
    protocol = build_protocol(
        repo_root, scratch_root, input_path, cases, arms, oracle_sha, folds
    )
    output_root = repo_root / "runs" / "e1_3"
    write_json(output_root / "router_fold_manifest.json", folds)
    write_json(output_root / "router_protocol.json", protocol)

    classes = Counter(row["cost_oracle_class"] for row in oracle_rows)
    print(
        json.dumps(
            {
                "status": "PROTOCOL_AND_FOLDS_FROZEN",
                "case_count": len(cases),
                "cost_oracle_classes": dict(sorted(classes.items())),
                "retrieval_benefit_cases": sum(r["retrieval_benefit"] for r in oracle_rows),
                "retrieval_benefit_rate": sum(r["retrieval_benefit"] for r in oracle_rows)
                / len(oracle_rows),
                "stratification": folds["stratification_method"],
                "fold_manifest_sha256": folds["manifest_sha256"],
                "protocol_sha256": protocol["protocol_sha256"],
                "bge_sha256": protocol["models"]["bge"]["weights_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
