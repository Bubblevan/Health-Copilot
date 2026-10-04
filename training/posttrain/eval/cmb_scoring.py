from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from eval.statistics import wilson_interval


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_stratified_ids(
    rows: Iterable[Mapping[str, Any]], *, target_n: int = 1024, seed: int = 20261004
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Use only id and subcategory metadata; ignore answer and question fields."""
    strata: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        row_id = str(row["id"])
        category = str(row["subcategory"])
        if row_id in seen:
            raise ValueError(f"Duplicate CMB row id: {row_id}")
        seen.add(row_id)
        strata[category].append(row_id)
    total = sum(len(values) for values in strata.values())
    if not strata or target_n <= 0 or target_n > total:
        raise ValueError(f"Invalid stratified sample size {target_n} for {total} rows")

    allocation = {name: target_n * len(values) // total for name, values in strata.items()}
    remaining = target_n - sum(allocation.values())
    quota_order = sorted(strata, key=lambda name: _sha(f"{seed}\0quota\0{name}"))
    for name in quota_order[:remaining]:
        allocation[name] += 1

    selected_by_category: dict[str, dict[str, Any]] = {}
    for name, ids in strata.items():
        ranked = sorted(ids, key=lambda row_id: _sha(f"{seed}\0row\0{name}\0{row_id}"))
        selected_by_category[name] = {
            "source_n": len(ids),
            "selected_n": allocation[name],
            "selected_ids": sorted(ranked[: allocation[name]]),
        }
    selected = sorted(
        row_id for values in selected_by_category.values() for row_id in values["selected_ids"]
    )
    if len(selected) != target_n:
        raise AssertionError(f"Selected {len(selected)} IDs; expected {target_n}")
    return selected, dict(sorted(selected_by_category.items()))


def normalized_labels(value: Any) -> str | None:
    text = str(value or "").upper()
    labels = [char for char in text if char in "ABCDEF"]
    return "".join(sorted(set(labels))) if labels else None


def binary_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(items)
    correct = sum(bool(item["correct"]) for item in items)
    lo, hi = wilson_interval(correct, n)
    parsed = sum(item["parsed"] for item in items)
    latencies = [float(item["latency"]) for item in items]
    output_tokens = [int(item["tokens"]) for item in items]
    sorted_latencies = sorted(latencies)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "wilson_95_ci": [lo, hi],
        "invalid_answer_rate": (n - parsed) / n if n else None,
        "mean_output_tokens": sum(output_tokens) / len(output_tokens) if output_tokens else None,
        "generation_latency_seconds": {
            "mean": statistics.mean(latencies) if latencies else None,
            "p50": statistics.median(latencies) if latencies else None,
            "p95": sorted_latencies[max(0, int(0.95 * len(sorted_latencies)) - 1)] if sorted_latencies else None,
        },
    }


def score_cmb_predictions(
    predictions: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    *,
    checkpoint: str,
    model_revision: str,
    benchmark: str = "cmb",
    ids_sha256: str | None = None,
) -> dict[str, Any]:
    gold_by_id = {str(row["id"]): row for row in gold_rows}
    pred_by_id = {str(row["id"]): row for row in predictions}
    if len(gold_by_id) != len(gold_rows) or len(pred_by_id) != len(predictions):
        raise ValueError("CMB gold and prediction IDs must be unique")
    if set(gold_by_id) != set(pred_by_id):
        raise ValueError("CMB prediction and scorer IDs are not a one-to-one exact match")

    items = []
    for row_id, gold in gold_by_id.items():
        pred = pred_by_id[row_id]
        expected = normalized_labels(gold["answer"])
        actual = pred.get("parsed_answer")
        items.append({
            "id": row_id,
            "correct": actual is not None and expected is not None and str(actual).upper() == expected,
            "parsed": bool(pred.get("parse_success")),
            "latency": pred["generation_latency_seconds"],
            "tokens": pred["output_tokens"],
            "subcategory": gold.get("subcategory"),
            "major_category": gold.get("major_category"),
            "question_type": gold.get("question_type"),
        })

    by_subcategory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_major: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_subcategory[str(item["subcategory"])].append(item)
        by_major[str(item["major_category"])].append(item)
    subcategory_metrics = {
        name: binary_metrics(rows) for name, rows in sorted(by_subcategory.items())
    }
    major_metrics = {name: binary_metrics(rows) for name, rows in sorted(by_major.items())}
    singles = [item for item in items if item["question_type"] in {"单项选择题", "C型选择题"}]
    multiples = [item for item in items if item["question_type"] == "多项选择题"]
    result = {
        "checkpoint": checkpoint,
        "model_revision": model_revision,
        "benchmark": benchmark,
        **binary_metrics(items),
        "macro_28_subcategory_accuracy": (
            sum(row["accuracy"] for row in subcategory_metrics.values()) / len(subcategory_metrics)
            if subcategory_metrics else None
        ),
        "subcategory_count": len(subcategory_metrics),
        "major_categories": major_metrics,
        "subcategories": subcategory_metrics,
        "single_choice": binary_metrics(singles),
        "multiple_answer": binary_metrics(multiples),
    }
    if ids_sha256 is not None:
        result["evaluation_ids_sha256"] = ids_sha256
    return result
