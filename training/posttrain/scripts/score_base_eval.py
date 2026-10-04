from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.statistics import wilson_interval

DATA_ROOT = Path("/root/gpufree-data/Health-Copilot-PT-E0-data")
RUNS = DATA_ROOT / "runs/posttrain/pt-e0"
POSTTRAIN_ROOT = Path(__file__).resolve().parents[1]
EVAL_PROTOCOL = POSTTRAIN_ROOT / "manifests/eval/eval_protocol.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_predictions(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "predictions.jsonl"
    manifest = json.loads((run_dir / "prediction_manifest.json").read_text(encoding="utf-8"))
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = manifest["predictions_sha256"]
    sidecar = (run_dir / "predictions.sha256").read_text(encoding="ascii").split()[0]
    if actual != expected or actual != sidecar:
        raise ValueError("Predictions failed frozen SHA256 verification")
    return read_jsonl(path)


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
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "wilson_95_ci": [lo, hi],
        "invalid_answer_rate": (n - parsed) / n if n else None,
        "mean_output_tokens": statistics.mean(output_tokens) if output_tokens else None,
        "generation_latency_seconds": {
            "mean": statistics.mean(latencies) if latencies else None,
            "p50": statistics.median(latencies) if latencies else None,
            "p95": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)] if latencies else None,
        },
    }


def score(benchmark: str) -> dict[str, Any]:
    protocol = json.loads(EVAL_PROTOCOL.read_text(encoding="utf-8"))
    if protocol["mcq_scorer_code_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("MCQ scorer code differs from the frozen evaluation protocol")
    if protocol["parser_sha256"] != hashlib.sha256((POSTTRAIN_ROOT / "eval/parsers.py").read_bytes()).hexdigest():
        raise ValueError("MCQ parser differs from the frozen evaluation protocol")
    run_dir = RUNS / benchmark
    prediction_manifest = json.loads((run_dir / "prediction_manifest.json").read_text(encoding="utf-8"))
    predictions = verify_predictions(run_dir)
    scorer_path = DATA_ROOT / "eval/prepared" / benchmark / "scorer_view.jsonl"
    gold_rows = read_jsonl(scorer_path)
    by_id = {str(row["id"]): row for row in gold_rows}
    pred_by_id = {str(row["id"]): row for row in predictions}
    if len(by_id) != len(gold_rows) or len(pred_by_id) != len(predictions) or set(by_id) != set(pred_by_id):
        raise ValueError("Prediction and scorer IDs are not a one-to-one exact match")

    items = []
    for row_id, gold in by_id.items():
        pred = pred_by_id[row_id]
        actual = pred.get("parsed_answer")
        if benchmark == "diagnosisarena":
            expected = normalized_labels(gold["answer"])
        else:
            expected = normalized_labels(gold["answer"])
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

    result: dict[str, Any] = {
        "checkpoint": prediction_manifest["checkpoint"],
        "model_revision": prediction_manifest["model_revision"],
        "benchmark": benchmark,
        **binary_metrics(items),
    }
    if benchmark == "diagnosisarena":
        result["per_specialty"] = None
        result["specialty_note"] = "The frozen DiagnosisArena test file has no specialty metadata."
    else:
        by_subcategory: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_major: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            by_subcategory[str(item["subcategory"])].append(item)
            by_major[str(item["major_category"])].append(item)
        subcategory_metrics = {name: binary_metrics(rows) for name, rows in sorted(by_subcategory.items())}
        major_metrics = {name: binary_metrics(rows) for name, rows in sorted(by_major.items())}
        singles = [item for item in items if item["question_type"] in {"单项选择题", "C型选择题"}]
        multiples = [item for item in items if item["question_type"] == "多项选择题"]
        result.update({
            "macro_28_subcategory_accuracy": statistics.mean(row["accuracy"] for row in subcategory_metrics.values()),
            "subcategory_count": len(subcategory_metrics),
            "major_categories": major_metrics,
            "subcategories": subcategory_metrics,
            "single_choice": binary_metrics(singles),
            "multiple_answer": binary_metrics(multiples),
        })
    (run_dir / "scores.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--benchmark", choices=["diagnosisarena", "cmb"], required=True)
    args = cli.parse_args()
    print(json.dumps(score(args.benchmark), ensure_ascii=False, indent=2))
