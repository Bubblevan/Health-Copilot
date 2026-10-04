from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.statistics import wilson_interval
from eval.cmb_scoring import score_cmb_predictions

DATA_ROOT = Path("/root/gpufree-data/Health-Copilot-PT-E0-data")
RUNS = DATA_ROOT / "runs/posttrain/pt-e0"
POSTTRAIN_ROOT = Path(__file__).resolve().parents[1]
EVAL_PROTOCOL = POSTTRAIN_ROOT / "manifests/eval/eval_protocol.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_predictions(
    run_dir: Path,
    prediction_file: Path | None = None,
    prediction_manifest_file: Path | None = None,
) -> list[dict[str, Any]]:
    path = prediction_file or run_dir / "predictions.jsonl"
    manifest_path = prediction_manifest_file or run_dir / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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


def score(
    benchmark: str,
    *,
    prediction_file: Path | None = None,
    prediction_manifest_file: Path | None = None,
    output_file: Path | None = None,
) -> dict[str, Any]:
    protocol = json.loads(EVAL_PROTOCOL.read_text(encoding="utf-8"))
    if protocol["mcq_scorer_code_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("MCQ scorer code differs from the frozen evaluation protocol")
    if protocol["parser_sha256"] != hashlib.sha256((POSTTRAIN_ROOT / "eval/parsers.py").read_bytes()).hexdigest():
        raise ValueError("MCQ parser differs from the frozen evaluation protocol")
    if protocol.get("cmb_scoring_code_sha256") != hashlib.sha256((POSTTRAIN_ROOT / "eval/cmb_scoring.py").read_bytes()).hexdigest():
        raise ValueError("Shared CMB scorer differs from the frozen evaluation protocol")
    default_run_dir = RUNS / benchmark
    run_dir = Path(prediction_file).parent if prediction_file is not None else default_run_dir
    predictions_path = Path(prediction_file) if prediction_file is not None else run_dir / "predictions.jsonl"
    manifest_path = (
        Path(prediction_manifest_file)
        if prediction_manifest_file is not None
        else run_dir / "prediction_manifest.json"
    )
    prediction_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    predictions = verify_predictions(run_dir, predictions_path, manifest_path)
    scorer_path = DATA_ROOT / "eval/prepared" / benchmark / "scorer_view.jsonl"
    dataset_manifest_path = POSTTRAIN_ROOT / "manifests/eval/eval_dataset_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    expected_scorer_hash = dataset_manifest["prepared_artifacts"].get(
        f"{benchmark}/scorer_view.jsonl"
    )
    actual_scorer_hash = hashlib.sha256(scorer_path.read_bytes()).hexdigest()
    expected_candidate_hash = dataset_manifest["prepared_artifacts"].get(
        f"{benchmark}/candidate_view.jsonl"
    )
    if actual_scorer_hash != expected_scorer_hash:
        raise ValueError("Scorer view differs from its frozen dataset manifest")
    if prediction_manifest.get("candidate_view_sha256") != expected_candidate_hash:
        raise ValueError("Predictions were not generated from the frozen candidate view")
    gold_rows = read_jsonl(scorer_path)
    by_id = {str(row["id"]): row for row in gold_rows}
    pred_by_id = {str(row["id"]): row for row in predictions}
    if benchmark == "diagnosisarena":
        ids_path = POSTTRAIN_ROOT / "manifests/eval/diagnosisarena915_ids.json"
        frozen_ids = {str(row_id) for row_id in json.loads(ids_path.read_text(encoding="utf-8"))}
        if frozen_ids != set(by_id):
            raise ValueError("DiagnosisArena scorer view differs from the frozen common-core ID list")
    if len(by_id) != len(gold_rows) or len(pred_by_id) != len(predictions) or set(by_id) != set(pred_by_id):
        raise ValueError("Prediction and scorer IDs are not a one-to-one exact match")

    if benchmark == "cmb":
        result = score_cmb_predictions(
            predictions,
            gold_rows,
            checkpoint=prediction_manifest["checkpoint"],
            model_revision=prediction_manifest["model_revision"],
            benchmark="cmb",
        )
    else:
        items = []
        for row_id, gold in by_id.items():
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
        result = {
            "checkpoint": prediction_manifest["checkpoint"],
            "model_revision": prediction_manifest["model_revision"],
            "benchmark": benchmark,
            **binary_metrics(items),
            "per_specialty": None,
            "specialty_note": "The frozen DiagnosisArena test file has no specialty metadata.",
        }
    result["prediction_sha256"] = prediction_manifest["predictions_sha256"]
    result["score_protocol_sha256"] = hashlib.sha256(EVAL_PROTOCOL.read_bytes()).hexdigest()
    score_path = Path(output_file) if output_file is not None else run_dir / "scores.json"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--benchmark", choices=["diagnosisarena", "cmb"], required=True)
    cli.add_argument("--prediction-file", type=Path, default=None)
    cli.add_argument("--prediction-manifest", type=Path, default=None)
    cli.add_argument("--output", type=Path, default=None)
    args = cli.parse_args()
    print(json.dumps(score(
        args.benchmark,
        prediction_file=args.prediction_file,
        prediction_manifest_file=args.prediction_manifest,
        output_file=args.output,
    ), ensure_ascii=False, indent=2))
