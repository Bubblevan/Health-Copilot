from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from eval.cmb_scoring import score_cmb_predictions

POSTTRAIN_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/root/gpufree-data/Health-Copilot-PT-E0-data")
RUNS = DATA_ROOT / "runs/posttrain/pt-e0"
PROTOCOL_PATH = POSTTRAIN_ROOT / "manifests/eval/eval_protocol.json"
CORE_PATH = POSTTRAIN_ROOT / "manifests/eval/common_eval_core.json"
IDS_PATH = POSTTRAIN_ROOT / "manifests/eval/cmb_common1024_ids.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--prediction-file", default=str(RUNS / "cmb/predictions.jsonl"))
    cli.add_argument("--prediction-manifest", default=None)
    cli.add_argument("--scorer-view", default=str(DATA_ROOT / "eval/prepared/cmb/scorer_view.jsonl"))
    cli.add_argument("--ids-manifest", default=str(IDS_PATH))
    cli.add_argument("--output", default=None)
    cli.add_argument("--output-dir", default=None)
    args = cli.parse_args()

    prediction_path = Path(args.prediction_file)
    prediction_manifest_path = Path(args.prediction_manifest) if args.prediction_manifest else prediction_path.parent / "prediction_manifest.json"
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["mcq_scorer_code_sha256"] != sha(POSTTRAIN_ROOT / "scripts/score_base_eval.py"):
        raise ValueError("Frozen MCQ scorer entrypoint hash mismatch")
    if protocol["cmb_scoring_code_sha256"] != sha(POSTTRAIN_ROOT / "eval/cmb_scoring.py"):
        raise ValueError("Frozen shared CMB scorer hash mismatch")
    if protocol["common_cmb_scorer_code_sha256"] != sha(Path(__file__)):
        raise ValueError("Frozen CMB common scorer entrypoint hash mismatch")
    if protocol["parser_sha256"] != sha(POSTTRAIN_ROOT / "eval/parsers.py"):
        raise ValueError("Frozen MCQ parser hash mismatch")
    if protocol["common_eval_core_manifest_sha256"] != sha(CORE_PATH):
        raise ValueError("Frozen Common Eval Core manifest hash mismatch")
    if protocol["common_eval_core"]["CMB-COMMON-1024"]["ids_manifest_sha256"] != sha(Path(args.ids_manifest)):
        raise ValueError("Frozen CMB common ID manifest hash mismatch")

    prediction_manifest = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
    prediction_hash = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    if prediction_hash != prediction_manifest["predictions_sha256"]:
        raise ValueError("Prediction file differs from its frozen SHA256")
    sidecar = prediction_path.with_name("predictions.sha256")
    if sidecar.exists() and sidecar.read_text(encoding="ascii").split()[0] != prediction_hash:
        raise ValueError("Prediction SHA256 sidecar mismatch")
    predictions = read_jsonl(prediction_path)
    ids_manifest = json.loads(Path(args.ids_manifest).read_text(encoding="utf-8"))
    dataset_manifest = json.loads(
        (POSTTRAIN_ROOT / "manifests/eval/eval_dataset_manifest.json").read_text(encoding="utf-8")
    )
    expected_full_scorer_hash = dataset_manifest["prepared_artifacts"]["cmb/scorer_view.jsonl"]
    actual_full_scorer_hash = sha(Path(args.scorer_view))
    if actual_full_scorer_hash != expected_full_scorer_hash:
        raise ValueError("CMB full scorer view differs from its frozen dataset manifest")
    if ids_manifest["source_scorer_view_sha256"] != expected_full_scorer_hash:
        raise ValueError("Common ID manifest was selected from a different CMB scorer view")
    core = json.loads(CORE_PATH.read_text(encoding="utf-8"))
    allowed_candidate_hashes = {
        core["benchmarks"]["CMB-COMMON-1024"]["candidate_view_sha256"],
        dataset_manifest["prepared_artifacts"]["cmb/candidate_view.jsonl"],
    }
    if prediction_manifest.get("candidate_view_sha256") not in allowed_candidate_hashes:
        raise ValueError("Predictions were not generated from a frozen full or common CMB candidate view")
    target_ids = set(str(row_id) for row_id in ids_manifest["ids"])
    if len(target_ids) != 1024:
        raise ValueError("Frozen CMB common ID list does not contain exactly 1024 unique IDs")
    gold_rows = read_jsonl(Path(args.scorer_view))
    gold_by_id = {str(row["id"]): row for row in gold_rows}
    pred_by_id = {str(row["id"]): row for row in predictions}
    if len(gold_by_id) != len(gold_rows) or not target_ids.issubset(gold_by_id):
        raise ValueError("Frozen CMB common IDs do not match the scorer view")
    if len(pred_by_id) != len(predictions) or not target_ids.issubset(pred_by_id):
        raise ValueError("Predictions do not cover every frozen CMB common ID")

    ordered_ids = sorted(target_ids)
    filtered_predictions = [pred_by_id[row_id] for row_id in ordered_ids]
    filtered_gold = [gold_by_id[row_id] for row_id in ordered_ids]
    output_dir = Path(args.output_dir) if args.output_dir else RUNS / "cmb_common1024"
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_prediction_path = output_dir / "predictions.jsonl"
    subset_prediction_bytes = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in filtered_predictions
    ).encode("utf-8")
    subset_prediction_path.write_bytes(subset_prediction_bytes)
    subset_prediction_hash = hashlib.sha256(subset_prediction_bytes).hexdigest()
    (output_dir / "predictions.sha256").write_text(
        f"{subset_prediction_hash}  predictions.jsonl\n", encoding="ascii"
    )
    subset_manifest = {
        **prediction_manifest,
        "schema_version": "pt-e0-derived-common-subset-prediction-v1",
        "benchmark": "CMB-COMMON-1024",
        "derived_from_prediction_sha256": prediction_hash,
        "evaluation_ids_sha256": ids_manifest["ids_sequence_sha256"],
        "ids_manifest_sha256": sha(Path(args.ids_manifest)),
        "candidate_view_sha256": protocol["common_eval_core"]["CMB-COMMON-1024"]["candidate_view_sha256"],
        "predictions_sha256": subset_prediction_hash,
        "n": len(filtered_predictions),
    }
    (output_dir / "prediction_manifest.json").write_text(
        json.dumps(subset_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    result = score_cmb_predictions(
        filtered_predictions,
        filtered_gold,
        checkpoint=prediction_manifest["checkpoint"],
        model_revision=prediction_manifest["model_revision"],
        benchmark="CMB-COMMON-1024",
        ids_sha256=ids_manifest["ids_sequence_sha256"],
    )
    result.update({
        "ids_manifest_sha256": sha(Path(args.ids_manifest)),
        "source_prediction_sha256": prediction_hash,
        "prediction_sha256": subset_prediction_hash,
        "candidate_view_sha256": protocol["common_eval_core"]["CMB-COMMON-1024"]["candidate_view_sha256"],
    })
    output_path = Path(args.output) if args.output else output_dir / "scores.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
