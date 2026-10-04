from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.healthbench_scoring import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    LENGTH_ADJUSTMENT_CENTER,
    LENGTH_PENALTY_PER_500_CHARS,
    clipped_bootstrap_ci,
    clipped_mean,
    length_adjusted_score,
    rubric_score,
)

DATA_ROOT = Path(os.environ.get("PT_E0_DATA_ROOT", "/root/gpufree-data/Health-Copilot-PT-E0-data"))
RUN_DIR = Path(os.environ.get("PT_E0_RUNS", str(DATA_ROOT / "runs/posttrain/pt-e0"))) / "hbpro"
SCORER_VIEW = DATA_ROOT / "eval/prepared/hbpro/scorer_view.jsonl"
POSTTRAIN_ROOT = Path(__file__).resolve().parents[1]
JUDGE_PROTOCOL_PATH = POSTTRAIN_ROOT / "manifests/eval/healthbench_local_judge.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_file(path: Path, manifest_hash: str, sidecar: Path) -> None:
    actual = sha256_file(path)
    recorded = sidecar.read_text(encoding="ascii").split()[0]
    if actual != manifest_hash or actual != recorded:
        raise ValueError(f"Frozen artifact failed SHA256 verification: {path.name}")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw = [row["raw_score"] for row in rows]
    adjusted = [row["length_adjusted_score"] for row in rows]
    return {
        "n": len(rows),
        "raw_mean": clipped_mean(raw),
        "raw_bootstrap_95_ci": list(clipped_bootstrap_ci(raw)) if rows else [None, None],
        "length_adjusted_mean": clipped_mean(adjusted),
        "length_adjusted_bootstrap_95_ci": list(clipped_bootstrap_ci(adjusted)) if rows else [None, None],
    }


def main() -> None:
    argparse.ArgumentParser().parse_args()
    judge_protocol = json.loads(JUDGE_PROTOCOL_PATH.read_text(encoding="utf-8"))
    if judge_protocol["score_calculation"]["scoring_code_sha256"] != sha256_file(POSTTRAIN_ROOT / "eval/healthbench_scoring.py"):
        raise ValueError("HealthBench scoring functions differ from the frozen judge protocol")
    if judge_protocol["score_calculation"]["scorer_entrypoint_sha256"] != sha256_file(Path(__file__)):
        raise ValueError("HealthBench scoring entrypoint differs from the frozen judge protocol")
    prediction_path = RUN_DIR / "predictions.jsonl"
    prediction_manifest = json.loads((RUN_DIR / "prediction_manifest.json").read_text(encoding="utf-8"))
    verify_file(prediction_path, prediction_manifest["predictions_sha256"], RUN_DIR / "predictions.sha256")

    grade_path = RUN_DIR / "rubric_grades.jsonl"
    grade_hash = sha256_file(grade_path)
    verify_file(grade_path, grade_hash, RUN_DIR / "rubric_grades.sha256")
    predictions = read_jsonl(prediction_path)
    scorer_rows = read_jsonl(SCORER_VIEW)
    grade_rows = read_jsonl(grade_path)
    predictions_by_id = {str(row["id"]): row for row in predictions}
    scorer_by_id = {str(row["id"]): row for row in scorer_rows}
    if len(predictions_by_id) != len(predictions) or set(predictions_by_id) != set(scorer_by_id):
        raise ValueError("Frozen predictions and HB-Pro scorer view IDs differ")

    grades_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_tasks = set()
    for row in grade_rows:
        if row["task_id"] in seen_tasks:
            raise ValueError("Duplicate criterion grader output")
        seen_tasks.add(row["task_id"])
        grades_by_id[str(row["id"])].append(row)

    case_rows = []
    expected_grade_count = 0
    for example_id in sorted(scorer_by_id):
        gold = scorer_by_id[example_id]
        items = gold["rubric_items"]
        grades = sorted(grades_by_id.get(example_id, []), key=lambda row: row["criterion_index"])
        expected_grade_count += len(items)
        if len(grades) != len(items) or [row["criterion_index"] for row in grades] != list(range(len(items))):
            raise ValueError(f"Incomplete rubric grades for case {example_id}")
        for item, grade in zip(items, grades, strict=True):
            if float(item["points"]) != float(grade["points"]) or item["criterion_text"] != grade["criterion_text"]:
                raise ValueError(f"Rubric item mismatch for case {example_id}")
        answer = str(predictions_by_id[example_id]["final_answer"])
        raw = rubric_score(items, grades)
        if raw is None:
            raise ValueError(f"No positive rubric points for case {example_id}")
        case_rows.append({
            "id": example_id,
            "raw_score": raw,
            "length_adjusted_score": length_adjusted_score(raw, answer),
            "answer_characters": len(answer),
            "use_case": gold.get("use_case"),
            "type": gold.get("type"),
            "difficulty": gold.get("difficulty"),
            "specialty": gold.get("specialty"),
        })
    if len(grade_rows) != expected_grade_count:
        raise ValueError("Unexpected extra rubric grade rows")

    subgroup_specs = {
        "use_case": {name: lambda row, name=name: row["use_case"] == name for name in ("consult", "writing", "research")},
        "type": {name: lambda row, name=name: str(row["type"]).casefold() == name for name in ("good_faith", "red_teaming")},
        "difficulty": {name: lambda row, name=name: str(row["difficulty"]).casefold() == name for name in ("typical", "difficult")},
    }
    subgroup_scores = {}
    for dimension, groups in subgroup_specs.items():
        subgroup_scores[dimension] = {
            name: summarize([row for row in case_rows if predicate(row)])
            for name, predicate in groups.items()
        }

    result = {
        "checkpoint": prediction_manifest["checkpoint"],
        "model_revision": prediction_manifest["model_revision"],
        "metric_name": "HB-Pro Local-Rubric Score",
        "official_healthbench_leaderboard_score": False,
        "n": len(case_rows),
        **summarize(case_rows),
        "subgroups": subgroup_scores,
        "specialty_count": len({row["specialty"] for row in case_rows if row["specialty"] is not None}),
        "criterion_grades": expected_grade_count,
        "score_contract": {
            "positive_denominator": "sum of positive rubric points",
            "numerator": "sum of signed points for all criteria marked met, including negative criteria",
            "aggregation": "mean clipped to [0,1]",
            "length_adjustment_center_characters": LENGTH_ADJUSTMENT_CENTER,
            "length_penalty_per_500_characters": LENGTH_PENALTY_PER_500_CHARS,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "rubric_grades_sha256": grade_hash,
            "predictions_sha256": prediction_manifest["predictions_sha256"],
        },
    }
    out_path = RUN_DIR / "scores.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
