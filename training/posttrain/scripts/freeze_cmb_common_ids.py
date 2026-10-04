from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from eval.cmb_scoring import select_stratified_ids

POSTTRAIN_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PT_E0_DATA_ROOT", "/root/gpufree-data/Health-Copilot-PT-E0-data"))
EXPECTED_CMB_REVISION = "935fbc09edf1303d89872b21265ff597f426ac0d"
EXPECTED_CMB_SCORER_SHA256 = "3127e55f1704c590c3efec6bfd4637d7bd06229a70540d48a701b5966cc23e64"
SEED = 20261004
TARGET_N = 1024


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    dataset_manifest = json.loads(
        (DATA_ROOT / "eval/prepared/dataset_manifest.json").read_text(encoding="utf-8")
    )
    source = dataset_manifest["sources"]["CMB-Exam"]
    if source["revision"] != EXPECTED_CMB_REVISION or source["rows"] != 11200:
        raise ValueError("CMB dataset revision/row count differs from the frozen PT-E0 source")
    candidate_path = DATA_ROOT / "eval/prepared/cmb/candidate_view.jsonl"
    scorer_path = DATA_ROOT / "eval/prepared/cmb/scorer_view.jsonl"
    candidate_rows = read_jsonl(candidate_path)
    scorer_rows = read_jsonl(scorer_path)
    scorer_hash = sha(scorer_path.read_bytes())
    if scorer_hash != EXPECTED_CMB_SCORER_SHA256 or len(scorer_rows) != 11200:
        raise ValueError("CMB scorer view differs from the frozen PT-E0 test file")
    candidate_ids = [str(row["id"]) for row in candidate_rows]
    scorer_ids = [str(row["id"]) for row in scorer_rows]
    if len(set(candidate_ids)) != 11200 or set(candidate_ids) != set(scorer_ids):
        raise ValueError("CMB candidate/scorer views do not have matching unique IDs")
    candidate_by_id = {str(row["id"]): row for row in candidate_rows}
    scorer_by_id = {str(row["id"]): row for row in scorer_rows}
    for row_id, candidate in candidate_by_id.items():
        scorer = scorer_by_id[row_id]
        if set(candidate) & {"answer", "gold_answer", "reference_answer"}:
            raise ValueError("CMB candidate view contains a scorer-only answer field")
        if candidate.get("question_type") != scorer.get("question_type"):
            raise ValueError(f"CMB candidate question type differs from scorer metadata for {row_id}")
        if not isinstance(candidate.get("valid_options"), list) or not candidate["valid_options"]:
            raise ValueError(f"CMB candidate lacks valid option labels for {row_id}")

    metadata_rows = [{"id": row["id"], "subcategory": row["subcategory"]} for row in scorer_rows]
    ids, per_subcategory = select_stratified_ids(metadata_rows, target_n=TARGET_N, seed=SEED)
    if len(per_subcategory) != 28 or any(item["source_n"] != 400 for item in per_subcategory.values()):
        raise ValueError("CMB test must contain 28 subcategories with 400 rows each")
    selected = set(ids)
    candidate_subset = [row for row in candidate_rows if str(row["id"]) in selected]
    scorer_subset = [row for row in scorer_rows if str(row["id"]) in selected]
    if len(candidate_subset) != TARGET_N or len(scorer_subset) != TARGET_N:
        raise AssertionError("CMB common subset does not contain exactly 1024 IDs")

    selector_hash = sha(
        (POSTTRAIN_ROOT / "eval/cmb_scoring.py").read_bytes()
        + Path(__file__).read_bytes()
    )
    id_manifest = {
        "schema_version": "health-copilot-common-eval-ids-v1",
        "benchmark": "CMB-COMMON-1024",
        "dataset_repo_id": source["repo_id"],
        "dataset_revision": source["revision"],
        "config": "exam",
        "split": "test",
        "source_scorer_view_sha256": scorer_hash,
        "source_candidate_view_sha256": sha(candidate_path.read_bytes()),
        "source_n": 11200,
        "source_fields_used_for_selection": ["id", "subcategory"],
        "target_n": TARGET_N,
        "seed": SEED,
        "selection_rule": "proportional largest-remainder quotas; assign leftover quotas by ascending SHA256(seed NUL quota NUL subcategory), then select rows within each subcategory by ascending SHA256(seed NUL row NUL subcategory NUL row_id)",
        "selector_code_sha256": selector_hash,
        "ids_sequence_sha256": sha(("\n".join(ids) + "\n").encode("utf-8")),
        "ids": ids,
        "per_subcategory": per_subcategory,
    }

    diagnosis_rows = read_jsonl(DATA_ROOT / "eval/prepared/diagnosisarena/scorer_view.jsonl")
    diagnosis_ids = sorted(str(row["id"]) for row in diagnosis_rows)
    if len(diagnosis_ids) != 915 or len(set(diagnosis_ids)) != 915:
        raise ValueError("DiagnosisArena full test IDs differ from the frozen 915-row test split")
    diagnosis_id_hash = sha((json.dumps(diagnosis_ids, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))

    prepared_dir = DATA_ROOT / "eval/prepared/cmb_common1024"
    write_jsonl(prepared_dir / "candidate_view.jsonl", candidate_subset)
    write_jsonl(prepared_dir / "scorer_view.jsonl", scorer_subset)
    candidate_subset_hash = sha((prepared_dir / "candidate_view.jsonl").read_bytes())
    scorer_subset_hash = sha((prepared_dir / "scorer_view.jsonl").read_bytes())
    data_manifest = {
        "schema_version": "health-copilot-common-eval-data-v1",
        "benchmark": "CMB-COMMON-1024",
        "candidate_view_sha256": candidate_subset_hash,
        "scorer_view_sha256": scorer_subset_hash,
        "rows": TARGET_N,
        "gold_fields_are_in_scorer_view_only": True,
    }
    write_json(prepared_dir / "data_manifest.json", data_manifest)

    manifest_dir = POSTTRAIN_ROOT / "manifests/eval"
    write_json(manifest_dir / "diagnosisarena915_ids.json", diagnosis_ids)
    write_json(manifest_dir / "cmb_common1024_ids.json", id_manifest)
    core = {
        "schema_version": "health-copilot-common-eval-core-v1",
        "name": "Health-Copilot Common Medical Evaluation Core",
        "shared_across": ["Single", "RAG", "Multi-Agent", "RAG+Multi-Agent", "post-trained checkpoints"],
        "shared_contract": "same frozen evaluation IDs, hidden gold answers, and answer parser/scorer; each system may use its intended execution path and module-specific prompt/context",
        "system_specific_context": "RAG may add retrieved evidence; Multi-Agent may add agent coordination/context; record these inputs per arm and never expose benchmark references or scorer metadata",
        "closed_book_comparability": "B0, P0, and R0 use the same standalone prompt and decoding settings",
        "system_arms": {
            "B0": "Qwen3-8B Base + Single + closed-book",
            "B1": "Qwen3-8B Base + RAG",
            "B2": "Qwen3-8B Base + Multi-Agent",
            "B3": "Qwen3-8B Base + RAG + Multi-Agent",
            "P0-P3": "same four system arms with the Medical SFT checkpoint",
            "R0-R3": "same four system arms with the SFT + GSPO/GDPO checkpoint",
        },
        "benchmarks": {
            "DiagnosisArena-915": {
                "repo_id": "SII-SPIRAL-MED/DiagnosisArena",
                "revision": "64bb873fe651e1c71cd8b0958104dd911f042872",
                "split": "test",
                "n": 915,
                "ids_file": "diagnosisarena915_ids.json",
                "ids_file_sha256": diagnosis_id_hash,
            },
            "CMB-COMMON-1024": {
                "repo_id": source["repo_id"],
                "revision": source["revision"],
                "config": "exam",
                "split": "test",
                "n": TARGET_N,
                "seed": SEED,
                "ids_file": "cmb_common1024_ids.json",
                "ids_manifest_sha256": sha((manifest_dir / "cmb_common1024_ids.json").read_bytes()),
                "ids_sequence_sha256": id_manifest["ids_sequence_sha256"],
                "candidate_view_sha256": candidate_subset_hash,
                "scorer_view_sha256": scorer_subset_hash,
                "allocation": {name: item["selected_n"] for name, item in per_subcategory.items()},
            },
        },
        "non_core_extensions": {
            "CMB-Exam-test-full-11200": "backbone-only extended evaluation; not the common system-level run",
            "HealthBench-Professional": "open-ended post-training backbone evaluation with frozen local rubric judge",
            "MedQA-1273": "historical MDAgents reproduction-specific benchmark",
            "R2MED": "retrieval-specific benchmark",
            "LoCoMo": "memory-specific benchmark",
            "LiveMedBench": "reserved final confirmation set; scores remain unopened",
        },
    }
    write_json(manifest_dir / "common_eval_core.json", core)

    reserved_dir = DATA_ROOT / "runs/posttrain/pt-e0/cmb_common1024"
    write_json(reserved_dir / "reserved1024_ids.json", ids)
    ids_path = reserved_dir / "reserved1024_ids.json"
    (reserved_dir / "reserved1024_ids.sha256").write_text(
        f"{sha(ids_path.read_bytes())}  reserved1024_ids.json\n", encoding="ascii"
    )
    write_json(reserved_dir / "id_manifest.json", id_manifest)
    write_json(reserved_dir / "data_manifest.json", data_manifest)
    note = (
        "# CMB-COMMON-1024 frozen IDs\n\n"
        "This deterministic subset is part of the shared Health-Copilot Common Medical Evaluation Core. "
        "Selection used only CMB row IDs and subcategory metadata with seed 20261004. "
        "Candidate generation must use candidate_view.jsonl; answers exist only in scorer_view.jsonl. "
        "Do not change these IDs or tune prompts/rewards against this subset.\n"
    )
    (reserved_dir / "README_RESERVED_DO_NOT_TUNE.md").write_text(note, encoding="utf-8")
    print(json.dumps({
        "n": TARGET_N,
        "ids_manifest_sha256": sha((manifest_dir / "cmb_common1024_ids.json").read_bytes()),
        "ids_sequence_sha256": id_manifest["ids_sequence_sha256"],
        "candidate_view_sha256": candidate_subset_hash,
        "scorer_view_sha256": scorer_subset_hash,
        "allocation": {name: item["selected_n"] for name, item in per_subcategory.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
