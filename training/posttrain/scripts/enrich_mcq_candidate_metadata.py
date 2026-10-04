from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from eval.prompts import cmb_prompt, diagnosisarena_prompt, option_labels

DATA_ROOT = Path(os.environ.get("PT_E0_DATA_ROOT", "/root/gpufree-data/Health-Copilot-PT-E0-data"))
PREPARED = DATA_ROOT / "eval/prepared"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    manifest_path = PREPARED / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    diag_file = DATA_ROOT / "eval/diagnosisarena/data/test-00000-of-00001.parquet"
    diag_source = manifest["sources"]["DiagnosisArena"]
    if sha256_file(diag_file) != diag_source["file_sha256"] or diag_source["rows"] != 915:
        raise ValueError("DiagnosisArena source differs from the frozen dataset manifest")
    diag_frame = pd.read_parquet(
        diag_file,
        columns=["id", "Case Information", "Physical Examination", "Diagnostic Tests", "Options"],
    )
    diag_candidates = []
    diag_ids = set()
    old_diag = {
        str(row["id"]): row
        for row in read_jsonl(PREPARED / "diagnosisarena/candidate_view.jsonl")
    }
    for row in diag_frame.to_dict(orient="records"):
        row_id = f"diagnosisarena:{row['id']}"
        diag_ids.add(row_id)
        prompt = diagnosisarena_prompt(row)
        fingerprint_text = "\n".join(
            str(row.get(field, ""))
            for field in ("Case Information", "Physical Examination", "Diagnostic Tests", "Options")
        )
        old = old_diag[row_id]
        if old["prompt"] != prompt or old["fingerprint_text"] != fingerprint_text:
            raise ValueError("DiagnosisArena prompt/fingerprint would change during metadata enrichment")
        diag_candidates.append({
            "id": row_id,
            "prompt": prompt,
            "fingerprint_text": fingerprint_text,
            "valid_options": option_labels(row["Options"]),
        })
    if diag_ids != set(old_diag):
        raise ValueError("DiagnosisArena candidate IDs changed during metadata enrichment")

    cmb_file = DATA_ROOT / "eval/cmb/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json"
    cmb_source = manifest["sources"]["CMB-Exam"]
    if sha256_file(cmb_file) != cmb_source["question_sha256"] or cmb_source["rows"] != 11200:
        raise ValueError("CMB source differs from the frozen dataset manifest")
    cmb_questions = json.loads(cmb_file.read_text(encoding="utf-8"))
    old_cmb = {
        str(row["id"]): row
        for row in read_jsonl(PREPARED / "cmb/candidate_view.jsonl")
    }
    cmb_candidates = []
    cmb_ids = set()
    for row in cmb_questions:
        row_id = f"cmb:{row['id']}"
        cmb_ids.add(row_id)
        prompt = cmb_prompt(row)
        fingerprint_text = str(row.get("question", ""))
        old = old_cmb[row_id]
        if old["prompt"] != prompt or old["fingerprint_text"] != fingerprint_text:
            raise ValueError("CMB prompt/fingerprint would change during metadata enrichment")
        question_type = str(row.get("question_type") or "")
        if question_type not in {"单项选择题", "多项选择题", "C型选择题"}:
            raise ValueError(f"Unknown CMB question type for {row_id}")
        cmb_candidates.append({
            "id": row_id,
            "prompt": prompt,
            "fingerprint_text": fingerprint_text,
            "question_type": question_type,
            "valid_options": option_labels(row.get("option")),
        })
    if cmb_ids != set(old_cmb):
        raise ValueError("CMB candidate IDs changed during metadata enrichment")

    hashes = {
        "diagnosisarena/candidate_view.jsonl": write_jsonl(
            PREPARED / "diagnosisarena/candidate_view.jsonl", diag_candidates
        ),
        "cmb/candidate_view.jsonl": write_jsonl(
            PREPARED / "cmb/candidate_view.jsonl", cmb_candidates
        ),
    }
    manifest["prepared_artifacts"].update(hashes)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "diagnosisarena_n": len(diag_candidates),
        "cmb_n": len(cmb_candidates),
        "cmb_question_types": {
            name: sum(row["question_type"] == name for row in cmb_candidates)
            for name in ("单项选择题", "多项选择题", "C型选择题")
        },
        "candidate_view_hashes": hashes,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
