from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from data.canonicalize import canonicalize, fingerprint
from eval.prompts import cmb_prompt, diagnosisarena_prompt, healthbench_messages, livemedbench_prompt, option_labels

DATA_ROOT = Path(os.environ.get("PT_E0_DATA_ROOT", "/root/gpufree-data/Health-Copilot-PT-E0-data"))
OUT = DATA_ROOT / "eval" / "prepared"
SEED = 20261004
RESERVED_N = 1024
LIVE_THEMES = {
    "Expertise-Tailored",
    "Responding under Uncertainty",
    "Context-Seeking",
    "Response Depth",
    "Emergency Referrals",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return sha256_file(path)


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sha256_file(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def detect_language(text: str) -> str:
    # Frozen deterministic metadata-only detector; no examples are displayed or hand-labelled.
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return "unknown"
    han = sum("\u3400" <= char <= "\u9fff" for char in letters)
    return "zh" if han / len(letters) >= 0.05 else "en"


def live_theme(value: Any) -> str:
    if isinstance(value, list):
        labels = [str(item).strip() for item in value if str(item).strip()]
        return labels[0] if labels else "unknown"
    return str(value or "unknown").strip()


def choose_reserved(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) < n:
        raise ValueError(f"LiveMedBench has {len(rows)} rows; cannot reserve {n}")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["language"], row["theme"])].append(row)
    if not LIVE_THEMES.issubset({theme for _, theme in groups}):
        raise ValueError("LiveMedBench snapshot does not contain every frozen behavioral theme")
    sizes = {key: len(values) for key, values in groups.items()}
    exact = {key: n * size / len(rows) for key, size in sizes.items()}
    quota = {key: int(math.floor(value)) for key, value in exact.items()}
    for key in sorted(sizes):
        quota[key] = min(quota[key], sizes[key])
    remaining = n - sum(quota.values())
    order = sorted(sizes, key=lambda key: (-(exact[key] - math.floor(exact[key])), key))
    while remaining:
        changed = False
        for key in order:
            if quota[key] < sizes[key]:
                quota[key] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            raise ValueError("Unable to allocate reserved subset quotas")
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: row["id"])
        rng.shuffle(group)
        selected.extend(group[:quota[key]])
    return sorted(selected, key=lambda row: row["id"])


def prepare() -> dict[str, Any]:
    diag_file = DATA_ROOT / "eval/diagnosisarena/data/test-00000-of-00001.parquet"
    cmb_file = DATA_ROOT / "eval/cmb/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json"
    cmb_answer_file = DATA_ROOT / "eval/cmb/CMB-test-choice-answer.json"
    hb_file = DATA_ROOT / "eval/hbpro/healthbench_professional_eval.jsonl"
    live_file = DATA_ROOT / "eval/livemedbench/LiveMedBench_v202604_new.json"

    diag_frame = pd.read_parquet(diag_file, columns=["id", "Case Information", "Physical Examination", "Diagnostic Tests", "Options"])
    diag_candidates = []
    diag_scorers = []
    for row in diag_frame.to_dict(orient="records"):
        row_id = str(row["id"])
        diag_candidates.append({"id": f"diagnosisarena:{row_id}", "prompt": diagnosisarena_prompt(row), "fingerprint_text": "\n".join(str(row.get(field, "")) for field in ("Case Information", "Physical Examination", "Diagnostic Tests", "Options")), "valid_options": option_labels(row["Options"])})
        diag_scorers.append({"id": f"diagnosisarena:{row_id}", "answer": None})
    # Right Option is loaded in a separate scorer-only read.
    diag_gold = pd.read_parquet(diag_file, columns=["id", "Right Option"])
    diag_answer = {str(row["id"]): row["Right Option"] for row in diag_gold.to_dict(orient="records")}
    for row in diag_scorers:
        row["answer"] = diag_answer[row["id"].split(":", 1)[1]]

    cmb_questions = json.loads(cmb_file.read_text(encoding="utf-8"))
    cmb_answers = json.loads(cmb_answer_file.read_text(encoding="utf-8"))
    answer_by_id = {str(row["id"]): row for row in cmb_answers}
    if len(answer_by_id) != len(cmb_answers) or len(cmb_questions) != len(answer_by_id):
        raise ValueError("CMB question and answer IDs are not one-to-one")
    cmb_candidates = []
    cmb_scorers = []
    for row in cmb_questions:
        row_id = str(row["id"])
        gold = answer_by_id[row_id]
        cmb_candidates.append({"id": f"cmb:{row_id}", "prompt": cmb_prompt(row), "fingerprint_text": str(row.get("question", "")), "question_type": row["question_type"], "valid_options": option_labels(row["option"])})
        cmb_scorers.append({
            "id": f"cmb:{row_id}",
            "answer": gold["answer"],
            "question_type": row["question_type"],
            "major_category": row["exam_type"],
            "subcategory": row["exam_class"],
            "exam_subject": row.get("exam_subject"),
        })

    hb_candidates = []
    hb_scorers = []
    for row in read_jsonl(hb_file):
        row_id = str(row["id"])
        messages = healthbench_messages(row["conversation"])
        hb_candidates.append({"id": f"hbpro:{row_id}", "messages": messages, "fingerprint_text": "\n".join(item["content"] for item in messages)})
        hb_scorers.append({
            "id": f"hbpro:{row_id}",
            "rubric_items": row["rubric_items"],
            "physician_response": row.get("physician_response"),
            "use_case": row.get("use_case"),
            "type": row.get("type"),
            "difficulty": row.get("difficulty"),
            "specialty": row.get("specialty"),
        })

    live_rows = json.loads(live_file.read_text(encoding="utf-8"))
    live_candidate_rows = []
    live_metadata = []
    for row in live_rows:
        prompt = livemedbench_prompt(row)
        row_id = str(row.get("case_id"))
        language = str(row.get("language") or detect_language(prompt)).lower()
        theme = live_theme(row.get("theme"))
        if theme not in LIVE_THEMES:
            raise ValueError(f"Unknown LiveMedBench theme label: {theme}")
        entry = {"id": f"livemedbench:{row_id}", "prompt": prompt, "fingerprint_text": prompt, "language": language, "theme": theme}
        live_candidate_rows.append(entry)
        live_metadata.append({"id": entry["id"], "language": language, "theme": theme})

    reserved = choose_reserved(live_candidate_rows, RESERVED_N, SEED)
    reserved_ids = [{"id": row["id"], "language": row["language"], "theme": row["theme"]} for row in reserved]

    files = {
        "diagnosisarena/candidate_view.jsonl": write_jsonl(OUT / "diagnosisarena/candidate_view.jsonl", diag_candidates),
        "diagnosisarena/scorer_view.jsonl": write_jsonl(OUT / "diagnosisarena/scorer_view.jsonl", diag_scorers),
        "cmb/candidate_view.jsonl": write_jsonl(OUT / "cmb/candidate_view.jsonl", cmb_candidates),
        "cmb/scorer_view.jsonl": write_jsonl(OUT / "cmb/scorer_view.jsonl", cmb_scorers),
        "hbpro/candidate_view.jsonl": write_jsonl(OUT / "hbpro/candidate_view.jsonl", hb_candidates),
        "hbpro/scorer_view.jsonl": write_jsonl(OUT / "hbpro/scorer_view.jsonl", hb_scorers),
        "livemedbench/reserved1024_candidate_view.jsonl": write_jsonl(OUT / "livemedbench/reserved1024_candidate_view.jsonl", reserved),
        "livemedbench/reserved1024_ids.json": write_json(OUT / "livemedbench/reserved1024_ids.json", {
            "dataset": "JuelieYann/LiveMedBench",
            "snapshot": "v202604_new",
            "seed": SEED,
            "n": RESERVED_N,
            "stratification": "language x primary theme; language detector = Han-letter fraction >= 0.05",
            "ids": reserved_ids,
        }),
    }
    manifest = {
        "schema_version": "pt-e0-eval-datasets-v1",
        "created_utc_date": "2026-10-04",
        "sources": {
            "DiagnosisArena": {
                "repo_id": "SII-SPIRAL-MED/DiagnosisArena",
                "revision": "64bb873fe651e1c71cd8b0958104dd911f042872",
                "file": str(diag_file),
                "file_sha256": sha256_file(diag_file),
                "rows": len(diag_candidates),
                "candidate_columns": ["id", "Case Information", "Physical Examination", "Diagnostic Tests", "Options"],
                "scorer_columns": ["id", "Right Option"],
            },
            "CMB-Exam": {
                "repo_id": "FreedomIntelligence/CMB",
                "revision": "935fbc09edf1303d89872b21265ff597f426ac0d",
                "question_file": str(cmb_file),
                "question_sha256": sha256_file(cmb_file),
                "answer_key_repo": "FreedomIntelligence/CMB",
                "answer_key_revision": "6c8ece46097dae736c6805dd3b831e1a38c08971",
                "answer_key_file": str(cmb_answer_file),
                "answer_key_sha256": sha256_file(cmb_answer_file),
                "rows": len(cmb_candidates),
            },
            "HealthBench Professional": {
                "repo_id": "openai/healthbench-professional",
                "revision": "349962fd46dd02343a0d8a606491baf59154ea1a",
                "file": str(hb_file),
                "file_sha256": sha256_file(hb_file),
                "rows": len(hb_candidates),
                "candidate_fields": ["conversation"],
                "scorer_fields": ["rubric_items", "physician_response", "use_case", "type", "difficulty", "specialty"],
            },
            "LiveMedBench": {
                "repo_id": "JuelieYann/LiveMedBench",
                "revision": "db7eb218959aeac83fc9d9353efd7d909b6616f0",
                "snapshot_file": str(live_file),
                "snapshot_sha256": sha256_file(live_file),
                "rows": len(live_candidate_rows),
                "reserved_n": len(reserved),
                "reserved_ids_sha256": files["livemedbench/reserved1024_ids.json"],
                "reserved_theme_counts": dict(Counter(row["theme"] for row in reserved)),
                "reserved_language_theme_counts": {f"{language}|{theme}": count for (language, theme), count in sorted(Counter((row["language"], row["theme"]) for row in reserved).items())},
                "score_status": "UNOPENED",
            },
        },
        "prepared_artifacts": files,
    }
    write_json(OUT / "dataset_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
