"""Pin ESL-Bench raw longitudinal state inputs without opening question/answer files."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_SOURCE_ROOT = Path("external/datasets/ESL-Bench")
BATCHES = {
    "202607": tuple(f"user{number}_AT_demo" for number in range(5200, 5220)),
    "202608": tuple(f"user{number}_AT_demo" for number in range(5300, 5320)),
}
STATE_FILES = ("profile.json", "timeline.json", "exam_data.json")
REQUIRED_FIELDS = {
    "profile.json": {"metadata", "demographics", "personality", "health_profile"},
    "timeline.json": {"user_id", "generated_at", "entry_count", "entries"},
    "exam_data.json": {"exam_date", "exam_type", "indicators"},
}


def build_manifest(source_root: Path) -> dict[str, Any]:
    root = source_root.resolve()
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    vcs_identity: str | None = None
    if (root / ".git").exists():
        try:
            vcs_identity = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            vcs_identity = None
    result: dict[str, Any] = {
        "schema_version": "rag-e5-esl-source-manifest-v1",
        "dataset_repo": "healthmemoryarena/ESL-Bench",
        "local_source_root": str(source_root.as_posix()),
        "upstream_git_commit": vcs_identity,
        "source_identity_basis": "git_commit" if vcs_identity else "manifest_and_state_file_hashes",
        "manifest_version": manifest["version"],
        "manifest_updated_at": manifest["updated_at"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "state_files_included": list(STATE_FILES),
        "question_or_answer_files_opened": False,
        "batches": {},
    }
    all_users: set[str] = set()
    for batch_id, expected_users in BATCHES.items():
        batch = manifest["batches"][batch_id]
        declared_users = tuple(batch["users"])
        if set(declared_users) != set(expected_users) or len(declared_users) != len(expected_users):
            raise ValueError(f"{batch_id} user set differs from the frozen expected cohort")
        if all_users.intersection(declared_users):
            raise ValueError("ESL development and holdout user sets overlap")
        all_users.update(declared_users)
        batch_root = root / "data" / batch_id
        actual_users = {
            child.name
            for child in batch_root.iterdir()
            if child.is_dir() and child.name.endswith("_AT_demo")
        }
        if actual_users != set(expected_users):
            raise ValueError(f"{batch_id} on-disk user directories differ from the pinned cohort")

        file_lines: list[str] = []
        schema_counts: dict[str, Counter[str]] = {}
        for user_id in expected_users:
            for filename in STATE_FILES:
                path = batch_root / user_id / filename
                raw = path.read_bytes()
                document = json.loads(raw)
                relative = f"{batch_id}/{user_id}/{filename}"
                file_lines.append(f"{relative}\t{hashlib.sha256(raw).hexdigest()}")
                _record_schemas(filename, document, schema_counts)
        canonical_index = "\n".join(sorted(file_lines)).encode("utf-8")
        result["batches"][batch_id] = {
            "role": "development" if batch_id == "202607" else "future_holdout",
            "published_batch_checksum": batch["checksum"],
            "evaluation_dataset_id": batch["eval_dataset"],
            "user_count": len(expected_users),
            "users": list(expected_users),
            "state_file_count": len(file_lines),
            "state_files_index_sha256": hashlib.sha256(canonical_index).hexdigest(),
            "schemas": {
                group: [
                    {"fields": json.loads(signature), "file_count": count}
                    for signature, count in sorted(counter.items())
                ]
                for group, counter in schema_counts.items()
            },
        }
    result["split"] = {
        "development_batch": "202607",
        "future_holdout_batch": "202608",
        "user_overlap": sorted(set(BATCHES["202607"]).intersection(BATCHES["202608"])),
        "user_level_split": True,
    }
    return result


def _record_schemas(filename: str, document: Any, counts: dict[str, Counter[str]]) -> None:
    if filename == "exam_data.json":
        if not isinstance(document, list):
            raise ValueError("exam_data.json must be a list")
        rows = document
        root_fields = ["array", f"length:{len(rows)}"]
    else:
        if not isinstance(document, dict):
            raise ValueError(f"{filename} must be a JSON object")
        root_fields = sorted(document)
        if not REQUIRED_FIELDS[filename].issubset(document):
            raise ValueError(f"{filename} is missing required top-level fields")
        if filename == "timeline.json":
            rows = document["entries"]
            if not isinstance(rows, list):
                raise ValueError("timeline.entries must be a list")
            if any(
                not isinstance(row, dict) or not {"time", "entry_type"}.issubset(row)
                for row in rows
            ):
                raise ValueError("timeline entries must have common time and entry_type fields")
        elif filename == "profile.json":
            for key in ("metadata", "demographics", "personality", "health_profile"):
                nested = document[key]
                _count_schema(counts, f"{filename}:{key}", _schema_description(nested))
            if not isinstance(document["demographics"], dict) or not isinstance(
                document["health_profile"], dict
            ):
                raise ValueError("profile demographics and health_profile must be objects")
            if not {"age", "gender"}.issubset(document["demographics"]):
                raise ValueError("profile demographics is missing shared identity fields")
            if not {"chronic_conditions", "past_medical_history", "lifestyle"}.issubset(
                document["health_profile"]
            ):
                raise ValueError("profile health_profile is missing shared state fields")
            rows = []
        else:
            rows = []
    if filename == "exam_data.json":
        required = REQUIRED_FIELDS[filename]
        if any(
            not isinstance(row, dict)
            or not required.issubset(row)
            or not isinstance(row["indicators"], dict)
            for row in rows
        ):
            raise ValueError("exam_data.json has a row missing common fields or indicator mapping")
    _count_schema(counts, f"{filename}:root", root_fields)
    for row in rows:
        if isinstance(row, dict):
            _count_schema(counts, f"{filename}:record", sorted(row))
            if filename == "exam_data.json":
                indicators = row.get("indicators")
                if isinstance(indicators, list):
                    for indicator in indicators:
                        if isinstance(indicator, dict):
                            _count_schema(counts, "exam_data.json:indicator", sorted(indicator))


def _count_schema(counts: dict[str, Counter[str]], group: str, fields: list[str]) -> None:
    counts.setdefault(group, Counter())[
        json.dumps(fields, ensure_ascii=False, separators=(",", ":"))
    ] += 1


def _schema_description(value: Any) -> list[str]:
    if isinstance(value, dict):
        return sorted(value)
    if isinstance(value, list):
        return ["$array"]
    if value is None:
        return ["$null"]
    if isinstance(value, bool):
        return ["$boolean"]
    if isinstance(value, int):
        return ["$integer"]
    if isinstance(value, float):
        return ["$number"]
    if isinstance(value, str):
        return ["$string"]
    return [f"$unsupported:{type(value).__name__}"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(build_manifest(args.source_root), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
