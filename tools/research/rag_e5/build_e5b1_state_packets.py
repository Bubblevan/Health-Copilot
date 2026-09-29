"""Build private v4 state packets and a Git-safe aggregate coverage report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

from eval.rag_e5.age import AGE_TEMPORAL_CONTRACT_SHA256
from eval.rag_e5.longitudinal_state import (
    PROFILE_TEMPORAL_CONTRACT_SHA256,
    STATE_PACKET_BUILDER_VERSION,
    STATE_PACKET_CONFIG_SHA256,
    build_longitudinal_state_packet,
)
from eval.rag_e5.temporal import (
    TEMPORAL_SEMANTICS_ID,
    TEMPORAL_SEMANTICS_SHA256,
    latest_decision_boundary,
)

DEVELOPMENT_BATCH = "202607"
DEVELOPMENT_USERS = tuple(f"user{number}_AT_demo" for number in range(5200, 5220))
DEFAULT_SOURCE_ROOT = Path("D:/MyLab/Jianli/external/datasets/ESL-Bench")
DEFAULT_SOURCE_MANIFEST = Path("runs/rag_e5/esl_source_manifest.json")
DEFAULT_PRIVATE_ROOT = Path("D:/MyLab/Jianli/external/rag_e5/e5b1")
DEFAULT_COVERAGE_PATH = Path("runs/rag_e5/e5b1_state_coverage_report.json")


def build_state_packets(
    *, source_root: Path, private_root: Path, coverage_path: Path,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
) -> dict[str, Any]:
    """Read only the allowlisted 202607 state files; persist patient artifacts externally."""
    _validate_source_manifest(_read_json(source_manifest_path))
    packet_root = private_root / "state_packets"
    packet_root.mkdir(parents=True, exist_ok=True)
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    packet_identity: list[dict[str, str]] = []
    chosen_metrics: Counter[str] = Counter()
    users_with_weight = 0
    users_with_bmi_fallback = 0
    users_with_neither = 0
    age_safe = 0
    age_unknown = 0

    for user_id in DEVELOPMENT_USERS:
        user_root = source_root / "data" / DEVELOPMENT_BATCH / user_id
        paths = {name: user_root / name for name in ("profile.json", "timeline.json", "exam_data.json")}
        payloads = {name: _read_json(path) for name, path in paths.items()}
        profile = payloads["profile.json"]
        timeline = payloads["timeline.json"]
        exams = payloads["exam_data.json"]
        if not isinstance(profile, dict) or not isinstance(timeline, dict):
            raise TypeError("202607 profile/timeline roots must be JSON objects")
        if timeline.get("user_id") != user_id:
            raise ValueError("202607 timeline identity differs from the pinned user")
        if not isinstance(exams, list):
            raise TypeError("202607 exam_data root must be a JSON array")

        boundary = latest_decision_boundary(user_id=user_id, timeline=timeline)
        hashes = {name: _file_sha256(path) for name, path in paths.items()}
        packet = build_longitudinal_state_packet(
            decision_boundary=boundary,
            profile=profile,
            timeline=timeline,
            exam_data=exams,
            source_file_hashes=hashes,
        )
        trends = dict(packet.recent_measurement_trends)
        if "body_weight" in trends:
            chosen_metrics["body_weight"] += 1
            users_with_weight += 1
        elif "body_mass_index" in trends:
            chosen_metrics["body_mass_index"] += 1
            users_with_bmi_fallback += 1
        else:
            users_with_neither += 1
        if packet.age_bucket == "unknown":
            age_unknown += 1
        else:
            age_safe += 1

        target = packet_root / f"{user_id}.json"
        _write_json(target, packet.to_dict())
        packet_identity.append(
            {
                "user_id": user_id,
                "packet_sha256": packet.packet_sha256,
                "decision_boundary": boundary.naive_timestamp,
            }
        )

    packet_set_sha = _canonical_sha256(packet_identity)
    users_with_required_trend = users_with_weight + users_with_bmi_fallback
    report = {
        "schema_version": "rag-e5-e5b1-state-coverage-v1",
        "batch_id": DEVELOPMENT_BATCH,
        "users_total": len(DEVELOPMENT_USERS),
        "users_with_weight_trend": users_with_weight,
        "users_with_bmi_fallback": users_with_bmi_fallback,
        "users_with_neither": users_with_neither,
        "users_failed": users_with_neither,
        "chosen_metric_counts": dict(sorted(chosen_metrics.items())),
        "age_safe_count": age_safe,
        "unknown_age_count": age_unknown,
        "state_packet_builder_version": STATE_PACKET_BUILDER_VERSION,
        "state_packet_config_sha256": STATE_PACKET_CONFIG_SHA256,
        "profile_temporal_contract_sha256": PROFILE_TEMPORAL_CONTRACT_SHA256,
        "age_temporal_contract_sha256": AGE_TEMPORAL_CONTRACT_SHA256,
        "temporal_semantics_id": TEMPORAL_SEMANTICS_ID,
        "temporal_semantics_sha256": TEMPORAL_SEMANTICS_SHA256,
        "state_packet_set_sha256": packet_set_sha,
        "coverage_gate": "PASS" if users_with_required_trend == 20 else "FAIL",
        "boundary_rule": "latest_unique_observed_timeline_timestamp_no_rescue",
        "raw_source_files_read_per_user": ["profile.json", "timeline.json", "exam_data.json"],
        "esl_source_manifest_sha256": _file_sha256(source_manifest_path),
        "native_esl_questions_opened": False,
        "native_esl_answers_opened": False,
        "kg_evaluation_queries_opened": False,
        "202608_opened": False,
        "patient_level_values_or_prose_included": False,
    }
    _write_json(coverage_path, report)
    return report


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_source_manifest(value: Any) -> None:
    if not isinstance(value, dict):
        raise TypeError("ESL source manifest must be an object")
    batch = value.get("batches", {}).get(DEVELOPMENT_BATCH)
    if not isinstance(batch, dict) or batch.get("role") != "development":
        raise ValueError("202607 must be the source manifest's development batch")
    if batch.get("user_count") != len(DEVELOPMENT_USERS):
        raise ValueError("202607 source manifest user count changed")
    if tuple(batch.get("users", ())) != DEVELOPMENT_USERS:
        raise ValueError("202607 source manifest user ordering/identity changed")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--coverage-path", type=Path, default=DEFAULT_COVERAGE_PATH)
    args = parser.parse_args()
    report = build_state_packets(
        source_root=args.source_root,
        private_root=args.private_root,
        coverage_path=args.coverage_path,
        source_manifest_path=args.source_manifest,
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    if report["coverage_gate"] != "PASS":
        raise SystemExit("E5-B1 state coverage gate failed; task construction is not authorized")


if __name__ == "__main__":
    main()
