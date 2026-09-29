"""Aggregate only 202607 longitudinal health-domain coverage.

This audit opens only manifest.json and each frozen 202607 user's profile.json,
timeline.json, and exam_data.json. It never reads benchmark question/answer
files, KG evaluation queries, or the 202608 batch. Patient prose and numeric
measurements are never emitted; output contains counts and indicator labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DEVELOPMENT_BATCH = "202607"
EXPECTED_USERS = tuple(f"user{number}_AT_demo" for number in range(5200, 5220))
ALLOWED_FILES = ("profile.json", "timeline.json", "exam_data.json")
DOMAIN_PATTERNS = {
    "hypertension": r"hypertension|blood[ _-]?pressure|\bbp\b|systolic|diastolic",
    "diabetes": r"diabetes|glucose|glyc|hba1c|insulin|blood[ _-]?sugar",
    "weight_metabolic": r"weight|bmi|body[ _-]?mass|obes|overweight|waist|lipid|cholesterol|fat",
    "physical_activity": r"physical[ _-]?activity|exercise|steps|sedentary|activity|active",
    "nutrition": r"diet|nutrition|food|calor|protein|carbohydrate|fiber|fibre|fat[ _-]?intake",
    "tobacco": r"tobacco|smok|nicotine",
    "alcohol": r"alcohol|drinking|ethanol",
    "sleep": r"sleep|insomnia|apnea|apnoea",
    "cognitive": r"cognit|dementia|memory|neuropsych",
    "cardiovascular": r"cardio|heart|cardiac|ecg|ekg|pulse|heart[ _-]?rate",
    "kidney": r"kidney|renal|creatinine|egfr|albuminuria",
    "respiratory": r"respirat|pulmonary|lung|spo2|oxygen|peak[ _-]?flow",
}
COMPILED_PATTERNS = {
    domain: re.compile(pattern, re.IGNORECASE) for domain, pattern in DOMAIN_PATTERNS.items()
}
SAFE_LABEL = re.compile(r"^[A-Za-z0-9_./ -]{1,64}$")


def _read_json(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    return json.loads(raw), raw


def _domain_hits(label: object) -> set[str]:
    if not isinstance(label, str):
        return set()
    return {
        domain for domain, pattern in COMPILED_PATTERNS.items() if pattern.search(label)
    }


def _structured_labels(value: Any) -> set[str]:
    """Read category-bearing structures without returning their source values."""
    labels: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(key, str):
                labels.add(key)
            if isinstance(nested, (dict, list, tuple)):
                labels.update(_structured_labels(nested))
            elif isinstance(nested, str):
                # Values are inspected only for coarse domain classification.
                # They are never copied into the output.
                labels.add(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            if isinstance(nested, str):
                labels.add(nested)
            elif isinstance(nested, (dict, list, tuple)):
                labels.update(_structured_labels(nested))
    elif isinstance(value, str):
        labels.add(value)
    return labels


def _safe_indicator(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.strip().split())
    if not SAFE_LABEL.fullmatch(normalized):
        return None
    return normalized


def build_inventory(source_root: Path) -> dict[str, Any]:
    root = source_root.resolve()
    manifest, manifest_bytes = _read_json(root / "manifest.json")
    batch = manifest["batches"][DEVELOPMENT_BATCH]
    declared_users = tuple(batch["users"])
    if declared_users != EXPECTED_USERS:
        raise ValueError("202607 user cohort differs from the frozen E5-A cohort")
    batch_root = root / "data" / DEVELOPMENT_BATCH
    actual_users = tuple(
        sorted(
            child.name
            for child in batch_root.iterdir()
            if child.is_dir() and child.name.endswith("_AT_demo")
        )
    )
    if set(actual_users) != set(EXPECTED_USERS) or len(actual_users) != len(EXPECTED_USERS):
        raise ValueError("202607 on-disk users differ from the frozen E5-A cohort")

    age_ranges: Counter[str] = Counter()
    chronic_users: Counter[str] = Counter()
    longitudinal_users: Counter[str] = Counter()
    state_users: Counter[str] = Counter()
    timeline_events: Counter[str] = Counter()
    timeline_event_users: dict[str, set[str]] = {}
    exam_users: dict[str, set[str]] = {}
    chronic_category_users: dict[str, set[str]] = {}
    lifestyle_fields: Counter[str] = Counter()
    measurement_indicators: Counter[str] = Counter()
    exam_indicators: Counter[str] = Counter()
    timeline_entry_types: Counter[str] = Counter()
    file_hashes: list[str] = []

    for user_id in EXPECTED_USERS:
        user_root = batch_root / user_id
        profile, profile_raw = _read_json(user_root / "profile.json")
        timeline, timeline_raw = _read_json(user_root / "timeline.json")
        exams, exams_raw = _read_json(user_root / "exam_data.json")
        for filename, raw in zip(ALLOWED_FILES, (profile_raw, timeline_raw, exams_raw), strict=True):
            file_hashes.append(
                f"{DEVELOPMENT_BATCH}/{user_id}/{filename}\t{hashlib.sha256(raw).hexdigest()}"
            )

        demographics = profile.get("demographics", {})
        age = demographics.get("age") if isinstance(demographics, dict) else None
        age_value = _numeric_age(age)
        age_range = _age_range(age_value)
        age_ranges[age_range] += 1

        health = profile.get("health_profile", {})
        if not isinstance(health, dict):
            raise TypeError("profile.health_profile must be an object")
        for field_name in ("chronic_conditions", "past_medical_history"):
            categories = set().union(
                *(_domain_hits(label) for label in _structured_labels(health.get(field_name)))
            ) if health.get(field_name) is not None else set()
            for domain in categories:
                chronic_category_users.setdefault(domain, set()).add(user_id)
                chronic_users[domain] += 1
        lifestyle = health.get("lifestyle")
        if isinstance(lifestyle, dict):
            for field_name in lifestyle:
                if isinstance(field_name, str):
                    lifestyle_fields[field_name] += 1

        history_domains: set[str] = set()
        entries = timeline.get("entries", []) if isinstance(timeline, dict) else []
        if not isinstance(entries, list):
            raise TypeError("timeline.entries must be a list")
        for row in entries:
            if not isinstance(row, dict):
                continue
            entry_type = row.get("entry_type")
            if isinstance(entry_type, str):
                timeline_entry_types[entry_type] += 1
            indicator = row.get("indicator")
            if indicator is None:
                continue
            safe_indicator = _safe_indicator(indicator)
            if safe_indicator is not None:
                measurement_indicators[safe_indicator] += 1
            domains = _domain_hits(indicator) | _domain_hits(entry_type)
            for domain in domains:
                timeline_events[domain] += 1
                timeline_event_users.setdefault(domain, set()).add(user_id)
                history_domains.add(domain)

        for exam in exams if isinstance(exams, list) else []:
            if not isinstance(exam, dict):
                continue
            indicators = exam.get("indicators")
            if not isinstance(indicators, dict):
                continue
            for indicator in indicators:
                safe_indicator = _safe_indicator(indicator)
                if safe_indicator is not None:
                    exam_indicators[safe_indicator] += 1
                for domain in _domain_hits(indicator):
                    exam_users.setdefault(domain, set()).add(user_id)
                    history_domains.add(domain)

        for domain in history_domains:
            longitudinal_users[domain] += 1
        profile_domains = set().union(
            *(
                _domain_hits(label)
                for field_name in ("chronic_conditions", "past_medical_history", "physical_measurements")
                for label in _structured_labels(health.get(field_name))
            )
        )
        for domain in history_domains | profile_domains:
            state_users[domain] += 1

    observed_hash_index = hashlib.sha256("\n".join(sorted(file_hashes)).encode("utf-8")).hexdigest()
    if observed_hash_index != "faa1f93cd9216fa218d33e41c6e8b2bddbe26ec1303027d3cc7d453ce8669137":
        raise ValueError("202607 raw state files no longer match the pinned E5-A audit")

    domains = sorted(set(longitudinal_users) | set(state_users))
    domain_rows = [
        {
            "domain": domain,
            "users_with_timeline_or_exam_signal": len(timeline_event_users.get(domain, set()) | exam_users.get(domain, set())),
            "timeline_event_count": timeline_events[domain],
            "exam_user_count": len(exam_users.get(domain, set())),
            "users_with_any_state_signal": state_users[domain],
            "profile_condition_user_count": len(chronic_category_users.get(domain, set())),
        }
        for domain in domains
    ]
    return {
        "schema_version": "rag-e5-dev-longitudinal-domain-inventory-v1",
        "batch_id": DEVELOPMENT_BATCH,
        "batch_role": "development_only",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "published_batch_checksum": batch["checksum"],
        "state_files_index_sha256": observed_hash_index,
        "user_count": len(EXPECTED_USERS),
        "files_read_per_user": list(ALLOWED_FILES),
        "forbidden_inputs_opened": [],
        "patient_prose_emitted": False,
        "numeric_measurements_emitted": False,
        "age_range_user_counts": dict(sorted(age_ranges.items())),
        "chronic_condition_domain_user_counts": {
            domain: len(users) for domain, users in sorted(chronic_category_users.items())
        },
        "lifestyle_field_user_counts": dict(sorted(lifestyle_fields.items())),
        "timeline_measurement_indicator_event_counts": dict(sorted(measurement_indicators.items())),
        "timeline_entry_type_counts": dict(sorted(timeline_entry_types.items())),
        "exam_indicator_record_counts": dict(sorted(exam_indicators.items())),
        "domain_coverage": domain_rows,
        "guideline_domain_gate": {
            "minimum_users_per_domain": 5,
            "domains_with_longitudinal_signal_in_at_least_5_users": [
                row["domain"]
                for row in domain_rows
                if row["users_with_timeline_or_exam_signal"] >= 5
            ],
        },
        "batch_checksum_identity_note": (
            "The published batch checksum identifies the benchmark batch; "
            "state_files_index_sha256 independently pins only allowed raw state files."
        ),
    }


def _numeric_age(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d{1,3})(?:\s*years?)?\s*", value, re.IGNORECASE)
        return float(match.group(1)) if match else None
    return None


def _age_range(age: float | None) -> str:
    if age is None:
        return "unknown"
    if age < 18:
        return "under_18"
    if age < 40:
        return "18_39"
    if age < 60:
        return "40_59"
    if age < 75:
        return "60_74"
    return "75_plus"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = build_inventory(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
