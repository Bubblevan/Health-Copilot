"""Deterministic E0 audits and metadata-only closeout reporting."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import (
    BenchmarkManifest,
    ReviewStatus,
    TaskProfile,
    canonical_hash,
    canonical_json,
    sha256_file,
)
from .registry import BenchmarkRegistry

EVAL_ONLY_FIELDS = {
    "task_family",
    "required_evidence_groups",
    "independent_subtasks",
    "dependency_edges",
    "serial_depth",
    "expected_architecture",
    "ood_label",
}


@dataclass(frozen=True)
class AuditReport:
    subject: str
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    details: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "details": self.details or {},
        }


def data_root(repository_root: str | Path) -> Path:
    configured = __import__("os").environ.get("HEALTH_COPILOT_BENCH_DATA")
    return Path(configured).expanduser().resolve() if configured else Path(repository_root) / ".health-bench-data"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def _normalized_text_hash(payload: dict[str, Any]) -> str:
    text = " ".join(str(payload.get(key, "")) for key in ("question", "query", "prompt"))
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return canonical_hash(normalized)


def audit_manifest(manifest: BenchmarkManifest) -> AuditReport:
    errors = list(manifest.validate_freeze_requirements())
    if manifest.license.license_review_status.value == "APPROVED" and manifest.status.value != "APPROVED":
        errors.append("license says APPROVED but manifest admissibility is not APPROVED")
    if not manifest.split_manifest:
        errors.append("missing split manifest")
    if not manifest.gold_contract:
        errors.append("missing gold contract")
    if not manifest.metric_protocol:
        errors.append("missing metric protocol")
    return AuditReport(subject=manifest.benchmark_id, errors=tuple(sorted(set(errors))))


def audit_normalized_dataset(manifest: BenchmarkManifest, normalized_root: Path) -> AuditReport:
    errors = list(audit_manifest(manifest).errors)
    cases_path = normalized_root / "cases.jsonl"
    identity_path = normalized_root / "identity.json"
    if not cases_path.exists():
        errors.append("normalized data unavailable")
        return AuditReport(manifest.benchmark_id, tuple(sorted(set(errors))))
    if not identity_path.exists():
        errors.append("normalized identity unavailable")
    else:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        for field in ("normalized_sha256", "split_manifest_sha256"):
            if not identity.get(field):
                errors.append(f"missing {field}")
        if identity.get("normalized_sha256") != sha256_file(cases_path):
            errors.append("normalized SHA mismatch")
    try:
        rows = _read_jsonl(cases_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"normalized cases unreadable: {exc}")
        rows = []
    ids = [row.get("case_id") for row in rows]
    if any(not isinstance(case_id, str) or not case_id for case_id in ids):
        errors.append("missing case ID")
    if len(ids) != len(set(ids)):
        errors.append("duplicate case ID")
    for row in rows:
        if not isinstance(row.get("gold"), dict) or not row["gold"]:
            errors.append(f"missing gold: {row.get('case_id')}")
    return AuditReport(
        manifest.benchmark_id,
        tuple(sorted(set(errors))),
        details={"case_count": len(rows), "normalized_root": str(normalized_root)},
    )


def load_research_pack(root: str | Path) -> tuple[list[dict[str, Any]], list[TaskProfile], dict[str, Any]]:
    pack_root = Path(root)
    cases = _read_jsonl(pack_root / "cases.jsonl")
    profiles = [TaskProfile.from_dict(row) for row in _read_jsonl(pack_root / "task_profiles.jsonl")]
    annotation_manifest = json.loads(
        (pack_root / "annotation_manifest.json").read_text(encoding="utf-8")
    )
    return cases, profiles, annotation_manifest


def audit_research_pack(root: str | Path) -> AuditReport:
    pack_root = Path(root)
    errors: list[str] = []
    warnings: list[str] = []
    try:
        cases, profiles, annotation = load_research_pack(pack_root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return AuditReport("research-architecture-v1", (f"pack unreadable: {exc}",))
    case_ids = [row.get("case_id") for row in cases]
    profile_ids = [profile.case_id for profile in profiles]
    if len(case_ids) != len(set(case_ids)):
        errors.append("duplicate case ID")
    if len(profile_ids) != len(set(profile_ids)):
        errors.append("duplicate task-profile case ID")
    if set(case_ids) != set(profile_ids):
        errors.append("case/profile ID mismatch")
    by_split: dict[str, set[str]] = {}
    family_splits: dict[tuple[str, str], set[str]] = {}
    text_hashes: dict[str, str] = {}
    category_counts: dict[str, int] = {}
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case.get("gold"), dict) or not case["gold"]:
            errors.append(f"missing gold: {case_id}")
        payload_keys = set(case.get("payload", {}))
        leaked = sorted(payload_keys & EVAL_ONLY_FIELDS)
        if leaked:
            errors.append(f"eval-only field leaked into payload {case_id}: {', '.join(leaked)}")
        text_hash = _normalized_text_hash(case.get("payload", {}))
        if text_hash in text_hashes:
            errors.append(f"exact/normalized text duplicate: {case_id} and {text_hashes[text_hash]}")
        text_hashes[text_hash] = str(case_id)
    for profile in profiles:
        category_counts[profile.task_family] = category_counts.get(profile.task_family, 0) + 1
        by_split.setdefault(profile.split, set()).add(profile.case_id)
        family_splits.setdefault((profile.source_group_id, profile.question_family_id), set()).add(profile.split)
        if profile.annotation_status != ReviewStatus.REVIEWED and profile.split == "TEST":
            errors.append(f"unreviewed internal research TEST gold: {profile.case_id}")
        serialized = canonical_json(profile.to_dict()).lower()
        if any(field in serialized for field in ("best_architecture", "team_should_win", "single_should_win")):
            errors.append(f"expected-winner field present: {profile.case_id}")
    if by_split.get("DEV", set()).intersection(by_split.get("TEST", set())):
        errors.append("overlapping DEV/TEST IDs")
    for family, splits in family_splits.items():
        if len(splits) > 1:
            errors.append(f"provenance leakage across DEV/TEST: {family[0]}/{family[1]}")
    if annotation.get("status") != "frozen":
        warnings.append("research pack is not frozen")
    if annotation.get("human_review", {}).get("status") != "complete":
        errors.append("pending human review")
    if not 48 <= len(cases) <= 60:
        errors.append("research pack must contain 48-60 cases")
    return AuditReport(
        "research-architecture-v1",
        tuple(sorted(set(errors))),
        tuple(sorted(set(warnings))),
        details={
            "case_count": len(cases),
            "category_counts": category_counts,
            "split_counts": {key: len(value) for key, value in by_split.items()},
            "annotation_status": annotation.get("status"),
        },
    )


def build_closeout(
    registry: BenchmarkRegistry,
    repository_root: str | Path,
    *,
    code_sha: str,
) -> dict[str, Any]:
    root = Path(repository_root)
    bench_data = data_root(root)
    external: list[dict[str, Any]] = []
    blockers: list[str] = []
    for manifest in registry.list():
        if manifest.benchmark_id == "research-architecture-v1":
            continue
        manifest_report = audit_manifest(manifest)
        raw_root = bench_data / "raw" / manifest.benchmark_id
        normalized_root = bench_data / "normalized" / manifest.benchmark_id
        raw_available = raw_root.exists() and any(raw_root.rglob("*"))
        normalized_available = (normalized_root / "cases.jsonl").exists()
        if not raw_available:
            blockers.append(f"{manifest.benchmark_id}: raw data unavailable")
        if not normalized_available:
            blockers.append(f"{manifest.benchmark_id}: normalized data unavailable")
        if manifest_report.errors:
            blockers.extend(f"{manifest.benchmark_id}: {error}" for error in manifest_report.errors)
        external.append(
            {
                "benchmark_id": manifest.benchmark_id,
                "manifest_hash": manifest.manifest_hash,
                "admissibility_status": manifest.status.value,
                "raw_available": raw_available,
                "normalized_available": normalized_available,
                "raw_sha256": [item.sha256 for item in manifest.raw_artifacts],
                "normalized_sha256": None,
                "license_review_status": manifest.license.license_review_status.value,
                "judge_protocol_hash": canonical_hash(manifest.judge_protocol)
                if manifest.judge_protocol
                else None,
            }
        )
    research_report = audit_research_pack(root / "benchmarks" / "research_architecture_v1")
    blockers.extend(f"research-architecture-v1: {error}" for error in research_report.errors)
    return {
        "schema_version": "e0-closeout-v1",
        "stage": "E0",
        "code_sha": code_sha,
        "registry_hash": canonical_hash(
            {item.benchmark_id: item.manifest_hash for item in registry.list()}
        ),
        "external_benchmarks": external,
        "research_pack": {
            "benchmark_id": "research-architecture-v1",
            "status": research_report.details.get("annotation_status") if research_report.details else None,
            "case_count": research_report.details.get("case_count") if research_report.details else None,
            "category_counts": research_report.details.get("category_counts", {})
            if research_report.details
            else {},
            "sha256": None,
            "audit": research_report.to_dict(),
        },
        "fairness_contracts": {
            "native_budget": "benchmarks/research_architecture_v1/fairness_contract.json",
            "cost_matched": "benchmarks/research_architecture_v1/fairness_contract.json",
            "deadline_matched": "benchmarks/research_architecture_v1/fairness_contract.json",
            "sha256": sha256_file(root / "benchmarks" / "research_architecture_v1" / "fairness_contract.json"),
        },
        "ready_for_e1": False,
        "ready_for_e2": False,
        "blockers": sorted(set(blockers)),
        "claims": {
            "external_evaluation_started": False,
            "healthbench_score": None,
            "mirage_score": None,
            "nfcorpus_new_result": None,
            "e2_started": False,
            "e3_started": False,
            "m11_started": False,
            "m12_started": False,
        },
    }


def write_closeout(registry: BenchmarkRegistry, repository_root: str | Path, code_sha: str) -> Path:
    root = Path(repository_root)
    output = root / "runs" / "e0" / "benchmark_foundation_closeout.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(build_closeout(registry, root, code_sha=code_sha)) + "\n", encoding="utf-8")
    return output
