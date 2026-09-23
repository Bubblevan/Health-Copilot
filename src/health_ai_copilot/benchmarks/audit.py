"""Deterministic E0 audits and metadata-only closeout reporting."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import (
    BenchmarkManifest,
    DatasetAdmissibility,
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
        if profile.annotation_status not in {ReviewStatus.REVIEWED, ReviewStatus.FROZEN} and profile.split == "TEST":
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
    if annotation.get("decision_summary", {}).get("REJECT", 0):
        errors.append("research pack contains rejected cases")
    if annotation.get("status") == "frozen":
        try:
            stored_hashes = annotation.get("hashes", {})
            expected_hashes = research_pack_hashes(pack_root)
            if any(stored_hashes.get(key) != value for key, value in expected_hashes.items()):
                errors.append("frozen research-pack hash mismatch")
        except (OSError, KeyError, ValueError) as exc:
            errors.append(f"frozen research-pack hash error: {exc}")
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


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _source_review_complete(root: Path, benchmark_id: str) -> bool:
    record = _read_json_if_exists(root / "runs" / "e0" / f"license_review_{benchmark_id}.json")
    decision = record.get("human_decision") if record else None
    return bool(
        isinstance(decision, dict)
        and decision.get("reviewer")
        and decision.get("review_date")
        and decision.get("final_admissibility") in {"APPROVED", "RESTRICTED", "UNAVAILABLE"}
    )


def _raw_identity(root: Path, benchmark_id: str) -> dict[str, Any] | None:
    return _read_json_if_exists(root / "raw" / benchmark_id / "raw_identity.json")


def _normalized_identity(root: Path, benchmark_id: str) -> dict[str, Any] | None:
    return _read_json_if_exists(root / "normalized" / benchmark_id / "identity.json")


def benchmark_stage_gate(
    manifest: BenchmarkManifest,
    repository_root: str | Path,
    benchmark_data_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root)
    bench_data = Path(benchmark_data_root) if benchmark_data_root else data_root(root)
    raw = _raw_identity(bench_data, manifest.benchmark_id)
    normalized = _normalized_identity(bench_data, manifest.benchmark_id)
    raw_by_name = {item.get("name"): item for item in raw.get("artifacts", [])} if raw else {}
    raw_identity_frozen = bool(raw and all(
        item.sha256 is not None and raw_by_name.get(item.name, {}).get("sha256") == item.sha256
        for item in manifest.raw_artifacts
    ))
    normalized_audit = audit_normalized_dataset(
        manifest,
        bench_data / "normalized" / manifest.benchmark_id,
    ) if normalized else AuditReport(manifest.benchmark_id, ("normalized data unavailable",))
    normalized_identity_frozen = bool(
        normalized
        and normalized.get("normalized_sha256")
        and normalized.get("split_manifest_sha256")
        and raw_identity_frozen
        and set(normalized.get("raw_artifact_sha256", [])) == {
            item.sha256 for item in manifest.raw_artifacts if item.sha256
        }
        and normalized_audit.ok
    )
    source_review_complete = _source_review_complete(root, manifest.benchmark_id)
    license_review_complete = (
        manifest.status in {
            DatasetAdmissibility.APPROVED,
            DatasetAdmissibility.RESTRICTED,
            DatasetAdmissibility.UNAVAILABLE,
        }
        and manifest.license.license_review_status in {
            DatasetAdmissibility.APPROVED,
            DatasetAdmissibility.RESTRICTED,
            DatasetAdmissibility.UNAVAILABLE,
        }
    )
    protocol_text = canonical_json(manifest.metric_protocol).lower()
    metric_protocol_ready = bool(manifest.metric_protocol) and "to_be_frozen" not in protocol_text
    judge_text = canonical_json(manifest.judge_protocol).lower() if manifest.judge_protocol else ""
    judge_protocol_ready = not manifest.judge_protocol or "to_be_frozen" not in judge_text
    e1_ready = bool(
        manifest.status == DatasetAdmissibility.APPROVED
        and source_review_complete
        and license_review_complete
        and raw_identity_frozen
        and normalized_identity_frozen
        and metric_protocol_ready
        and judge_protocol_ready
        and normalized_audit.ok
    )
    blockers: list[str] = []
    checks = {
        "source_review_complete": source_review_complete,
        "license_review_complete": license_review_complete,
        "raw_identity_frozen": raw_identity_frozen,
        "normalized_identity_frozen": normalized_identity_frozen,
        "metric_protocol_ready": metric_protocol_ready,
        "judge_protocol_ready": judge_protocol_ready,
    }
    for name, passed in checks.items():
        if not passed:
            blockers.append(name)
    if manifest.status != DatasetAdmissibility.APPROVED:
        blockers.append(f"admissibility={manifest.status.value}")
    return {
        "benchmark_id": manifest.benchmark_id,
        "review_status": "COMPLETE" if source_review_complete else "REVIEW_REQUIRED",
        "admissibility": manifest.status.value,
        **checks,
        "e1_ready": e1_ready,
        "raw_sha256": [item.sha256 for item in manifest.raw_artifacts],
        "normalized_sha256": normalized.get("normalized_sha256") if normalized else None,
        "normalized_identity": normalized,
        "manifest_hash": manifest.manifest_hash,
        "judge_protocol_hash": canonical_hash(manifest.judge_protocol)
        if manifest.judge_protocol
        else None,
        "blockers": sorted(set(blockers)),
    }


def research_pack_hashes(root: str | Path) -> dict[str, str]:
    pack_root = Path(root)
    cases = _read_jsonl(pack_root / "cases.jsonl")
    annotation = json.loads((pack_root / "annotation_manifest.json").read_text(encoding="utf-8"))
    gold = [{"case_id": row["case_id"], "gold": row["gold"]} for row in cases]
    split = [
        {"case_id": row["case_id"], "split": row.get("metadata", {}).get("split")}
        for row in cases
    ]
    annotation_without_hashes = {key: value for key, value in annotation.items() if key != "hashes"}
    values = {
        "cases_sha256": sha256_file(pack_root / "cases.jsonl"),
        "gold_sha256": canonical_hash(gold),
        "task_profiles_sha256": sha256_file(pack_root / "task_profiles.jsonl"),
        "split_sha256": canonical_hash(split),
        "annotation_manifest_sha256": canonical_hash(annotation_without_hashes),
    }
    values["aggregate_benchmark_sha256"] = canonical_hash(values)
    return values


def _research_gate(root: Path) -> dict[str, Any]:
    pack_root = root / "benchmarks" / "research_architecture_v1"
    report = audit_research_pack(pack_root)
    annotation = _read_json_if_exists(pack_root / "annotation_manifest.json") or {}
    hashes = research_pack_hashes(pack_root)
    stored_hashes = annotation.get("hashes", {})
    hashes_frozen = annotation.get("status") == "frozen" and all(
        stored_hashes.get(key) == value for key, value in hashes.items()
    )
    ready = bool(report.ok and hashes_frozen)
    blockers = list(report.errors)
    if not hashes_frozen:
        blockers.append("research pack hashes not frozen")
    if not (root / "benchmarks" / "research_architecture_v1" / "fairness_contract.json").exists():
        blockers.append("fairness contract unavailable")
    return {
        "benchmark_id": "research-architecture-v1",
        "human_review_status": annotation.get("human_review", {}).get("status"),
        "annotation_status": annotation.get("status"),
        "case_count": report.details.get("case_count") if report.details else None,
        "category_counts": report.details.get("category_counts", {}) if report.details else {},
        "hashes": hashes,
        "hashes_frozen": hashes_frozen,
        "e2_ready": ready,
        "audit": report.to_dict(),
        "blockers": sorted(set(blockers)),
    }


def build_closeout(
    registry: BenchmarkRegistry,
    repository_root: str | Path,
    *,
    functional_code_sha: str,
    final_documentation_sha: str | None = None,
) -> dict[str, Any]:
    root = Path(repository_root)
    bench_data = data_root(root)
    external = [
        benchmark_stage_gate(manifest, root, bench_data)
        for manifest in registry.list()
        if manifest.benchmark_id != "research-architecture-v1"
    ]
    research = _research_gate(root)
    blockers = [
        f"{item['benchmark_id']}: {blocker}"
        for item in external
        for blocker in item["blockers"]
    ]
    blockers.extend(f"research-architecture-v1: {item}" for item in research["blockers"])
    all_external_reviews_complete = all(
        item["source_review_complete"] and item["license_review_complete"] for item in external
    )
    fairness_path = root / "benchmarks" / "research_architecture_v1" / "fairness_contract.json"
    fairness_sha = sha256_file(fairness_path) if fairness_path.exists() else None
    e0_complete = bool(all_external_reviews_complete and research["e2_ready"] and fairness_sha)
    ready_for_e1 = any(item["e1_ready"] for item in external)
    ready_for_e2 = research["e2_ready"]
    return {
        "schema_version": "e0-closeout-v2",
        "stage": "E0",
        "code_identity": {
            "functional_code_sha": functional_code_sha,
            "final_documentation_sha": final_documentation_sha,
        },
        "registry_hash": canonical_hash(
            {item.benchmark_id: item.manifest_hash for item in registry.list()}
        ),
        "external_benchmarks": external,
        "research_pack": research,
        "fairness_contract_sha256": fairness_sha,
        "e0_complete": e0_complete,
        "ready_for_e1": ready_for_e1,
        "ready_for_e2": ready_for_e2,
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


def write_closeout(
    registry: BenchmarkRegistry,
    repository_root: str | Path,
    functional_code_sha: str,
    final_documentation_sha: str | None = None,
) -> Path:
    root = Path(repository_root)
    output = root / "runs" / "e0" / "benchmark_foundation_closeout_v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_closeout(
        registry,
        root,
        functional_code_sha=functional_code_sha,
        final_documentation_sha=final_documentation_sha,
    )
    text = canonical_json(payload) + "\n"
    output.write_text(text, encoding="utf-8")
    (output.with_name("benchmark_foundation_closeout.json")).write_text(text, encoding="utf-8")
    return output
