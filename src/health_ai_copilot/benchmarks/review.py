"""Structured human-review decisions for the E0 research pack."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audit import audit_research_pack, load_research_pack, research_pack_hashes
from .contracts import ReviewStatus, TaskProfile, canonical_json

REVIEW_CHECKS = (
    "question_ok",
    "evidence_groups_ok",
    "source_families_ok",
    "task_family_ok",
    "dependency_structure_ok",
    "conflict_structure_ok",
    "temporal_structure_ok",
    "architecture_neutral",
    "safety_boundary_ok",
    "private_data_absent",
)
ALLOWED_EDIT_KEYS = {"question", "gold", "task_profile"}


class ReviewDecisionError(ValueError):
    """Raised when review coverage or an explicit edit is invalid."""


def _text_hash(question: str) -> str:
    normalized = re.sub(r"\s+", " ", question.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResearchCaseReviewDecision:
    case_id: str
    decision: str
    checks: dict[str, bool]
    notes: str
    proposed_edits: dict[str, Any]
    reviewer: str
    review_date: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ResearchCaseReviewDecision:
        required = {"case_id", "decision", "notes", "proposed_edits", "reviewer", "review_date"}
        missing = sorted(required - set(value))
        if missing:
            raise ReviewDecisionError(f"review decision missing fields: {', '.join(missing)}")
        decision = value["decision"]
        if decision not in {"APPROVE", "EDIT", "REJECT"}:
            raise ReviewDecisionError(f"unknown review decision: {decision}")
        checks = {name: value.get(name) for name in REVIEW_CHECKS}
        if any(not isinstance(item, bool) for item in checks.values()):
            raise ReviewDecisionError(f"{value['case_id']}: every review checklist field must be boolean")
        if not isinstance(value["proposed_edits"], dict):
            raise ReviewDecisionError(f"{value['case_id']}: proposed_edits must be an object")
        unknown_edits = sorted(set(value["proposed_edits"]) - ALLOWED_EDIT_KEYS)
        if unknown_edits:
            raise ReviewDecisionError(
                f"{value['case_id']}: unknown proposed edit(s): {', '.join(unknown_edits)}"
            )
        if decision == "EDIT" and not value["proposed_edits"]:
            raise ReviewDecisionError(f"{value['case_id']}: EDIT requires proposed_edits")
        if decision == "REJECT" and not str(value["notes"]).strip():
            raise ReviewDecisionError(f"{value['case_id']}: REJECT requires notes")
        if not str(value["reviewer"]).strip() or not str(value["review_date"]).strip():
            raise ReviewDecisionError(f"{value['case_id']}: reviewer and review_date are required")
        return cls(
            case_id=str(value["case_id"]),
            decision=decision,
            checks=checks,
            notes=str(value["notes"]),
            proposed_edits=dict(value["proposed_edits"]),
            reviewer=str(value["reviewer"]),
            review_date=str(value["review_date"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "decision": self.decision,
            **self.checks,
            "notes": self.notes,
            "proposed_edits": self.proposed_edits,
            "reviewer": self.reviewer,
            "review_date": self.review_date,
        }


def load_review_decisions(path: str | Path) -> list[ResearchCaseReviewDecision]:
    decisions: list[ResearchCaseReviewDecision] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReviewDecisionError(f"invalid review JSON at line {line_number}") from exc
        if not isinstance(value, dict):
            raise ReviewDecisionError(f"review line {line_number} must be an object")
        decisions.append(ResearchCaseReviewDecision.from_dict(value))
    return decisions


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def apply_review(
    source_root: str | Path,
    decisions_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Apply exactly one complete decision per case into an explicit staging root."""

    cases, profiles, annotation = load_research_pack(source_root)
    decisions = load_review_decisions(decisions_path)
    case_ids = {row["case_id"] for row in cases}
    decision_ids = [decision.case_id for decision in decisions]
    if len(decision_ids) != len(set(decision_ids)):
        raise ReviewDecisionError("duplicate review decision case ID")
    if set(decision_ids) != case_ids:
        missing = sorted(case_ids - set(decision_ids))
        extra = sorted(set(decision_ids) - case_ids)
        raise ReviewDecisionError(f"review coverage mismatch; missing={missing}; extra={extra}")
    decision_by_id = {decision.case_id: decision for decision in decisions}
    case_map = {row["case_id"]: json.loads(json.dumps(row, ensure_ascii=False)) for row in cases}
    profile_map = {profile.case_id: profile.to_dict() for profile in profiles}
    for case_id in sorted(case_ids):
        decision = decision_by_id[case_id]
        case = case_map[case_id]
        profile = profile_map[case_id]
        edits = decision.proposed_edits
        if "question" in edits:
            if not isinstance(edits["question"], str) or not edits["question"].strip():
                raise ReviewDecisionError(f"{case_id}: edited question must be non-empty text")
            case["payload"]["question"] = edits["question"]
            profile["text_hash"] = _text_hash(edits["question"])
        if "gold" in edits:
            if not isinstance(edits["gold"], dict) or not edits["gold"]:
                raise ReviewDecisionError(f"{case_id}: edited gold must be a non-empty object")
            case["gold"] = edits["gold"]
        if "task_profile" in edits:
            if not isinstance(edits["task_profile"], dict):
                raise ReviewDecisionError(f"{case_id}: edited task_profile must be an object")
            proposed = dict(profile)
            proposed.update(edits["task_profile"])
            if proposed.get("case_id") != case_id or proposed.get("split") != profile.get("split"):
                raise ReviewDecisionError(f"{case_id}: review cannot change case_id or split")
            profile = proposed
        profile["annotation_status"] = ReviewStatus.REVIEWED.value
        profile["reviewer"] = decision.reviewer
        profile_map[case_id] = TaskProfile.from_dict(profile).to_dict()
        case["metadata"]["review_decision"] = decision.decision
        case["metadata"]["reviewer"] = decision.reviewer
        case["metadata"]["review_date"] = decision.review_date
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "cases.jsonl", [case_map[key] for key in sorted(case_map)])
    _write_jsonl(output / "task_profiles.jsonl", [profile_map[key] for key in sorted(profile_map)])
    _write_jsonl(output / "review_decisions.jsonl", [decision.to_dict() for decision in decisions])
    updated_annotation = json.loads(json.dumps(annotation, ensure_ascii=False))
    counts = {name: sum(decision.decision == name for decision in decisions) for name in ("APPROVE", "EDIT", "REJECT")}
    updated_annotation["status"] = "reviewed"
    updated_annotation["human_review"] = {
        "status": "complete",
        "reviewer": sorted({decision.reviewer for decision in decisions}),
        "review_date": sorted({decision.review_date for decision in decisions}),
        "review_notes": "applied from explicit per-case review decisions",
    }
    updated_annotation["decision_summary"] = counts
    updated_annotation.pop("hashes", None)
    (output / "annotation_manifest.json").write_text(
        canonical_json(updated_annotation) + "\n", encoding="utf-8"
    )
    return {
        "output_root": str(output),
        "case_count": len(case_map),
        "decision_summary": counts,
        "split_changed": False,
    }


def freeze_research_pack(pack_root: str | Path) -> dict[str, Any]:
    """Freeze only a fully reviewed staging pack and persist all identity hashes."""

    root = Path(pack_root)
    annotation_path = root / "annotation_manifest.json"
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    if annotation.get("status") != "reviewed":
        raise ReviewDecisionError("research pack freeze requires status=reviewed")
    if annotation.get("human_review", {}).get("status") != "complete":
        raise ReviewDecisionError("research pack freeze requires complete human review")
    if annotation.get("decision_summary", {}).get("REJECT", 0):
        raise ReviewDecisionError("research pack contains rejected cases")
    report = audit_research_pack(root)
    if report.errors:
        raise ReviewDecisionError("research pack audit failed: " + "; ".join(report.errors))
    _, profiles, _ = load_research_pack(root)
    profile_map = {profile.case_id: profile.to_dict() for profile in profiles}
    for profile in profiles:
        profile_dict = profile.to_dict()
        profile_dict["annotation_status"] = ReviewStatus.FROZEN.value
        profile_dict["reviewer"] = profile.reviewer
        profile_map[profile.case_id] = profile_dict
    _write_jsonl(root / "task_profiles.jsonl", [profile_map[key] for key in sorted(profile_map)])
    annotation["status"] = "frozen"
    # Persist the status before hashing so annotation_manifest_sha256 covers
    # the exact frozen annotation rather than the pre-freeze reviewed state.
    annotation_path.write_text(canonical_json(annotation) + "\n", encoding="utf-8")
    hashes = research_pack_hashes(root)
    annotation["hashes"] = hashes
    annotation_path.write_text(canonical_json(annotation) + "\n", encoding="utf-8")
    return {"frozen": True, "hashes": hashes, "case_count": report.details.get("case_count", 0)}
