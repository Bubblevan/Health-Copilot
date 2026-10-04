"""Harness-owned ledgers and artifact validation for MA-MVP2."""

from __future__ import annotations

import json
import re
from uuid import uuid4

from .contracts import (
    ArtifactFact,
    CoverageItem,
    CoverageLedger,
    CoverageStatus,
    PlannedAspect,
    TaskAspect,
    TaskAssignment,
    TaskLedger,
    WorkerArtifact,
    WorkerReport,
    WorkerRole,
    WorkerStatus,
)
from .shared_context import EvidenceLedgerEntry
from .skills import SkillResult

_TOKEN = re.compile(r"SYN(?:KEY|VAL)-[A-Z0-9]+", re.IGNORECASE)
_MAX_ASPECTS = 12
_MAX_TEXT = 1200


def _strip_fence(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        text = text[3:].strip()
        if text.casefold().startswith("json"):
            text = text[4:].strip()
        if text.endswith(fence):
            text = text[:-3].strip()
    return text


def parse_aspect_plan(content: str, *, allowed_workers: tuple[WorkerRole, ...]) -> tuple[PlannedAspect, ...]:
    """Parse Lead text; identities and acceptance policy are always added by Harness."""
    try:
        payload = json.loads(_strip_fence(content))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("LEAD_ASPECT_PLAN_INVALID_JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"aspects"}:
        raise ValueError("LEAD_ASPECT_PLAN_INVALID_SHAPE")
    rows = payload["aspects"]
    if not isinstance(rows, list) or not rows or len(rows) > _MAX_ASPECTS:
        raise ValueError("LEAD_ASPECT_PLAN_ASPECT_LIMIT")
    allowed = set(allowed_workers)
    result: list[PlannedAspect] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"aspect", "worker", "expected_output"}:
            raise ValueError("LEAD_ASPECT_PLAN_INVALID_ASPECT")
        if not all(isinstance(row[name], str) and row[name].strip()
                   for name in ("aspect", "worker", "expected_output")):
            raise ValueError("LEAD_ASPECT_PLAN_EMPTY_FIELD")
        try:
            worker = WorkerRole(row["worker"].strip())
        except ValueError as exc:
            raise ValueError("LEAD_ASPECT_PLAN_UNKNOWN_WORKER") from exc
        if worker not in allowed:
            raise ValueError("LEAD_ASPECT_PLAN_WORKER_OUTSIDE_TRIAGE_SCOPE")
        result.append(PlannedAspect(
            aspect=row["aspect"].strip()[:_MAX_TEXT],
            worker=worker,
            expected_output=row["expected_output"].strip()[:_MAX_TEXT],
        ))
    return tuple(result)


def fallback_aspects(query: str, workers: tuple[WorkerRole, ...]) -> tuple[PlannedAspect, ...]:
    """Conservative query-only decomposition used when Lead planning is unavailable."""
    pieces = [item.strip(" \t-*•") for item in re.split(
        r"\n+|[;；]+|(?<=[?？])\s+", query
    ) if item.strip(" \t-*•")]
    if not pieces:
        pieces = [query.strip()]
    pieces = pieces[:_MAX_ASPECTS]
    return tuple(PlannedAspect(
        aspect=text[:_MAX_TEXT],
        worker=workers[min(index, len(workers) - 1)],
        expected_output="直接回答该用户子问题；保留所有经 Harness 核验的事实和来源。",
    ) for index, text in enumerate(pieces))


def harness_task_ledger(
    request_id: str,
    planned: tuple[PlannedAspect, ...],
    worker_ids: dict[WorkerRole, str],
) -> tuple[TaskLedger, dict[WorkerRole, tuple[str, ...]]]:
    aspects: list[TaskAspect] = []
    assignments: list[TaskAssignment] = []
    aspect_ids_by_worker: dict[WorkerRole, list[str]] = {}
    criteria = (
        "Answer this aspect explicitly; do not omit a requested subpart.",
        "Use only Harness-observed patient records and external evidence.",
        "Preserve every verified synthetic fact token exactly.",
        "If required information is absent, state what remains unresolved.",
    )
    for row in planned:
        worker_id = worker_ids.get(row.worker)
        if not worker_id:
            raise ValueError("TASK_LEDGER_WORKER_ID_NOT_HARNESS_ASSIGNED")
        aspect_id = f"aspect-{uuid4().hex[:12]}"
        assignment_id = f"assignment-{uuid4().hex[:12]}"
        aspects.append(TaskAspect(
            aspect_id, row.aspect, row.worker, row.expected_output, criteria,
        ))
        assignments.append(TaskAssignment(
            assignment_id, aspect_id, worker_id, row.worker, row.expected_output,
        ))
        aspect_ids_by_worker.setdefault(row.worker, []).append(aspect_id)
    return (
        TaskLedger(request_id, tuple(aspects), tuple(assignments)),
        {role: tuple(ids) for role, ids in aspect_ids_by_worker.items()},
    )


def artifact_from_worker_output(
    *,
    worker_id: str,
    role: WorkerRole,
    status: WorkerStatus,
    raw_output: str,
    skill_results: list[SkillResult],
    evidence_entries: tuple[EvidenceLedgerEntry, ...],
    aspect_ids: tuple[str, ...],
) -> WorkerArtifact:
    """Create a provenance-safe artifact; the model never supplies any IDs."""
    output = _strip_fence(raw_output)
    parsed: dict[str, object] | None = None
    invalid_structured_output = False
    try:
        value = json.loads(output)
        if isinstance(value, dict) and set(value) == {
            "findings", "facts", "unresolved", "answer_fragment",
        }:
            parsed = value
        elif isinstance(value, (dict, list)):
            invalid_structured_output = True
    except (TypeError, json.JSONDecodeError):
        pass

    findings = _string_rows(parsed.get("findings")) if parsed else []
    model_facts = _string_rows(parsed.get("facts")) if parsed else []
    unresolved = _string_rows(parsed.get("unresolved")) if parsed else []
    answer = parsed.get("answer_fragment") if parsed else ("" if invalid_structured_output else output)
    answer_fragment = answer.strip()[:_MAX_TEXT * 2] if isinstance(answer, str) else ""
    if not findings and answer_fragment:
        findings = [answer_fragment]

    sources = [
        (source.source_id, source.tool_id, source.excerpt)
        for result in skill_results for source in result.sources
    ]
    facts: list[ArtifactFact] = []
    seen: set[tuple[str, str]] = set()

    def add_fact(text: str, source_id: str, tool_id: str) -> None:
        normalized = text.strip()
        key = (source_id, normalized.casefold())
        if not normalized or key in seen:
            return
        seen.add(key)
        facts.append(ArtifactFact(
            fact_id=f"fact-{uuid4().hex[:12]}",
            text=normalized[:_MAX_TEXT],
            source_id=source_id,
            tool_id=tool_id,
        ))

    for claimed in model_facts:
        for source_id, tool_id, excerpt in sources:
            if claimed.casefold() in excerpt.casefold():
                add_fact(claimed, source_id, tool_id)
                break
    for source_id, tool_id, excerpt in sources:
        for line in re.split(r"\r?\n|(?<=[。.!?])\s+", excerpt):
            if _TOKEN.search(line):
                add_fact(line, source_id, tool_id)

    evidence_refs = tuple(sorted({
        entry.source_id for entry in evidence_entries
        if entry.worker_id == worker_id and entry.tool_id == "external_retrieval"
    }))
    patient_refs = tuple(sorted({
        entry.source_id for entry in evidence_entries
        if entry.worker_id == worker_id and entry.tool_id == "memory_read"
    }))
    if not parsed:
        unresolved.append(
            "worker_output_invalid_structured_json" if invalid_structured_output
            else "worker_output_not_structured_json"
        )
    unresolved.extend(result.error for result in skill_results if result.error)
    return WorkerArtifact(
        worker_id=worker_id,
        role=role,
        findings=tuple(dict.fromkeys(item[:_MAX_TEXT] for item in findings)),
        facts=tuple(facts),
        evidence_refs=evidence_refs,
        patient_record_refs=patient_refs,
        unresolved=tuple(dict.fromkeys(item[:_MAX_TEXT] for item in unresolved)),
        answer_fragment=answer_fragment,
        status=status,
        aspect_ids=aspect_ids,
    )


def make_coverage_ledger(
    task_ledger: TaskLedger,
    reports: tuple[WorkerReport, ...] | list[WorkerReport],
    *,
    statuses: tuple[CoverageStatus, ...] | None = None,
    judge: str = "harness+bounded-model",
) -> CoverageLedger:
    by_worker = {report.worker_id: report for report in reports}
    items: list[CoverageItem] = []
    for index, aspect in enumerate(task_ledger.aspects):
        assigned = [item for item in task_ledger.assignments
                    if item.aspect_id == aspect.aspect_id]
        supporting = [by_worker[item.worker_id] for item in assigned
                      if item.worker_id in by_worker]
        if statuses is not None and index < len(statuses):
            status = statuses[index]
        else:
            usable = [report for report in supporting
                      if report.status in {WorkerStatus.COMPLETE, WorkerStatus.PARTIAL}
                      and report.answer_text.strip()]
            status = CoverageStatus.PARTIAL if usable else CoverageStatus.MISSING
        worker_ids = tuple(dict.fromkeys(report.worker_id for report in supporting))
        refs = tuple(sorted({ref for report in supporting if report.artifact
                             for ref in (
                                 *report.artifact.evidence_refs,
                                 *report.artifact.patient_record_refs,
                                 *(fact.source_id for fact in report.artifact.facts),
                             )}))
        items.append(CoverageItem(
            aspect_id=aspect.aspect_id,
            status=status,
            supporting_worker_ids=worker_ids,
            evidence_refs=refs,
            reason=("worker artifact contains support" if status != CoverageStatus.MISSING
                    else "no usable worker result for assigned aspect"),
        ))
    return CoverageLedger(task_ledger.request_id, tuple(items), judge)


def coverage_judgment_prompt(task_ledger: TaskLedger, reports: tuple[WorkerReport, ...]) -> str:
    """Only task and observed worker artifacts enter the bounded model judgment."""
    aspects = []
    for index, aspect in enumerate(task_ledger.aspects):
        assigned_ids = {item.worker_id for item in task_ledger.assignments
                        if item.aspect_id == aspect.aspect_id}
        rows = []
        for report in reports:
            if report.worker_id not in assigned_ids:
                continue
            rows.append({
                "status": report.status.value,
                "answer_fragment": report.artifact.answer_fragment if report.artifact else "",
                "findings": list(report.artifact.findings) if report.artifact else [],
                "verified_fact_text": [fact.text for fact in report.artifact.facts]
                if report.artifact else [],
                "unresolved": list(report.artifact.unresolved) if report.artifact else [],
            })
        aspects.append({
            "index": index,
            "aspect": aspect.aspect,
            "expected_output": aspect.expected_output,
            "worker_artifacts": rows,
        })
    return json.dumps({"aspects": aspects}, ensure_ascii=False, separators=(",", ":"))


def parse_coverage_judgment(content: str, aspect_count: int) -> tuple[CoverageStatus, ...]:
    try:
        payload = json.loads(_strip_fence(content))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("COVERAGE_JUDGMENT_INVALID_JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"statuses"}:
        raise ValueError("COVERAGE_JUDGMENT_INVALID_SHAPE")
    values = payload["statuses"]
    if not isinstance(values, list) or len(values) != aspect_count:
        raise ValueError("COVERAGE_JUDGMENT_COUNT_MISMATCH")
    try:
        return tuple(CoverageStatus(str(value)) for value in values)
    except ValueError as exc:
        raise ValueError("COVERAGE_JUDGMENT_INVALID_STATUS") from exc


def task_ledger_with_coverage(ledger: TaskLedger, coverage: CoverageLedger) -> TaskLedger:
    status_by_id = {item.aspect_id: item.status.value.lower() for item in coverage.items}
    return TaskLedger(
        request_id=ledger.request_id,
        aspects=tuple(TaskAspect(
            item.aspect_id, item.aspect, item.worker, item.expected_output,
            item.acceptance_criteria, status_by_id.get(item.aspect_id, item.status),
        ) for item in ledger.aspects),
        assignments=ledger.assignments,
    )


def _string_rows(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip()[:_MAX_TEXT] for item in value
            if isinstance(item, str) and item.strip()]


__all__ = [
    "artifact_from_worker_output",
    "coverage_judgment_prompt",
    "fallback_aspects",
    "harness_task_ledger",
    "make_coverage_ledger",
    "parse_aspect_plan",
    "parse_coverage_judgment",
    "task_ledger_with_coverage",
]
