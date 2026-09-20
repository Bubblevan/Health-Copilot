"""Inspectable deterministic metric aggregation for M7 evaluation artifacts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from .schema import CaseRunRecord, CaseRunStatus, GraderResult, GraderStatus, MetricResult

DEFAULT_METRIC_DEFINITION_VERSION = "m7-metrics-v1"


def metric_from_binary(
    metric_id: str,
    values: Iterable[bool],
    *,
    definition_version: str = DEFAULT_METRIC_DEFINITION_VERSION,
    scope: str = "all",
) -> MetricResult:
    values = tuple(values)
    return MetricResult.ratio(
        metric_id,
        definition_version,
        sum(values),
        len(values),
        scope=scope,
    )


def metric_results_from_grader_results(
    results: Sequence[GraderResult],
    *,
    definition_version: str = DEFAULT_METRIC_DEFINITION_VERSION,
) -> dict[str, MetricResult]:
    """Aggregate only PASS/FAIL results; UNGRADED and ERROR stay visible in counts."""

    grouped: dict[str, list[GraderResult]] = defaultdict(list)
    for result in results:
        grouped[result.grader_id].append(result)
    metrics: dict[str, MetricResult] = {}
    for grader_id, rows in sorted(grouped.items()):
        scored = [row for row in rows if row.status in {GraderStatus.PASS, GraderStatus.FAIL}]
        metrics[f"grader.{grader_id}.pass_rate"] = MetricResult.ratio(
            f"grader.{grader_id}.pass_rate",
            definition_version,
            sum(row.status == GraderStatus.PASS for row in scored),
            len(scored),
            scope=grader_id,
        )
        metrics[f"grader.{grader_id}.scored_count"] = MetricResult(
            f"grader.{grader_id}.scored_count",
            definition_version,
            len(scored),
            len(scored),
            len(rows),
            "count",
            grader_id,
            "scored/all",
        )
    return metrics


def trial_metrics(
    records: Sequence[CaseRunRecord],
    grader_results: Sequence[GraderResult],
    *,
    definition_version: str = DEFAULT_METRIC_DEFINITION_VERSION,
) -> dict[str, MetricResult]:
    """Return trial-aware quality, completeness, and consistency metrics."""

    metrics = metric_results_from_grader_results(
        grader_results, definition_version=definition_version
    )
    complete = [record for record in records if record.status == CaseRunStatus.COMPLETE]
    metrics["trajectory.completeness"] = MetricResult.ratio(
        "trajectory.completeness",
        definition_version,
        len(complete),
        len(records),
    )
    by_case: dict[str, list[CaseRunRecord]] = defaultdict(list)
    for record in records:
        by_case[record.case_id].append(record)
    all_trials_passed: list[bool] = []
    for case_id, case_records in sorted(by_case.items()):
        case_grades = [row for row in grader_results if row.case_id == case_id]
        all_trials_passed.append(
            bool(case_records)
            and all(row.status == CaseRunStatus.COMPLETE for row in case_records)
            and bool(case_grades)
            and all(
                row.status == GraderStatus.PASS
                for row in case_grades
                if row.status in {GraderStatus.PASS, GraderStatus.FAIL}
            )
            and not any(row.status == GraderStatus.ERROR for row in case_grades)
        )
    metrics["case.all_trials_pass_rate"] = metric_from_binary(
        "case.all_trials_pass_rate",
        all_trials_passed,
        definition_version=definition_version,
        scope="case",
    )
    route_groups: dict[str, list[str | None]] = defaultdict(list)
    for record in records:
        route_groups[record.case_id].append(record.route)
    metrics["case.route_consistency"] = metric_from_binary(
        "case.route_consistency",
        [len(set(routes)) <= 1 for routes in route_groups.values()],
        definition_version=definition_version,
        scope="case",
    )
    harness_groups: dict[str, list[str | None]] = defaultdict(list)
    for record in records:
        harness_groups[record.case_id].append(record.harness_disposition)
    metrics["case.harness_disposition_consistency"] = metric_from_binary(
        "case.harness_disposition_consistency",
        [len(set(dispositions)) <= 1 for dispositions in harness_groups.values()],
        definition_version=definition_version,
        scope="case",
    )
    metrics["run.case_count"] = MetricResult(
        "run.case_count",
        definition_version,
        len(by_case),
        len(by_case),
        len(by_case),
        "count",
        "run",
        "count",
    )
    metrics["run.trial_count"] = MetricResult(
        "run.trial_count",
        definition_version,
        len(records),
        len(records),
        len(records),
        "count",
        "run",
        "count",
    )
    return metrics


def metrics_to_dict(metrics: Mapping[str, MetricResult]) -> dict[str, object]:
    return {key: value.to_dict() for key, value in sorted(metrics.items())}
