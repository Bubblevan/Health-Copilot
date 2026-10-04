"""Tune MA-MVP2 on TRAIN_TUNE, validate on DEV, freeze, then run reserved TEST."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ma_mvp2_data import materialize_reserved_test
from ma_mvp2_protocol import (
    CONFIG_VERSION,
    MODEL_SHA256,
    MVP1_BASE_COMMIT,
    MVP1_COMPLEX_TEAM_SUCCESS,
    MVP1_EXACT_WORKER_SET,
    MVP1_OVERALL_TEAM_FACT_COVERAGE,
    RESERVED_TEST_RELATIVE,
    TRAIN_TUNE_SELECTION,
    canonical_hash,
    code_identity,
    file_sha256,
    prompt_hashes,
    validate_frozen_config,
)

from health_ai_copilot.multi_agent.contracts import (
    MedicalAgentRequest,
    RouteMode,
    TriageDecision,
    WorkerRole,
)
from health_ai_copilot.multi_agent.data import (
    DATASET_ID,
    DATASET_ROOT_HASH,
    EvaluationRecord,
    load_records,
)
from health_ai_copilot.multi_agent.evaluation import (
    aggregate_metrics,
    compare_systems,
    score_execution,
)
from health_ai_copilot.multi_agent.providers import LocalLlamaCppProvider
from health_ai_copilot.multi_agent.routing import (
    JevTriageProvider,
    LocalTriageProvider,
    MedicalRouter,
    RouteDecision,
    route_from_triage,
)
from health_ai_copilot.multi_agent.runtime import (
    MedicalAgentRuntime,
    RuntimeConfig,
)
from health_ai_copilot.routing.jev import JevConfig

_COMPLEX_FAMILIES = frozenset({
    "MEMORY_EXTERNAL_JOIN", "MEMORY_EXTERNAL_CONFLICT", "EXTERNAL_MULTI_SOURCE",
    "MEMORY_MULTI_RECORD", "COMPOSITIONAL_MULTI_FACT",
})
_MAX_EPISODE_CONCURRENCY = 1  # Isolate per-request latency; specialist workers still run in parallel.
_ROLE_LABELS = (
    ("need_patient_context", WorkerRole.PATIENT_CONTEXT),
    ("need_external_evidence", WorkerRole.EVIDENCE),
    ("need_care_analysis", WorkerRole.CARE),
)
_CAPABILITY_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
_COMPLEXITY_THRESHOLDS = (0.25, 0.4, 0.5, 0.6, 0.75)
_MODEL_TOKEN_PATTERN = r"(?u)\b[\w-]+\b"


class StrongSingleRouter(MedicalRouter):
    def decide(self, query: str) -> RouteDecision:
        routed = super().decide(query)
        return RouteDecision(
            RouteMode.SINGLE,
            "strong_single_baseline; all registered skills remain available",
            routed.predicted_capabilities,
            routed.unique_key_count,
        )


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        stream.write("\n")


def _read_jsonl_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        if index == len(lines) - 1 and not line.endswith(("\n", "\r")):
            # A killed process can leave only its last append incomplete. The
            # result file is the commit marker, so an unterminated tail is safe
            # to regenerate from the runtime.
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"MA_MVP2_CHECKPOINT_CORRUPT:{path.name}:{index + 1}") from exc
        if not isinstance(row, dict):
            raise TypeError(f"MA_MVP2_CHECKPOINT_ROW_INVALID:{path.name}:{index + 1}")
        rows.append(row)
    return rows


def _rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".repair.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_directory(run_id: str) -> Path:
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
                         for char in run_id):
        raise ValueError("run_id may contain only letters, digits, dash, underscore, and dot")
    return REPOSITORY_ROOT / "runs" / "multi_agent" / f"ma-mvp2-{run_id}"


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        max_model_turns_per_worker=1,
        max_tool_calls_per_worker=2,
        max_worker_calls=3,
        max_repair_workers=2,
        worker_timeout_seconds=120.0,
        planning_timeout_seconds=60.0,
        synthesis_timeout_seconds=120.0,
        max_worker_output_tokens=256,
        max_lead_output_tokens=320,
        max_single_output_tokens=96,
        max_coverage_output_tokens=128,
    )


def _provider(args) -> LocalLlamaCppProvider:
    return LocalLlamaCppProvider(base_url=args.base_url, model=args.model)


async def _healthcheck(provider: LocalLlamaCppProvider) -> tuple[str, ...]:
    try:
        served = await provider.healthcheck()
    except Exception as exc:
        raise RuntimeError(
            "LOCAL_LLAMA_CPP_UNAVAILABLE; check --base-url and use the repository .venv"
        ) from exc
    if not served:
        raise RuntimeError("LOCAL_LLAMA_CPP_RETURNED_NO_MODELS")
    return served


def _oracle_capabilities(record: EvaluationRecord) -> frozenset[WorkerRole]:
    oracle = record.truth.get("capability_requirement_oracle", {})
    expected: set[WorkerRole] = set()
    if oracle.get("memory_required"):
        expected.add(WorkerRole.PATIENT_CONTEXT)
    if oracle.get("external_retrieval_required"):
        expected.add(WorkerRole.EVIDENCE)
    if not oracle.get("answerability", True) and not expected:
        expected.add(WorkerRole.CARE)
    return frozenset(expected)


def _fit_local_triage_model(records: tuple[EvaluationRecord, ...]) -> dict[str, Any]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - research extra is installed in this repo env
        raise RuntimeError("scikit-learn is required; run this through the repository uv environment") from exc

    queries = [record.case.episode.query for record in records]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000,
        token_pattern=_MODEL_TOKEN_PATTERN,
        binary=True,
        norm="l2",
        use_idf=True,
    )
    matrix = vectorizer.fit_transform(queries)
    labels: dict[str, list[int]] = {name: [] for name, _role in _ROLE_LABELS}
    labels["complexity"] = []
    for record in records:
        expected = _oracle_capabilities(record)
        for name, role in _ROLE_LABELS:
            labels[name].append(int(role in expected))
        labels["complexity"].append(int(record.scenario_family in _COMPLEX_FAMILIES))

    classifiers: dict[str, Any] = {}
    for name, target in labels.items():
        estimator = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
        estimator.fit(matrix, target)
        positive_index = list(estimator.classes_).index(1)
        classifiers[name] = {
            "intercept": float(estimator.intercept_[0]),
            "coefficients": [float(value) for value in estimator.coef_[0]],
            "positive_class_index": positive_index,
        }
    return {
        "schema_version": "local-tfidf-logistic-triage-v1",
        "version": "local-tfidf-logistic-triage-v1",
        "training_split": "TRAIN_TUNE",
        "training_episode_count": len(records),
        "vocabulary": {str(key): int(value) for key, value in vectorizer.vocabulary_.items()},
        "idf": [float(value) for value in vectorizer.idf_],
        "classifiers": classifiers,
        "vectorizer": {
            "token_pattern": _MODEL_TOKEN_PATTERN,
            "ngram_range": [1, 2],
            "lowercase": True,
            "binary": True,
            "norm": "l2",
            "use_idf": True,
        },
    }


def _expected_complex(record: EvaluationRecord) -> bool:
    return record.scenario_family in _COMPLEX_FAMILIES


def _routing_metrics(
    records: tuple[EvaluationRecord, ...],
    decisions: list[TriageDecision],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    if len(records) != len(decisions):
        raise ValueError("ROUTING_METRIC_ROW_COUNT_MISMATCH")
    exact = 0
    intersection = predicted_count = expected_count = 0
    team_count = fast_count = 0
    complexity_tp = complexity_fp = complexity_fn = 0
    role_counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name, _role in _ROLE_LABELS}
    for record, decision in zip(records, decisions, strict=True):
        route = route_from_triage(decision, thresholds=thresholds)
        predicted = set(route.predicted_capabilities)
        expected = set(_oracle_capabilities(record))
        exact += predicted == expected
        intersection += len(predicted & expected)
        predicted_count += len(predicted)
        expected_count += len(expected)
        team_count += route.mode.value == "TEAM"
        fast_count += route.mode.value == "SINGLE"
        complex_predicted = decision.complexity >= thresholds["complexity"]
        complex_expected = _expected_complex(record)
        complexity_tp += complex_predicted and complex_expected
        complexity_fp += complex_predicted and not complex_expected
        complexity_fn += not complex_predicted and complex_expected
        for name, role in _ROLE_LABELS:
            is_predicted = role in predicted
            is_expected = role in expected
            role_counts[name]["tp"] += is_predicted and is_expected
            role_counts[name]["fp"] += is_predicted and not is_expected
            role_counts[name]["fn"] += not is_predicted and is_expected
    return {
        "episode_count": len(records),
        "worker_set_precision": intersection / predicted_count if predicted_count else None,
        "worker_set_recall": intersection / expected_count if expected_count else None,
        "worker_set_exact_specialist_match": exact / len(records) if records else None,
        "single_fast_path_rate": fast_count / len(records) if records else None,
        "team_activation_rate": team_count / len(records) if records else None,
        "per_worker_counts": role_counts,
        "complexity_binary": {
            "precision": complexity_tp / (complexity_tp + complexity_fp)
            if complexity_tp + complexity_fp else None,
            "recall": complexity_tp / (complexity_tp + complexity_fn)
            if complexity_tp + complexity_fn else None,
        },
    }


def _tune_thresholds(
    records: tuple[EvaluationRecord, ...], decisions: list[TriageDecision],
) -> tuple[dict[str, float], dict[str, Any]]:
    best: tuple[tuple[float, float, float, float], dict[str, float], dict[str, Any]] | None = None
    for patient in _CAPABILITY_THRESHOLDS:
        for evidence in _CAPABILITY_THRESHOLDS:
            for care in _CAPABILITY_THRESHOLDS:
                for complexity in _COMPLEXITY_THRESHOLDS:
                    candidate = {
                        "need_patient_context": patient,
                        "need_external_evidence": evidence,
                        "need_care_analysis": care,
                        "complexity": complexity,
                    }
                    metrics = _routing_metrics(records, decisions, candidate)
                    exact = float(metrics["worker_set_exact_specialist_match"] or 0.0)
                    precision = float(metrics["worker_set_precision"] or 0.0)
                    recall = float(metrics["worker_set_recall"] or 0.0)
                    team_rate = float(metrics["team_activation_rate"] or 0.0)
                    key = (exact, (precision + recall) / 2, metrics["complexity_binary"]["recall"] or 0.0,
                           -team_rate)
                    if best is None or key > best[0]:
                        best = (key, candidate, metrics)
    if best is None:
        raise AssertionError("threshold grid unexpectedly empty")
    return best[1], best[2]


def _observable_context(record: EvaluationRecord) -> str:
    observable = record.case.episode.observable_state
    payload = {
        "history_exists": observable.history_exists,
        "history_length_bucket": observable.history_length_bucket,
        "history_time_span": observable.history_time_span,
        "available_personal_state_types": list(observable.available_personal_state_types),
        "available_external_source_families": list(observable.available_external_source_families),
        "available_tool_ids": list(observable.available_tool_ids),
        "available_worker_capabilities": list(observable.available_worker_capabilities),
        "budget_class": observable.budget_class,
        "deadline_class": observable.deadline_class,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _jev_dotenv_path(args) -> Path | None:
    configured = getattr(args, "jev_dotenv", None)
    if configured:
        return Path(configured).expanduser().resolve()
    sibling_config = REPOSITORY_ROOT.parent / "Health-Copilot" / ".env"
    return sibling_config if sibling_config.is_file() else None


async def _ablate_router(
    records: tuple[EvaluationRecord, ...], *, dotenv_path: Path | None,
) -> dict[str, Any]:
    baseline = LocalTriageProvider()
    jev = JevTriageProvider(local_fallback=baseline, dotenv_path=dotenv_path)
    current_rows = [
        await baseline.triage(record.case.episode.query, _observable_context(record))
        for record in records
    ]
    semaphore = asyncio.Semaphore(8)

    async def classify(index: int, record: EvaluationRecord) -> tuple[int, TriageDecision]:
        async with semaphore:
            decision = await jev.triage(
                record.case.episode.query, _observable_context(record),
            )
            return index, decision

    selected_rows: list[TriageDecision | None] = [None] * len(records)
    completed = 0
    for chunk_start in range(0, len(records), 32):
        chunk_end = min(len(records), chunk_start + 32)
        pending = [asyncio.create_task(classify(index, records[index]))
                   for index in range(chunk_start, chunk_end)]
        for future in asyncio.as_completed(pending):
            index, decision = await future
            selected_rows[index] = decision
            completed += 1
        print(json.dumps({"stage": "router_ablation", "completed": completed,
                          "total": len(records)}, ensure_ascii=False))
        if chunk_end < len(records):
            await asyncio.sleep(0.25)
    jev_rows = [item for item in selected_rows if item is not None]
    if len(jev_rows) != len(records):
        raise RuntimeError("MA_MVP2_JEV_ABLATION_INCOMPLETE")
    current_thresholds, current_metrics = _tune_thresholds(records, current_rows)
    jev_thresholds, jev_metrics = _tune_thresholds(records, jev_rows)
    jev_used = sum(item.provider == "jev" for item in jev_rows)
    fallback_reasons = Counter(item.fallback_reason or "" for item in jev_rows if item.provider != "jev")
    current_cost_tokens = sum(item.input_tokens + item.output_tokens for item in current_rows)
    jev_cost_tokens = sum(item.input_tokens + item.output_tokens for item in jev_rows)
    jev_cost_rows = [item.cost_usd for item in jev_rows if item.cost_usd is not None]
    gain = float(jev_metrics["worker_set_exact_specialist_match"] or 0.0) - float(
        current_metrics["worker_set_exact_specialist_match"] or 0.0
    )
    jev_selected = (
        jev_used / len(jev_rows) >= 0.95 and gain >= 0.03
        and float(jev_metrics["worker_set_precision"] or 0.0)
            + float(jev_metrics["worker_set_recall"] or 0.0)
        >= float(current_metrics["worker_set_precision"] or 0.0)
            + float(current_metrics["worker_set_recall"] or 0.0)
    )
    return {
        "training_split": "TRAIN_TUNE",
        "baseline": {"provider_version": baseline.version, "thresholds": current_thresholds,
                     "metrics": current_metrics, "total_triage_tokens": current_cost_tokens},
        "jev": {
            "provider_version": jev.version,
            "model": next((item.model for item in jev_rows if item.model), None),
            "calls_used": jev_used,
            "fallback_count": len(jev_rows) - jev_used,
            "fallback_reasons": dict(fallback_reasons),
            "thresholds": jev_thresholds,
            "metrics": jev_metrics,
            "total_triage_tokens": jev_cost_tokens,
            "total_cost_usd": sum(jev_cost_rows) if jev_cost_rows else None,
            "cost_reported_calls": len(jev_cost_rows),
            "exact_match_gain_vs_current": gain,
        },
        "decision": "jev" if jev_selected else "local",
        "selection_rule": "Use Jev only with >=95% call availability, >=3pp exact-match gain, and no precision+recall regression.",
    }


async def _tune_router(args) -> None:
    run_dir = _run_directory(args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "system_config.json").exists():
        raise FileExistsError("cannot tune a frozen MA-MVP2 run")
    records = load_records(REPOSITORY_ROOT, "train_tune")
    dotenv_path = _jev_dotenv_path(args)
    try:
        jev_config = JevConfig.from_env(dotenv_path=dotenv_path)
        print(json.dumps({"stage": "router_ablation", "jev_config_loaded": True,
                          "dotenv_file_present": bool(dotenv_path and dotenv_path.is_file()),
                          "api_mode": jev_config.api_mode, "model": jev_config.model},
                         ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001 - local router remains available without Jev.
        print(json.dumps({"stage": "router_ablation", "jev_config_loaded": False,
                          "error_type": type(exc).__name__}, ensure_ascii=False))
    ablation = await _ablate_router(records, dotenv_path=dotenv_path)
    model = _fit_local_triage_model(records)
    trained_provider = LocalTriageProvider(trained_model=model)
    train_decisions = [
        await trained_provider.triage(record.case.episode.query, _observable_context(record))
        for record in records
    ]
    trained_thresholds, trained_metrics = _tune_thresholds(records, train_decisions)
    local_gain = float(trained_metrics["worker_set_exact_specialist_match"] or 0.0) - float(
        ablation["baseline"]["metrics"]["worker_set_exact_specialist_match"] or 0.0
    )
    selected_provider = "jev" if ablation["decision"] == "jev" else "local-trained"
    thresholds = ablation["jev"]["thresholds"] if selected_provider == "jev" else trained_thresholds
    _write_json(run_dir / "router_ablation.json", ablation)
    _write_json(run_dir / "local_triage_model.json", model)
    selected = {
        "schema_version": "ma-mvp2-train-tune-selection-v1",
        "training_split": "TRAIN_TUNE",
        "training_selection": TRAIN_TUNE_SELECTION,
        "provider": selected_provider,
        "provider_version": (JevTriageProvider.version if selected_provider == "jev"
                             else trained_provider.version),
        "thresholds": thresholds,
        "trained_local_router": {
            "thresholds": trained_thresholds,
            "metrics": trained_metrics,
            "exact_gain_vs_current": local_gain,
            "model_sha256": file_sha256(run_dir / "local_triage_model.json"),
        },
        "tuning_round": 1,
        "maximum_tuning_rounds": 3,
    }
    _write_json(run_dir / "selected_router.json", selected)
    print(json.dumps({"status": "ROUTER_TUNE_COMPLETE", "run_dir": str(run_dir),
                      "router_ablation": ablation, "trained_local": selected["trained_local_router"],
                      "selected_provider": selected_provider, "thresholds": thresholds},
                     ensure_ascii=False, sort_keys=True))


def _load_router_selection(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_path = run_dir / "selected_router.json"
    model_path = run_dir / "local_triage_model.json"
    if not selected_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("run TRAIN_TUNE router selection first")
    selected = _read_json(selected_path)
    model = _read_json(model_path)
    if file_sha256(model_path) != selected["trained_local_router"]["model_sha256"]:
        raise ValueError("MA_MVP2_TRAINED_ROUTER_MODEL_HASH_MISMATCH")
    if selected.get("training_split") != "TRAIN_TUNE" or selected.get("tuning_round") > 3:
        raise ValueError("MA_MVP2_TRAINING_SELECTION_INVALID")
    return selected, model


def _selected_triage_provider(
    selected: dict[str, Any], model: dict[str, Any], *, dotenv_path: Path | None,
):
    local = LocalTriageProvider(trained_model=model)
    if selected["provider"] == "jev":
        return JevTriageProvider(local_fallback=local, dotenv_path=dotenv_path)
    return local


async def _run_system(
    records: tuple[EvaluationRecord, ...],
    runtime: MedicalAgentRuntime,
    system: str,
    stage: str,
    result_rows: list[dict[str, Any]],
    run_dir: Path,
) -> None:
    system_tag = system.casefold()
    result_path = run_dir / f"{stage}_{system_tag}_results.jsonl"
    routing_path = run_dir / f"{stage}_{system_tag}_routing.jsonl"
    workers_path = run_dir / f"{stage}_{system_tag}_worker_artifacts.jsonl"
    traces_path = run_dir / f"{stage}_{system_tag}_traces.jsonl"
    expected_ids = {record.episode_id for record in records}
    completed_rows = _read_jsonl_checkpoint(result_path)
    completed_ids: set[str] = set()
    for row in completed_rows:
        episode_id = str(row.get("episode_id", ""))
        if (episode_id not in expected_ids or episode_id in completed_ids
                or row.get("system") != system):
            raise ValueError("MA_MVP2_RESULT_CHECKPOINT_MISMATCH")
        completed_ids.add(episode_id)
    if result_path.exists():
        _rewrite_jsonl(result_path, completed_rows)

    sidecars = (routing_path, workers_path, traces_path)
    sidecar_rows: dict[Path, list[dict[str, Any]]] = {}
    for path in sidecars:
        rows = _read_jsonl_checkpoint(path)
        for row in rows:
            if str(row.get("episode_id", "")) not in expected_ids:
                raise ValueError(f"MA_MVP2_SIDECAR_CHECKPOINT_MISMATCH:{path.name}")
        retained = [row for row in rows if str(row.get("episode_id")) in completed_ids]
        if path.exists():
            _rewrite_jsonl(path, retained)
        sidecar_rows[path] = retained
    if completed_ids:
        if {str(row["episode_id"]) for row in sidecar_rows[traces_path]} != completed_ids:
            raise ValueError("MA_MVP2_TRACE_CHECKPOINT_INCOMPLETE")
        if (system == "MA_MVP2_ROUTED_TEAM"
                and {str(row["episode_id"]) for row in sidecar_rows[routing_path]}
                != completed_ids):
            raise ValueError("MA_MVP2_ROUTING_CHECKPOINT_INCOMPLETE")
    result_rows.extend(completed_rows)
    semaphore = asyncio.Semaphore(_MAX_EPISODE_CONCURRENCY)
    completed = len(completed_ids)

    async def run_one(record: EvaluationRecord) -> None:
        nonlocal completed
        if record.episode_id in completed_ids:
            return
        async with semaphore:
            execution = await runtime.execute(MedicalAgentRequest(
                query=record.case.episode.query,
                request_id=f"ma-mvp2-{stage}-{system.lower()}-{record.episode_id}",
                patient_id=record.case.episode.subject_id,
                as_of_time=record.case.episode.decision_time,
                episode=record.case.episode,
                resources=record.case.resources,
            ))
        scored = score_execution(record, execution, system)
        if system == "MA_MVP2_ROUTED_TEAM":
            _append_jsonl(routing_path, {
                "episode_id": record.episode_id,
                "route_decision": execution.route_decision.to_dict(),
                "task_ledger": execution.task_ledger.to_dict() if execution.task_ledger else None,
                "coverage_ledger": execution.coverage_ledger.to_dict()
                if execution.coverage_ledger else None,
                "repair_wave": execution.repair_wave,
                "expected_capabilities": scored["expected_capabilities"],
                "predicted_capabilities": scored["predicted_capabilities"],
                "route_precision_numerator": scored["route_precision_numerator"],
                "route_precision_denominator": scored["route_precision_denominator"],
                "route_recall_numerator": scored["route_recall_numerator"],
                "route_recall_denominator": scored["route_recall_denominator"],
                "exact_worker_set_match": scored["exact_worker_set_match"],
            })
        for report in execution.worker_reports:
            _append_jsonl(workers_path, {"episode_id": record.episode_id, "system": system,
                                         **report.to_dict()})
        _append_jsonl(traces_path, {
            "episode_id": record.episode_id,
            "system": system,
            "trace": execution.trace,
            "trajectory": execution.trajectory,
        })
        scored = score_execution(record, execution, system)
        _append_jsonl(result_path, scored)
        result_rows.append(scored)
        completed_ids.add(record.episode_id)
        completed += 1
        if completed % 16 == 0 or completed == len(records):
            print(json.dumps({"stage": stage, "system": system, "completed": completed,
                              "total": len(records), "episode_id": record.episode_id},
                             ensure_ascii=False))

    await asyncio.gather(*(run_one(record) for record in records))


def _metrics_by_system(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        name: aggregate_metrics([row for row in results if row["system"] == name])
        for name in ("STRONG_SINGLE", "MA_MVP2_ROUTED_TEAM")
    }


async def _run_pair(
    *,
    records: tuple[EvaluationRecord, ...],
    provider: LocalLlamaCppProvider,
    selected: dict[str, Any],
    local_model: dict[str, Any],
    args,
    stage: str,
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], tuple[str, ...]]:
    served_models = await _healthcheck(provider)
    results: list[dict[str, Any]] = []
    config = _runtime_config()
    single = MedicalAgentRuntime(provider, router=StrongSingleRouter(), config=config)
    team = MedicalAgentRuntime(
        provider,
        config=config,
        triage_provider=_selected_triage_provider(
            selected, local_model, dotenv_path=_jev_dotenv_path(args),
        ),
        triage_thresholds=selected["thresholds"],
    )
    await _run_system(records, single, "STRONG_SINGLE", stage, results, run_dir)
    await _run_system(records, team, "MA_MVP2_ROUTED_TEAM", stage, results, run_dir)
    return results, _metrics_by_system(results), served_models


def _internal_dev_gate(metrics: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    team_rows = [row for row in results if row["system"] == "MA_MVP2_ROUTED_TEAM"]
    complex_rows = [row for row in team_rows if row["slice_complex"]]
    complex_success = aggregate_metrics(complex_rows).get("task_success", 0.0)
    team_fact = metrics["MA_MVP2_ROUTED_TEAM"]["required_fact_coverage"]
    exact = metrics["MA_MVP2_ROUTED_TEAM"]["exact_worker_set_match"]
    checks = {
        "complex_success_above_ma_mvp1_team": complex_success > MVP1_COMPLEX_TEAM_SUCCESS,
        "required_fact_coverage_above_ma_mvp1_team": team_fact > MVP1_OVERALL_TEAM_FACT_COVERAGE,
        "routing_exact_match_above_ma_mvp1": exact > MVP1_EXACT_WORKER_SET,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "complex_task_success": complex_success,
            "required_fact_coverage": team_fact,
            "exact_worker_set_match": exact,
        },
        "ma_mvp1_reference": {
            "complex_task_success": MVP1_COMPLEX_TEAM_SUCCESS,
            "required_fact_coverage": MVP1_OVERALL_TEAM_FACT_COVERAGE,
            "exact_worker_set_match": MVP1_EXACT_WORKER_SET,
        },
    }


async def _train_tune(args) -> None:
    run_dir = _run_directory(args.run_id)
    if (run_dir / "system_config.json").exists():
        raise FileExistsError("TRAIN_TUNE cannot change a frozen run")
    selected, model = _load_router_selection(run_dir)
    records = load_records(REPOSITORY_ROOT, "train_tune")
    _results, metrics, served = await _run_pair(
        records=records,
        provider=_provider(args),
        selected=selected,
        local_model=model,
        args=args,
        stage="train_tune",
        run_dir=run_dir,
    )
    payload = {
        "dataset_id": DATASET_ID,
        "dataset_root_sha256": DATASET_ROOT_HASH,
        "split": "TRAIN_TUNE",
        "selection": TRAIN_TUNE_SELECTION,
        "episode_count_per_system": len(records),
        "reserved_test_materialized": False,
        "tuning_rounds_used": 3,
        "tuning_log": [
            {"round": 1, "split": "TRAIN_TUNE", "scope": "router thresholds; Jev ablation"},
            {"round": 2, "split": "TRAIN_TUNE", "scope": "lead plan and worker artifact prompts; output budgets"},
            {"round": 3, "split": "TRAIN_TUNE", "scope": "deduplicate Lead aspects; match verified fact mappings in coverage; emit repeated facts once in finalization"},
        ],
        "systems": metrics,
        "served_model_ids": list(served),
        "selected_router": selected,
    }
    _write_json(run_dir / "train_tune_metrics.json", payload)
    print(json.dumps({"status": "TRAIN_TUNE_COMPLETE", "systems": metrics,
                      "reserved_test_materialized": False}, ensure_ascii=False, sort_keys=True))


async def _dev(args) -> None:
    run_dir = _run_directory(args.run_id)
    if (run_dir / "system_config.json").exists():
        raise FileExistsError("DEV is a pre-freeze diagnostic stage")
    selected, model = _load_router_selection(run_dir)
    records = load_records(REPOSITORY_ROOT, "dev")
    results, systems, served = await _run_pair(
        records=records,
        provider=_provider(args),
        selected=selected,
        local_model=model,
        args=args,
        stage="dev",
        run_dir=run_dir,
    )
    slices: dict[str, Any] = {}
    for label, selected_rows in (
        ("ALL_DEV", results),
        ("COMPLEX", [row for row in results if row["slice_complex"]]),
        ("SIMPLE", [row for row in results if row["slice_simple"]]),
    ):
        left = aggregate_metrics([row for row in selected_rows if row["system"] == "STRONG_SINGLE"])
        right = aggregate_metrics([row for row in selected_rows
                                   if row["system"] == "MA_MVP2_ROUTED_TEAM"])
        slices[label] = compare_systems(left, right)
    gate = _internal_dev_gate(systems, results)
    by_pool = {}
    for pool in sorted({row["split"] for row in records}):
        episode_ids = {record.episode_id for record in records if record.split == pool}
        by_pool[pool] = {
            name: aggregate_metrics([row for row in results
                                     if row["episode_id"] in episode_ids and row["system"] == name])
            for name in ("STRONG_SINGLE", "MA_MVP2_ROUTED_TEAM")
        }
    payload = {
        "dataset_id": DATASET_ID,
        "dataset_root_sha256": DATASET_ROOT_HASH,
        "split": "DEV_IID+DEV_STRUCTURAL",
        "episode_count_per_system": len(records),
        "reserved_test_materialized": False,
        "systems": systems,
        "slices": slices,
        "by_split": by_pool,
        "internal_dev_gate": gate,
        "development_targets": {
            "complex_success_delta_vs_single_at_least": 0.05,
            "team_fact_coverage_delta_vs_single_at_least": -0.02,
            "exact_worker_set_match_at_least": 0.60,
            "parallel_speedup_at_least": 1.5,
        },
        "served_model_ids": list(served),
        "tuning_rounds_used": 3,
        "tuning_log": [
            {"round": 1, "split": "TRAIN_TUNE", "scope": "router thresholds; Jev ablation"},
            {"round": 2, "split": "TRAIN_TUNE", "scope": "lead plan and worker artifact prompts; output budgets"},
            {"round": 3, "split": "TRAIN_TUNE", "scope": "deduplicate Lead aspects; match verified fact mappings in coverage; emit repeated facts once in finalization"},
        ],
        "selected_router": selected,
    }
    _write_json(run_dir / "dev_metrics.json", payload)
    _write_json(run_dir / "dev_internal_gate.json", gate)
    status = "DEV_GATE_PASSED" if gate["passed"] else "DEV_GATE_FAILED_STOP_AND_DIAGNOSE"
    print(json.dumps({"status": status, "gate": gate, "systems": systems,
                      "slices": slices, "by_split": by_pool}, ensure_ascii=False, sort_keys=True))
    if not gate["passed"]:
        raise SystemExit(2)


def _verify_pre_freeze(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    train_path = run_dir / "train_tune_metrics.json"
    dev_path = run_dir / "dev_metrics.json"
    if not train_path.is_file() or not dev_path.is_file():
        raise FileNotFoundError("TRAIN_TUNE and DEV must complete before MA_MVP2_CONFIG_V1 freeze")
    train = _read_json(train_path)
    dev = _read_json(dev_path)
    if train.get("episode_count_per_system") != 512 or train.get("reserved_test_materialized") is not False:
        raise ValueError("MA_MVP2_TRAIN_TUNE_REQUIREMENTS_NOT_MET")
    if dev.get("episode_count_per_system") != 1024 or dev.get("reserved_test_materialized") is not False:
        raise ValueError("MA_MVP2_DEV_REQUIREMENTS_NOT_MET")
    gate = dev.get("internal_dev_gate", {})
    if gate.get("passed") is not True:
        raise ValueError("MA_MVP2_DEV_GATE_FAILED_DO_NOT_FREEZE_OR_OPEN_RESERVED_TEST")
    selected, model = _load_router_selection(run_dir)
    return train, dev, {"selected": selected, "model": model}


def _freeze(args) -> None:
    run_dir = _run_directory(args.run_id)
    if (run_dir / "system_config.json").exists():
        raise FileExistsError("MA_MVP2_CONFIG_V1 already frozen")
    _train, dev, router_data = _verify_pre_freeze(run_dir)
    plan_path = REPOSITORY_ROOT / RESERVED_TEST_RELATIVE
    code = code_identity(REPOSITORY_ROOT)
    selected = router_data["selected"]
    model_path = run_dir / "local_triage_model.json"
    payload = {
        "config_version": CONFIG_VERSION,
        "frozen": True,
        "frozen_at_utc": _utc_now(),
        "baseline_commit": MVP1_BASE_COMMIT,
        "git_commit_at_freeze": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True,
        ).strip(),
        "code": {"sha256": code["code_sha256"], "files": code["code_file_sha256"]},
        "prompts": {"sha256": prompt_hashes()},
        "tuning": {
            "rounds_used": 2,
            "maximum_rounds": 3,
            "rounds": [
                {"round": 1, "split": "TRAIN_TUNE", "scope": "router thresholds; Jev ablation"},
                {"round": 2, "split": "TRAIN_TUNE", "scope": "lead plan and worker artifact prompts; output budgets"},
            ],
        },
        "triage": {
            "provider": selected["provider"],
            "provider_version": selected["provider_version"],
            "jev_model": _read_json(run_dir / "router_ablation.json")["jev"].get("model"),
            "jev_cost_usd_on_train_tune": _read_json(run_dir / "router_ablation.json")[
                "jev"
            ].get("total_cost_usd"),
            "local_model_schema": router_data["model"].get("schema_version"),
            "local_model_sha256": file_sha256(model_path),
            "thresholds": selected["thresholds"],
            "tuning_split": "TRAIN_TUNE",
            "train_tune_selection": TRAIN_TUNE_SELECTION,
            "tuning_rounds": 1,
            "maximum_tuning_rounds": 3,
            "jev_ablation_sha256": file_sha256(run_dir / "router_ablation.json"),
        },
        "model": {
            "provider": "local llama.cpp OpenAI-compatible API",
            "name": args.model,
            "artifact_sha256": MODEL_SHA256,
            "base_url": args.base_url,
            "device_policy": "CPU_ONLY; llama-server --n-gpu-layers 0 --no-op-offload",
            "reasoning_mode": "Qwen3 /no_think; server --reasoning off",
            "server_parallel_sequences": 3,
            "server_context_tokens": 16384,
            "server_threads": 24,
            "temperature": 0,
        },
        "dataset": {
            "dataset_id": DATASET_ID,
            "root_sha256": DATASET_ROOT_HASH,
            "train_tune_metrics_sha256": file_sha256(run_dir / "train_tune_metrics.json"),
            "dev_metrics_sha256": file_sha256(run_dir / "dev_metrics.json"),
            "dev_gate": dev["internal_dev_gate"],
        },
        "generation": {
            "max_worker_model_turns": 1,
            "max_tool_calls_per_worker": 2,
            "max_initial_workers": 3,
            "max_repair_wave_workers": 2,
            "max_repair_waves": 1,
            "parallel_independent_workers": True,
            "two_wave_dependency": True,
            "worker_output_tokens": 256,
            "coverage_output_tokens": 128,
            "finalizer_output_tokens": 320,
            "single_output_tokens": 96,
            "temperature": 0,
        },
        "evaluation": {
            "max_concurrent_episodes_per_system": _MAX_EPISODE_CONCURRENCY,
            "same_limit_applied_to_both_systems": True,
        },
        "reserved_test": {
            "episode_count": 1024,
            "plan_sha256": file_sha256(plan_path),
            "pool_episode_counts": {"IID_TEST": 128, "OOD_PATIENT": 128, "OOD_TASK": 128,
                                     "OOD_TEMPORAL": 128, "OOD_SOURCE": 128,
                                     "OOD_COMPOSITION": 384},
            "truth_materialized": False,
            "opens_only_after_this_config_is_frozen": True,
        },
        "train_tune": {
            "episode_count_per_system": 512,
            "selection": TRAIN_TUNE_SELECTION,
            "reserved_split_materialized": False,
            "metrics_sha256": file_sha256(run_dir / "train_tune_metrics.json"),
        },
        "dev": {
            "episode_count_per_system": 1024,
            "splits": ["DEV_IID", "DEV_STRUCTURAL"],
            "reserved_split_materialized": False,
            "metrics_sha256": file_sha256(run_dir / "dev_metrics.json"),
        },
    }
    payload["config_sha256"] = canonical_hash(payload)
    _write_json(run_dir / "system_config.json", payload)
    validate_frozen_config(_read_json(run_dir / "system_config.json"))
    print(json.dumps({"status": "FROZEN", "config_version": CONFIG_VERSION,
                      "config_sha256": payload["config_sha256"],
                      "code_sha256": code["code_sha256"], "run_dir": str(run_dir)},
                     ensure_ascii=False, sort_keys=True))


async def _test(args) -> None:
    run_dir = _run_directory(args.run_id)
    config_path = run_dir / "system_config.json"
    if not config_path.is_file():
        raise FileNotFoundError("freeze MA_MVP2_CONFIG_V1 before reserved TEST")
    config = _read_json(config_path)
    validate_frozen_config(config)
    current_code = code_identity(REPOSITORY_ROOT)
    if current_code["code_sha256"] != config["code"]["sha256"]:
        raise ValueError("MA_MVP2_CODE_CHANGED_AFTER_FREEZE")
    if prompt_hashes() != config["prompts"]["sha256"]:
        raise ValueError("MA_MVP2_PROMPT_CHANGED_AFTER_FREEZE")
    if args.base_url != config["model"]["base_url"] or args.model != config["model"]["name"]:
        raise ValueError("MA_MVP2_MODEL_ENDPOINT_CHANGED_AFTER_FREEZE")
    if file_sha256(run_dir / "local_triage_model.json") != config["triage"]["local_model_sha256"]:
        raise ValueError("MA_MVP2_LOCAL_ROUTER_MODEL_CHANGED_AFTER_FREEZE")
    reserved_dir = run_dir / "reserved_test"
    records = materialize_reserved_test(
        run_dir=reserved_dir,
        config_path=config_path,
        repository_root=REPOSITORY_ROOT,
    )
    if len(records) != config["reserved_test"]["episode_count"]:
        raise ValueError("MA_MVP2_FRESH_RESERVED_TEST_SIZE_MISMATCH")
    selected, local_model = _load_router_selection(run_dir)
    results, systems, served = await _run_pair(
        records=records,
        provider=_provider(args),
        selected=selected,
        local_model=local_model,
        args=args,
        stage="reserved_test",
        run_dir=run_dir,
    )
    pool_metrics = {}
    pool_names = sorted({record.split for record in records})
    for pool in pool_names:
        episode_ids = {record.episode_id for record in records if record.split == pool}
        pool_metrics[pool] = {
            name: aggregate_metrics([row for row in results
                                     if row["episode_id"] in episode_ids and row["system"] == name])
            for name in ("STRONG_SINGLE", "MA_MVP2_ROUTED_TEAM")
        }
    complex_single_rows = [row for row in results
                           if row["system"] == "STRONG_SINGLE" and row["slice_complex"]]
    complex_team_rows = [row for row in results
                         if row["system"] == "MA_MVP2_ROUTED_TEAM" and row["slice_complex"]]
    latency = _latency_metrics(results)
    report = {
        "schema_version": "ma-mvp2-reserved-test-results-v1",
        "run_id": args.run_id,
        "evaluated_at_utc": _utc_now(),
        "config_version": config["config_version"],
        "config_sha256": config["config_sha256"],
        "code_sha256": config["code"]["sha256"],
        "dataset_id": DATASET_ID,
        "dataset_root_sha256": DATASET_ROOT_HASH,
        "reserved_test_materialized_after_freeze": True,
        "reserved_plan_sha256": config["reserved_test"]["plan_sha256"],
        "episode_count_per_system": len(records),
        "systems": systems,
        "delta_team_minus_single": compare_systems(
            systems["STRONG_SINGLE"], systems["MA_MVP2_ROUTED_TEAM"],
        )["delta"],
        "complex_task_success_delta_team_minus_single": (
            aggregate_metrics(complex_team_rows)["task_success"]
            - aggregate_metrics(complex_single_rows)["task_success"]
        ),
        "by_reserved_pool": pool_metrics,
        "latency_and_parallelism": latency,
        "served_model_ids": list(served),
        "truth_policy": "Truth was read only by the post-freeze evaluator; never sent to runtime prompts.",
    }
    _write_json(run_dir / "reserved_test_metrics.json", report)
    _write_json(run_dir / "reserved_test_manifest.json", {
        "config_sha256": config["config_sha256"],
        "reserved_materialization_manifest_sha256": file_sha256(
            reserved_dir / "materialization_manifest.json"
        ),
        "test_metrics_sha256": file_sha256(run_dir / "reserved_test_metrics.json"),
        "episode_count_per_system": len(records),
        "systems": ["STRONG_SINGLE", "MA_MVP2_ROUTED_TEAM"],
        "reserved_test_accessed_after_freeze": True,
        "truth_was_not_used_for_tuning": True,
    })
    _write_json(run_dir / "reserved_test_latency_metrics.json", latency)
    print(json.dumps({"status": "RESERVED_TEST_COMPLETE", "run_dir": str(run_dir),
                      "systems": systems, "delta_team_minus_single": report["delta_team_minus_single"],
                      "complex_task_success_delta_team_minus_single": report[
                          "complex_task_success_delta_team_minus_single"],
                      "by_reserved_pool": pool_metrics, "latency_and_parallelism": latency},
                     ensure_ascii=False, sort_keys=True))


def _latency_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for system in ("STRONG_SINGLE", "MA_MVP2_ROUTED_TEAM"):
        subset = [row for row in rows if row["system"] == system]
        latencies = sorted(int(row["latency_ms"]) for row in subset)
        output[system] = {
            "count": len(latencies),
            "mean_ms": sum(latencies) / len(latencies) if latencies else None,
            "p50_ms": latencies[int((len(latencies) - 1) * 0.5)] if latencies else None,
            "p95_ms": latencies[int((len(latencies) - 1) * 0.95)] if latencies else None,
            "total_tokens": sum(int(row["total_tokens"]) for row in subset),
        }
    team_rows = [row for row in rows if row["system"] == "MA_MVP2_ROUTED_TEAM"]
    team = output["MA_MVP2_ROUTED_TEAM"]
    team["single_fast_path_rate"] = sum(row["single_fast_path"] for row in team_rows) / len(team_rows)
    team["team_activation_rate"] = sum(row["team_activated"] for row in team_rows) / len(team_rows)
    team_latency = sorted(int(row["latency_ms"]) for row in team_rows if row["team_activated"])
    team["team_only_p95_ms"] = (
        team_latency[int((len(team_latency) - 1) * 0.95)] if team_latency else None
    )
    active = [row for row in team_rows if row["team_activated"]]
    parallel = aggregate_metrics(active).get("parallel_speedup") if active else None
    team["mean_parallel_speedup"] = parallel
    team["p95_latency_ms"] = team["p95_ms"]
    return output


async def _main(args) -> None:
    stages = {
        "tune-router": _tune_router,
        "train-tune": _train_tune,
        "dev": _dev,
        "freeze": _freeze,
        "test": _test,
    }
    if args.stage == "all":
        for stage in ("tune-router", "train-tune", "dev", "freeze", "test"):
            print(json.dumps({"status": "STAGE_START", "stage": stage}, ensure_ascii=False))
            await stages[stage](args)
    else:
        await stages[args.stage](args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("tune-router", "train-tune", "dev", "freeze", "test", "all"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-url", default=os.environ.get(
        "MA_MVP2_BASE_URL", "http://127.0.0.1:8082/v1"
    ))
    parser.add_argument("--model", default=os.environ.get(
        "MA_MVP2_MODEL", "Qwen3-8B-Q4_K_M.gguf"
    ))
    parser.add_argument("--jev-dotenv", default=os.environ.get("JEV_DOTENV_PATH"))
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
