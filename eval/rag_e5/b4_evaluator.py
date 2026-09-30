"""Privileged post-freeze scorer for RAG-E5-B4 materialized responses."""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from eval.rag_e5.b4_execution import (ACTION_ORDER, DEFAULT_B4_PRIVATE_ROOT,
                                      DEFAULT_EXECUTION_MANIFEST_PATH,
                                      DEFAULT_PROTOCOL_PATH, _read_json,
                                      load_verified_b2, verify_b3_identity,
                                      verify_frozen_code, verify_protocol_lock)
from eval.rag_e5.b4_materializer import extract_citations, strip_aliases
from eval.rag_e5.counterfactual import canonical_sha256
from eval.rag_e5.e5b3_recovery import read_jsonl, sha256_file
from eval.rag_e5.overlay import score_case

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_JSON = ROOT / "runs/rag_e5/e5b4_cpu_counterfactual_report.json"
DEFAULT_REPORT_MARKDOWN = ROOT / "docs/research/rag_e5/e5b4_cpu_counterfactual_report.md"
TEACHER_PATH = Path(r"D:\MyLab\Jianli\external\rag_e5\e5b1\teacher_cases.jsonl")
B2_REPORT_PATH = ROOT / "runs/rag_e5/e5b2_counterfactual_report.json"
BOOTSTRAP_SEED = 20260930
BOOTSTRAP_SAMPLES = 10000


def score_materialized_case(
    *,
    teacher: dict[str, Any],
    state_claims: list[dict[str, str]],
    guidance_text: str,
    cited_chunk_ids: list[str],
    supplied_chunks: list[dict[str, Any]],
) -> dict[str, float]:
    """Action-blind case scorer: it has no action argument or action-dependent branch."""
    state_facts = [
        {"field": row.get("field", ""), "value": row.get("value", "")}
        for row in state_claims
    ]
    clean_guidance = strip_aliases(guidance_text)
    guidance_facts = [{"statement": clean_guidance}] if clean_guidance else []
    return score_case(
        teacher=teacher,
        reader_output={
            "state_facts": state_facts,
            "guidance_facts": guidance_facts,
            "citations": cited_chunk_ids,
        },
        supplied_chunks=supplied_chunks,
    )


def _verify_manifest(
    *,
    lock: dict[str, Any],
    manifest_path: Path,
    private_root: Path,
    protocol_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_manifest_path = lock.get(
        "execution_manifest_path", DEFAULT_EXECUTION_MANIFEST_PATH.relative_to(ROOT).as_posix()
    )
    _require_committed_manifest(manifest_path, expected_manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError("B4 execution manifest must exist before scoring")
    manifest = _read_json(manifest_path)
    body = {key: value for key, value in manifest.items() if key != "execution_manifest_sha256"}
    if canonical_sha256(body) != manifest.get("execution_manifest_sha256"):
        raise ValueError("B4 execution manifest self-hash mismatch")
    if (
        manifest.get("status") != "ALL_B4_ARMS_FROZEN_BEFORE_SCORING"
        or manifest.get("execution_id") != lock.get("execution_id")
        or manifest.get("runtime_backend") != lock["runtime"].get("backend")
        or manifest.get("protocol_lock_sha256") != lock["protocol_lock_sha256"]
        or manifest.get("protocol_lock_file_sha256") != sha256_file(protocol_path)
        or manifest.get("arms_expected") != 180
        or manifest.get("arms_completed") != 180
        or manifest.get("guidance_calls_expected") != 120
        or manifest.get("guidance_calls_completed") != 120
        or manifest.get("new_retrieval_calls") != 0
        or manifest.get("new_bridge_calls") != 0
        or manifest.get("teacher_opened") is not False
        or manifest.get("202608_opened") is not False
    ):
        raise ValueError("B4 execution is incomplete or violates the pre-score freeze contract")
    runtime = manifest.get("inference_runtime")
    locked_runtime = lock["runtime"]
    if not isinstance(runtime, dict) or any(
        runtime.get(field) != locked_runtime.get(locked_field)
        for field, locked_field in (
            ("backend", "backend"),
            ("server_pid", "server_pid"),
            ("build_info", "llama_cpp_build_info"),
            ("server_context_capacity", "server_context_capacity"),
            ("model_path", "model_path"),
            ("server_executable", "server_executable"),
            ("server_executable_sha256", "server_executable_sha256"),
            ("gpu_offload_layers", "gpu_offload_layers"),
            ("op_offload", "op_offload"),
            ("cpu_threads", "cpu_threads"),
        )
    ):
        raise ValueError("B4 execution runtime differs from the frozen CPU protocol lock")
    expected_command_sha = hashlib.sha256(
        locked_runtime["server_process_command_line"].encode("utf-8")
    ).hexdigest()
    if runtime.get("process_command_line_sha256") != expected_command_sha:
        raise ValueError("B4 execution server flags differ from the frozen CPU protocol lock")
    recorded_root = Path(manifest.get("external_arm_root", "")).resolve()
    if recorded_root != private_root.resolve():
        raise ValueError("B4 manifest points to an unexpected private arm directory")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or len(entries) != 180:
        raise ValueError("B4 manifest must freeze exactly 180 arm artifacts")
    if canonical_sha256(entries) != manifest.get("artifact_set_sha256"):
        raise ValueError("B4 artifact-set hash mismatch")
    expected_order = [
        (case_id, action)
        for case_id in sorted({row["case_id"] for row in entries})
        for action in ACTION_ORDER
    ]
    observed_order = [(row.get("case_id"), row.get("action")) for row in entries]
    if observed_order != expected_order:
        raise ValueError("B4 manifest arm order/identity is incomplete or duplicated")
    ledger_path = private_root / "call_ledger.jsonl"
    if not ledger_path.is_file() or sha256_file(ledger_path) != manifest.get("call_ledger_sha256"):
        raise ValueError("B4 append-only call ledger changed after execution freeze")
    call_rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    started = [row for row in call_rows if row.get("status") == "STARTED"]
    completed = [row for row in call_rows if row.get("status") == "COMPLETED"]
    if len(started) != 120 or len(completed) != 120:
        raise ValueError("B4 ledger must contain 120 starts and 120 completions")
    start_by_id = {row.get("call_id"): row for row in started}
    done_by_id = {row.get("call_id"): row for row in completed}
    if len(start_by_id) != 120 or set(start_by_id) != set(done_by_id):
        raise ValueError("B4 call ledger contains missing or repeated calls")
    for call_id, start_row in start_by_id.items():
        done_row = done_by_id[call_id]
        for field in ("case_id", "action", "run_id", "prompt_sha256", "call_ordinal"):
            if start_row.get(field) != done_row.get(field):
                raise ValueError("B4 call completion does not match its frozen start record")
    arms: list[dict[str, Any]] = []
    for entry in entries:
        path = private_root / entry["relative_path"]
        if sha256_file(path) != entry.get("file_sha256"):
            raise ValueError("B4 arm file changed after execution freeze")
        payload = _read_json(path)
        completion_sha = payload.pop("completion_sha256", None)
        if (
            completion_sha != entry.get("completion_sha256")
            or canonical_sha256(payload) != completion_sha
            or payload.get("run_id") != entry.get("run_id")
            or payload.get("protocol_lock_sha256") != lock["protocol_lock_sha256"]
            or payload.get("new_model_calls") != entry.get("new_model_calls")
            or payload.get("new_retrieval_calls") != 0
            or payload.get("new_bridge_calls") != 0
        ):
            raise ValueError("B4 arm payload identity/hash mismatch")
        payload["completion_sha256"] = completion_sha
        arms.append(payload)
    if sum(row["new_model_calls"] for row in arms) != 120:
        raise ValueError("B4 arm artifacts do not represent exactly 120 generation calls")
    if manifest.get("transport_failures") != 0:
        raise ValueError("B4 execution contains transport failures")
    return manifest, arms


def _require_committed_manifest(
    manifest_path: Path, expected_relative_path: str | None = None
) -> None:
    """Require the frozen execution manifest to be tracked and clean before labels open."""
    try:
        relative_path = manifest_path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("B4 execution manifest must be committed inside the repository") from exc
    expected_path = expected_relative_path or DEFAULT_EXECUTION_MANIFEST_PATH.relative_to(
        ROOT
    ).as_posix()
    if relative_path != expected_path:
        raise ValueError("B4 scoring only accepts the canonical committed execution manifest")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative_path],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        raise ValueError("B4 execution manifest must be committed before teacher labels are opened")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", relative_path],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise ValueError("B4 execution manifest must be committed and clean before scoring")


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return statistics.fmean(values) if values else None


def _percentile(rows: list[dict[str, Any]], key: str, percentile: float) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return float(np.percentile(values, percentile)) if values else None


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    state_rows = [row for row in rows if row["task_family"] in {"T0", "T2"}]
    guidance_rows = [row for row in rows if row["task_family"] in {"T1", "T2"}]
    model_rows = [row for row in rows if row["model_calls"]]
    return {
        "cases": len(rows),
        "end_to_end_quality": _mean(rows, "end_to_end_quality"),
        "content_only_quality": _mean(rows, "content_only_quality"),
        "state_score": _mean(state_rows, "state_score"),
        "guideline_content_score": _mean(guidance_rows, "guideline_content_score"),
        "guidance_quality": _mean(guidance_rows, "guideline_content_score"),
        "grounding_score": _mean(guidance_rows, "grounding_score"),
        "model_calls": sum(row["model_calls"] for row in rows),
        "input_tokens": sum(row["input_tokens"] or 0 for row in rows),
        "output_tokens": sum(row["output_tokens"] or 0 for row in rows),
        "token_usage_complete": all(row["token_usage_complete"] for row in rows if row["model_calls"]),
        "mean_latency_ms": _mean(rows, "total_latency_ms"),
        "p50_latency_ms": _percentile(rows, "total_latency_ms", 50),
        "p95_latency_ms": _percentile(rows, "total_latency_ms", 95),
        "mean_guidance_call_latency_ms": _mean(model_rows, "guidance_latency_ms"),
        "p95_guidance_call_latency_ms": _percentile(model_rows, "guidance_latency_ms", 95),
        "b2_historical_retrieval_calls_reused": sum(
            row["b2_historical_retrieval_calls"] for row in rows
        ),
        "b2_historical_bridge_calls_reused": sum(
            row["b2_historical_bridge_calls"] for row in rows
        ),
        "b2_historical_bridge_input_tokens_reused": sum(
            row["b2_historical_bridge_input_tokens"] or 0 for row in rows
        ),
        "b2_historical_bridge_output_tokens_reused": sum(
            row["b2_historical_bridge_output_tokens"] or 0 for row in rows
        ),
        "b2_historical_retrieval_latency_ms_reused": sum(
            row["b2_historical_retrieval_latency_ms"] for row in rows
        ),
        "b2_historical_bridge_latency_ms_reused": sum(
            row["b2_historical_bridge_latency_ms"] for row in rows
        ),
        "invented_evidence_aliases": sum(row["invented_evidence_alias_count"] for row in rows),
        "new_retrieval_calls": sum(row["new_retrieval_calls"] for row in rows),
        "new_bridge_calls": sum(row["new_bridge_calls"] for row in rows),
    }


def _paired_user_bootstrap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    users = sorted({row["user_id"] for row in rows})
    if len(users) != 20:
        raise ValueError("B4 paired bootstrap requires exactly 20 users")
    indexed = {(row["user_id"], row["action"], row["task_family"]): row for row in rows}
    families = ("T0", "T1", "T2")
    user_e2e = np.empty((20, 3, 3), dtype=np.float64)
    user_guidance = np.empty((20, 3, 2), dtype=np.float64)
    for user_index, user in enumerate(users):
        for action_index, action in enumerate(ACTION_ORDER):
            for family_index, family in enumerate(families):
                row = indexed.get((user, action, family))
                if row is None:
                    raise ValueError("B4 bootstrap found a missing paired user/action/task arm")
                user_e2e[user_index, action_index, family_index] = row[
                    "end_to_end_quality"
                ]
                if family_index > 0:
                    user_guidance[user_index, action_index, family_index - 1] = row[
                        "guideline_content_score"
                    ]
    user_action_e2e = user_e2e.mean(axis=2)
    user_action_guidance = user_guidance.mean(axis=2)
    user_oracle_3 = user_e2e.max(axis=1).mean(axis=1)
    user_oracle_2 = user_e2e[:, :2, :].max(axis=1).mean(axis=1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(users), size=(BOOTSTRAP_SAMPLES, len(users)))

    def interval(samples: np.ndarray, observed: float) -> dict[str, float]:
        return {
            "mean_delta": float(observed),
            "lower_95": float(np.percentile(samples, 2.5)),
            "upper_95": float(np.percentile(samples, 97.5)),
        }

    comparisons: dict[str, Any] = {}
    for left, right in ((1, 0), (2, 0), (2, 1)):
        name = f"{ACTION_ORDER[left]}_minus_{ACTION_ORDER[right]}"
        e2e_samples = (user_action_e2e[:, left] - user_action_e2e[:, right])[draws].mean(axis=1)
        guidance_samples = (
            user_action_guidance[:, left] - user_action_guidance[:, right]
        )[draws].mean(axis=1)
        comparisons[name] = {
            "e2e": interval(
                e2e_samples,
                float(user_action_e2e[:, left].mean() - user_action_e2e[:, right].mean()),
            ),
            "guidance_quality": interval(
                guidance_samples,
                float(
                    user_action_guidance[:, left].mean()
                    - user_action_guidance[:, right].mean()
                ),
            ),
        }
    fixed_samples = user_action_e2e[draws].mean(axis=1).max(axis=1)
    oracle3_samples = user_oracle_3[draws].mean(axis=1)
    oracle2_samples = user_oracle_2[draws].mean(axis=1)
    comparisons["ORACLE3_minus_BEST_FIXED"] = {
        "e2e": interval(
            oracle3_samples - fixed_samples,
            float(user_oracle_3.mean() - user_action_e2e.mean(axis=0).max()),
        )
    }
    comparisons["ORACLE2_minus_BEST_FIXED_OFF_STANDARD"] = {
        "e2e": interval(
            oracle2_samples - user_action_e2e[:, :2].mean(axis=0).max(),
            float(user_oracle_2.mean() - user_action_e2e[:, :2].mean(axis=0).max()),
        )
    }
    return {
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_SAMPLES,
        "resampling_unit": "user_id; each draw carries T0/T1/T2 and all action outcomes",
        "users": len(users),
        "comparisons": comparisons,
    }


def _oracle_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_case[row["case_id"]][row["action"]] = row
    counts: Counter[str] = Counter()
    users_by_action: dict[str, set[str]] = defaultdict(set)
    unique_wins = Counter()
    oracle_values: list[float] = []
    oracle2_values: list[float] = []
    for actions in by_case.values():
        best_value = max(actions[action]["end_to_end_quality"] for action in ACTION_ORDER)
        chosen = next(
            action for action in ACTION_ORDER
            if actions[action]["end_to_end_quality"] == best_value
        )
        counts[chosen] += 1
        users_by_action[chosen].add(actions[chosen]["user_id"])
        oracle_values.append(best_value)
        if (
            actions["STRONG"]["end_to_end_quality"]
            > max(actions["OFF"]["end_to_end_quality"], actions["STANDARD"]["end_to_end_quality"])
        ):
            unique_wins["STRONG"] += 1
        best_two = max(
            actions[action]["end_to_end_quality"] for action in ("OFF", "STANDARD")
        )
        oracle2_values.append(best_two)
    fixed = {
        action: statistics.fmean(
            row["end_to_end_quality"] for row in rows if row["action"] == action
        )
        for action in ACTION_ORDER
    }
    fixed2 = max(fixed["OFF"], fixed["STANDARD"])
    oracle3 = statistics.fmean(oracle_values)
    oracle2 = statistics.fmean(oracle2_values)
    best_fixed = max(ACTION_ORDER, key=lambda action: (fixed[action], -ACTION_ORDER.index(action)))
    diverse3 = all(counts[action] >= 6 and len(users_by_action[action]) >= 3 for action in ACTION_ORDER)
    two_candidate = (
        counts["OFF"] >= 6
        and counts["STANDARD"] >= 6
        and len(users_by_action["OFF"]) >= 3
        and len(users_by_action["STANDARD"]) >= 3
        and unique_wins["STRONG"] < 3
        and fixed["STRONG"] - fixed["STANDARD"] <= 0
    )
    return {
        "fixed_action_e2e": fixed,
        "best_fixed_action": best_fixed,
        "oracle3_e2e": oracle3,
        "oracle3_headroom": oracle3 - fixed[best_fixed],
        "oracle2_off_standard_e2e": oracle2,
        "oracle2_headroom": oracle2 - fixed2,
        "oracle3_action_counts": {action: counts[action] for action in ACTION_ORDER},
        "oracle3_users_per_action": {
            action: len(users_by_action[action]) for action in ACTION_ORDER
        },
        "strong_unique_oracle_wins": unique_wins["STRONG"],
        "action_diversity_gate": "PASS" if diverse3 else "FAIL",
        "two_action_candidate": two_candidate,
        "strong_collapse_rule": (
            fixed["STRONG"] - fixed["STANDARD"] <= 0 and unique_wins["STRONG"] < 3
        ),
        "oracle3_minus_oracle2": oracle3 - oracle2,
    }


def score_frozen_run(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    manifest_path: Path = DEFAULT_EXECUTION_MANIFEST_PATH,
    private_root: Path = DEFAULT_B4_PRIVATE_ROOT,
    report_json_path: Path = DEFAULT_REPORT_JSON,
    report_markdown_path: Path = DEFAULT_REPORT_MARKDOWN,
) -> dict[str, Any]:
    lock = _read_json(protocol_path)
    lock_sha = verify_protocol_lock(lock)
    verify_frozen_code(lock)
    frozen = load_verified_b2(lock)
    b3_identity = verify_b3_identity(lock)
    manifest, arms = _verify_manifest(
        lock=lock,
        manifest_path=manifest_path,
        private_root=private_root,
        protocol_path=protocol_path,
    )
    if manifest.get("b2_artifact_set_sha256") != frozen.artifact_set_sha256:
        raise ValueError("B4 run no longer matches the frozen B2 evidence set")

    # Privileged labels are deliberately opened only after the full B4 arm/ledger freeze passed.
    teachers = read_jsonl(TEACHER_PATH)
    teacher_by_id = {row["case_id"]: row for row in teachers}
    if len(teacher_by_id) != 60:
        raise ValueError("post-freeze teacher plane must have exactly 60 unique case IDs")
    contexts = {(row.case_id, row.action): row for row in frozen.arm_contexts}
    rows: list[dict[str, Any]] = []
    for arm in arms:
        teacher = teacher_by_id.get(arm["case_id"])
        if teacher is None:
            raise ValueError("post-freeze teacher identity differs from the runtime case set")
        context = contexts[(arm["case_id"], arm["action"])]
        if (
            arm.get("b2_source") != context.reuse_manifest_row()
            or arm.get("b2_retrieval_ranking_sha256") != context.retrieval_ranking_sha256
            or arm.get("b2_supplied_chunks_sha256") != context.supplied_chunks_sha256
        ):
            raise ValueError("B4 arm does not reuse its exact B2 evidence provenance")
        task_kind = arm["runtime_task_kind"]
        guidance_text = (
            arm["guidance_raw_response"]["text"]
            if arm.get("guidance_raw_response") is not None
            else ""
        )
        resolved, invented = extract_citations(guidance_text, arm["evidence_aliases"])
        if resolved != arm["resolved_citation_chunk_ids"] or invented != arm[
            "invented_evidence_aliases"
        ]:
            raise ValueError("B4 citation alias resolution is not reproducible")
        score = score_materialized_case(
            teacher=teacher,
            state_claims=arm["state_claims"],
            guidance_text=guidance_text,
            cited_chunk_ids=resolved,
            supplied_chunks=arm["evidence_chunks"],
        )
        expected_kind = {
            "T0": "STATE_ONLY",
            "T1": "GUIDANCE_ONLY",
            "T2": "STATE_AND_GUIDANCE",
        }[teacher["task_family"]]
        if task_kind != expected_kind:
            raise ValueError("runtime-visible task classifier disagrees with post-freeze task stratum")
        response = arm.get("guidance_raw_response") or {}
        rows.append(
            {
                "case_id": arm["case_id"],
                "user_id": teacher["user_id"],
                "task_family": teacher["task_family"],
                "action": arm["action"],
                **score,
                "guidance_finish_reason": response.get("finish_reason"),
                "guidance_input_tokens": response.get("input_tokens"),
                "guidance_output_tokens": response.get("output_tokens"),
                "guidance_latency_ms": arm["guidance_latency_ms"],
                "model_calls": arm["new_model_calls"],
                "input_tokens": arm["guidance_input_tokens"],
                "output_tokens": arm["guidance_output_tokens"],
                "token_usage_complete": (
                    arm["new_model_calls"] == 0
                    or (
                        isinstance(arm["guidance_input_tokens"], int)
                        and isinstance(arm["guidance_output_tokens"], int)
                    )
                ),
                "total_latency_ms": arm["total_latency_ms"],
                "invented_evidence_alias_count": len(invented),
                "b2_historical_retrieval_calls": arm["b2_historical_retrieval_calls"],
                "b2_historical_bridge_calls": arm["b2_historical_bridge_calls"],
                "b2_historical_bridge_input_tokens": arm[
                    "b2_historical_bridge_input_tokens"
                ],
                "b2_historical_bridge_output_tokens": arm[
                    "b2_historical_bridge_output_tokens"
                ],
                "b2_historical_retrieval_latency_ms": arm[
                    "b2_historical_retrieval_latency_ms"
                ],
                "b2_historical_bridge_latency_ms": arm[
                    "b2_historical_bridge_latency_ms"
                ],
                "new_retrieval_calls": arm["new_retrieval_calls"],
                "new_bridge_calls": arm["new_bridge_calls"],
            }
        )
    groups = {
        action: [row for row in rows if row["action"] == action]
        for action in ACTION_ORDER
    }
    matrix = {action: _metric_summary(group) for action, group in groups.items()}
    by_family = {
        family: {
            action: _metric_summary(
                [row for row in rows if row["task_family"] == family and row["action"] == action]
            )
            for action in ACTION_ORDER
        }
        for family in ("T0", "T1", "T2")
    }
    bootstrap = _paired_user_bootstrap(rows)
    oracle = _oracle_diagnostics(rows)
    execution_health = (
        manifest["guidance_calls_completed"] == 120
        and manifest["empty_outputs"] == 0
        and manifest["transport_failures"] == 0
        and manifest["finish_reason_length"] == 0
        and manifest["token_usage_complete"] is True
        and manifest["teacher_opened"] is False
        and manifest["202608_opened"] is False
    )
    comparisons = bootstrap["comparisons"]

    def positive_ci(metric: str, comparison: str) -> bool:
        row = comparisons[comparison][metric]
        return row["mean_delta"] > 0 and row["lower_95"] > 0

    standard_value = positive_ci("e2e", "STANDARD_minus_OFF")
    strong_value = positive_ci("e2e", "STRONG_minus_STANDARD")
    retrieval_value = positive_ci("e2e", "STANDARD_minus_OFF") or positive_ci(
        "e2e", "STRONG_minus_OFF"
    )
    oracle_headroom_pass = oracle["oracle3_headroom"] >= 0.05
    two_headroom_pass = oracle["oracle2_headroom"] >= 0.05
    if not execution_health:
        action_space = "OFF"
    elif oracle["action_diversity_gate"] == "PASS" and oracle_headroom_pass:
        action_space = "OFF|STANDARD|STRONG"
    elif oracle["two_action_candidate"] and two_headroom_pass:
        action_space = "OFF|STANDARD"
    elif standard_value:
        action_space = "OFF|STANDARD"
    else:
        action_space = "OFF"
    post_training_worth = bool(
        execution_health
        and retrieval_value
        and (oracle["action_diversity_gate"] == "PASS" or oracle["two_action_candidate"])
        and (oracle_headroom_pass or two_headroom_pass)
    )
    report = {
        "schema_version": "rag-e5-e5b4-counterfactual-report-v1",
        "status": "COMPLETE_FROZEN_HARNESS_NATIVE_COUNTERFACTUAL",
        "execution_id": lock["execution_id"],
        "protocol_lock_sha256": lock_sha,
        "execution_manifest_sha256": manifest["execution_manifest_sha256"],
        "inference_runtime": manifest["inference_runtime"],
        "task_set_sha256": lock["task_set_sha256"],
        "state_set_sha256": lock["state_set_sha256"],
        "b2_artifact_set_sha256": frozen.artifact_set_sha256,
        "b3_unchanged_identity": b3_identity,
        "cases": 60,
        "users": 20,
        "arms": 180,
        "guidance_calls": 120,
        "new_retrieval_calls": 0,
        "new_bridge_calls": 0,
        "teacher_opened_after_freeze": True,
        "202608_opened": False,
        "scoring": {
            "method": "existing deterministic anchor scorer; no LLM judge",
            "guideline_content": "existing required-anchor coverage over full plain-text guidance",
            "grounding": "fraction of required recommendations with at least one cited supplied chunk",
            "action_blind_case_scorer": True,
        },
        "fixed_action_matrix": matrix,
        "by_task_family": by_family,
        "paired_user_bootstrap": bootstrap,
        "oracle": oracle,
        "b2_retrieval_coverage_reused": {
            "source_report_sha256": sha256_file(B2_REPORT_PATH),
            "matrix": _read_json(B2_REPORT_PATH)["retrieval_target_matrix"],
            "new_retrieval_run": False,
        },
        "execution_health": {
            "guidance_calls_120_of_120": manifest["guidance_calls_completed"] == 120,
            "empty_outputs": manifest["empty_outputs"],
            "transport_failures": manifest["transport_failures"],
            "length_finishes": manifest["finish_reason_length"],
            "token_usage_complete": manifest["token_usage_complete"],
            "MEASUREMENT_HEALTH": "PASS" if execution_health else "FAIL",
        },
        "conditional_value": {
            "EXTERNAL_RETRIEVAL_CONDITIONAL_VALUE": "YES" if retrieval_value else "NO",
            "STANDARD_VALUE": "YES" if standard_value else "NO",
            "STRONG_INCREMENTAL_VALUE": "YES" if strong_value else "NO",
            "definitions": {
                "retrieval_value": "positive fixed-action E2E delta vs OFF with user-paired 95% CI lower bound > 0",
                "standard_value": "same gate for STANDARD vs OFF",
                "strong_incremental_value": "same gate for STRONG vs STANDARD",
            },
        },
        "recommended_action_space": action_space,
        "POST_TRAINING_DATA_WORTH_BUILDING": "YES" if post_training_worth else "NO",
        "next_stage": "E5-C0 Execution Policy Training-View Transfer"
        if post_training_worth
        else "STOP; no post-training dataset authorized by this result",
    }
    report["report_sha256"] = canonical_sha256(report)
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_markdown(report_markdown_path, report)
    return report


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    matrix = report["fixed_action_matrix"]
    lines = [
        "# RAG-E5-B4 CPU — Harness-Native Counterfactual Recovery",
        "",
        "State claims are deterministically materialized from frozen longitudinal packets; the same Qwen3-8B CPU guidance generator receives only the case question and the action-specific frozen B2 evidence. The scorer is deterministic and opens teacher labels only after all B4 artifacts pass their freeze checks.",
        "",
        f"- Protocol lock: `{report['protocol_lock_sha256']}`",
        f"- Frozen B2 artifact set: `{report['b2_artifact_set_sha256']}`",
        f"- Cases / users / arms / new guidance calls: {report['cases']} / {report['users']} / {report['arms']} / {report['guidance_calls']}",
        f"- New retrieval / bridge calls: {report['new_retrieval_calls']} / {report['new_bridge_calls']}",
        f"- Measurement health: **{report['execution_health']['MEASUREMENT_HEALTH']}**",
        "",
        "## Fixed-action matrix",
        "",
        "| Metric | OFF | STANDARD | STRONG |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("end_to_end_quality", "E2E quality"),
        ("content_only_quality", "Content-only quality"),
        ("guidance_quality", "Guidance quality (T1/T2)"),
        ("state_score", "State score (T0/T2)"),
        ("grounding_score", "Grounding (T1/T2)"),
        ("model_calls", "New model calls"),
        ("input_tokens", "Input tokens"),
        ("output_tokens", "Output tokens"),
        ("mean_latency_ms", "Mean guidance latency (ms)"),
        ("p95_latency_ms", "P95 guidance latency (ms)"),
        ("b2_historical_retrieval_calls_reused", "Historical B2 retrieval calls reused"),
        ("b2_historical_bridge_calls_reused", "Historical B2 bridge calls reused"),
        ("b2_historical_retrieval_latency_ms_reused", "Historical B2 retrieval latency sum (ms)"),
        ("b2_historical_bridge_latency_ms_reused", "Historical B2 bridge latency sum (ms)"),
        ("mean_guidance_call_latency_ms", "Mean latency per new guidance call (ms)"),
        ("p95_guidance_call_latency_ms", "P95 latency per new guidance call (ms)"),
    ):
        lines.append(
            f"| {label} | " + " | ".join(_fmt(matrix[action].get(key)) for action in ACTION_ORDER) + " |"
        )
    lines.extend(
        [
            "",
            "## By task family",
            "",
            "| Family | Action | E2E | Content-only | State | Guidance | Grounding | Calls |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for family in ("T0", "T1", "T2"):
        for action in ACTION_ORDER:
            row = report["by_task_family"][family][action]
            lines.append(
                f"| {family} | {action} | {_fmt(row['end_to_end_quality'])} | "
                f"{_fmt(row['content_only_quality'])} | {_fmt(row['state_score'])} | "
                f"{_fmt(row['guidance_quality'])} | {_fmt(row['grounding_score'])} | "
                f"{row['model_calls']} |"
            )
    lines.extend(
        [
            "",
            "## Paired user bootstrap",
            "",
            "10,000 user-level paired resamples; each user's T0/T1/T2 outcomes remain together.",
            "",
            "| Comparison | E2E Δ [95% CI] | Guidance-quality Δ [95% CI] |",
            "|---|---:|---:|",
        ]
    )
    for name in ("STANDARD_minus_OFF", "STRONG_minus_OFF", "STRONG_minus_STANDARD"):
        row = report["paired_user_bootstrap"]["comparisons"][name]
        e2e = row["e2e"]
        guidance = row["guidance_quality"]
        lines.append(
            f"| {name.replace('_minus_', ' − ')} | {_fmt(e2e['mean_delta'])} "
            f"[{_fmt(e2e['lower_95'])}, {_fmt(e2e['upper_95'])}] | "
            f"{_fmt(guidance['mean_delta'])} [{_fmt(guidance['lower_95'])}, "
            f"{_fmt(guidance['upper_95'])}] |"
        )
    oracle = report["oracle"]
    lines.extend(
        [
            "",
            "## Oracle and decision gates",
            "",
            f"- Best fixed action: **{oracle['best_fixed_action']}** ({_fmt(oracle['fixed_action_e2e'][oracle['best_fixed_action']])}).",
            f"- Oracle-3 E2E / headroom: {_fmt(oracle['oracle3_e2e'])} / {_fmt(oracle['oracle3_headroom'])}.",
            f"- Oracle-2 (OFF|STANDARD) E2E / headroom: {_fmt(oracle['oracle2_off_standard_e2e'])} / {_fmt(oracle['oracle2_headroom'])}.",
            f"- Tie-broken oracle counts: `{json.dumps(oracle['oracle3_action_counts'], sort_keys=True)}`.",
            f"- Action diversity: **{oracle['action_diversity_gate']}**; two-action candidate: **{oracle['two_action_candidate']}**.",
            f"- Strong collapse rule: **{oracle['strong_collapse_rule']}**; unique STRONG oracle wins: {oracle['strong_unique_oracle_wins']}.",
            f"- Recommended action space: **{report['recommended_action_space']}**.",
            f"- External retrieval conditional value: **{report['conditional_value']['EXTERNAL_RETRIEVAL_CONDITIONAL_VALUE']}**.",
            f"- Post-training data worth building: **{report['POST_TRAINING_DATA_WORTH_BUILDING']}**.",
            "",
            "## Measurement boundaries",
            "",
            "T0 is deterministic state-only materialization and makes zero model calls. T1/T2 guidance outputs are unstructured plain text; `[E1]`–`[E5]` are resolved by the runtime to the original frozen B2 chunk IDs and provenance. Unknown aliases are ignored and counted. No new retrieval or bridge generation was run. The scorer checks the existing deterministic rubric anchors and citations; it does not assess unlisted medical facts or use an LLM judge. Historical B2 retrieval coverage is reused as a frozen diagnostic, not recomputed.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


__all__ = ["score_frozen_run", "score_materialized_case"]
