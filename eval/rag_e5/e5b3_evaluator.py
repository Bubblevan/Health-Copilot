"""Measurement-gated, post-freeze evaluation for E5-B3 Reader V2 recovery."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from eval.rag_e5.counterfactual import ACTION_ORDER, verify_complete_arm
from eval.rag_e5.e5b2_evaluator import (
    _action_diagnostics,
    _bootstrap,
    _dependency_gate,
    _failure_counts,
    _metric_summary,
    _read_jsonl,
    _target_diagnostics,
    _write_json,
)
from eval.rag_e5.e5b3_reader import ReaderStateView, parse_reader_v2_output
from eval.rag_e5.e5b3_recovery import (
    DEFAULT_B2_ARTIFACT_ROOT,
    DEFAULT_CORPUS_ROOT,
    DEFAULT_PRIVATE_ROOT,
    b3_run_id,
    canonical_sha256,
    load_and_verify_b2_inputs,
    read_json,
    reader_v2_to_scorer,
    sha256_file,
)
from eval.rag_e5.overlay import score_case

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = ROOT / "runs/rag_e5/e5b3_recovery_lock.json"
DEFAULT_MANIFEST = ROOT / "runs/rag_e5/e5b3_execution_manifest.json"
DEFAULT_REPORT_JSON = ROOT / "runs/rag_e5/e5b3_measurement_recovery_report.json"
DEFAULT_REPORT_MD = ROOT / "docs/research/rag_e5/e5b3_measurement_recovery.md"
DEFAULT_B2_REPORT = ROOT / "runs/rag_e5/e5b2_counterfactual_report.json"
DEFAULT_PRIVATE_LEDGER = Path(
    r"D:\MyLab\Jianli\external\rag_e5\e5b3\e5b3_private_score_ledger.json"
)
DEFAULT_B2_CALL_LEDGER = Path(r"D:\MyLab\Jianli\external\rag_e5\e5b2\call_ledger.jsonl")


def _verify_source_lock(lock: dict[str, Any]) -> None:
    for relative, expected in lock.get("source_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"B3 locked source changed before scoring: {relative}")


def _verify_git_frozen(paths: Sequence[Path]) -> None:
    relative = [str(path.relative_to(ROOT)).replace("\\", "/") for path in paths]
    result = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--", *relative],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise ValueError("B3 lock, execution manifest, and locked code must be committed before scoring")


def _state_contract(output: Mapping[str, Any] | None, view: ReaderStateView | None) -> dict[str, Any]:
    expected = set(view.recent_measurement_trends) if view is not None else set()
    facts = output.get("state_facts", []) if output is not None else []
    observed: list[tuple[str, str]] = []
    for fact in facts:
        if isinstance(fact, Mapping):
            field, value = fact.get("field"), fact.get("value")
            if isinstance(field, str) and isinstance(value, str):
                observed.append((field, value))
    matches = bool(observed) and bool(expected) and all(item in expected for item in observed)
    return {
        "has_state_fact": bool(observed),
        "all_emitted_facts_match_runtime_projection": matches,
        "observed_fact_count": len(observed),
        "matching_fact_count": sum(item in expected for item in observed),
    }


def _measurement_rows(
    *,
    frozen: Any,
    b2_artifact_root: Path,
    b2_lock_sha256: str,
    b3_arms: Mapping[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_by_id = {row["case_id"]: row for row in frozen.runtime_cases}
    stateful_ids = {
        row["case_id"] for row in frozen.runtime_cases if row.get("state_packet_ref")
    }
    if len(stateful_ids) != 40:
        raise ValueError("B2 runtime state-bearing case count changed from the frozen 40")

    def collect_b3() -> dict[str, Any]:
        all_valid = 0
        lengths = 0
        state_arm_count = 0
        present = 0
        contract_matches = 0
        fact_count = 0
        fact_matches = 0
        by_action: dict[str, dict[str, int]] = {}
        for action in ACTION_ORDER:
            action_arms = [arm for (case_id, arm_action), arm in b3_arms.items() if arm_action == action]
            action_valid = sum(bool(arm["reader_json_valid"]) for arm in action_arms)
            action_lengths = sum(arm.get("reader_finish_reason") == "length" for arm in action_arms)
            by_action[action] = {
                "arms": len(action_arms),
                "json_valid": action_valid,
                "finish_reason_length": action_lengths,
            }
        for (case_id, action), arm in b3_arms.items():
            all_valid += int(bool(arm["reader_json_valid"]))
            lengths += int(arm.get("reader_finish_reason") == "length")
            if case_id not in stateful_ids:
                continue
            state_arm_count += 1
            case = case_by_id[case_id]
            packet = frozen.state_packets_by_ref[case["state_packet_ref"]]
            view = ReaderStateView.from_packet(packet)
            state_result = _state_contract(arm.get("reader_parsed"), view)
            present += int(state_result["has_state_fact"])
            contract_matches += int(state_result["all_emitted_facts_match_runtime_projection"])
            fact_count += state_result["observed_fact_count"]
            fact_matches += state_result["matching_fact_count"]
        return {
            "arms": len(b3_arms),
            "reader_json_valid": all_valid,
            "reader_json_valid_rate": all_valid / len(b3_arms),
            "finish_reason_length": lengths,
            "finish_reason_length_rate": lengths / len(b3_arms),
            "by_action": by_action,
            "stateful_arms": state_arm_count,
            "state_fact_present_arms": present,
            "state_contract_matching_arms": contract_matches,
            "state_contract_match_rate": contract_matches / state_arm_count,
            "state_facts": fact_count,
            "state_facts_matching_runtime_projection": fact_matches,
            "state_fact_projection_match_rate": fact_matches / fact_count if fact_count else None,
        }

    b3_measurement = collect_b3()
    b2_valid = 0
    b2_lengths = 0
    b2_state_present = 0
    b2_state_matches = 0
    b2_state_facts = 0
    b2_state_fact_matches = 0
    b2_reader_arms = 0
    for context in frozen.arm_contexts:
        path = (
            b2_artifact_root
            / "arms"
            / context.case_id
            / context.action
            / f"{context.b2_run_id}.json"
        )
        source = verify_complete_arm(
            path,
            expected_run_id=context.b2_run_id,
            expected_lock_sha256=b2_lock_sha256,
        )
        b2_reader_arms += 1
        b2_valid += int(bool(source.get("reader_json_valid")))
        raw = source.get("reader_raw_response") or {}
        b2_lengths += int(raw.get("finish_reason") == "length")
        if context.case_id in stateful_ids:
            case = case_by_id[context.case_id]
            packet = frozen.state_packets_by_ref[case["state_packet_ref"]]
            view = ReaderStateView.from_packet(packet)
            state_result = _state_contract(source.get("reader_parsed"), view)
            b2_state_present += int(state_result["has_state_fact"])
            b2_state_matches += int(state_result["all_emitted_facts_match_runtime_projection"])
            b2_state_facts += state_result["observed_fact_count"]
            b2_state_fact_matches += state_result["matching_fact_count"]
    b2_measurement = {
        "arms": b2_reader_arms,
        "reader_json_valid": b2_valid,
        "reader_json_valid_rate": b2_valid / b2_reader_arms,
        "finish_reason_length": b2_lengths,
        "finish_reason_length_rate": b2_lengths / b2_reader_arms,
        "stateful_arms": len(stateful_ids) * len(ACTION_ORDER),
        "state_fact_present_arms": b2_state_present,
        "state_contract_matching_arms": b2_state_matches,
        "state_contract_match_rate": b2_state_matches / (len(stateful_ids) * len(ACTION_ORDER)),
        "state_facts": b2_state_facts,
        "state_facts_matching_runtime_projection": b2_state_fact_matches,
        "state_fact_projection_match_rate": (
            b2_state_fact_matches / b2_state_facts if b2_state_facts else None
        ),
        "comparison_scope": "reader measurement only; no B2 teacher score recomputed",
    }
    return b2_measurement, b3_measurement


def _load_frozen_run(
    *,
    lock_path: Path,
    manifest_path: Path,
    private_root: Path,
    corpus_root: Path,
    b2_artifact_root: Path,
    b2_call_ledger: Path,
    b2_report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, dict[str, Any]], dict[str, Any]]:
    lock = read_json(lock_path)
    lock_sha = lock.get("b3_recovery_lock_sha256")
    lock_body = {key: value for key, value in lock.items() if key != "b3_recovery_lock_sha256"}
    if not isinstance(lock_sha, str) or canonical_sha256(lock_body) != lock_sha:
        raise ValueError("B3 recovery lock self-hash mismatch")
    if lock.get("status") != "FROZEN_BEFORE_B3_READER_CALLS":
        raise ValueError("B3 recovery protocol lock is not frozen")
    _verify_source_lock(lock)
    qualification = read_json(ROOT / "runs/rag_e5/e5b3_reader_qualification.json")
    if (
        qualification.get("reader_contract_qualified") is not True
        or qualification.get("qualification_report_sha256")
        != lock.get("qualification_report_sha256")
    ):
        raise ValueError("B3 reader qualification identity mismatch")
    if not b2_call_ledger.is_file() or sha256_file(b2_call_ledger) != lock.get(
        "b2_call_ledger_sha256"
    ):
        raise ValueError("B2 call ledger changed after B3 protocol freeze")
    if not b2_report_path.is_file() or sha256_file(b2_report_path) != lock.get("b2_report_sha256"):
        raise ValueError("frozen B2 result report changed after B3 protocol freeze")

    frozen = load_and_verify_b2_inputs(
        private_root=private_root,
        corpus_root=corpus_root,
        b2_artifact_root=b2_artifact_root,
    )
    for field, actual in (
        ("b2_lock_file_sha256", frozen.b2_lock_file_sha256),
        ("b2_execution_manifest_file_sha256", frozen.b2_execution_manifest_file_sha256),
        ("b2_artifact_set_sha256", frozen.artifact_set_sha256),
        ("b2_reuse_manifest_sha256", frozen.reuse_manifest_sha256),
    ):
        if lock.get(field) != actual:
            raise ValueError(f"B3 lock no longer matches the frozen B2 source: {field}")

    manifest = read_json(manifest_path)
    manifest_sha = manifest.get("execution_manifest_sha256")
    manifest_body = {key: value for key, value in manifest.items() if key != "execution_manifest_sha256"}
    if not isinstance(manifest_sha, str) or canonical_sha256(manifest_body) != manifest_sha:
        raise ValueError("B3 execution manifest self-hash mismatch")
    if (
        manifest.get("status") != "ALL_B3_READER_ARMS_FROZEN_BEFORE_SCORING"
        or manifest.get("b3_recovery_lock_sha256") != lock_sha
        or manifest.get("b3_recovery_lock_file_sha256") != sha256_file(lock_path)
        or manifest.get("completed_arms") != 180
        or manifest.get("expected_arms") != 180
        or manifest.get("reader_calls") != 180
        or manifest.get("new_reader_calls") != 180
        or manifest.get("new_retrieval_calls") != 0
        or manifest.get("new_bridge_calls") != 0
        or manifest.get("teacher_opened") is not False
        or manifest.get("202608_opened") is not False
        or manifest.get("b2_results_rescored") is not False
    ):
        raise ValueError("B3 execution manifest violates the frozen recovery contract")
    for key in (
        "b2_artifact_set_sha256",
        "b2_reuse_manifest_sha256",
        "qualification_lock_sha256",
        "qualification_report_sha256",
        "reader_schema_sha256",
        "reader_prompt_sha256",
        "reader_state_projection_sha256",
        "reader_output_budget",
        "model",
        "llama_cpp",
    ):
        if manifest.get(key) != lock.get(key):
            raise ValueError(f"B3 execution manifest identity mismatch: {key}")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 180:
        raise ValueError("B3 execution manifest must freeze exactly 180 arms")
    if canonical_sha256(artifacts) != manifest.get("artifact_set_sha256"):
        raise ValueError("B3 arm artifact-set hash mismatch")
    if canonical_sha256([row["run_id"] for row in artifacts]) != manifest.get(
        "run_id_set_sha256"
    ):
        raise ValueError("B3 run-ID set hash mismatch")

    source_rows = {
        (row["case_id"], row["action"]): row for row in frozen.reuse_manifest_rows
    }
    expected: list[tuple[str, str, str]] = []
    for case_id in sorted(row["case_id"] for row in frozen.runtime_cases):
        for action in ACTION_ORDER:
            expected.append((case_id, action, b3_run_id(case_id, action, lock_sha)))
    observed = [(row.get("case_id"), row.get("action"), row.get("run_id")) for row in artifacts]
    if observed != expected:
        raise ValueError("B3 artifact order, task identity, or lock-derived run IDs differ")

    b3_arms: dict[tuple[str, str], dict[str, Any]] = {}
    for row in artifacts:
        case_id, action = row["case_id"], row["action"]
        source = source_rows[(case_id, action)]
        if row.get("b2_source_arm_file_sha256") != source["source_arm_file_sha256"] or row.get(
            "b2_source_completion_sha256"
        ) != source["source_completion_sha256"]:
            raise ValueError("B3 arm does not bind to its exact B2 source artifact")
        path = b2_artifact_root / "arms" / case_id / action / f"{source['b2_run_id']}.json"
        # The source context was already verified above; this explicit hash binds the B3 reuse row.
        source_arm = verify_complete_arm(
            path, expected_run_id=source["b2_run_id"], expected_lock_sha256=frozen.lock[
                "counterfactual_lock_sha256"
            ]
        )
        b3_path = Path(manifest["call_ledger_path_external"]).parent / row["path"]
        if sha256_file(b3_path) != row.get("file_sha256"):
            raise ValueError(f"B3 arm artifact changed after freeze: {case_id}/{action}")
        arm = verify_complete_arm(
            b3_path, expected_run_id=row["run_id"], expected_lock_sha256=lock_sha
        )
        if (
            arm.get("status") != "COMPLETE"
            or arm.get("b2_source") != source
            or arm.get("b2_reader_output_reused") is not False
            or arm.get("b2_retrieval_and_bridge_artifacts_reused") is not True
            or arm.get("new_retrieval_calls") != 0
            or arm.get("new_bridge_calls") != 0
            or arm.get("new_reader_calls") != 1
            or arm.get("supplied_chunks") != source_arm.get("supplied_chunks")
            or arm.get("retrieval", {}).get("ranking")
            != source_arm.get("retrieval", {}).get("ranking")
        ):
            raise ValueError("B3 arm altered evidence or violated the no-retrieval/no-bridge contract")
        response = arm.get("reader_raw_response") or {}
        if arm.get("reader_raw_response_sha256") != canonical_sha256(response):
            raise ValueError("B3 raw reader response hash mismatch")
        parsed, valid = parse_reader_v2_output(response.get("text", ""))
        if arm.get("reader_json_valid") is not valid or arm.get("reader_parsed") != parsed:
            raise ValueError("B3 reader parse record is not reproducible from its frozen raw output")
        b3_arms[(case_id, action)] = arm

    ledger_path = Path(manifest["call_ledger_path_external"])
    if sha256_file(ledger_path) != manifest.get("call_ledger_sha256"):
        raise ValueError("B3 append-only reader call ledger changed after freeze")
    call_rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(call_rows) != 360:
        raise ValueError("B3 call ledger must contain exactly 180 STARTED and 180 COMPLETED rows")
    starts = [row for row in call_rows if row.get("status") == "STARTED"]
    completions = [row for row in call_rows if row.get("status") == "COMPLETED"]
    if len(starts) != 180 or len(completions) != 180:
        raise ValueError("B3 call ledger has a missing or repeated model call")
    started_by_id = {row.get("call_id"): row for row in starts}
    complete_by_id = {row.get("call_id"): row for row in completions}
    if len(started_by_id) != 180 or set(started_by_id) != set(complete_by_id):
        raise ValueError("B3 call ledger call IDs are incomplete or duplicated")
    for call_id, started in started_by_id.items():
        done = complete_by_id[call_id]
        if any(started.get(key) != done.get(key) for key in (
            "case_id", "action", "run_id", "prompt_sha256", "call_ordinal"
        )):
            raise ValueError("B3 call ledger completion does not match its STARTED record")

    b2_report = read_json(b2_report_path)
    if b2_report.get("counterfactual_lock_sha256") != frozen.lock.get(
        "counterfactual_lock_sha256"
    ):
        raise ValueError("B2 frozen result report refers to a different protocol")
    _verify_git_frozen(
        [lock_path, manifest_path, *(ROOT / relative for relative in lock["source_sha256"])]
    )
    return lock, manifest, frozen, b3_arms, b2_report


def _score_b3_rows(
    *,
    arms: Mapping[tuple[str, str], Mapping[str, Any]],
    teachers: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    teacher_by_id = {row["case_id"]: row for row in teachers}
    chunk_by_id = {row["chunk_id"]: row for row in chunks}
    if len(teacher_by_id) != 60:
        raise ValueError("frozen B2 teacher artifact must contain 60 unique cases")
    rows: list[dict[str, Any]] = []
    for (case_id, action), arm in sorted(
        arms.items(), key=lambda item: (item[0][0], ACTION_ORDER.index(item[0][1]))
    ):
        teacher = teacher_by_id.get(case_id)
        if teacher is None:
            raise ValueError("teacher identity does not match the frozen runtime case set")
        parsed = arm["reader_parsed"] if arm["reader_json_valid"] else None
        projection = reader_v2_to_scorer(parsed or {})
        supplied = arm.get("supplied_chunks", [])
        scores = score_case(teacher=teacher, reader_output=projection, supplied_chunks=supplied)
        family = teacher["task_family"]
        retrieval_diagnostics = None
        failures: list[str] = []
        if not arm["reader_json_valid"]:
            failures.append("JSON_FAILURE")
        if arm.get("reader_finish_reason") == "length":
            failures.append("LENGTH_FAILURE")
        if teacher.get("expected_state_fields") and scores["state_score"] < 1.0:
            failures.append("STATE_MISS")
        if teacher.get("required_external_source_id"):
            if action != "OFF":
                retrieval_diagnostics = _target_diagnostics(
                    arm["retrieval"]["ranking"],
                    required_source=teacher["required_external_source_id"],
                    required_recommendations=set(teacher.get("required_recommendation_ids", [])),
                    chunk_by_id=chunk_by_id,
                )
            if retrieval_diagnostics is not None:
                if not retrieval_diagnostics["required_recommendation_hit_at_100"]:
                    failures.append("RETRIEVAL_MISS")
                elif not retrieval_diagnostics["required_recommendation_hit_at_5"]:
                    failures.append("RANKING_MISS")
                if (
                    retrieval_diagnostics["required_recommendation_hit_at_5"]
                    and scores["guideline_content_score"] < 1.0
                ):
                    failures.append("READER_MISS")
                if scores["guideline_content_score"] == 1.0 and scores["grounding_score"] == 0.0:
                    failures.append("GROUNDING_MISS")
        if arm.get("bridge_fallback_original_query"):
            failures.append("BRIDGE_FALLBACK")
        rows.append(
            {
                "case_id": case_id,
                "user_id": teacher["user_id"],
                "task_family": family,
                "action": action,
                **scores,
                "reader_json_valid": arm["reader_json_valid"],
                "reader_finish_reason": arm.get("reader_finish_reason"),
                "reader_input_tokens": arm.get("reader_input_tokens"),
                "reader_output_tokens": arm.get("reader_output_tokens"),
                "bridge_input_tokens": arm.get("bridge_input_tokens"),
                "bridge_output_tokens": arm.get("bridge_output_tokens"),
                "retrieval_latency_ms": arm["retrieval_latency_ms"],
                "bridge_latency_ms": arm["bridge_latency_ms"],
                "reader_latency_ms": arm["reader_latency_ms"],
                "total_latency_ms": arm["total_latency_ms"],
                "retrieval_calls": arm["retrieval_calls"],
                "bridge_calls": arm["bridge_call_count"],
                "reader_calls": arm["reader_call_count"],
                "invented_citation_count": len(arm.get("invented_citations", [])),
                "retrieval_diagnostics": retrieval_diagnostics,
                "failure_flags": failures,
                "cost_basis": "COMPOSED_COMPONENT_COST",
            }
        )
    return rows


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# RAG-E5-B3 — Reader Contract Measurement Recovery",
        "",
        "This is a new reader-measurement protocol over exact frozen B2 task, state, retrieval, and bridge artifacts. B2 is not rescored or overwritten.",
        "",
        f"- B3 recovery lock: `{report['b3_recovery_lock_sha256']}`",
        f"- Execution manifest: `{report.get('execution_manifest_sha256', 'not frozen')}`",
        f"- B2 artifact-set SHA-256: `{report['b2_artifact_set_sha256']}`",
        f"- Cases / arms / new reader calls: {report['cases']} / {report['arms']} / {report['new_reader_calls']}",
        f"- New retrieval / bridge calls: {report['new_retrieval_calls']} / {report['new_bridge_calls']}",
        f"- Measurement recovery gate: **{report['MEASUREMENT_RECOVERY_GATE']}**",
        "",
        "## Reader measurement comparison (no B2 quality rescore)",
        "",
        "| Measurement | B2 | B3 |",
        "|---|---:|---:|",
    ]
    for key, label in (
        ("reader_json_valid_rate", "JSON-valid rate"),
        ("finish_reason_length_rate", "finish_reason=length rate"),
        ("state_contract_match_rate", "state facts exactly match runtime projection / stateful arms"),
    ):
        left = report["reader_measurement_comparison"]["B2"][key]
        right = report["reader_measurement_comparison"]["B3"][key]
        lines.append(f"| {label} | {left:.3f} | {right:.3f} |")
    if report["MEASUREMENT_RECOVERY_GATE"] == "PASS":
        matrix = report["fixed_action_matrix"]
        lines.extend(
            [
                "",
                "## B3 scored quality",
                "",
                "| Metric | OFF | STANDARD | STRONG | ORACLE |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for key, title in (
            ("end_to_end_quality", "E2E quality"),
            ("content_only_quality", "Content-only quality"),
            ("guideline_content_score", "Guideline content"),
            ("grounding_score", "Grounding"),
        ):
            values = [matrix[action].get(key) for action in ACTION_ORDER]
            values.append(report["oracle"].get(key))
            lines.append(
                f"| {title} | "
                + " | ".join("—" if value is None else f"{value:.3f}" for value in values)
                + " |"
            )
        lines.extend(
            [
                "",
                f"- Action diversity: **{report['ACTION_DIVERSITY_GATE']}**; oracle headroom: {report['oracle']['headroom']:.3f} (**{report['ORACLE_HEADROOM_GATE']}**).",
                f"- T2 dependency: **{report['T2_DEPENDENCY_GATE']}**.",
                f"- E5-C worth investigating: **{report['E5C_RESEARCH_QUESTION_WORTH_CONTINUING']}**; E5-C started: **NO**.",
                "- Cost figures are composed frozen B2 retrieval/bridge plus new B3 reader component costs, not a single-run wall-clock measurement.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "The measurement gate failed; no teacher artifact was opened and no conditional-value or quality result is reported.",
            ]
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "B2 outcome artifacts remain immutable. This report does not establish that B3 proves B2 wrong; it reports a separately frozen reader-contract measurement.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def score_recovery(
    *,
    lock_path: Path = DEFAULT_LOCK,
    manifest_path: Path = DEFAULT_MANIFEST,
    private_root: Path = DEFAULT_PRIVATE_ROOT,
    corpus_root: Path = DEFAULT_CORPUS_ROOT,
    b2_artifact_root: Path = DEFAULT_B2_ARTIFACT_ROOT,
    b2_call_ledger: Path = DEFAULT_B2_CALL_LEDGER,
    b2_report_path: Path = DEFAULT_B2_REPORT,
    private_score_path: Path = DEFAULT_PRIVATE_LEDGER,
    report_json_path: Path = DEFAULT_REPORT_JSON,
    report_markdown_path: Path = DEFAULT_REPORT_MD,
    seed: int = 20260930,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    if report_json_path.exists():
        raise FileExistsError(f"B3 result report is immutable: {report_json_path}")
    lock, manifest, frozen, arms, b2_report = _load_frozen_run(
        lock_path=lock_path,
        manifest_path=manifest_path,
        private_root=private_root,
        corpus_root=corpus_root,
        b2_artifact_root=b2_artifact_root,
        b2_call_ledger=b2_call_ledger,
        b2_report_path=b2_report_path,
    )
    b2_measurement, b3_measurement = _measurement_rows(
        frozen=frozen,
        b2_artifact_root=b2_artifact_root,
        b2_lock_sha256=frozen.lock["counterfactual_lock_sha256"],
        b3_arms=arms,
    )
    measurement_pass = (
        b3_measurement["reader_json_valid_rate"] >= 0.95
        and b3_measurement["finish_reason_length"] == 0
    )
    report: dict[str, Any] = {
        "schema_version": "rag-e5-e5b3-measurement-recovery-report-v1",
        "status": "MEASUREMENT_RECOVERY_COMPLETE" if measurement_pass else "MEASUREMENT_RECOVERY_GATE_FAIL",
        "b3_recovery_lock_sha256": lock["b3_recovery_lock_sha256"],
        "execution_manifest_sha256": manifest["execution_manifest_sha256"],
        "qualification_report_sha256": lock["qualification_report_sha256"],
        "task_set_sha256": lock["task_set_sha256"],
        "state_packet_set_sha256": lock["state_packet_set_sha256"],
        "b2_artifact_set_sha256": lock["b2_artifact_set_sha256"],
        "b2_reuse_manifest_sha256": lock["b2_reuse_manifest_sha256"],
        "scorer_version": lock["scorer_version"],
        "scorer_module_sha256": lock["scorer_module_sha256"],
        "scoring_contract_sha256": lock["scoring_contract_sha256"],
        "cases": 60,
        "arms": 180,
        "new_reader_calls": 180,
        "new_retrieval_calls": 0,
        "new_bridge_calls": 0,
        "reader_output_budget": lock["reader_output_budget"],
        "reader_model": lock["model"],
        "reader_measurement_comparison": {"B2": b2_measurement, "B3": b3_measurement},
        "measurement_recovery_gate": "PASS" if measurement_pass else "FAIL",
        "MEASUREMENT_RECOVERY_GATE": "PASS" if measurement_pass else "FAIL",
        "b2_results_rescored": False,
        "b2_artifacts_modified": False,
        "202608_opened": False,
        "E5C_STARTED": False,
    }
    if not measurement_pass:
        report.update(
            {
                "teacher_opened": False,
                "conditional_value_evaluated": False,
                "ACTION_DIVERSITY_GATE": "NOT_EVALUATED",
                "ORACLE_HEADROOM_GATE": "NOT_EVALUATED",
                "T2_DEPENDENCY_GATE": "NOT_EVALUATED",
                "E5C_RESEARCH_QUESTION_WORTH_CONTINUING": False,
                "reason": "Reader JSON validity below 95% or at least one length finish; teacher was not opened.",
            }
        )
        report["report_sha256"] = canonical_sha256(report)
        _write_json(report_json_path, report)
        _write_markdown(report_markdown_path, report)
        return report

    # No teacher bytes are read until the complete 180-arm manifest and the measurement gate pass.
    teacher_path = private_root / "teacher_cases.jsonl"
    if sha256_file(teacher_path) != frozen.lock.get("teacher_cases_file_sha256"):
        raise ValueError("B2 teacher artifact differs from the pre-execution frozen hash")
    teachers = _read_jsonl(teacher_path)
    chunks = _read_jsonl(corpus_root / "chunks.jsonl")
    rows = _score_b3_rows(arms=arms, teachers=teachers, chunks=chunks)
    grouped = {action: [row for row in rows if row["action"] == action] for action in ACTION_ORDER}
    per_family = {
        family: {
            action: _metric_summary(
                [row for row in grouped[action] if row["task_family"] == family]
            )
            for action in ACTION_ORDER
        }
        for family in ("T0", "T1", "T2")
    }
    action_summary = {action: _metric_summary(grouped[action]) for action in ACTION_ORDER}
    diagnostics = _action_diagnostics(rows)
    bootstrap = _bootstrap(rows, seed=seed, samples=bootstrap_samples)
    failure_counts = _failure_counts(rows)
    t2_dependency = _dependency_gate(teachers, chunks)
    retrieval_coverage = b2_report["retrieval_target_matrix"]
    for action in ("STANDARD", "STRONG"):
        target = retrieval_coverage[action]
        if (
            target.get("eligible_cases") != 40
            or target.get("required_source_hit_at_5", {}).get("hits") != 40
            or target.get("required_recommendation_hit_at_5", {}).get("hits") != 40
        ):
            raise ValueError("preexisting B2 retrieval coverage no longer matches the frozen gate")

    private_rows = {
        "schema_version": "rag-e5-e5b3-private-score-ledger-v1",
        "b3_recovery_lock_sha256": lock["b3_recovery_lock_sha256"],
        "execution_manifest_sha256": manifest["execution_manifest_sha256"],
        "cases": rows,
        "oracle_by_case": {row["case_id"]: row["action"] for row in diagnostics["oracle_rows"]},
    }
    if private_score_path.exists():
        raise FileExistsError(f"B3 private score ledger is immutable: {private_score_path}")
    _write_json(private_score_path, private_rows)
    private_scores_sha = sha256_file(private_score_path)
    unresolved = sum(
        max(
            next(row["end_to_end_quality"] for row in grouped[action] if row["case_id"] == case_id)
            for action in ACTION_ORDER
        ) < 1.0
        for case_id in {row["case_id"] for row in rows}
    )
    oracle_rows = diagnostics["oracle_rows"]
    oracle_summary = {
        "end_to_end_quality": diagnostics["oracle_e2e"],
        "content_only_quality": fmean(row["content_only_quality"] for row in oracle_rows),
        "guideline_content_score": fmean(
            row["guideline_content_score"] for row in oracle_rows
            if row["task_family"] in {"T1", "T2"}
        ),
        "grounding_score": fmean(
            row["grounding_score"] for row in oracle_rows
            if row["task_family"] in {"T1", "T2"}
        ),
        "best_fixed_action": diagnostics["best_fixed_action"],
        "best_fixed_e2e": diagnostics["fixed_action_means"][diagnostics["best_fixed_action"]],
        "headroom": diagnostics["oracle_headroom"],
        "action_counts": diagnostics["oracle_counts"],
        "users_per_action": diagnostics["oracle_users_per_action"],
        "unresolved_cases_max_quality_below_1": unresolved,
    }
    gates = {
        "ACTION_DIVERSITY_GATE": diagnostics["action_diversity_gate"],
        "ORACLE_HEADROOM_GATE": diagnostics["oracle_headroom_gate"],
        "T2_DEPENDENCY_GATE": t2_dependency["gate"],
    }
    authorized = all(value == "PASS" for value in gates.values())
    report.update(
        {
            "status": "COMPLETE_FROZEN_READER_MEASUREMENT_RECOVERY",
            "teacher_opened": True,
            "teacher_cases_file_sha256": frozen.lock["teacher_cases_file_sha256"],
            "private_score_ledger_sha256": private_scores_sha,
            "users": 20,
            "task_family_counts": {
                family: sum(row["task_family"] == family and row["action"] == "OFF" for row in rows)
                for family in ("T0", "T1", "T2")
            },
            "fixed_action_matrix": action_summary,
            "by_task_family": per_family,
            "oracle": oracle_summary,
            "paired_user_bootstrap": bootstrap,
            "retrieval_coverage_reused_from_frozen_b2_report": retrieval_coverage,
            "retrieval_coverage_rescored": False,
            "retrieval_value_flags_vs_off": diagnostics["value_flags_vs_off"],
            "failure_attribution_counts": failure_counts,
            "t2_dependency_synthetic_gate": t2_dependency,
            "action_costs": {
                action: {
                    "cost_basis": "COMPOSED_COMPONENT_COST",
                    "retrieval_calls_from_b2_component": action_summary[action]["external_retrieval_calls"],
                    "bridge_calls_from_b2_component": action_summary[action]["bridge_calls"],
                    "new_reader_calls": action_summary[action]["reader_calls"],
                    "new_retrieval_calls": 0,
                    "new_bridge_calls": 0,
                    "reader_input_tokens": action_summary[action]["reader_input_tokens_total"],
                    "reader_output_tokens": action_summary[action]["reader_output_tokens_total"],
                    "bridge_input_tokens": action_summary[action]["bridge_input_tokens_total"],
                    "bridge_output_tokens": action_summary[action]["bridge_output_tokens_total"],
                    "composed_component_latency_p50_ms": action_summary[action]["p50_latency_ms"],
                    "composed_component_latency_p95_ms": action_summary[action]["p95_latency_ms"],
                }
                for action in ACTION_ORDER
            },
            "oracle_headroom": diagnostics["oracle_headroom"],
            "STANDARD_MINUS_OFF": bootstrap["comparisons"]["STANDARD_minus_OFF"]["mean_delta"],
            "STRONG_MINUS_OFF": bootstrap["comparisons"]["STRONG_minus_OFF"]["mean_delta"],
            "STRONG_MINUS_STANDARD": bootstrap["comparisons"]["STRONG_minus_STANDARD"]["mean_delta"],
            "CONTENT_RESCUE_COUNT": {
                action: diagnostics["value_flags_vs_off"][action]["CONTENT_RESCUE"]
                for action in ("STANDARD", "STRONG")
            },
            "GROUNDING_ONLY_GAIN_COUNT": {
                action: diagnostics["value_flags_vs_off"][action]["GROUNDING_ONLY_GAIN"]
                for action in ("STANDARD", "STRONG")
            },
            "HARMFUL_RETRIEVAL_COUNT": {
                action: diagnostics["value_flags_vs_off"][action]["HARMFUL_RETRIEVAL"]
                for action in ("STANDARD", "STRONG")
            },
            "OFF_ORACLE_COUNT": diagnostics["oracle_counts"].get("OFF", 0),
            "STANDARD_ORACLE_COUNT": diagnostics["oracle_counts"].get("STANDARD", 0),
            "STRONG_ORACLE_COUNT": diagnostics["oracle_counts"].get("STRONG", 0),
            "ACTION_DIVERSITY_GATE": gates["ACTION_DIVERSITY_GATE"],
            "ORACLE_HEADROOM_GATE": gates["ORACLE_HEADROOM_GATE"],
            "T2_DEPENDENCY_GATE": gates["T2_DEPENDENCY_GATE"],
            "E5C_RESEARCH_QUESTION_WORTH_CONTINUING": authorized,
            "E5C_STARTED": False,
            "202608_opened": False,
            "measurement_recovery_gate": "PASS",
            "MEASUREMENT_RECOVERY_GATE": "PASS",
            "B2_results_rescored": False,
            "interpretation_boundary": "B3 is a separately frozen reader measurement; it does not replace or retroactively invalidate B2.",
        }
    )
    report["report_sha256"] = canonical_sha256(report)
    _write_json(report_json_path, report)
    _write_markdown(report_markdown_path, report)
    return report


__all__ = ["score_recovery"]
