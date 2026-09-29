"""Post-freeze, teacher-aware scoring and aggregate analysis for E5-B2."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from eval.rag_e5.counterfactual import canonical_sha256, expected_arm_specs, verify_complete_arm
from eval.rag_e5.overlay import ACTION_ORDER, score_case


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def _verify_execution_freeze(
    *,
    lock: dict[str, Any],
    lock_sha256: str,
    execution_manifest: dict[str, Any],
    artifact_root: Path,
) -> list[dict[str, Any]]:
    if execution_manifest.get("status") != "ALL_ARMS_FROZEN_BEFORE_SCORING":
        raise ValueError("teacher scoring is forbidden until every arm is frozen")
    manifest_body = {
        key: value for key, value in execution_manifest.items() if key != "execution_manifest_sha256"
    }
    if canonical_sha256(manifest_body) != execution_manifest.get("execution_manifest_sha256"):
        raise ValueError("execution manifest self-hash mismatch")
    if execution_manifest.get("counterfactual_lock_sha256") != lock_sha256:
        raise ValueError("execution manifest is bound to another B2 lock")
    if execution_manifest.get("completed_arms") != 180 or execution_manifest.get("expected_arms") != 180:
        raise ValueError("exactly 180 completed arms are required before scoring")
    if (
        execution_manifest.get("reader_calls") != 180
        or execution_manifest.get("bridge_calls") != 60
        or execution_manifest.get("model_calls") != 240
        or execution_manifest.get("202608_opened") is not False
    ):
        raise ValueError("frozen execution call budget/cohort boundary does not match B2")
    artifact_rows = execution_manifest["artifacts"]
    if canonical_sha256(artifact_rows) != execution_manifest.get("artifact_set_sha256"):
        raise ValueError("execution artifact-set hash mismatch")
    if canonical_sha256([row["run_id"] for row in artifact_rows]) != execution_manifest.get("run_id_set_sha256"):
        raise ValueError("execution run-ID set hash mismatch")
    case_ids = sorted({row["case_id"] for row in artifact_rows})
    if len(case_ids) != 60:
        raise ValueError("execution manifest must contain exactly 60 unique cases")
    specs = expected_arm_specs(
        case_ids, lock_sha256
    )
    if len(specs) != 180:
        raise ValueError("execution manifest does not identify 60 complete cases")
    expected_keys = [(row["case_id"], row["action"], row["run_id"]) for row in specs]
    observed_keys = [(row["case_id"], row["action"], row["run_id"]) for row in artifact_rows]
    if observed_keys != expected_keys:
        raise ValueError("execution manifest arm order/identity differs from the frozen run order")
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec, record in zip(specs, artifact_rows, strict=True):
        path = artifact_root / "arms" / spec["case_id"] / spec["action"] / f"{spec['run_id']}.json"
        if _sha256_file(path) != record.get("file_sha256"):
            raise ValueError("arm artifact changed after the execution manifest was frozen")
        arm = verify_complete_arm(
            path, expected_run_id=spec["run_id"], expected_lock_sha256=lock_sha256
        )
        by_case[spec["case_id"]].append(arm)
    for arms in by_case.values():
        if len(arms) != 3 or len({arm["question_sha256"] for arm in arms}) != 1:
            raise ValueError("counterfactual arms do not share the same question input")
        if len({arm.get("state_packet_sha256") for arm in arms}) != 1:
            raise ValueError("counterfactual arms do not share the same state packet input")
        if len({arm.get("runtime_capability_context_sha256") for arm in arms}) != 1:
            raise ValueError("counterfactual arms do not share the same capability context")
    return specs


def _target_diagnostics(
    ranking: Sequence[Mapping[str, Any]],
    *,
    required_source: str,
    required_recommendations: set[str],
    chunk_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    ids = [str(row["chunk_id"]) for row in ranking]

    def source_hit(depth: int) -> bool:
        return any(
            chunk_by_id.get(chunk_id, {}).get("source_id") == required_source
            for chunk_id in ids[:depth]
        )

    def recommendation_hit(depth: int) -> bool:
        return any(
            chunk_by_id.get(chunk_id, {}).get("source_id") == required_source
            and chunk_by_id.get(chunk_id, {}).get("recommendation_id") in required_recommendations
            for chunk_id in ids[:depth]
        )

    target_ranks = [
        rank
        for rank, chunk_id in enumerate(ids, start=1)
        if chunk_by_id.get(chunk_id, {}).get("source_id") == required_source
        and chunk_by_id.get(chunk_id, {}).get("recommendation_id") in required_recommendations
    ]
    return {
        "required_source_hit_at_5": source_hit(5),
        "required_source_hit_at_100": source_hit(100),
        "required_recommendation_hit_at_5": recommendation_hit(5),
        "required_recommendation_hit_at_100": recommendation_hit(100),
        "best_target_rank": min(target_ranks) if target_ranks else None,
    }


def _score_rows(
    *,
    specs: list[dict[str, str]],
    lock_sha256: str,
    teacher_path: Path,
    artifact_root: Path,
    chunks_path: Path,
) -> list[dict[str, Any]]:
    # This function is called only after _verify_execution_freeze has checked all 180 hashes.
    teachers = _read_jsonl(teacher_path)
    teacher_by_id = {row["case_id"]: row for row in teachers}
    if len(teacher_by_id) != 60:
        raise ValueError("teacher artifact must contain exactly 60 unique case IDs")
    chunks = _read_jsonl(chunks_path)
    chunk_by_id = {row["chunk_id"]: row for row in chunks}
    rows = []
    for spec in specs:
        path = artifact_root / "arms" / spec["case_id"] / spec["action"] / f"{spec['run_id']}.json"
        arm = verify_complete_arm(
            path,
            expected_run_id=spec["run_id"],
            expected_lock_sha256=lock_sha256,
        )
        teacher = teacher_by_id.get(spec["case_id"])
        if teacher is None:
            raise ValueError("teacher and runtime case identities differ")
        reader_output = arm["reader_parsed"] if arm["reader_json_valid"] else {}
        scores = score_case(
            teacher=teacher,
            reader_output=reader_output or {},
            supplied_chunks=arm.get("supplied_chunks", []),
        )
        family = teacher["task_family"]
        diagnostics: dict[str, Any] | None = None
        failure_flags: list[str] = []
        if not arm["reader_json_valid"]:
            failure_flags.append("JSON_FAILURE")
        if teacher.get("expected_state_fields") and scores["state_score"] < 1.0:
            failure_flags.append("STATE_MISS")
        if teacher.get("required_external_source_id"):
            diagnostics = _target_diagnostics(
                arm["retrieval"]["ranking"],
                required_source=teacher["required_external_source_id"],
                required_recommendations=set(teacher.get("required_recommendation_ids", [])),
                chunk_by_id=chunk_by_id,
            ) if spec["action"] != "OFF" else None
            if diagnostics is not None:
                if not diagnostics["required_recommendation_hit_at_100"]:
                    failure_flags.append("RETRIEVAL_MISS")
                elif not diagnostics["required_recommendation_hit_at_5"]:
                    failure_flags.append("RANKING_MISS")
                if diagnostics["required_recommendation_hit_at_5"] and scores["guideline_content_score"] < 1.0:
                    failure_flags.append("READER_MISS")
                if scores["guideline_content_score"] == 1.0 and scores["grounding_score"] == 0.0:
                    failure_flags.append("GROUNDING_MISS")
        if arm.get("bridge", {}).get("fallback_original_query"):
            failure_flags.append("BRIDGE_FALLBACK")
        rows.append(
            {
                "case_id": spec["case_id"],
                "user_id": teacher["user_id"],
                "task_family": family,
                "action": spec["action"],
                **scores,
                "reader_json_valid": arm["reader_json_valid"],
                "reader_input_tokens": arm.get("reader_input_tokens"),
                "reader_output_tokens": arm.get("reader_output_tokens"),
                "bridge_input_tokens": arm.get("bridge", {}).get("raw_response", {}).get("input_tokens"),
                "bridge_output_tokens": arm.get("bridge", {}).get("raw_response", {}).get("output_tokens"),
                "retrieval_latency_ms": arm["retrieval_latency_ms"],
                "bridge_latency_ms": arm["bridge_latency_ms"],
                "reader_latency_ms": arm["reader_latency_ms"],
                "total_latency_ms": arm["total_latency_ms"],
                "retrieval_calls": arm["retrieval_calls"],
                "bridge_calls": arm["bridge_call_count"],
                "reader_calls": arm["reader_call_count"],
                "invented_citation_count": len(arm.get("invented_citations", [])),
                "retrieval_diagnostics": diagnostics,
                "failure_flags": failure_flags,
            }
        )
    return rows


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return fmean(values) if values else None


def _percentile(rows: Sequence[Mapping[str, Any]], key: str, p: float) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return float(np.percentile(values, p)) if values else None


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    state_rows = [row for row in rows if row["task_family"] in {"T0", "T2"}]
    guideline_rows = [row for row in rows if row["task_family"] in {"T1", "T2"}]
    return {
        "cases": len(rows),
        "end_to_end_quality": _mean(rows, "end_to_end_quality"),
        "content_only_quality": _mean(rows, "content_only_quality"),
        "state_score": _mean(state_rows, "state_score"),
        "guideline_content_score": _mean(guideline_rows, "guideline_content_score"),
        "grounding_score": _mean(guideline_rows, "grounding_score"),
        "reader_json_valid_rate": _mean(
            [{"valid": float(row["reader_json_valid"])} for row in rows], "valid"
        ),
        "reader_input_tokens_total": _sum_or_none(rows, "reader_input_tokens"),
        "reader_output_tokens_total": _sum_or_none(rows, "reader_output_tokens"),
        "bridge_input_tokens_total": (
            _sum_or_none([row for row in rows if row["bridge_calls"]], "bridge_input_tokens")
            if any(row["bridge_calls"] for row in rows)
            else 0
        ),
        "bridge_output_tokens_total": (
            _sum_or_none([row for row in rows if row["bridge_calls"]], "bridge_output_tokens")
            if any(row["bridge_calls"] for row in rows)
            else 0
        ),
        "mean_latency_ms": _mean(rows, "total_latency_ms"),
        "p50_latency_ms": _percentile(rows, "total_latency_ms", 50),
        "p95_latency_ms": _percentile(rows, "total_latency_ms", 95),
        "external_retrieval_calls": sum(int(row["retrieval_calls"]) for row in rows),
        "bridge_calls": sum(int(row["bridge_calls"]) for row in rows),
        "reader_calls": sum(int(row["reader_calls"]) for row in rows),
        "invented_citations": sum(int(row["invented_citation_count"]) for row in rows),
    }


def _sum_or_none(rows: Sequence[Mapping[str, Any]], key: str) -> int | None:
    actual = [row.get(key) for row in rows]
    if any(value is None for value in actual):
        return None
    observed = [value for value in actual if isinstance(value, int)]
    return sum(observed) if observed else None


def _bootstrap(
    rows: Sequence[Mapping[str, Any]], *, seed: int, samples: int
) -> dict[str, Any]:
    users = sorted({str(row["user_id"]) for row in rows})
    by_user = {
        user: [row for row in rows if str(row["user_id"]) == user]
        for user in users
    }
    if len(users) != 20 or any(len(by_user[user]) != 9 for user in users):
        raise ValueError("paired bootstrap requires 20 users with three tasks across all three actions")
    expected_pairs = {
        (action, family)
        for action in ACTION_ORDER
        for family in ("T0", "T1", "T2")
    }
    for user in users:
        observed_pairs = [(row["action"], row["task_family"]) for row in by_user[user]]
        if set(observed_pairs) != expected_pairs or len(observed_pairs) != len(expected_pairs):
            raise ValueError("paired bootstrap found missing or duplicate user/action/task rows")
    e2e = {
        action: np.asarray(
            [
                [
                    next(row["end_to_end_quality"] for row in by_user[user] if row["action"] == action and row["task_family"] == family)
                    for family in ("T0", "T1", "T2")
                ]
                for user in users
            ],
            dtype=np.float64,
        )
        for action in ACTION_ORDER
    }
    per_user_action = {action: values.mean(axis=1) for action, values in e2e.items()}
    per_user_oracle = np.max(np.stack([values for values in e2e.values()], axis=0), axis=0).mean(axis=1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(users), size=(samples, len(users)))

    def interval(values: np.ndarray) -> dict[str, float]:
        return {
            "mean_delta": float(np.mean(values)),
            "lower_95": float(np.percentile(values, 2.5)),
            "upper_95": float(np.percentile(values, 97.5)),
        }

    comparisons = {}
    for left, right in (("STANDARD", "OFF"), ("STRONG", "OFF"), ("STRONG", "STANDARD")):
        diffs = per_user_action[left] - per_user_action[right]
        comparisons[f"{left}_minus_{right}"] = interval(diffs[draws].mean(axis=1))
    fixed_stack = np.stack([per_user_action[action] for action in ACTION_ORDER], axis=1)
    sampled_fixed = fixed_stack[draws].mean(axis=1).max(axis=1)
    sampled_oracle = per_user_oracle[draws].mean(axis=1)
    comparisons["ORACLE_minus_BEST_FIXED"] = interval(sampled_oracle - sampled_fixed)
    return {
        "seed": seed,
        "resamples": samples,
        "resampling_unit": "user_id; three related tasks sampled together",
        "comparisons": comparisons,
    }


def _action_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["case_id"])][str(row["action"])] = row
    oracle_rows = []
    oracle_counts: Counter[str] = Counter()
    oracle_users: dict[str, set[str]] = defaultdict(set)
    for actions in grouped.values():
        chosen = max(ACTION_ORDER, key=lambda action: (actions[action]["end_to_end_quality"], -ACTION_ORDER.index(action)))
        oracle_counts[chosen] += 1
        oracle_users[chosen].add(str(actions[chosen]["user_id"]))
        oracle_rows.append(actions[chosen])
    fixed_means = {
        action: _mean([row for row in rows if row["action"] == action], "end_to_end_quality")
        for action in ACTION_ORDER
    }
    best_fixed = max(ACTION_ORDER, key=lambda action: (fixed_means[action], -ACTION_ORDER.index(action)))
    oracle_e2e = _mean(oracle_rows, "end_to_end_quality")
    best_fixed_mean = fixed_means[best_fixed]
    headroom = float(oracle_e2e - best_fixed_mean)
    diversity = all(oracle_counts[action] >= 6 and len(oracle_users[action]) >= 3 for action in ACTION_ORDER)
    by_case = {case_id: actions for case_id, actions in grouped.items()}
    value_flags: dict[str, dict[str, int]] = {}
    for action in ("STANDARD", "STRONG"):
        counts = Counter()
        for actions in by_case.values():
            current = actions[action]
            off = actions["OFF"]
            counts["CONTENT_RESCUE"] += int(
                current["content_only_quality"] > off["content_only_quality"]
            )
            counts["GROUNDING_ONLY_GAIN"] += int(
                current["end_to_end_quality"] > off["end_to_end_quality"]
                and current["content_only_quality"] == off["content_only_quality"]
            )
            counts["HARMFUL_RETRIEVAL"] += int(
                current["end_to_end_quality"] < off["end_to_end_quality"]
            )
            counts["NEUTRAL"] += int(
                current["end_to_end_quality"] == off["end_to_end_quality"]
                and current["content_only_quality"] == off["content_only_quality"]
            )
        value_flags[action] = dict(counts)
    return {
        "fixed_action_means": fixed_means,
        "best_fixed_action": best_fixed,
        "oracle_e2e": oracle_e2e,
        "oracle_headroom": headroom,
        "oracle_counts": dict(oracle_counts),
        "oracle_users_per_action": {action: len(oracle_users[action]) for action in ACTION_ORDER},
        "action_diversity_gate": "PASS" if diversity else "FAIL",
        "oracle_headroom_gate": "PASS" if headroom >= 0.05 else "FAIL",
        "oracle_rows": oracle_rows,
        "value_flags_vs_off": value_flags,
    }


def _retrieval_matrix(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for action in ("STANDARD", "STRONG"):
        candidates = [
            row for row in rows
            if row["action"] == action
            and row["task_family"] in {"T1", "T2"}
            and row["retrieval_diagnostics"] is not None
        ]
        result[action] = {"eligible_cases": len(candidates)}
        for key in (
            "required_source_hit_at_5",
            "required_recommendation_hit_at_5",
            "required_source_hit_at_100",
            "required_recommendation_hit_at_100",
        ):
            result[action][key] = {
                "hits": sum(bool(row["retrieval_diagnostics"][key]) for row in candidates),
                "rate": _mean(
                    [
                        {"hit": float(bool(row["retrieval_diagnostics"][key]))}
                        for row in candidates
                    ],
                    "hit",
                ),
            }
    return result


def _failure_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    result = {}
    labels = (
        "RETRIEVAL_MISS", "RANKING_MISS", "READER_MISS", "GROUNDING_MISS",
        "STATE_MISS", "JSON_FAILURE", "HARMFUL_RETRIEVAL", "BRIDGE_FALLBACK",
    )
    for action in ACTION_ORDER:
        action_rows = [row for row in rows if row["action"] == action]
        counts = Counter(flag for row in action_rows for flag in row["failure_flags"])
        if action != "OFF":
            counts["HARMFUL_RETRIEVAL"] = sum(
                row["end_to_end_quality"]
                < next(
                    other["end_to_end_quality"]
                    for other in rows
                    if other["case_id"] == row["case_id"] and other["action"] == "OFF"
                )
                for row in action_rows
            )
        result[action] = {label: int(counts[label]) for label in labels}
    return result


def _paired_mean_token_delta(
    strong: Mapping[str, Mapping[str, Any]],
    standard: Mapping[str, Mapping[str, Any]],
    reader_key: str,
    bridge_key: str,
) -> int | None:
    deltas = []
    for case_id in strong:
        reader_strong = strong[case_id].get(reader_key)
        bridge_tokens = strong[case_id].get(bridge_key)
        reader_standard = standard[case_id].get(reader_key)
        if any(not isinstance(value, int) for value in (reader_strong, bridge_tokens, reader_standard)):
            return None
        deltas.append(reader_strong + bridge_tokens - reader_standard)
    return round(fmean(deltas)) if deltas else None


def _dependency_gate(teacher_cases: Sequence[Mapping[str, Any]], chunks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {row["chunk_id"]: row for row in chunks}
    t2 = [row for row in teacher_cases if row.get("task_family") == "T2"]
    maxima = {"full": [], "without_state": [], "without_evidence": []}
    for teacher in t2:
        if not teacher.get("expected_state_fields") or not teacher.get("required_external_source_id") or not teacher.get("required_recommendation_ids"):
            return {"gate": "FAIL", "reason": "T2 teacher dependency missing", "cases": len(t2)}
        metric, trend = next(iter(teacher["expected_state_fields"].items()))
        target_chunk = next(
            by_id[chunk_id]
            for chunk_id in teacher["required_chunk_ids"]
            if chunk_id in by_id
            and by_id[chunk_id].get("source_id") == teacher["required_external_source_id"]
            and by_id[chunk_id].get("recommendation_id") in teacher["required_recommendation_ids"]
        )
        output = {
            "state_facts": [{"field": metric, "value": trend}],
            "guidance_facts": [
                {"statement": "Adults aged 20 years and older should limit total fat to 30% of total energy or less."},
                {"statement": "Adults already below 30% should not increase fat intake to reach the threshold."},
            ],
            "citations": [target_chunk["chunk_id"]],
            "answer": "Synthetic scoring dependency fixture.",
        }
        full = score_case(teacher=teacher, reader_output=output, supplied_chunks=[target_chunk])
        no_state = score_case(
            teacher=teacher, reader_output=output, supplied_chunks=[target_chunk], remove_state=True
        )
        no_evidence = score_case(
            teacher=teacher, reader_output=output, supplied_chunks=[target_chunk],
            remove_evidence_and_citations=True,
        )
        maxima["full"].append(full["end_to_end_quality"])
        maxima["without_state"].append(no_state["end_to_end_quality"])
        maxima["without_evidence"].append(no_evidence["end_to_end_quality"])
    passed = len(t2) == 20 and all(value == 1.0 for value in maxima["full"]) and all(
        value == 0.5 for value in maxima["without_state"]
    ) and all(value == 0.75 for value in maxima["without_evidence"])
    return {
        "gate": "PASS" if passed else "FAIL",
        "cases": len(t2),
        "max_full": max(maxima["full"], default=None),
        "max_without_state": max(maxima["without_state"], default=None),
        "max_without_evidence": max(maxima["without_evidence"], default=None),
    }


def analyze_frozen_run(
    *,
    lock_path: Path,
    execution_manifest_path: Path,
    private_root: Path,
    corpus_root: Path,
    artifact_root: Path,
    private_output_path: Path,
    report_json_path: Path,
    matrix_json_path: Path,
    report_markdown_path: Path,
    seed: int = 20260930,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    lock = _read_json(lock_path)
    lock_sha = lock.get("counterfactual_lock_sha256")
    lock_body = {key: value for key, value in lock.items() if key != "counterfactual_lock_sha256"}
    if not isinstance(lock_sha, str) or canonical_sha256(lock_body) != lock_sha:
        raise ValueError("B2 lock self-hash mismatch")
    execution = _read_json(execution_manifest_path)
    specs = _verify_execution_freeze(
        lock=lock,
        lock_sha256=lock_sha,
        execution_manifest=execution,
        artifact_root=artifact_root,
    )
    if _sha256_file(corpus_root / "chunks.jsonl") != lock["external_corpus_chunks_sha256"]:
        raise ValueError("active corpus changed between execution and evaluation")

    # The teacher artifact is first opened only after all arm and manifest hashes pass.
    rows = _score_rows(
        specs=specs,
        lock_sha256=lock_sha,
        teacher_path=private_root / "teacher_cases.jsonl",
        artifact_root=artifact_root,
        chunks_path=corpus_root / "chunks.jsonl",
    )
    teachers = _read_jsonl(private_root / "teacher_cases.jsonl")
    chunks = _read_jsonl(corpus_root / "chunks.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = {
        action: [row for row in rows if row["action"] == action] for action in ACTION_ORDER
    }
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
    retrieval = _retrieval_matrix(rows)
    failures = _failure_counts(rows)
    t2_dependency = _dependency_gate(teachers, chunks)
    str_rows = {row["case_id"]: row for row in grouped["STRONG"]}
    std_rows = {row["case_id"]: row for row in grouped["STANDARD"]}
    off_rows = {row["case_id"]: row for row in grouped["OFF"]}
    strong_standard = {
        "content_only_gain": fmean(
            str_rows[case_id]["content_only_quality"] - std_rows[case_id]["content_only_quality"]
            for case_id in str_rows
        ),
        "grounding_gain": fmean(
            str_rows[case_id]["grounding_score"] - std_rows[case_id]["grounding_score"]
            for case_id in str_rows
            if str_rows[case_id]["task_family"] in {"T1", "T2"}
        ),
        "harm_rate": fmean(
            float(str_rows[case_id]["end_to_end_quality"] < std_rows[case_id]["end_to_end_quality"])
            for case_id in str_rows
        ),
        "mean_total_latency_delta_ms": fmean(
            str_rows[case_id]["total_latency_ms"] - std_rows[case_id]["total_latency_ms"]
            for case_id in str_rows
        ),
        "bridge_calls_added": sum(row["bridge_calls"] for row in grouped["STRONG"])
        - sum(row["bridge_calls"] for row in grouped["STANDARD"]),
        "mean_input_tokens_delta": _paired_mean_token_delta(
            str_rows, std_rows, "reader_input_tokens", "bridge_input_tokens"
        ),
        "mean_output_tokens_delta": _paired_mean_token_delta(
            str_rows, std_rows, "reader_output_tokens", "bridge_output_tokens"
        ),
        "retrieval_action_invocation_delta": sum(row["retrieval_calls"] for row in grouped["STRONG"])
        - sum(row["retrieval_calls"] for row in grouped["STANDARD"]),
    }
    private_scores = {
        "schema_version": "rag-e5-e5b2-private-score-ledger-v1",
        "counterfactual_lock_sha256": lock_sha,
        "execution_manifest_sha256": execution["execution_manifest_sha256"],
        "cases": rows,
        "oracle_by_case": {
            row["case_id"]: row["action"] for row in diagnostics["oracle_rows"]
        },
    }
    _write_json(private_output_path, private_scores)
    private_scores_sha = _sha256_file(private_output_path)
    unresolved = sum(
        max(
            off_rows[case_id]["end_to_end_quality"],
            std_rows[case_id]["end_to_end_quality"],
            str_rows[case_id]["end_to_end_quality"],
        ) < 1.0
        for case_id in off_rows
    )
    t2_observed = {
        action: {
            "state_score": _mean([row for row in grouped[action] if row["task_family"] == "T2"], "state_score"),
            "guideline_content_score": _mean([row for row in grouped[action] if row["task_family"] == "T2"], "guideline_content_score"),
            "grounding_score": _mean([row for row in grouped[action] if row["task_family"] == "T2"], "grounding_score"),
        }
        for action in ACTION_ORDER
    }
    gates = {
        "ACTION_DIVERSITY_GATE": diagnostics["action_diversity_gate"],
        "ORACLE_HEADROOM_GATE": diagnostics["oracle_headroom_gate"],
        "T2_DEPENDENCY_GATE": t2_dependency["gate"],
    }
    e5c_authorized = all(value == "PASS" for value in gates.values())
    report = {
        "schema_version": "rag-e5-e5b2-counterfactual-report-v1",
        "status": "COMPLETE_FROZEN_COUNTERFACTUAL_CHARACTERIZATION",
        "base_main": lock["base_main"],
        "counterfactual_lock_sha256": lock_sha,
        "execution_manifest_sha256": execution["execution_manifest_sha256"],
        "private_score_ledger_sha256": private_scores_sha,
        "task_set_sha256": lock["ordered_case_ids_sha256"],
        "state_packet_set_sha256": lock["state_packet_set_sha256"],
        "external_corpus_identity": lock["external_corpus_identity"],
        "scorer_version": lock["scorer_version"],
        "scorer_module_sha256": lock["scorer_module_sha256"],
        "scoring_contract_sha256": lock["scoring_contract_sha256"],
        "cases": 60,
        "users": 20,
        "task_family_counts": {family: sum(row["task_family"] == family for row in rows if row["action"] == "OFF") for family in ("T0", "T1", "T2")},
        "arms": execution["completed_arms"],
        "reader_calls": execution["reader_calls"],
        "bridge_calls": execution["bridge_calls"],
        "total_model_calls": execution["model_calls"],
        "bridge_fallbacks": execution["bridge_fallbacks"],
        "invalid_reader_json": execution["invalid_reader_outputs"],
        "202608_opened": False,
        "fixed_action_matrix": action_summary,
        "by_task_family": per_family,
        "oracle": {
            "end_to_end_quality": diagnostics["oracle_e2e"],
            "content_only_quality": _mean(diagnostics["oracle_rows"], "content_only_quality"),
            "best_fixed_action": diagnostics["best_fixed_action"],
            "best_fixed_e2e": diagnostics["fixed_action_means"][diagnostics["best_fixed_action"]],
            "headroom": diagnostics["oracle_headroom"],
            "action_counts": diagnostics["oracle_counts"],
            "users_per_action": diagnostics["oracle_users_per_action"],
            "unresolved_cases_max_quality_below_1": unresolved,
        },
        "paired_user_bootstrap": bootstrap,
        "retrieval_target_matrix": retrieval,
        "retrieval_value_flags_vs_off": diagnostics["value_flags_vs_off"],
        "failure_attribution_counts": failures,
        "t2_observed_components": t2_observed,
        "t2_dependency_synthetic_gate": t2_dependency,
        "strong_vs_standard": strong_standard,
        "gates": gates,
        "E5C_AUTHORIZED": e5c_authorized,
        "E5C_STARTED": False,
        "quality_definition_note": "CONTENT_RESCUE/GROUNDING_ONLY_GAIN/HARMFUL_RETRIEVAL counts are independent flags and may overlap.",
    }
    matrix = {
        "schema_version": "rag-e5-e5b2-counterfactual-matrix-v1",
        "counterfactual_lock_sha256": lock_sha,
        "fixed_action_matrix": action_summary,
        "by_task_family": per_family,
        "oracle": report["oracle"],
        "paired_user_bootstrap": bootstrap,
        "retrieval_target_matrix": retrieval,
        "failure_attribution_counts": failures,
        "gates": gates,
        "E5C_AUTHORIZED": e5c_authorized,
        "E5C_STARTED": False,
    }
    _write_json(report_json_path, report)
    _write_json(matrix_json_path, matrix)
    _write_report_markdown(report_markdown_path, report)
    return report


def _write_report_markdown(path: Path, report: Mapping[str, Any]) -> None:
    matrix = report["fixed_action_matrix"]
    lines = [
        "# RAG-E5-B2 — Frozen Counterfactual Characterization",
        "",
        "This report compares OFF, STANDARD, and STRONG over the same 60 frozen tasks (20 users; T0/T1/T2). Scoring separates guideline-content correctness from evidence grounding. No router was trained.",
        "",
        f"- Task-set SHA-256: `{report['task_set_sha256']}`",
        f"- State-packet-set SHA-256: `{report['state_packet_set_sha256']}`",
        f"- Active corpus identity: `{report['external_corpus_identity']}`",
        f"- Scorer: `{report['scorer_version']}`",
        f"- Frozen arms / model calls: {report['arms']} / {report['total_model_calls']} ({report['reader_calls']} readers, {report['bridge_calls']} bridges)",
        f"- Invalid reader JSON / bridge fallbacks: {report['invalid_reader_json']} / {report['bridge_fallbacks']}",
        "",
        "## Overall fixed-action matrix",
        "",
        "| Metric | OFF | STANDARD | STRONG | ORACLE |",
        "|---|---:|---:|---:|---:|",
    ]
    oracle = report["oracle"]
    for key, label in (
        ("end_to_end_quality", "E2E quality"),
        ("content_only_quality", "Content-only quality"),
        ("state_score", "State score (T0/T2)"),
        ("guideline_content_score", "Guideline content (T1/T2)"),
        ("grounding_score", "Grounding (T1/T2)"),
        ("reader_json_valid_rate", "Reader JSON valid rate"),
        ("mean_latency_ms", "Mean total latency (ms)"),
        ("p50_latency_ms", "P50 total latency (ms)"),
        ("p95_latency_ms", "P95 total latency (ms)"),
    ):
        cells = [matrix[action].get(key) for action in ACTION_ORDER]
        if key == "end_to_end_quality":
            cells.append(oracle["end_to_end_quality"])
        elif key == "content_only_quality":
            cells.append(oracle["content_only_quality"])
        else:
            cells.append(None)
        lines.append("| " + label + " | " + " | ".join(_fmt(value) for value in cells) + " |")
    lines.extend(["", "## By task family", "", "| Family | Action | E2E | Content-only | State | Guideline content | Grounding |", "|---|---|---:|---:|---:|---:|---:|"])
    for family in ("T0", "T1", "T2"):
        for action in ACTION_ORDER:
            row = report["by_task_family"][family][action]
            lines.append(
                f"| {family} | {action} | {_fmt(row['end_to_end_quality'])} | {_fmt(row['content_only_quality'])} | {_fmt(row['state_score'])} | {_fmt(row['guideline_content_score'])} | {_fmt(row['grounding_score'])} |"
            )
    lines.extend(["", "## Paired user-level bootstrap", "", "10,000 resamples; each draw samples 20 users with replacement and carries each user's T0/T1/T2 tasks together.", "", "| Comparison | Mean delta | 95% CI |", "|---|---:|---:|"])
    for name, result in report["paired_user_bootstrap"]["comparisons"].items():
        lines.append(f"| {name.replace('_minus_', ' − ')} | {_fmt(result['mean_delta'])} | [{_fmt(result['lower_95'])}, {_fmt(result['upper_95'])}] |")
    lines.extend(["", "## Retrieval target coverage (T1/T2)", "", "| Action | Cases | Source @5 | Recommendation @5 | Source @100 | Recommendation @100 |", "|---|---:|---:|---:|---:|---:|"])
    for action in ("STANDARD", "STRONG"):
        row = report["retrieval_target_matrix"][action]
        lines.append(
            f"| {action} | {row['eligible_cases']} | {_fmt(row['required_source_hit_at_5']['rate'])} ({row['required_source_hit_at_5']['hits']}) | {_fmt(row['required_recommendation_hit_at_5']['rate'])} ({row['required_recommendation_hit_at_5']['hits']}) | {_fmt(row['required_source_hit_at_100']['rate'])} ({row['required_source_hit_at_100']['hits']}) | {_fmt(row['required_recommendation_hit_at_100']['rate'])} ({row['required_recommendation_hit_at_100']['hits']}) |"
        )
    lines.extend(["", "## Oracle and gates", "", f"- Best fixed action: {oracle['best_fixed_action']} ({_fmt(oracle['best_fixed_e2e'])}).", f"- Per-case oracle E2E: {_fmt(oracle['end_to_end_quality'])}; headroom: {_fmt(oracle['headroom'])}.", f"- Tie-broken oracle action counts: `{json.dumps(oracle['action_counts'], sort_keys=True)}`.", f"- Unresolved cases (best E2E < 1.0): {oracle['unresolved_cases_max_quality_below_1']}.", f"- Gates: `{json.dumps(report['gates'], sort_keys=True)}`.", f"- E5-C authorized: **{report['E5C_AUTHORIZED']}**. E5-C started: **NO**.", "", "## Interpretation boundaries", "", "Content-only quality isolates answer-content anchors from grounding credit; E2E quality retains the preregistered weights. Retrieval hit@100/hit@5 separates retrieval misses from rank-cutoff misses. Reader misses and grounding misses are counted separately. Content-rescue, grounding-only-gain, and harmful-retrieval flags can overlap. Costs are reported as raw calls, token usage, and latency; no post-hoc cost utility is applied.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _fmt(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"
