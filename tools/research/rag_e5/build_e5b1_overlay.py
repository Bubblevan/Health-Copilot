"""Freeze E5-B1 runtime/teacher overlays and B2 execution protocol, without execution."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

from eval.rag_e5.age import AGE_TEMPORAL_CONTRACT_SHA256
from eval.rag_e5.longitudinal_state import (
    PROFILE_TEMPORAL_CONTRACT_SHA256,
    STATE_PACKET_BUILDER_VERSION,
    STATE_PACKET_CONFIG_SHA256,
)
from eval.rag_e5.overlay import (
    ACTION_ORDER,
    BATCH_ID,
    CAPABILITY_CONTEXT,
    CORPUS_IDENTITY,
    READER_PROMPT,
    READER_SCHEMA,
    SCORER_VERSION,
    TASK_TEMPLATE_VERSION,
    TASK_TEMPLATES,
    audit_questions,
    build_overlay_cases,
    score_case,
)
from eval.rag_e5.temporal import TEMPORAL_SEMANTICS_ID, TEMPORAL_SEMANTICS_SHA256

DEFAULT_PRIVATE_ROOT = Path("D:/MyLab/Jianli/external/rag_e5/e5b1")
DEFAULT_CORPUS_ROOT = Path("D:/MyLab/Jianli/external/rag_e5/e5a3/corpus/public_health_plus_guideline")
DEFAULT_CORPUS_MANIFEST = Path("runs/rag_e5/external_corpus_manifest.json")
DEFAULT_PROFILES = Path("runs/rag_e5/retrieval_action_profiles.json")
DEFAULT_COVERAGE = Path("runs/rag_e5/e5b1_state_coverage_report.json")
TASK_MANIFEST_PATH = Path("runs/rag_e5/e5b1_task_manifest.json")
LOCK_PATH = Path("runs/rag_e5/e5b_counterfactual_lock.json")
LEAKAGE_PATH = Path("runs/rag_e5/e5b1_leakage_audit.json")
DEVELOPMENT_USERS = tuple(f"user{number}_AT_demo" for number in range(5200, 5220))
RELEVANT_CODE_FILES = (
    "eval/rag_e5/age.py",
    "eval/rag_e5/temporal.py",
    "eval/rag_e5/longitudinal_state.py",
    "eval/rag_e5/overlay.py",
    "tools/research/rag_e5/build_e5b1_state_packets.py",
    "tools/research/rag_e5/build_e5b1_overlay.py",
)

def build_freeze(
    *,
    private_root: Path,
    corpus_root: Path,
    corpus_manifest_path: Path,
    profiles_path: Path,
    coverage_path: Path,
) -> dict[str, Any]:
    coverage = _read_json(coverage_path)
    _validate_coverage(coverage)
    corpus_manifest = _read_json(corpus_manifest_path)
    _validate_corpus(corpus_manifest)
    profiles = _read_json(profiles_path)
    standard, strong = _validate_profiles(profiles)

    state_packets = []
    for user_id in DEVELOPMENT_USERS:
        path = private_root / "state_packets" / f"{user_id}.json"
        packet = _read_json(path)
        _validate_packet(packet, expected_user_id=user_id)
        state_packets.append(packet)
    actual_packet_set_sha = _packet_set_sha(state_packets)
    if actual_packet_set_sha != coverage.get("state_packet_set_sha256"):
        raise ValueError("private state packet set does not match the coverage freeze")

    chunks_path = corpus_root / "chunks.jsonl"
    all_chunks = _read_jsonl(chunks_path)
    eligible_source_ids = {
        row["source_id"]
        for row in corpus_manifest["guideline_candidates"]
        if row.get("task_authoring_eligible") is True
    }
    if eligible_source_ids != {
        "who-physical-activity-sedentary-2020",
        "who-total-fat-weight-gain-2023",
    }:
        raise ValueError("owner-approved task-authoring eligible source set changed")
    eligible_chunks = [chunk for chunk in all_chunks if chunk.get("source_id") in eligible_source_ids]
    runtime_cases, teacher_cases = build_overlay_cases(
        state_packets=state_packets, eligible_chunks=eligible_chunks
    )
    if len(runtime_cases) != 60 or len(teacher_cases) != 60:
        raise ValueError("overlay case count must be exactly 60")
    if {case.user_id for case in runtime_cases} != set(DEVELOPMENT_USERS):
        raise ValueError("overlay user set differs from the frozen 202607 cohort")

    private_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(private_root / "runtime_cases.jsonl", (case.to_dict() for case in runtime_cases))
    _write_jsonl(private_root / "teacher_cases.jsonl", (case.to_dict() for case in teacher_cases))
    _write_json(private_root / "capability_context.json", CAPABILITY_CONTEXT)
    _write_json(private_root / "reader_schema.json", READER_SCHEMA)
    (private_root / "reader_prompt.txt").write_text(READER_PROMPT, encoding="utf-8")
    _write_json(private_root / "task_templates.json", TASK_TEMPLATES)

    rubric = _build_rubric(teacher_cases, eligible_chunks)
    rubric_path = private_root / "scoring_rubric.json"
    _write_json(rubric_path, rubric)
    # The evaluator's lookup contains IDs and provenance only, never passage text.
    chunk_registry = [
        {
            "chunk_id": chunk["chunk_id"],
            "source_id": chunk.get("source_id"),
            "recommendation_id": chunk.get("recommendation_id"),
            "source_family": chunk.get("source_family"),
        }
        for chunk in all_chunks
    ]
    _write_json(private_root / "chunk_provenance_registry.json", chunk_registry)

    questions = [case.question for case in runtime_cases]
    question_audit = audit_questions(questions)
    context_refs = {case.runtime_capability_context_ref for case in runtime_cases}
    runtime_keys = set(runtime_cases[0].to_dict())
    teacher_only_keys = {
        "task_family", "expected_state_fields", "required_external_source_id",
        "required_recommendation_ids", "required_chunk_ids", "scoring_rubric_ref",
    }
    if runtime_keys.intersection(teacher_only_keys):
        raise AssertionError("runtime/teacher separation failed")

    ordered_case_ids = [case.case_id for case in runtime_cases]
    ordered_user_ids = list(DEVELOPMENT_USERS)
    case_counts = Counter(case.task_family for case in teacher_cases)
    task_template_sha = _canonical_sha256(TASK_TEMPLATES)
    reader_schema_sha = _canonical_sha256(READER_SCHEMA)
    reader_prompt_sha = sha256(READER_PROMPT.encode("utf-8")).hexdigest()
    rubric_sha = _canonical_sha256(rubric)
    code_sha = _code_tree_sha()
    code_commit = _git_head()
    model = strong["frozen_config"]["generator"]
    llama_cpp_version = _llama_cpp_version()

    task_manifest = {
        "schema_version": "rag-e5-e5b1-task-manifest-v1",
        "batch_id": BATCH_ID,
        "user_count": 20,
        "case_count": 60,
        "task_family_counts": {family: case_counts[family] for family in ("T0", "T1", "T2")},
        "ordered_user_ids_sha256": _canonical_sha256(ordered_user_ids),
        "ordered_case_ids_sha256": _canonical_sha256(ordered_case_ids),
        "state_packet_set_sha256": actual_packet_set_sha,
        "state_packet_builder_version": STATE_PACKET_BUILDER_VERSION,
        "state_packet_config_sha256": STATE_PACKET_CONFIG_SHA256,
        "profile_temporal_contract_sha256": PROFILE_TEMPORAL_CONTRACT_SHA256,
        "age_temporal_contract_sha256": AGE_TEMPORAL_CONTRACT_SHA256,
        "temporal_semantics_id": TEMPORAL_SEMANTICS_ID,
        "temporal_semantics_sha256": TEMPORAL_SEMANTICS_SHA256,
        "task_template_version": TASK_TEMPLATE_VERSION,
        "task_template_sha256": task_template_sha,
        "reader_schema_sha256": reader_schema_sha,
        "reader_prompt_sha256": reader_prompt_sha,
        "scorer_version": SCORER_VERSION,
        "scoring_rubric_sha256": rubric_sha,
        "external_corpus_identity": CORPUS_IDENTITY,
        "external_corpus_manifest_sha256": _file_sha256(corpus_manifest_path),
        "action_profiles_manifest_sha256": _file_sha256(profiles_path),
        "esl_source_manifest_sha256": _file_sha256(Path("runs/rag_e5/esl_source_manifest.json")),
        "standard_profile_config_sha256": standard["config_sha256"],
        "strong_profile_config_sha256": strong["config_sha256"],
        "code_commit": code_commit,
        "code_tree_sha256": code_sha,
        "task_construction_inputs": ["v4_state_packets", "active_corpus_manifest", "approved_chunks"],
        "native_esl_questions_opened": False,
        "native_esl_answers_opened": False,
        "kg_evaluation_queries_opened": False,
        "202608_opened": False,
        "model_calls": 0,
        "retrieval_calls": 0,
        "counterfactual_outcomes_created": False,
    }

    dependency_audit = _synthetic_dependency_audit(teacher_cases, eligible_chunks)
    leakage_report = {
        "schema_version": "rag-e5-e5b1-leakage-audit-v1",
        "runtime_case_schema": "PASS_EXACT_ALLOWLIST",
        "runtime_teacher_separation": "PASS",
        "question_audit": question_audit,
        "capability_context_identical": len(context_refs) == 1,
        "capability_context_sha256": _canonical_sha256(CAPABILITY_CONTEXT),
        "state_packet_refs": "NULL_FOR_T1_ONLY",
        "native_esl_questions_opened": False,
        "native_esl_answers_opened": False,
        "kg_evaluation_queries_opened": False,
        "202608_profile_timeline_exam_questions_answers_opened": False,
        "counterfactual_outcomes_created": False,
        "model_calls": 0,
        "retrieval_calls": 0,
        "synthetic_t2_dependency_test": dependency_audit,
        "leakage_gate": "PASS"
        if question_audit["question_leakage_gate"] == "PASS"
        and len(context_refs) == 1
        and dependency_audit["gate"] == "PASS"
        else "FAIL",
    }

    lock = {
        "schema_version": "rag-e5-e5b-counterfactual-lock-v1",
        "status": "FROZEN_PROTOCOL_NO_OUTCOMES",
        "batch_id": BATCH_ID,
        "case_count": 60,
        "ordered_case_ids_sha256": task_manifest["ordered_case_ids_sha256"],
        "state_packet_set_sha256": actual_packet_set_sha,
        "task_template_sha256": task_template_sha,
        "runtime_capability_context_sha256": _canonical_sha256(CAPABILITY_CONTEXT),
        "answer_model": {
            "model": model["model"],
            "revision": model["revision"],
            "file": model["file"],
            "sha256": model["sha256"],
            "temperature": model["temperature"],
            "reasoning": model["reasoning"],
            "max_output_tokens": model["max_output_tokens"],
            "calls_per_case": 1,
            "retry_on_error": False,
            "endpoint_binding": "loopback_only",
            "llama_cpp_version": llama_cpp_version,
        },
        "reader_prompt_sha256": reader_prompt_sha,
        "reader_schema_sha256": reader_schema_sha,
        "reader_evidence_context": "top_five_fused_chunks_in_rank_order",
        "counterfactual_input_invariance": "same_question_state_answer_model_prompt_and_evaluator;_only_retrieved_evidence_differs",
        "strong_generator": {
            "method": strong["frozen_config"]["method"],
            "prompt_template_sha256": strong["frozen_config"]["prompt_binding"]["template_sha256"],
            "calls_per_case": model["calls_per_query"],
            "feedback_depth": strong["frozen_config"]["feedback_depth"],
            "retry_on_error": model["retry_on_error"],
            "generation_config": {
                "model_sha256": model["sha256"],
                "temperature": model["temperature"],
                "reasoning": model["reasoning"],
                "max_output_tokens": model["max_output_tokens"],
                "calls_per_query": model["calls_per_query"],
            },
            "input_boundary": "current_question_and_same_query_bm25_top10_only",
            "fallback": "original_query_on_generation_failure",
        },
        "retrieval_actions_in_execution_order": list(ACTION_ORDER),
        "counterfactual_execution_order": "case_id_sorted_then_OFF_STANDARD_STRONG",
        "external_corpus_identity": CORPUS_IDENTITY,
        "standard_profile_config_sha256": standard["config_sha256"],
        "strong_profile_config_sha256": strong["config_sha256"],
        "scorer_version": SCORER_VERSION,
        "scoring_rubric_sha256": rubric_sha,
        "temporal_semantics_id": TEMPORAL_SEMANTICS_ID,
        "temporal_semantics_sha256": TEMPORAL_SEMANTICS_SHA256,
        "profile_temporal_contract_sha256": PROFILE_TEMPORAL_CONTRACT_SHA256,
        "age_temporal_contract_sha256": AGE_TEMPORAL_CONTRACT_SHA256,
        "source_cohort": "202607_only",
        "model_calls": 0,
        "retrieval_calls": 0,
        "outcomes_or_oracle_action_created": False,
        "code_commit": code_commit,
        "code_tree_sha256": code_sha,
    }

    _write_json(TASK_MANIFEST_PATH, task_manifest)
    _write_json(LOCK_PATH, lock)
    _write_json(LEAKAGE_PATH, leakage_report)
    if leakage_report["leakage_gate"] != "PASS":
        raise SystemExit("E5-B1 leakage/dependency audit failed")
    return {
        "task_manifest": task_manifest,
        "lock": lock,
        "leakage_report": leakage_report,
        "case_count": len(runtime_cases),
        "task_family_counts": task_manifest["task_family_counts"],
        "task_set_sha256": task_manifest["ordered_case_ids_sha256"],
        "state_packet_set_sha256": actual_packet_set_sha,
    }


def _build_rubric(
    teacher_cases: list[Any], chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    sources = sorted(
        {
            case.required_external_source_id
            for case in teacher_cases
            if case.required_external_source_id
        }
    )
    recommendations: dict[str, Any] = {}
    for source_id in sources:
        recommendation_ids = sorted(
            {
                recommendation_id
                for case in teacher_cases
                if case.required_external_source_id == source_id
                for recommendation_id in case.required_recommendation_ids
            }
        )
        recommendations[source_id] = {}
        for recommendation_id in recommendation_ids:
            recommendations[source_id][recommendation_id] = {
                "approved_chunk_ids": sorted(
                    chunk["chunk_id"]
                    for chunk in chunks
                    if chunk.get("source_id") == source_id
                    and chunk.get("recommendation_id") == recommendation_id
                ),
                "keypoints": _rubric_keypoints(source_id, recommendation_id),
            }
    return {
        "scorer_version": SCORER_VERSION,
        "normalization": "casefold_and_collapse_whitespace; numbers_match_integer_token",
        "guideline_credit_gate": "at_least_one_supplied_correct_source_and_recommendation_chunk_is_cited",
        "state_score": "exact_required_field_enum_match_fraction",
        "grounding_score": "cited_supplied_correct_recommendation_fraction",
        "weights": {
            "T0": {"state": 1.0},
            "T1": {"guideline": 0.75, "grounding": 0.25},
            "T2": {"state": 0.5, "guideline": 0.25, "grounding": 0.25},
        },
        "recommendations": recommendations,
    }


def _rubric_keypoints(source_id: str, recommendation_id: str) -> list[dict[str, Any]]:
    # Keep the committed rubric artifact content-free; rules live in the scorer module.
    from eval.rag_e5.overlay import _RUBRIC_TEMPLATES

    return _RUBRIC_TEMPLATES[source_id][recommendation_id]


def _synthetic_dependency_audit(
    teacher_cases: list[Any], chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    t2_cases = [case for case in teacher_cases if case.task_family == "T2"]
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    maxima_without_state: list[float] = []
    maxima_without_evidence: list[float] = []
    maxima_full: list[float] = []
    # The fixture is synthetic and intentionally contains no user-specific values.
    output = {
        "state_facts": [{"field": "body_weight", "value": "rising"}],
        "guidance_facts": [
            {"statement": "Adults aged 20 years and older should limit total fat to 30% of total energy or less."},
            {"statement": "Adults already below 30% should not increase fat intake to reach the threshold."},
        ],
        "citations": [],
        "answer": "Synthetic scoring fixture.",
    }
    for teacher in t2_cases:
        metric, expected_trend = next(iter(teacher.expected_state_fields.items()))
        output["state_facts"] = [{"field": metric, "value": expected_trend}]
        valid_chunk = next(
            by_id[chunk_id]
            for chunk_id in teacher.required_chunk_ids
            if chunk_id in by_id
            and by_id[chunk_id].get("source_id") == teacher.required_external_source_id
            and by_id[chunk_id].get("recommendation_id") in teacher.required_recommendation_ids
        )
        output["citations"] = [valid_chunk["chunk_id"]]
        full = score_case(
            teacher=teacher.to_dict(), reader_output=output, supplied_chunks=[valid_chunk]
        )
        no_state = score_case(
            teacher=teacher.to_dict(), reader_output=output, supplied_chunks=[valid_chunk], remove_state=True
        )
        no_evidence = score_case(
            teacher=teacher.to_dict(), reader_output=output, supplied_chunks=[valid_chunk],
            remove_evidence_and_citations=True,
        )
        maxima_full.append(full["quality"])
        maxima_without_state.append(no_state["quality"])
        maxima_without_evidence.append(no_evidence["quality"])
    passed = (
        len(t2_cases) == 20
        and all(score == 1.0 for score in maxima_full)
        and all(score <= 0.5 for score in maxima_without_state)
        and all(score <= 0.5 for score in maxima_without_evidence)
    )
    return {
        "synthetic_cases": len(t2_cases),
        "maximum_full_score": max(maxima_full, default=0.0),
        "maximum_without_required_state": max(maxima_without_state, default=1.0),
        "maximum_without_required_evidence_and_citation": max(maxima_without_evidence, default=1.0),
        "gate": "PASS" if passed else "FAIL",
    }


def _validate_coverage(value: dict[str, Any]) -> None:
    if value.get("batch_id") != BATCH_ID or value.get("coverage_gate") != "PASS":
        raise ValueError("state coverage report is not a passing 202607 coverage freeze")
    if value.get("users_total") != 20 or value.get("users_with_neither") != 0:
        raise ValueError("state coverage does not meet the 20-user no-rescue gate")


def _validate_corpus(value: dict[str, Any]) -> None:
    if value.get("status") != "ACTIVE_FROZEN":
        raise ValueError("external corpus is not ACTIVE_FROZEN")
    if value.get("active_external_corpus_identity") != CORPUS_IDENTITY:
        raise ValueError("external corpus identity differs from the E5-A3.1 activation")


def _validate_profiles(value: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if value.get("external_corpus_identity") != CORPUS_IDENTITY:
        raise ValueError("retrieval profiles do not bind the active corpus identity")
    by_action = {item["action"]: item for item in value.get("profiles", [])}
    standard, strong = by_action.get("STANDARD"), by_action.get("STRONG")
    if not standard or not strong:
        raise ValueError("STANDARD and STRONG profiles are required")
    for profile in (standard, strong):
        digest = _canonical_sha256(profile["frozen_config"])
        if digest != profile.get("config_sha256"):
            raise ValueError(f"{profile['action']} retrieval profile hash mismatch")
        if set(profile.get("source_families", [])) != {"public_health", "reviewed_guideline"}:
            raise ValueError(f"{profile['action']} source-family capability changed")
    return standard, strong


def _validate_packet(value: dict[str, Any], *, expected_user_id: str) -> None:
    if value.get("user_id") != expected_user_id:
        raise ValueError("private state packet user identity mismatch")
    if value.get("summary_builder_version") != STATE_PACKET_BUILDER_VERSION:
        raise ValueError("private state packet is not v4")
    if value.get("summary_config_sha256") != STATE_PACKET_CONFIG_SHA256:
        raise ValueError("private state packet config SHA mismatch")
    packet_hash = value.get("packet_sha256")
    payload = {key: item for key, item in value.items() if key != "packet_sha256"}
    if _canonical_sha256(payload) != packet_hash:
        raise ValueError("private state packet content hash mismatch")


def _packet_set_sha(packets: list[dict[str, Any]]) -> str:
    identity = [
        {
            "user_id": packet["user_id"],
            "packet_sha256": packet["packet_sha256"],
            "decision_boundary": packet["decision_boundary"]["naive_timestamp"],
        }
        for packet in packets
    ]
    return _canonical_sha256(identity)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("active corpus chunks must be JSON objects")
                rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _canonical_sha256(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _code_tree_sha() -> str:
    records = []
    for relative in RELEVANT_CODE_FILES:
        digest = sha256(Path(relative).read_bytes()).hexdigest()
        records.append({"path": relative, "sha256": digest})
    return _canonical_sha256(records)


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _llama_cpp_version() -> str:
    import shutil

    executable = shutil.which("llama-server")
    if executable is None:
        return "NOT_INSTALLED_AT_B1_FREEZE"
    result = subprocess.run(
        [executable, "--version"], check=True, capture_output=True, text=True
    )
    return " ".join((result.stdout or result.stderr).split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--corpus-manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    args = parser.parse_args()
    result = build_freeze(
        private_root=args.private_root,
        corpus_root=args.corpus_root,
        corpus_manifest_path=args.corpus_manifest,
        profiles_path=args.profiles,
        coverage_path=args.coverage,
    )
    print(
        json.dumps(
            {
                "case_count": result["case_count"],
                "task_family_counts": result["task_family_counts"],
                "task_set_sha256": result["task_set_sha256"],
                "state_packet_set_sha256": result["state_packet_set_sha256"],
                "question_leakage_gate": result["leakage_report"]["question_audit"]["question_leakage_gate"],
                "dependency_gate": result["leakage_report"]["synthetic_t2_dependency_test"]["gate"],
                "model_calls": 0,
                "retrieval_calls": 0,
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
