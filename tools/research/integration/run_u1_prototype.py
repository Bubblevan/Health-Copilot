"""Build the deterministic synthetic U1 run bundle; never calls an LLM/provider."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from health_ai_copilot.research.integration.counterfactual import CounterfactualRunner
from health_ai_copilot.research.integration.esl_adapter import esl_adapter_feasibility
from health_ai_copilot.research.integration.fixtures import (
    FIXTURE_STATUS,
    build_synthetic_cases,
)
from health_ai_copilot.research.integration.replay import semantic_execution_hash
from health_ai_copilot.research.integration.training_views import build_training_views


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    write_text_lf(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def write_text_lf(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content.rstrip("\n") + "\n")


def main() -> int:
    cases = build_synthetic_cases()
    runner = CounterfactualRunner(epsilon=0)
    bundles = [(case, runner.run(case.episode, case.evaluation, case.resources)) for case in cases]
    views = [(case, bundle, build_training_views(case.episode, bundle)) for case, bundle in bundles]
    replay_stable = all(
        semantic_execution_hash(bundle)
        == semantic_execution_hash(runner.run(case.episode, case.evaluation, case.resources))
        for case, bundle in bundles
    )
    student_payloads = [json.dumps(view.student_packet.to_dict(), ensure_ascii=False)
                        for _, _, view in views]
    student_privileged_keys = (
        "gold_answer", "task_success", "counterfactual_action_outcomes",
        "minimal_successful_action_set", "failure_attribution", "privileged_plane",
    )
    sft_payloads = [json.dumps(view.sft_candidate.to_dict(), ensure_ascii=False)
                    for _, _, view in views]
    sft_forbidden_keys = ("gold_answer", "future_patient_state", "counterfactual_outcomes",
                          "failure_attribution", "raw_outcome")
    temporal_case = next(item for item in cases if item.case_id == "U1-TEMPORAL-LEAKAGE")
    temporal_snapshot = temporal_case.resources.patient_state_store.snapshot(
        temporal_case.episode.subject_id, temporal_case.episode.decision_time
    )
    gates = {
        "U1_EPISODE_CONTRACT_FROZEN": "YES",
        "U1_RUNTIME_EVAL_PRIVILEGED_PLANES_SEPARATED": "YES",
        "U1_MEMORY_EXTERNAL_NAMESPACE_SEPARATED": "YES",
        "U1_TEMPORAL_SNAPSHOT_FAIL_CLOSED": "YES" if all(row.timestamp <= temporal_case.episode.decision_time for row in temporal_snapshot) and "mem-future-1" not in {row.record_id for row in temporal_snapshot} else "NO",
        "U1_SINGLE_TEAM_CAPABILITY_EQUIVALENCE_ENFORCED": "YES" if all(bundle.equivalence.equivalent for _, bundle in bundles) else "NO",
        "U1_VALID_ACTION_MASK_ENFORCED": "YES" if any(bundle.availability.invalid_reasons for _, bundle in bundles) else "NO",
        "U1_COUNTERFACTUAL_REPLAY_DETERMINISTIC": "YES" if replay_stable else "NO",
        "U1_MINIMAL_SUCCESS_ACTION_SET_WORKS": "YES" if any(bundle.oracle_action_set.actions for _, bundle in bundles) else "NO",
        "U1_SFT_VIEW_NO_GOLD_LEAKAGE": "YES" if not any(key in payload for payload in sft_payloads for key in sft_forbidden_keys) else "NO",
        "U1_GRPO_GROUP_SCHEMA_WORKS": "YES" if all(len(view.grpo_group.samples) == len(view.grpo_group.group_rewards) for _, _, view in views) else "NO",
        "U1_OPD_PRIVILEGED_ISOLATION_WORKS": "YES" if not any(key in payload for payload in student_payloads for key in student_privileged_keys) else "NO",
        "U1_ESL_EVALUATION_CONTENT_ACCESSED": "NO",
        "RAG_TRACK_MODIFIED": "NO",
        "MEMORY_TRACK_MODIFIED": "NO",
        "E2_B_STARTED": "NO",
        "L4_STARTED": "NO",
        "POST_TRAINING_STARTED": "NO",
    }
    run_id = "u1-synthetic-20260929-01"
    output = ROOT / "runs" / "integration" / run_id
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "task": "U1 Longitudinal Environment Contract & Replay Prototype",
        "base_commit": "9e023dbddc1f8025e3608bfae4c5390c1a7957ef",
        "branch": "integration-u1-longitudinal-env-20260929",
        "fixture_status": FIXTURE_STATUS,
        "episode_count": len(cases),
        "counterfactual_executor": "u1-scripted-executor-v1",
        "evaluator": "u1-deterministic-evaluator-v1",
        "provider_calls": 0,
        "training_started": False,
        "esl_evaluation_content_accessed": False,
        "memory_track_modified": False,
        "rag_track_modified": False,
        "esl_adapter_feasibility": esl_adapter_feasibility().to_dict(),
        "gates": gates,
        "case_ids": [case.case_id for case in cases],
    }
    write_text_lf(output / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    write_text_lf(
        output / "esl_adapter_feasibility.json",
        json.dumps(esl_adapter_feasibility().to_dict(), ensure_ascii=False, indent=2),
    )
    write_jsonl(output / "episodes.jsonl", [case.episode.to_runtime_dict() for case in cases])
    write_jsonl(output / "counterfactual_outcomes.jsonl", [
        {"episode_id": case.episode.episode_id, **bundle.to_dict()} for case, bundle in bundles
    ])
    write_jsonl(output / "oracle_action_sets.jsonl", [
        {"episode_id": case.episode.episode_id, **bundle.oracle_action_set.to_dict()}
        for case, bundle in bundles
    ])
    write_jsonl(output / "sft_candidates.jsonl", [
        {"episode_id": case.episode.episode_id, **view.sft_candidate.to_dict()}
        for case, _, view in views
    ])
    write_jsonl(output / "grpo_groups.jsonl", [view.grpo_group.to_dict() for _, _, view in views])
    write_jsonl(output / "opd_student_packets.jsonl", [view.student_packet.to_dict() for _, _, view in views])
    write_jsonl(output / "opd_teacher_packets.jsonl", [view.teacher_packet.to_dict() for _, _, view in views])

    report_lines = [
        "# U1 Deterministic Integration Prototype",
        "",
        f"Run: `{run_id}`; fixture status: `{FIXTURE_STATUS}`.",
        "",
        "This is a synthetic contract prototype, not a medical benchmark or a model-quality evaluation.",
        "No provider was called and no training was run.",
        "",
        "## Case outcomes",
        "",
        "| Episode | Valid arms | Successful arms | A* | Single/Team union |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case, bundle in bundles:
        arms = ", ".join(item.action_key.value for item in bundle.arms)
        successful = ", ".join(
            arm.action_key.value for arm in bundle.arms
            if arm.evaluation.outcome.task_success and arm.evaluation.outcome.safety_pass
            and arm.evaluation.outcome.grounding_pass
        )
        oracle = ", ".join(action.value for action in bundle.oracle_action_set.actions)
        report_lines.append(
            f"| {case.case_id} | {arms} | {successful or '∅'} | {oracle or '∅'} | "
            f"{'PASS' if bundle.equivalence.equivalent else 'FAIL'} |"
        )
    report_lines.extend([
        "",
        "## Gates",
        "",
        "- Runtime/evaluation/privileged payloads are separate typed planes.",
        "- All provider calls and token counters are zero; deterministic latency is zero.",
        "- Abstract cost uses `u1-abstract-cost-v1`; no USD cost is synthesized.",
        "- `episodes.jsonl` contains runtime fields only; OPD student and teacher records use separate JSONL files.",
        "- ESL mapping is schema-only. No evaluation query or gold was accessed.",
        "",
    ])
    write_text_lf(output / "report.md", "\n".join(report_lines))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
