"""Build the U1.1 synthetic fairness bundle without external data or providers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from health_ai_copilot.research.integration.contracts import FailureCategory
from health_ai_copilot.research.integration.counterfactual import CounterfactualRunner
from health_ai_copilot.research.integration.fixtures import FIXTURE_STATUS, build_synthetic_cases
from health_ai_copilot.research.integration.training_views import (
    StudentActionSource,
    build_training_views,
)

RUN_ID = "20260929-01"
OUTPUT = ROOT / "runs" / "integration" / f"u1-1-fairness-{RUN_ID}"
U1_BASE = "c6df93da827a505c757ecdf064fffc7301854680"
BRANCH = "integration-u1-1-exec-fairness-20260929"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8", newline="\n")


def _successful(arm) -> bool:
    outcome = arm.evaluation.outcome
    return outcome.task_success and outcome.safety_pass and outcome.grounding_pass


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite an existing U1.1 run: {OUTPUT}")
    cases = build_synthetic_cases()
    runner = CounterfactualRunner(epsilon=0)
    bundles = [(case, runner.run(case.episode, case.evaluation, case.resources)) for case in cases]
    views = [(case, bundle, build_training_views(
        case.episode,
        bundle,
        student_action="NONE",
        student_action_source=StudentActionSource.SCRIPTED_PROBE,
    )) for case, bundle in bundles]

    team_case = next(case for case in cases if case.case_id == "U1-TEAM")
    team_bundle = next(bundle for case, bundle in bundles if case.case_id == "U1-TEAM")
    team_arms = {item.action_key.value: item for item in team_bundle.arms}
    def tool_observation_signature(arm):
        return [(
            item.tool_id, item.output, item.resource_ids, item.resource_versions,
            item.input_hash, item.implementation_hash, item.output_hash,
        ) for item in arm.execution.tool_observations]
    executable_shared_tools = (
        tool_observation_signature(team_arms["NONE"])
        == tool_observation_signature(team_arms["TEAM"])
        and [item.tool_id for item in team_arms["NONE"].execution.tool_observations]
        == ["query_part_a", "query_part_b"]
    )
    single_team_success_equal = (
        _successful(team_arms["NONE"]) == _successful(team_arms["TEAM"])
        and team_arms["NONE"].evaluation.outcome.answer == team_arms["TEAM"].evaluation.outcome.answer
        and team_arms["NONE"].evaluation.outcome.used_evidence_ids
        == team_arms["TEAM"].evaluation.outcome.used_evidence_ids
        and team_arms["NONE"].evaluation.outcome.safety_pass
        == team_arms["TEAM"].evaluation.outcome.safety_pass
        and team_arms["NONE"].evaluation.outcome.grounding_pass
        == team_arms["TEAM"].evaluation.outcome.grounding_pass
    )
    declared_equal = all(bundle.equivalence.equivalent for _, bundle in bundles)
    executable_equal = all(bundle.executable_equivalence.equivalent for _, bundle in bundles)
    all_budgeted = all(
        arm.execution.budget_audit is not None
        and arm.execution.budget_audit.within_global_budget
        and arm.execution.budget_audit.worker_budgets_sum_to_global
        for _, bundle in bundles for arm in bundle.arms
    )
    training_sources_explicit = all(
        view.student_packet.student_action_source == StudentActionSource.SCRIPTED_PROBE
        for _, _, view in views
    )
    student_json = [json.dumps(view.student_packet.to_dict(), ensure_ascii=False) for _, _, view in views]
    student_forbidden = ("gold_answer", "task_success", "safety_pass", "grounding_pass",
                         "counterfactual_action_outcomes", "minimal_successful_action_set",
                         "arm_failure_categories", "bundle_counterfactual_attributions")
    student_isolated = not any(key in row for row in student_json for key in student_forbidden)
    runtime_json = [json.dumps(case.episode.to_runtime_dict(), ensure_ascii=False) for case in cases]
    sft_json = [json.dumps(view.sft_candidate.to_dict(), ensure_ascii=False) for _, _, view in views]
    runtime_sft_isolated = not any(
        key in row
        for row in runtime_json + sft_json
        for key in ("gold_answer", "required_facts", "counterfactual_outcomes", "failure_attribution")
    )
    opd_schema_only = all(view.teacher_packet.opd_data_status.value == "SCHEMA_PROBE"
                          for _, _, view in views)
    grpo_counterfactual = all(view.grpo_group.rollout_source.value == "COUNTERFACTUAL_ENUMERATION"
                              for _, _, view in views)
    single_tools_executable = executable_shared_tools and all(
        not bundle.executable_equivalence.single.unimplemented_tool_ids
        and not bundle.executable_equivalence.team_union.unimplemented_tool_ids
        for _, bundle in bundles
    )
    gates = {
        "U1_1_DECLARED_CAPABILITY_EQUIVALENCE": "YES" if declared_equal else "NO",
        "U1_1_EXECUTABLE_CAPABILITY_EQUIVALENCE": "YES" if executable_equal else "NO",
        "U1_1_SINGLE_SHARED_TOOLS_EXECUTABLE": "YES" if single_tools_executable else "NO",
        "U1_1_TEAM_HAS_NO_EXTRA_DATA_AUTHORITY": "YES" if executable_equal else "NO",
        "U1_1_ARCHITECTURE_AGNOSTIC_EVALUATOR": "YES" if single_team_success_equal else "NO",
        "U1_1_REQUIRES_TEAM_GOLD_REMOVED_OR_QUARANTINED": "YES" if all(
            not hasattr(case.evaluation, "requires_team")
            and "requires_team" not in case.evaluation.to_evaluation_dict()
            for case in cases
        ) else "NO",
        "U1_1_SINGLE_TOOL_COST_ACCOUNTED": "YES" if all(
            arm.execution.outcome.tool_calls == arm.cost.tool_units
            for _, bundle in bundles for arm in bundle.arms
        ) else "NO",
        "U1_1_GLOBAL_BUDGET_PARITY": "YES" if all_budgeted else "NO",
        "U1_1_TEAM_VALUE_COUNTERFACTUAL_ONLY": "YES" if (
            "MISSING_TEAM" not in FailureCategory.__members__
            and all(isinstance(item.category.value, str)
                    for _, bundle in bundles for item in bundle.attributions)
        ) else "NO",
        "U1_1_STUDENT_ACTION_SOURCE_EXPLICIT": "YES" if training_sources_explicit else "NO",
        "U1_1_NO_POLICY_SAMPLE_TAG_GENERATED": "YES" if training_sources_explicit else "NO",
        "U1_1_OPD_STATUS": "SCHEMA_PROBE" if opd_schema_only else "NO",
        "U1_1_GRPO_ROLLOUT_SOURCE": "COUNTERFACTUAL_ENUMERATION" if grpo_counterfactual else "NO",
        "U1_1_RUNTIME_EVAL_PRIVILEGED_ISOLATION": "YES"
        if student_isolated and runtime_sft_isolated else "NO",
        "U1_1_EXTERNAL_TEST_CONTENT_ACCESSED": "NO",
        "U0_REMOTE_PROVENANCE": "LOCAL_ONLY",
        "MEMORY_TRACK_MODIFIED": "NO",
        "RAG_TRACK_MODIFIED": "NO",
        "E2_B_STARTED": "NO",
        "L4_STARTED": "NO",
        "POST_TRAINING_STARTED": "NO",
    }
    required_yes = (
        "U1_1_DECLARED_CAPABILITY_EQUIVALENCE",
        "U1_1_EXECUTABLE_CAPABILITY_EQUIVALENCE",
        "U1_1_SINGLE_SHARED_TOOLS_EXECUTABLE",
        "U1_1_TEAM_HAS_NO_EXTRA_DATA_AUTHORITY",
        "U1_1_ARCHITECTURE_AGNOSTIC_EVALUATOR",
        "U1_1_REQUIRES_TEAM_GOLD_REMOVED_OR_QUARANTINED",
        "U1_1_SINGLE_TOOL_COST_ACCOUNTED",
        "U1_1_GLOBAL_BUDGET_PARITY",
        "U1_1_TEAM_VALUE_COUNTERFACTUAL_ONLY",
        "U1_1_STUDENT_ACTION_SOURCE_EXPLICIT",
        "U1_1_NO_POLICY_SAMPLE_TAG_GENERATED",
        "U1_1_RUNTIME_EVAL_PRIVILEGED_ISOLATION",
    )
    if any(gates[name] != "YES" for name in required_yes):
        raise AssertionError(f"U1.1 completion gate failed: {gates}")
    if gates["U1_1_OPD_STATUS"] != "SCHEMA_PROBE":
        raise AssertionError("U1.1 OPD gate failed")
    if gates["U1_1_GRPO_ROLLOUT_SOURCE"] != "COUNTERFACTUAL_ENUMERATION":
        raise AssertionError("U1.1 GRPO gate failed")

    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / "manifest.json", {
        "run_id": RUN_ID,
        "task": "U1.1 Execution Semantics Fairness & Training-View Hardening",
        "base_commit": U1_BASE,
        "branch": BRANCH,
        "fixture_status": FIXTURE_STATUS,
        "episode_count": len(cases),
        "executor": runner.executor.version,
        "evaluator": runner.evaluator.version,
        "provider_calls": 0,
        "training_started": False,
        "external_test_content_accessed": False,
        "u0_remote_provenance": "LOCAL_ONLY",
        "opd_data_status": "SCHEMA_PROBE",
        "grpo_rollout_source": "COUNTERFACTUAL_ENUMERATION",
        "student_action_sources": sorted({view.student_packet.student_action_source.value
                                            for _, _, view in views}),
        "gates": gates,
        "case_ids": [case.case_id for case in cases],
        "historical_u1_run_mutated": False,
    })
    write_json(OUTPUT / "executable_equivalence.json", {
        "schema": "u1.1-executable-equivalence-v1",
        "all_cases_equivalent": executable_equal,
        "cases": [{"episode_id": case.episode.episode_id,
                   **bundle.executable_equivalence.to_dict()}
                  for case, bundle in bundles],
    })
    write_jsonl(OUTPUT / "counterfactual_outcomes.jsonl", [
        {"episode_id": case.episode.episode_id, **bundle.to_dict()}
        for case, bundle in bundles
    ])
    write_jsonl(OUTPUT / "cost_accounting.jsonl", [
        {"episode_id": case.episode.episode_id,
         "action": arm.action_key.value,
         "architecture": arm.action.architecture.value,
         "budget": case.episode.budget.to_dict(),
         "activation_cost": arm.cost.activation.to_dict(),
         "observed_usage": arm.cost.observed.to_dict(),
         "cost": arm.cost.to_dict(),
         "budget_audit": arm.execution.budget_audit.to_dict()
         if arm.execution.budget_audit else None}
        for case, bundle in bundles for arm in bundle.arms
    ])
    write_jsonl(OUTPUT / "training_view_provenance.jsonl", [
        {"episode_id": case.episode.episode_id,
         "sft_label_source": dict(view.sft_candidate.provenance)["label_source"],
         "sft_provenance": dict(view.sft_candidate.provenance),
         "student_action": view.student_packet.student_action,
         "student_action_source": view.student_packet.student_action_source.value,
         "student_provenance": dict(view.student_packet.provenance),
         "opd_data_status": view.teacher_packet.opd_data_status.value,
         "grpo_rollout_source": view.grpo_group.rollout_source.value,
         "privileged_data_in_student_packet": False}
        for case, _, view in views
    ])

    lines = [
        "# U1.1 Execution Semantics Fairness Run",
        "",
        f"Run: `{RUN_ID}`. Fixture: `{FIXTURE_STATUS}`.",
        "",
        (
            "This bundle uses synthetic deterministic resources only. It is not a medical benchmark, "
            "calls no provider, performs no training, and reads no external benchmark rows."
        ),
        "",
        "## Counterfactual results",
        "",
        "| Episode | Successful arms | A* | Executable parity |",
        "| --- | --- | --- | --- |",
    ]
    for case, bundle in bundles:
        successful = ", ".join(item.action_key.value for item in bundle.arms if _successful(item)) or "∅"
        selected = ", ".join(item.value for item in bundle.oracle_action_set.actions) or "∅"
        lines.append(f"| {case.case_id} | {successful} | {selected} | "
                     f"{'PASS' if bundle.executable_equivalence.equivalent else 'FAIL'} |")
    lines.extend([
        "",
        "## Fairness interpretation",
        "",
        (
            f"For `{team_case.case_id}`, Single and Team both execute `query_part_a` and `query_part_b` "
            f"through the same registered implementations and return `{team_arms['NONE'].execution.outcome.answer}`. "
            f"Single cost is {team_arms['NONE'].cost.total_units} abstract units; Team cost is "
            f"{team_arms['TEAM'].cost.total_units}. The corrected A* is `NONE` (Single), because the "
            "old Team-only success came from evaluator/fixture semantics and a Team-only execution path. "
            "This change repairs the contract; it is not a regression and it does not demonstrate a real Team gain."
        ),
        "",
        "## Provenance",
        "",
        "- Tool calls record implementation hashes, input hashes, resource versions, and output hashes.",
        "- Total abstract cost is activation cost plus observed workers/tools/reads/retrievals; no USD is inferred.",
        "- One global budget is shared across architectures; Team worker shares partition that same cap.",
        "- Student actions are `SCRIPTED_PROBE`; OPD status is `SCHEMA_PROBE`; GRPO source is `COUNTERFACTUAL_ENUMERATION`.",
        "- SFT labels come from the deterministic counterfactual oracle and list backend, evaluator, cost model, epsilon, and arms.",
        "- U0 commit `df8cac1` is `LOCAL_ONLY`; it was not included in this run or branch.",
        "",
    ])
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
