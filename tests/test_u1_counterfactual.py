from health_ai_copilot.research.integration.counterfactual import CounterfactualRunner
from health_ai_copilot.research.integration.fixtures import build_synthetic_cases


def test_required_minimal_actions_and_valid_arm_mask() -> None:
    expected = {
        "U1-NONE": ("NONE",),
        "U1-MEM": ("MEMORY",),
        "U1-RAG": ("RAG",),
        "U1-MEM-RAG": ("MEMORY+RAG",),
        "U1-TEAM": ("NONE",),
        "U1-MEM-TEAM": ("MEMORY",),
        "U1-ALL": ("MEMORY+RAG",),
        "U1-OOD-INSUFFICIENT": ("NONE",),
    }
    runner = CounterfactualRunner(epsilon=0)
    for case in build_synthetic_cases():
        bundle = runner.run(case.episode, case.evaluation, case.resources)
        assert all(item.action_key in bundle.availability.valid for item in bundle.arms)
        if case.case_id in expected:
            assert tuple(item.value for item in bundle.oracle_action_set.actions) == expected[case.case_id]
        for arm in bundle.arms:
            assert arm.evaluation.outcome.provider_calls == 0
            assert arm.evaluation.outcome.input_tokens == 0
            assert arm.evaluation.outcome.output_tokens == 0
        identities = [arm.identity.to_dict() for arm in bundle.arms]
        shared_fields = set(identities[0]) - {"action_hash"}
        assert all(all(row[name] == identities[0][name] for name in shared_fields) for row in identities)


def test_invalid_action_is_masked_when_no_history_or_evidence_world() -> None:
    case = next(item for item in build_synthetic_cases() if item.case_id == "U1-NONE")
    mask = CounterfactualRunner().run(case.episode, case.evaluation, case.resources).availability
    assert "MEMORY_READ_REQUIRES_SUBJECT_HISTORY" == mask.rejection_reason("MEMORY")
    assert "EXTERNAL_RETRIEVAL_REQUIRES_EVIDENCE_WORLD" == mask.rejection_reason("RAG")


def test_epsilon_keeps_near_equal_successful_action_set() -> None:
    case = next(item for item in build_synthetic_cases() if item.case_id == "U1-MEM")
    bundle = CounterfactualRunner(epsilon=3).run(case.episode, case.evaluation, case.resources)
    assert {item.value for item in bundle.oracle_action_set.actions} == {"MEMORY", "MEMORY+TEAM"}
