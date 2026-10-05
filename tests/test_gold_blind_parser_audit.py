from health_ai_copilot.harness.contracts import AnswerSchema
from tools.eval.audit_gold_blind_parse_failures import _classify_unparsed


def test_gold_blind_audit_separates_invalid_classifier_from_runtime_failure() -> None:
    category, details = _classify_unparsed(
        "目前无法可靠完成这项分析。",
        ["reasoning_failure:ValueError"],
        AnswerSchema.SINGLE_CHOICE,
        {
            "fallback_reason": "ValueError:invalid_complexity_output",
            "validation_errors": ["ValueError:invalid_complexity_output"],
        },
    )
    assert category == "F_ADAPTIVE_CLASSIFIER_OUTPUT_INVALID"
    assert details["strategy_failure_reason"] == "ValueError:invalid_complexity_output"


def test_gold_blind_audit_separates_recruitment_and_transport_failures() -> None:
    recruitment, _ = _classify_unparsed(
        "目前无法可靠完成这项分析。",
        ["reasoning_failure:TypeError"],
        AnswerSchema.SINGLE_CHOICE,
        {"fallback_reason": "TypeError:invalid_team_recruitment"},
    )
    transport, _ = _classify_unparsed(
        "目前无法可靠完成这项分析。",
        ["reasoning_failure:APIConnectionError"],
        AnswerSchema.SINGLE_CHOICE,
        {"fallback_reason": "APIConnectionError:connectionerror"},
    )
    assert recruitment == "G_ADAPTIVE_RECRUITMENT_OUTPUT_INVALID"
    assert transport == "H_PROVIDER_TRANSPORT_FAILURE"
