from health_ai_copilot.contracts import Route
from health_ai_copilot.safety import route_question


def test_urgent_question_is_not_sent_to_the_model() -> None:
    result = route_question("我持续胸痛并且呼吸困难，应该怎么办？")

    assert result is not None
    assert result.route == Route.URGENT_CARE
    assert "急救" in result.message


def test_prescription_request_is_routed_to_human_review() -> None:
    result = route_question("可以给我开药并告诉我剂量吗？")

    assert result is not None
    assert result.route == Route.HUMAN_REVIEW


def test_patient_education_question_can_continue_to_retrieval() -> None:
    assert route_question("高血压患者日常低盐饮食有哪些原则？") is None

