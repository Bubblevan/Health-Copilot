"""Deterministic safety checks that run before any LLM call.

This is deliberately a narrow prototype policy. A production medical product
requires clinical governance, validation and formal operational processes.
"""

from .contracts import AssistantResponse, Route


URGENT_MARKERS = (
    "胸痛",
    "呼吸困难",
    "意识不清",
    "昏迷",
    "大出血",
    "自杀",
)

PRESCRIPTION_MARKERS = ("开药", "处方", "剂量", "停药", "加药")


def route_question(question: str) -> AssistantResponse | None:
    """Return a safe fixed response when the request must not reach the LLM."""
    normalized = question.replace(" ", "")

    urgent_hits = [marker for marker in URGENT_MARKERS if marker in normalized]
    if urgent_hits:
        return AssistantResponse(
            route=Route.URGENT_CARE,
            message=(
                "你描述的情况可能需要紧急医疗帮助。请立即联系当地急救服务或前往急诊；"
                "不要等待在线回复。"
            ),
            safety_reasons=[f"urgent_marker:{marker}" for marker in urgent_hits],
        )

    prescription_hits = [marker for marker in PRESCRIPTION_MARKERS if marker in normalized]
    if prescription_hits:
        return AssistantResponse(
            route=Route.HUMAN_REVIEW,
            message=(
                "这个原型不提供用药或处方决策。请咨询有资质的医生或药师，"
                "并携带完整病史、检查结果与正在使用的药物信息。"
            ),
            safety_reasons=[f"prescription_marker:{marker}" for marker in prescription_hits],
        )

    return None

