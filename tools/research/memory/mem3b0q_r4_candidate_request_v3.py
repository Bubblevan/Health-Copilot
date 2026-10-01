"""R4 v3 prompt clarifies event value versus temporal expression."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tools.research.memory import mem3b0q_r4_candidate_request_v2 as request_v2


REQUEST_BUILDER_VERSION = "mem3b0q-r4-candidate-request-v3-event-value-boundary"
EVENT_VALUE_INSTRUCTION = (
    " For a PURCHASE_EVENT attribute, value_span must name the event/action that "
    "occurred, not a phrase that only says when it occurred. For example, 'last "
    "month' is a time expression, not an event value. Use only an explicit event "
    "phrase present in the source; do not invent one."
)


def build_candidate_request(case: Mapping[str, Any], *, model: str) -> dict[str, Any]:
    request = request_v2.build_candidate_request(case, model=model)
    request["messages"][0]["content"] += EVENT_VALUE_INSTRUCTION
    request["response_format"]["json_schema"]["name"] = (
        "mem3b0q_r4_candidate_proposal_v3"
    )
    return request
