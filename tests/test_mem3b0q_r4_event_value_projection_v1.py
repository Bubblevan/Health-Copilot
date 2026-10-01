from __future__ import annotations

import json

import pytest

from tools.research.memory.mem3b0q_r4_event_value_projection_v1 import (
    EventProjectionError,
    project_event_value_spans,
)
from tools.research.memory import mem3b0q_r4 as frozen_r4
from tools.research.memory import mem3b0q_r4_candidate_builder_v1 as candidate_builder


@pytest.mark.parametrize(
    ("source", "value"),
    [
        ("My sister confirmed that I completed the purchase of my bicycle last month.", "completed the purchase"),
        ("My doctor remembers I completed the purchase of my bicycle two weeks ago.", "completed the purchase"),
        ("My diary records that I completed the purchase of my bicycle on May 14.", "completed the purchase"),
    ],
)
def test_event_value_is_projected_to_exact_source_cue(source: str, value: str) -> None:
    manifest = candidate_builder.build_candidates(
        source, frozen_r4.ALIASES, source_id="EVTCONF-X"
    )["candidates"]
    object_candidate = next(row for row in manifest["object"] if row["canonical_id"] == "BICYCLE")
    attribute_candidate = next(row for row in manifest["attribute"] if row["canonical_id"] == "PURCHASE_EVENT")
    nearest_owner = max(
        (row for row in manifest["owner"] if row["end"] <= object_candidate["start"]),
        key=lambda row: (row["end"], row["start"]),
    )
    content = json.dumps(
        {
            "source_id": "EVTCONF-X",
            "atoms": [
                {
                    "owner_candidate_id": "owner:SELF:0:2",
                    "object_candidate_id": object_candidate["candidate_id"],
                    "attribute_candidate_id": attribute_candidate["candidate_id"],
                    "value_span": "last month",
                }
            ],
            "abstention_reason": "NONE",
        }
    )
    projected, audit = project_event_value_spans(
        content,
        {"source_id": "EVTCONF-X", "proposition_text": source},
    )

    assert json.loads(projected)["atoms"][0]["value_span"] == value
    assert json.loads(projected)["atoms"][0]["owner_candidate_id"] == nearest_owner["candidate_id"]
    assert audit[0]["projected_value_span"] == value
    assert audit[0]["projected_owner_candidate_id"] == nearest_owner["candidate_id"]


def test_event_value_projection_fails_closed_without_a_cue() -> None:
    source = "I am considering the purchase of my bicycle."
    manifest = candidate_builder.build_candidates(
        source, frozen_r4.ALIASES, source_id="EVTCONF-X"
    )["candidates"]
    object_candidate = next(row for row in manifest["object"] if row["canonical_id"] == "BICYCLE")
    attribute_candidate = next(row for row in manifest["attribute"] if row["canonical_id"] == "PURCHASE_EVENT")
    content = json.dumps(
        {
            "source_id": "EVTCONF-X",
            "atoms": [
                {
                    "owner_candidate_id": "owner:SELF:0:2",
                    "object_candidate_id": object_candidate["candidate_id"],
                    "attribute_candidate_id": attribute_candidate["candidate_id"],
                    "value_span": "last month",
                }
            ],
            "abstention_reason": "NONE",
        }
    )
    with pytest.raises(EventProjectionError, match="event_cue_not_unique"):
        project_event_value_spans(
            content,
            {"source_id": "EVTCONF-X", "proposition_text": source},
        )
