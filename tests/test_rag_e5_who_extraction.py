from __future__ import annotations

import pytest

from eval.rag_e5.corpus import RecommendationBlock
from eval.rag_e5.who_extraction import (
    audit_recommendation_retention,
    canonical_retained_section_ids,
)


def test_physical_activity_retention_keeps_disability_population_blocks() -> None:
    expected = (
        "children_5_17_physical_activity",
        "children_5_17_sedentary",
        "adults_18_64_physical_activity",
        "adults_18_64_sedentary",
        "older_adults_65_physical_activity",
        "older_adults_65_sedentary",
        "pregnant_postpartum_physical_activity",
        "pregnant_postpartum_sedentary",
        "chronic_conditions_physical_activity",
        "chronic_conditions_sedentary",
        "disability_physical_activity",
        "disability_sedentary",
    )
    blocks = tuple(
        RecommendationBlock(
            section_path=("population", "topic", section_id),
            recommendation_id=block_id,
            paragraphs=("A population-level recommendation statement.",),
        )
        for section_id, block_id in (
            *((section_id, section_id) for section_id in expected if not section_id.startswith("disability_")),
            ("disability_physical_activity", "disability_physical_activity-children"),
            ("disability_physical_activity", "disability_physical_activity-adults"),
            ("disability_sedentary", "disability_sedentary-children"),
            ("disability_sedentary", "disability_sedentary-adults"),
        )
    )

    retained = canonical_retained_section_ids(
        source_id="who-physical-activity-sedentary-2020",
        declared_section_ids=expected,
        blocks=blocks,
    )
    audit = audit_recommendation_retention(
        expected_section_ids=expected,
        retained_section_ids=retained,
    )

    assert len(blocks) == 14
    assert len(retained) == 12
    assert audit == {
        "missing_section_ids": [],
        "unexpected_section_ids": [],
        "duplicate_section_ids": [],
    }


def test_retention_audit_detects_missing_unexpected_and_duplicate_sections() -> None:
    audit = audit_recommendation_retention(
        expected_section_ids=("rec-1", "rec-2"),
        retained_section_ids=("rec-1", "rec-1", "other"),
    )

    assert audit == {
        "missing_section_ids": ["rec-2"],
        "unexpected_section_ids": ["other"],
        "duplicate_section_ids": ["rec-1"],
    }


def test_canonical_retention_rejects_dropped_hypertension_section() -> None:
    blocks = tuple(
        RecommendationBlock(
            section_path=("hypertension", str(index)),
            recommendation_id=f"recommendation-{index}",
            paragraphs=("A recommendation statement.",),
        )
        for index in range(1, 8)
    )

    with pytest.raises(ValueError, match="block count does not match"):
        canonical_retained_section_ids(
            source_id="who-hypertension-pharmacological-2021",
            declared_section_ids=tuple(f"3.{index}" for index in range(1, 9)),
            blocks=blocks,
        )
