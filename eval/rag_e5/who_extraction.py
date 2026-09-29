"""Deterministic extraction of WHO recommendation prose for the E5-A3 corpus."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping

from eval.rag_e5.corpus import RecommendationBlock

WHO_EXTRACTOR_VERSION = "who-recommendation-prose-poppler-raw-v1"

_ACTIVITY_PAGE_SPECS = (
    ("children_5_17_physical_activity", "Children and adolescents aged 5–17", 35, 1),
    ("children_5_17_sedentary", "Children and adolescents aged 5–17", 39, 1),
    ("adults_18_64_physical_activity", "Adults aged 18–64", 42, 1),
    ("adults_18_64_sedentary", "Adults aged 18–64", 48, 1),
    ("older_adults_65_physical_activity", "Older adults aged 65+", 53, 1),
    ("older_adults_65_sedentary", "Older adults aged 65+", 56, 1),
    ("pregnant_postpartum_physical_activity", "Pregnant and postpartum women", 57, 1),
    ("pregnant_postpartum_sedentary", "Pregnant and postpartum women", 61, 1),
    ("chronic_conditions_physical_activity", "Adults with chronic conditions", 62, 1),
    ("chronic_conditions_sedentary", "Adults with chronic conditions", 68, 1),
    ("disability_physical_activity", "People living with disability", 70, 1),
    ("disability_physical_activity", "People living with disability", 71, 1),
    ("disability_sedentary", "People living with disability", 74, 2),
)

_HYPERTENSION_SECTIONS = (
    (
        "3.1 Blood pressure threshold for initiation of pharmacological treatment",
        "1",
        "Blood pressure threshold for pharmacological treatment",
    ),
    (
        "3.2 Laboratory testing before and during pharmacological treatment",
        "2",
        "Laboratory testing during pharmacological treatment",
    ),
    (
        "3.3 Cardiovascular disease risk assessment as guide to initiation of antihypertensive medications",
        "3",
        "Cardiovascular risk assessment",
    ),
    (
        "3.4 Drug classes to be used as first-line agents",
        "4",
        "First-line antihypertensive drug classes",
    ),
    ("3.5 Combination therapy", "5", "Combination therapy"),
    ("3.6 Target blood pressure", "6", "Blood-pressure treatment targets"),
    ("3.7 Frequency of re-assessment", "7", "Treatment reassessment frequency"),
    (
        "3.8 Administration of treatment by nonphysician professionals",
        "8",
        "Administration by nonphysician professionals",
    ),
)

_GRADE = re.compile(r"(?:Strong|Conditional) recommendation\b[^\n]*", re.IGNORECASE)
_PAGE_OR_FOOTER = re.compile(
    r"^(?:\d+|Recommendations|WHO guidelines(?: on)?|Available online at .*)$", re.IGNORECASE
)


def extract_who_blocks(
    *, source_id: str, full_text: str, pages: Mapping[int, str] | None = None
) -> tuple[RecommendationBlock, ...]:
    """Extract only explicit WHO recommendation statements and source remarks.

    No rationale, evidence tables, figures, or bibliography are returned. Activity
    recommendations are page-scoped to the recommendation panels in the pinned
    2020 edition; the two disability populations remain separate blocks.
    """
    if source_id == "who-hypertension-pharmacological-2021":
        return _extract_hypertension(full_text)
    if source_id == "who-total-fat-weight-gain-2023":
        scoped_text = "\n\f\n".join(pages[page] for page in (30, 31, 32)) if pages else full_text
        return _extract_total_fat(scoped_text)
    if source_id == "who-physical-activity-sedentary-2020":
        if pages is None:
            raise ValueError("physical-activity extraction requires pinned recommendation pages")
        return _extract_activity(pages)
    raise ValueError(f"unsupported WHO source: {source_id}")


def canonical_retained_section_ids(
    *, source_id: str, declared_section_ids: tuple[str, ...], blocks: tuple[RecommendationBlock, ...]
) -> tuple[str, ...]:
    """Map extracted blocks to the source manifest's canonical section IDs."""
    expected = tuple(declared_section_ids)
    if not expected or any(not section_id for section_id in expected):
        raise ValueError("declared recommendation section IDs must be non-empty")
    if len(expected) != len(set(expected)):
        raise ValueError(f"duplicate declared recommendation sections for {source_id}")

    if source_id in {"who-hypertension-pharmacological-2021", "who-total-fat-weight-gain-2023"}:
        if len(blocks) != len(expected):
            raise ValueError(f"recommendation block count does not match declared sections for {source_id}")
        for index, block in enumerate(blocks, start=1):
            if block.recommendation_id != f"recommendation-{index}":
                raise ValueError(f"non-canonical recommendation order for {source_id}")
        return expected

    if source_id == "who-physical-activity-sedentary-2020":
        if any(not block.section_path for block in blocks):
            raise ValueError("physical-activity extraction produced a block without a section ID")
        return tuple(sorted({block.section_path[-1] for block in blocks}))

    raise ValueError(f"unsupported WHO source: {source_id}")


def audit_recommendation_retention(
    *, expected_section_ids: tuple[str, ...], retained_section_ids: tuple[str, ...]
) -> dict[str, list[str]]:
    """Return exact section retention discrepancies for a source."""
    expected = set(expected_section_ids)
    retained_counts = Counter(retained_section_ids)
    retained = set(retained_counts)
    return {
        "missing_section_ids": sorted(expected - retained),
        "unexpected_section_ids": sorted(retained - expected),
        "duplicate_section_ids": sorted(
            section_id for section_id, count in retained_counts.items() if count > 1
        ),
    }


def _extract_hypertension(text: str) -> tuple[RecommendationBlock, ...]:
    flat = " ".join(text.split())
    blocks: list[RecommendationBlock] = []
    section_positions: list[tuple[int, str, str, str]] = []
    for heading, recommendation_number, title in _HYPERTENSION_SECTIONS:
        position = flat.rfind(heading)
        if position < 0:
            raise ValueError(f"hypertension recommendation section not found: {heading}")
        section_positions.append((position, heading, recommendation_number, title))
    section_positions.sort()
    for index, (start, heading, recommendation_number, title) in enumerate(section_positions):
        end = section_positions[index + 1][0] if index + 1 < len(section_positions) else len(flat)
        section = flat[start:end]
        label = re.search(
            rf"\b{recommendation_number}\.\s+RECOMMENDATIONS?\s+ON\s+", section, re.IGNORECASE
        )
        if label is None:
            raise ValueError(f"recommendation label missing for {heading}")
        body_start = label.end()
        evidence_start = re.search(
            r"\b(?:Evidence and rationale|There was a minimal number of comparative studies)\b",
            section[body_start:],
            re.IGNORECASE,
        )
        if evidence_start is None:
            raise ValueError(f"rationale boundary missing for {heading}")
        body = section[body_start : body_start + evidence_start.start()].strip(" :;.")
        body = re.sub(r"\s*\[\s*\[\s*", " ", body)
        body = re.sub(r"\s*\]\s*\]\s*", " ", body)
        body = re.sub(r"\s+", " ", body).strip()
        if not body or not re.search(r"\b(Strong|Conditional) recommendation\b", body, re.IGNORECASE):
            raise ValueError(f"recommendation text or grade missing for {heading}")
        blocks.append(
            RecommendationBlock(
                section_path=("WHO hypertension pharmacological treatment", title),
                recommendation_id=f"recommendation-{recommendation_number}",
                paragraphs=(body,),
            )
        )
    if len(blocks) != 8:
        raise ValueError("expected all eight WHO hypertension recommendation sections")
    return tuple(blocks)


def _extract_total_fat(text: str) -> tuple[RecommendationBlock, ...]:
    recommendations_start = text.find("WHO recommendations")
    rec_one_start = re.search(r"\b1\.\s+To reduce the risk of unhealthy weight gain", text)
    rec_two_start = re.search(r"\b2\.\s+Fat consumed should be primarily unsaturated", text)
    if recommendations_start < 0 or rec_one_start is None or rec_two_start is None:
        raise ValueError("WHO total-fat recommendation list was not found")
    rationale_one = _find_after(text, "Rationale for recommendation 1", recommendations_start)
    remarks_one = _find_after(text, "Remarks for recommendation 1", rationale_one)
    rationale_two = _find_after(text, "Rationale for recommendation 2", remarks_one)
    remarks_two = _find_after(text, "Remarks for recommendation 2", rationale_two)

    rec_one = re.sub(
        r"^\s*\d+\.\s*", "", " ".join(text[rec_one_start.start() : rec_two_start.start()].split())
    ).strip()
    rec_two = re.sub(
        r"^\s*\d+\.\s*", "", " ".join(text[rec_two_start.start() : rationale_one].split())
    ).strip()
    remarks_one_rows = _bullet_paragraphs(text[remarks_one + len("Remarks for recommendation 1") : rationale_two])
    remarks_two_rows = _bullet_paragraphs(
        text[remarks_two + len("Remarks for recommendation 2") :]
    )
    if not rec_one or not rec_two or not remarks_one_rows or not remarks_two_rows:
        raise ValueError("total-fat recommendations or their scope remarks were not fully extracted")
    return (
        RecommendationBlock(
            section_path=("WHO total-fat guideline", "Recommendation 1: unhealthy weight-gain prevention"),
            recommendation_id="recommendation-1",
            paragraphs=(rec_one, *remarks_one_rows),
        ),
        RecommendationBlock(
            section_path=("WHO total-fat guideline", "Recommendation 2: fat quality"),
            recommendation_id="recommendation-2",
            paragraphs=(rec_two, *remarks_two_rows),
        ),
    )


def _extract_activity(pages: Mapping[int, str]) -> tuple[RecommendationBlock, ...]:
    blocks: list[RecommendationBlock] = []
    for section_id, population, page_number, expected_groups in _ACTIVITY_PAGE_SPECS:
        page = pages.get(page_number)
        if not isinstance(page, str) or not page.strip():
            raise ValueError(f"required physical-activity page is missing: {page_number}")
        starts = list(re.finditer(r"It is recommended that", page, re.IGNORECASE))
        if len(starts) != expected_groups:
            raise ValueError(
                f"expected {expected_groups} recommendation group(s) on page {page_number}, "
                f"found {len(starts)}"
            )
        for group_index, match in enumerate(starts):
            region_end = starts[group_index + 1].start() if group_index + 1 < len(starts) else len(page)
            region = page[match.start() : region_end]
            boundary = re.search(
                r"\b(?:GOOD\s+PRACTICE\s+STATEMENTS?|PHYSICAL\s+ACTIVITY\s+RECOMMENDATION|"
                r"SEDENTARY\s+BEHAVIOUR|Supporting\s+evidence\s+and\s+rationale|"
                r"For\s+.+?\s+(?:physical\s+activity\s+can\s+be\s+undertaken|"
                r"sedentary\s+behaviour\s+is\s+defined))\b",
                region,
                re.IGNORECASE,
            )
            if boundary is not None:
                region = region[: boundary.start()]
            grades = list(_GRADE.finditer(region))
            if not grades:
                raise ValueError(
                    f"recommendation grade missing for {section_id} on page {page_number}"
                )
            region = region[: grades[-1].end()]
            cleaned = _clean_page_region(region)
            if len(cleaned.split()) < 7:
                raise ValueError(f"extracted physical-activity recommendation is too short: {section_id}")
            subgroup = _activity_subgroup(section_id, page_number, group_index)
            blocks.append(
                RecommendationBlock(
                    section_path=(population, _activity_topic(section_id), section_id),
                    recommendation_id=subgroup,
                    paragraphs=(cleaned,),
                )
            )
    if len(blocks) != 14:
        raise ValueError("expected 14 recommendation blocks across 12 WHO activity sections")
    return tuple(blocks)


def _activity_topic(section_id: str) -> str:
    return "sedentary behaviour" if section_id.endswith("sedentary") else "physical activity"


def _activity_subgroup(section_id: str, page_number: int, group_index: int) -> str:
    if section_id == "disability_physical_activity":
        population = "children" if page_number == 70 else "adults"
        return f"{section_id}-{population}"
    if section_id == "disability_sedentary":
        population = "children" if group_index == 0 else "adults"
        return f"{section_id}-{population}"
    return section_id


def _clean_page_region(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.replace("\f", "\n").splitlines():
        line = " ".join(raw_line.split())
        if not line or _PAGE_OR_FOOTER.fullmatch(line):
            continue
        if line.startswith(
            ("See the section Evidence to recommendations", "Infants should be exclusively breastfed")
        ):
            continue
        if line.casefold() in {"it is recommended that:", "it is recommended that"}:
            if not lines:
                lines.append("It is recommended that")
            continue
        if line in {"It is recommended that:", "It is recommended that"}:
            continue
        lines.append(line)
    return " ".join(lines).strip()


def _find_after(text: str, marker: str, start: int) -> int:
    position = text.find(marker, max(start, 0))
    if position < 0:
        raise ValueError(f"required extraction boundary not found: {marker}")
    return position


def _bullet_paragraphs(text: str) -> list[str]:
    rows: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line or _PAGE_OR_FOOTER.fullmatch(line):
            continue
        if re.fullmatch(r"\d+\.?", line):
            continue
        bullet = re.match(r"^(?:▶+|•|▪|–|—)\s*(.*)$", line)
        if bullet:
            if current:
                rows.append(" ".join(current).strip())
            current = [bullet.group(1)] if bullet.group(1) else []
        elif current:
            current.append(line)
    if current:
        rows.append(" ".join(current).strip())
    cleaned = [row for row in rows if row]
    return [
        re.sub(r"\s+", " ", row).strip()
        for row in cleaned
        if not row.startswith("Infants should be exclusively breastfed")
        and not row.startswith("See the section Evidence to recommendations")
    ]


__all__ = [
    "WHO_EXTRACTOR_VERSION",
    "audit_recommendation_retention",
    "canonical_retained_section_ids",
    "extract_who_blocks",
]
