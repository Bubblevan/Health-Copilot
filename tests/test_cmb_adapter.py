import json

from health_ai_copilot.evaluation.contracts import EvalCase
from health_ai_copilot.evaluation.datasets import CMBAdapter, stratified_case_ids
from health_ai_copilot.harness.contracts import AnswerSchema


class TwoCategoryCMB(CMBAdapter):
    expected_count = 2
    expected_category_count = 2


def test_cmb_maps_single_and_multiple_answers_and_category(tmp_path) -> None:
    path = tmp_path / "cmb.jsonl"
    rows = [
        {"id": "1", "question": "single?", "options": {"A": "x", "B": "y"},
         "answer": "A", "subcategory": "one"},
        {"id": "2", "question": "multi?", "options": {"A": "x", "C": "z"},
         "answer": ["A", "C"], "subcategory": "two"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    cases = TwoCategoryCMB(path, source_revision="rev1").cases()
    assert cases[0].gold == "A"
    assert cases[1].gold == ("A", "C")


def test_cmb_common_selection_is_deterministic_and_even() -> None:
    cases = tuple(
        EvalCase(f"{category}-{index}", "question", AnswerSchema.SINGLE_CHOICE, "A",
                 {"subcategory": f"cat-{category:02d}"})
        for category in range(28) for index in range(40)
    )
    selected = stratified_case_ids(cases, size=1024, seed=20261004)
    assert selected == stratified_case_ids(cases, size=1024, seed=20261004)
    counts = {name: 0 for name in {case.metadata["subcategory"] for case in cases}}
    by_id = {case.case_id: case for case in cases}
    for case_id in selected:
        counts[by_id[case_id].metadata["subcategory"]] += 1
    assert set(counts.values()).issubset({36, 37})
