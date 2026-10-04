from __future__ import annotations

import unittest

from eval.cmb_scoring import select_stratified_ids
from eval.cmb_scoring import score_cmb_predictions
from eval.parsers import extract_mcq_answer
from eval.prompts import cmb_prompt, option_labels


class CMBCommonCoreTests(unittest.TestCase):
    def test_stratified_ids_are_deterministic_and_ignore_answer_fields(self) -> None:
        rows = [
            {"id": f"{category}-{index}", "subcategory": category, "answer": "A"}
            for category in ("alpha", "beta", "gamma")
            for index in range(4)
        ]
        ids, strata = select_stratified_ids(rows, target_n=7, seed=20261004)
        changed = [{**row, "answer": "F", "question": "ignored content"} for row in rows]
        ids_after_change, strata_after_change = select_stratified_ids(
            changed, target_n=7, seed=20261004
        )
        self.assertEqual(ids, ids_after_change)
        self.assertEqual(strata, strata_after_change)
        self.assertEqual(len(ids), 7)
        self.assertEqual(sorted(item["selected_n"] for item in strata.values()), [2, 2, 3])

    def test_multiple_answer_remains_exact_match(self) -> None:
        gold = [
            {"id": "1", "answer": "A", "subcategory": "one", "major_category": "m1", "question_type": "单项选择题"},
            {"id": "2", "answer": "AC", "subcategory": "two", "major_category": "m2", "question_type": "多项选择题"},
        ]
        predictions = [
            {"id": "1", "parsed_answer": "A", "parse_success": True, "generation_latency_seconds": 1.0, "output_tokens": 10},
            {"id": "2", "parsed_answer": "A", "parse_success": True, "generation_latency_seconds": 1.2, "output_tokens": 12},
        ]
        result = score_cmb_predictions(predictions, gold, checkpoint="test", model_revision="local")
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["multiple_answer"]["correct"], 0)
        self.assertEqual(result["subcategory_count"], 2)


    def test_invalid_option_is_not_parsed(self) -> None:
        answer = extract_mcq_answer("Final answer: F", {"A", "B", "C", "D"})
        self.assertIsNone(answer)

    def test_candidate_metadata_drives_cmb_prompt_mode_and_valid_options(self) -> None:
        options = {"A": "first", "B": "second", "C": "third"}
        single = cmb_prompt({
            "question": "question",
            "option": options,
            "question_type": "单项选择题",
        })
        multiple = cmb_prompt({
            "question": "question",
            "option": options,
            "question_type": "多项选择题",
        })
        self.assertIn("请选择一个最佳选项", single)
        self.assertIn("请选择所有正确选项", multiple)
        self.assertEqual(option_labels(options), ["A", "B", "C"])

    def test_mcq_parser_prefers_final_explicit_answer(self) -> None:
        answer = extract_mcq_answer(
            "Initially A seems possible, but the final answer is C.",
            set("ABCDEF"),
        )
        self.assertEqual(answer, "C")

    def test_cmb_parser_extracts_all_multiple_answer_labels(self) -> None:
        answer = extract_mcq_answer(
            "分析后，最终答案：A、C",
            set("ABCDEF"),
            allow_multiple=True,
        )
        self.assertEqual(answer, "AC")


if __name__ == "__main__":
    unittest.main()
