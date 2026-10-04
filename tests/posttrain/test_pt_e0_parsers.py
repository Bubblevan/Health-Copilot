from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "training" / "posttrain"
sys.path.insert(0, str(ROOT))

from data.canonicalize import canonicalize, fingerprint
from scripts.audit_training_sources import include_rl_stage, linked_family_stage
from eval.parsers import extract_mcq_answer, final_response
from eval.healthbench_judge import build_judge_prompt
from eval.healthbench_scoring import clipped_mean, length_adjusted_score, rubric_score
from scripts.run_healthbench_local_judge import ensure_local_url, parse_grade
from eval.prompts import cmb_prompt, diagnosisarena_prompt, healthbench_messages
from eval.statistics import wilson_interval


class PTE0ParserTests(unittest.TestCase):
    def test_mcq_parser_uses_final_answer_after_reasoning(self) -> None:
        raw = "Initially A seems possible because of the first finding. Final answer: C"
        self.assertEqual(extract_mcq_answer(raw, set("ABCD")), "C")

    def test_mcq_parser_discards_qwen_thinking_block(self) -> None:
        raw = "<think>Option A may fit, but reconsider.</think>Final answer: C<|im_end|>"
        self.assertEqual(final_response(raw), "Final answer: C")
        self.assertEqual(extract_mcq_answer(raw, set("ABCD")), "C")

    def test_cmb_parser_supports_multiple_exact_labels(self) -> None:
        self.assertEqual(extract_mcq_answer("分析后，最终答案：B、D", set("ABCDE"), allow_multiple=True), "BD")

    def test_parser_returns_none_for_invalid_output(self) -> None:
        self.assertIsNone(extract_mcq_answer("I cannot determine this.", set("ABCD")))

    def test_diagnosisarena_candidate_prompt_excludes_gold_fields(self) -> None:
        row = {
            "Case Information": "case text",
            "Physical Examination": "exam text",
            "Diagnostic Tests": "test text",
            "Options": {"A": "alpha", "B": "beta"},
            "Final Diagnosis": "HIDDEN_DIAGNOSIS",
            "Right Option": "HIDDEN_ANSWER",
        }
        prompt = diagnosisarena_prompt(row)
        self.assertIn("case text", prompt)
        self.assertIn("alpha", prompt)
        self.assertNotIn("HIDDEN_DIAGNOSIS", prompt)
        self.assertNotIn("HIDDEN_ANSWER", prompt)

    def test_cmb_candidate_prompt_excludes_answer(self) -> None:
        prompt = cmb_prompt({"question": "question", "option": {"A": "a"}, "answer": "HIDDEN"})
        self.assertIn("question", prompt)
        self.assertNotIn("HIDDEN", prompt)

    def test_healthbench_prompt_keeps_only_conversation_turns(self) -> None:
        messages = healthbench_messages({"messages": [
            {"role": "user", "content": "patient question"},
            {"role": "assistant", "content": "prior turn"},
        ]})
        self.assertEqual(len(messages), 1)
        self.assertIn("patient question", messages[0]["content"])
        self.assertIn("prior turn", messages[0]["content"])
        self.assertNotIn("rubric", messages[0]["content"].lower())

    def test_canonicalization_is_stable_for_fullwidth_punctuation_and_whitespace(self) -> None:
        self.assertEqual(canonicalize("  A：  Test\r\n\n B  "), canonicalize("a: test\nb"))
        self.assertEqual(fingerprint("Question", options=True), fingerprint("question", options=True))

    def test_wilson_interval_is_bounded_and_contains_point_estimate(self) -> None:
        low, high = wilson_interval(50, 100)
        self.assertTrue(0 <= low < 0.5 < high <= 1)

    def test_linked_prompt_family_uses_frozen_stage_bucket(self) -> None:
        self.assertEqual(linked_family_stage("0000000000000000"), "sft")
        self.assertEqual(linked_family_stage("0000000000000031"), "sft")
        self.assertEqual(linked_family_stage("0000000000000032"), "rl_train")
        self.assertEqual(linked_family_stage("0000000000000060"), "rl_train")
        self.assertEqual(linked_family_stage("0000000000000061"), "rl_dev")
        self.assertEqual(linked_family_stage("0000000000000063"), "rl_dev")

    def test_linked_family_is_assigned_to_only_one_training_stage(self) -> None:
        for bucket in range(100):
            family_hash = f"{bucket:016x}" + "0" * 48
            assigned = linked_family_stage(family_hash)
            sft_included = assigned == "sft"
            rl_included = include_rl_stage(assigned)
            self.assertNotEqual(sft_included, rl_included)

    def test_rl_candidate_builder_drops_sft_assigned_shared_families(self) -> None:
        self.assertFalse(include_rl_stage("withheld_sft_family"))
        self.assertTrue(include_rl_stage("rl_train"))
        self.assertTrue(include_rl_stage("rl_dev"))

    def test_healthbench_local_rubric_score_uses_signed_points(self) -> None:
        items = [{"points": 2, "criterion_text": "positive"}, {"points": -1, "criterion_text": "negative"}]
        self.assertEqual(rubric_score(items, [{"criteria_met": True}, {"criteria_met": True}]), 0.5)
        self.assertEqual(rubric_score(items, [{"criteria_met": True}, {"criteria_met": False}]), 1.0)
        self.assertEqual(clipped_mean([-0.25, 1.25]), 0.5)

    def test_healthbench_length_adjustment_is_frozen(self) -> None:
        self.assertAlmostEqual(length_adjusted_score(0.5, "x" * 2500), 0.4853)

    def test_healthbench_judge_prompt_uses_only_candidate_context_and_rubric(self) -> None:
        prompt = build_judge_prompt(
            [{"role": "user", "content": "patient asks a question"}],
            "answer from candidate",
            {"points": 1, "criterion_text": "addresses the question"},
        )
        self.assertIn("patient asks a question", prompt)
        self.assertIn("answer from candidate", prompt)
        self.assertIn("addresses the question", prompt)
        self.assertNotIn("physician_response", prompt)
        self.assertNotIn("difficulty", prompt)

    def test_local_judge_parser_requires_boolean_grade(self) -> None:
        self.assertEqual(parse_grade('{"criteria_met":true,"explanation":"met"}'), {
            "criteria_met": True,
            "explanation": "met",
        })
        self.assertIsNone(parse_grade('{"criteria_met":"true","explanation":"not a boolean"}'))

    def test_local_judge_endpoint_must_be_loopback(self) -> None:
        self.assertEqual(ensure_local_url("http://127.0.0.1:8080/"), "http://127.0.0.1:8080")
        with self.assertRaises(ValueError):
            ensure_local_url("https://api.example.com")


if __name__ == "__main__":
    unittest.main()
