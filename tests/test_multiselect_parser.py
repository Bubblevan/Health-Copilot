from health_ai_copilot.evaluation.parser import canonical_option_set
from health_ai_copilot.harness.contracts import AnswerSchema
from health_ai_copilot.harness.verification import parse_answer


def test_multiselect_accepts_compact_and_separated_labels() -> None:
    assert parse_answer("AC", AnswerSchema.MULTI_SELECT) == ("A", "C")
    assert parse_answer("A, C because ...", AnswerSchema.MULTI_SELECT) == ("A", "C")
    assert canonical_option_set("AC") == ("A", "C")


def test_parser_reads_schema_named_json_and_final_chinese_choice_lines() -> None:
    assert parse_answer(
        '```json\n{"single_choice":"A"}\n```', AnswerSchema.SINGLE_CHOICE,
    ) == "A"
    assert parse_answer(
        "长解释在前。\n\n选项字母：C", AnswerSchema.SINGLE_CHOICE,
    ) == "C"
    assert parse_answer(
        '{"multi_select":["A","C"]}', AnswerSchema.MULTI_SELECT,
    ) == ("A", "C")


def test_single_choice_json_value_may_include_choice_text() -> None:
    assert parse_answer('{"answer":"D. 肾"}', AnswerSchema.SINGLE_CHOICE) == "D"
    assert parse_answer(
        '{"answer":"The correct answer is C"}', AnswerSchema.SINGLE_CHOICE,
    ) == "C"
    assert parse_answer(
        '{"answer":"E", "rationale":"选项E的描述不正确。根据原则"}',
        AnswerSchema.SINGLE_CHOICE,
    ) == "E"
    assert parse_answer(
        "最佳选项：**E. 冲头与模孔吻合性不好**", AnswerSchema.SINGLE_CHOICE,
    ) == "E"
    assert parse_answer(
        "**Correct options: B. 丁香**", AnswerSchema.SINGLE_CHOICE,
    ) == "B"
    assert parse_answer(
        "single_choice: C. Tracheal fistula (post-thyroidectomy)",
        AnswerSchema.SINGLE_CHOICE,
    ) == "C"
    assert parse_answer("单选题答案：E. 以上均是", AnswerSchema.SINGLE_CHOICE) == "E"
    assert parse_answer("**Correct options: B. 丁香**", AnswerSchema.MULTI_SELECT) == ("B",)
