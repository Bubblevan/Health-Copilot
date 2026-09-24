from eval.e1_2_answer_provider import build_answer_messages, parse_answer


def test_all_answer_arms_share_a_prompt_template_with_evidence_as_only_context_change():
    case = {"question": "Which option is correct?", "options": {"A": "first", "B": "second"}}
    closed_book = build_answer_messages(case, [])
    rag = build_answer_messages(case, [{"id": "doc-1", "title": "Text", "content": "Fact."}])
    assert closed_book[0] == rag[0]
    assert "Which option is correct?" in closed_book[1]["content"]
    assert "No retrieved evidence" in closed_book[1]["content"]
    assert "doc-1" in rag[1]["content"]


def test_answer_parser_accepts_only_a_supplied_option_label():
    assert parse_answer('{"answer":"b"}', {"A", "B"}) == ("B", False)
    assert parse_answer("answer: A", {"A", "B"}) == ("A", False)
    assert parse_answer('{"answer":"C"}', {"A", "B"}) == ("C", True)
    assert parse_answer("not a selection", {"A", "B"}) == (None, True)
