from tools.research.memory.run_mem3b1_dev_change_mention_admission_v1 import (
    _owner_assertion_span,
)


def test_owner_evidence_is_limited_to_the_sentence_with_the_change_values() -> None:
    text = "I like tea. The patient takes 2:30 pm instead of 3 pm."

    evidence = _owner_assertion_span(text, "2:30 pm", "3 pm")

    assert evidence == "The patient takes 2:30 pm instead of 3 pm"
    assert "I like tea" not in evidence


def test_first_person_successor_sentence_is_preserved_as_evidence() -> None:
    text = "I have tea at 2:30 pm instead of 3 pm. Another sentence follows."

    evidence = _owner_assertion_span(text, "2:30 pm", "3 pm")

    assert evidence == "I have tea at 2:30 pm instead of 3 pm"
