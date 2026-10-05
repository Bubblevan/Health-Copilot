from health_ai_copilot.evaluation.runner import summarize_records
from health_ai_copilot.harness.verification import ANSWER_PARSER_REVISION


def test_summary_separates_safety_and_nonanswer_failures_from_format_failures() -> None:
    records = [
        {
            "score": {"correct": False, "parse_success": False, "category": None},
            "response": {
                "safety_flags": ["urgent_marker:胸痛"],
                "latency_ms": 1.0,
                "provider_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            },
            "trace_summary": {},
        },
        {
            "score": {"correct": False, "parse_success": False, "category": None},
            "response": {
                "answer_text": "unable to determine",
                "safety_flags": [],
                "latency_ms": 2.0,
                "provider_calls": 1,
                "input_tokens": 10,
                "output_tokens": 3,
            },
            "trace_summary": {},
        },
        {
            "score": {"correct": False, "parse_success": False, "category": None},
            "response": {
                "answer_text": "safe fallback",
                "safety_flags": ["reasoning_failure:TypeError"],
                "latency_ms": 3.0,
                "provider_calls": 2,
                "input_tokens": 20,
                "output_tokens": 4,
            },
            "trace_summary": {},
        },
        {
            "score": {"correct": False, "parse_success": False, "category": None},
            "response": {
                "answer_text": "unstructured answer without a label",
                "safety_flags": [],
                "latency_ms": 4.0,
                "provider_calls": 1,
                "input_tokens": 10,
                "output_tokens": 3,
            },
            "trace_summary": {},
        },
    ]

    summary = summarize_records(records)

    assert summary["safety_route_abstentions"] == 1
    assert summary["model_abstentions"] == 1
    assert summary["reasoning_failures"] == 1
    assert summary["answer_format_failures"] == 1
    assert summary["unparsed_non_safety"] == 1
    assert summary["total_tokens_per_case"] == 12.5
    assert summary["parse_success"] == 0
    assert summary["scoring_revision"] == ANSWER_PARSER_REVISION
