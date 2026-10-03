from __future__ import annotations

import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "research" / "memory" / "run_memora_qwen_locomo.py"
SPEC = importlib.util.spec_from_file_location("memora_qwen_locomo", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_official_memora_metrics_and_empty_prediction():
    values = MODULE._official_metrics("The blue sky!", "blue sky")
    assert values["official_f1"] == 0.8
    assert values["official_exact_match"] == 0
    assert values["normalized_em"] == 1
    assert MODULE._official_metrics("", "blue")["official_f1"] == 0.0
    assert MODULE._official_metrics("", "blue")["official_exact_match"] == 0


def test_paired_bootstrap_is_deterministic_and_keeps_question_pairs():
    semantic = [
        {"question_id": "q1", "category": 1, "official_f1": 0.2},
        {"question_id": "q2", "category": 2, "official_f1": 0.4},
        {"question_id": "q3", "category": 2, "official_f1": 0.6},
    ]
    prompt = [
        {"question_id": "q1", "category": 1, "official_f1": 0.1},
        {"question_id": "q2", "category": 2, "official_f1": 0.5},
        {"question_id": "unpaired", "category": 1, "official_f1": 1.0},
    ]
    first = MODULE._bootstrap_delta(semantic, prompt)
    second = MODULE._bootstrap_delta(semantic, prompt)
    assert first == second
    assert first["n"] == 2
    assert abs(first["delta"]) < 1e-12
    assert first["ci95"][0] <= first["delta"] <= first["ci95"][1]


def test_provider_health_is_scoped_to_strategy(tmp_path):
    ledger = tmp_path / "calls.jsonl"
    ledger.write_text(
        "\n".join([
            '{"question_id":"q1","system":"Memora-semantic","role":"reader_answer","success":true}',
            '{"question_id":"q1","system":"Memora-semantic","role":"embedding","success":true}',
            '{"question_id":"q1","system":"Memora-prompt","role":"reader_answer","success":true}',
        ]) + "\n",
        encoding="utf-8",
    )
    health = MODULE._question_call_health(ledger, "q1", "prompt")
    assert health["successes_by_role"] == {"reader_answer": 1}
    assert health["missing_success_roles"] == ["embedding", "memory_reasoning"]


def test_local_usage_does_not_impute_unknown_token_usage_as_zero():
    usage = MODULE._summarize_call_usage([
        {"prompt_tokens": 12, "completion_tokens": None, "latency_ms": 1250},
        {"prompt_tokens": None, "completion_tokens": 4, "latency_ms": 750},
    ])
    assert usage["prompt_tokens"] == 12
    assert usage["prompt_token_calls_known"] == 1
    assert usage["completion_tokens"] == 4
    assert usage["completion_token_calls_known"] == 1
    assert usage["latency_seconds"] == 2.0


def test_parallel_jsonl_appends_remain_parseable_and_unique(tmp_path):
    path = tmp_path / "parallel.jsonl"
    rows = [{"question_id": f"q{index}", "answer": str(index)} for index in range(128)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda row: MODULE._append_jsonl(path, row), rows))
    actual = MODULE._read_jsonl(path)
    assert len(actual) == len(rows)
    assert {row["question_id"] for row in actual} == {row["question_id"] for row in rows}


def test_local_judge_accepts_official_json_and_qwen_label_suffix():
    parse = MODULE._parse_local_judge_output
    extract = lambda text: text.strip()
    assert parse('{"label":"CORRECT"}', extract) == ("CORRECT", False)
    assert parse("The answer matches the reference.\nCORRECT", extract) == ("CORRECT", True)
    assert parse("The dates differ.\nWRONG", extract) == ("WRONG", True)


def test_local_judge_rejects_ambiguous_or_missing_suffix():
    parse = MODULE._parse_local_judge_output
    extract = lambda text: text.strip()
    for output in ("CORRECT and WRONG", "The answer seems reasonable."):
        try:
            parse(output, extract)
        except RuntimeError:
            continue
        raise AssertionError(f"Ambiguous judge output should not be scored: {output!r}")
