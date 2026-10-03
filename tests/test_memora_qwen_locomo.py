from __future__ import annotations

import importlib.util
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).parents[1] / "tools" / "research" / "memory" / "run_memora_qwen_locomo.py"
SPEC = importlib.util.spec_from_file_location("memora_qwen_locomo", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

JUDGE_COMPAT_SCRIPT = Path(__file__).parents[1] / "tools" / "research" / "memory" / "run_memora_qwen_locomo_judge_compat.py"
JUDGE_COMPAT_SPEC = importlib.util.spec_from_file_location("memora_qwen_judge_compat", JUDGE_COMPAT_SCRIPT)
assert JUDGE_COMPAT_SPEC and JUDGE_COMPAT_SPEC.loader
JUDGE_COMPAT = importlib.util.module_from_spec(JUDGE_COMPAT_SPEC)
JUDGE_COMPAT_SPEC.loader.exec_module(JUDGE_COMPAT)


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


def test_each_locomo_conversation_gets_an_isolated_chroma_directory(tmp_path):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "general": {"memory_store_path": "old"},
        "memory": {"persist_path": "old"},
    })
    first = MODULE._conversation_config(cfg, tmp_path, 0)
    second = MODULE._conversation_config(cfg, tmp_path, 1)
    assert first.memory.persist_path != second.memory.persist_path
    assert first.general.memory_store_path == first.memory.persist_path
    assert second.general.memory_store_path == second.memory.persist_path
    assert cfg.memory.persist_path == "old"


def test_each_strategy_conversation_pair_gets_an_isolated_upstream_trace_file(tmp_path):
    semantic_first = MODULE._conversation_output_path(tmp_path, "semantic", 0)
    semantic_second = MODULE._conversation_output_path(tmp_path, "semantic", 1)
    prompt_first = MODULE._conversation_output_path(tmp_path, "prompt", 0)
    assert len({semantic_first, semantic_second, prompt_first}) == 3


def test_chroma_query_failures_are_audited_and_terminal_failures_are_counted(tmp_path):
    logger = logging.getLogger("memora.core.local_memory_store")
    context = SimpleNamespace(question_id="q-1", system="Memora-semantic", role="memory_reasoning")
    event_path = MODULE._install_storage_event_handler(tmp_path, "identity-1", lambda: context)
    try:
        logger.warning("Query attempt 1/3 failed, retrying in 0.1s: transient")
        logger.error("Query failed after 3 attempts: terminal")
        rows = MODULE._read_jsonl(event_path)
        assert [row["event_type"] for row in rows] == ["query_retry", "query_terminal_failure"]
        assert all(row["question_id"] == "q-1" for row in rows)
        assert all(row["run_identity"] == "identity-1" for row in rows)
    finally:
        for handler in list(logger.handlers):
            if getattr(handler, "_healthcopilot_storage_event_path", None) == str(event_path):
                logger.removeHandler(handler)
                handler.close()


def test_judge_compat_accepts_json_label_with_trailing_rationale():
    parse = JUDGE_COMPAT.parse_local_judge_output
    extract = lambda text: text.strip()
    assert parse('{"label":"WRONG"}\nThe dates differ.', extract) == ("WRONG", True, "")
    assert parse('The answer matches.\n{"label":"CORRECT"}', extract) == ("CORRECT", True, "")
    assert parse("Reasoning\nlabel: CORRECT", extract) == ("CORRECT", True, "")


def test_judge_compat_keeps_strict_label_and_rejects_conflicts():
    parse = JUDGE_COMPAT.parse_local_judge_output
    extract = lambda text: text.strip()
    assert parse('{"label":"CORRECT"}', extract) == ("CORRECT", False, "")
    assert parse("Reasoning\nWRONG", extract) == ("WRONG", True, "")
    for value in (
        '{"label":"WRONG"}\n{"label":"CORRECT"}',
        "Unlabeled judge reasoning only.",
    ):
        try:
            parse(value, extract)
        except RuntimeError:
            continue
        raise AssertionError(f"Ambiguous judge output must not be scored: {value!r}")
