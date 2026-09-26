from __future__ import annotations

import ast
from pathlib import Path

from tools.generate_r2med_crb_compact_repair import (
    COMPACT_JSON_SCHEMA,
    compact_prompt,
    normalize_compact_payload,
)


def test_compact_schema_normalizes_to_crb_fields() -> None:
    normalized = normalize_compact_payload(
        {
            "q": "Treatment options for anti-MDA5 juvenile dermatomyositis with rapidly progressive ILD",
            "t": ["anti-MDA5", "juvenile dermatomyositis", "interstitial lung disease"],
            "e": "Evidence describes immunosuppressive treatment strategies for rapidly progressive interstitial lung disease in anti-MDA5 dermatomyositis.",
        }
    )

    assert normalized == {
        "canonical_query": "Treatment options for anti-MDA5 juvenile dermatomyositis with rapidly progressive ILD",
        "key_concepts": ["anti-MDA5", "juvenile dermatomyositis", "interstitial lung disease"],
        "disambiguating_terms": [],
        "pseudo_evidence": "Evidence describes immunosuppressive treatment strategies for rapidly progressive interstitial lung disease in anti-MDA5 dermatomyositis.",
    }


def test_compact_schema_rejects_extra_fields_and_overlong_passages() -> None:
    valid = {"q": "medical evidence search query", "t": ["diagnosis"], "e": "A short evidence passage."}
    assert normalize_compact_payload({**valid, "answer": "not allowed"}) is None
    assert normalize_compact_payload({**valid, "e": "word " * 41}) is None
    assert normalize_compact_payload({**valid, "t": ["a", "b", "c", "d", "e"]}) is None


def test_compact_prompt_is_query_only_and_schema_is_closed() -> None:
    prompt = compact_prompt("How is condition X treated?")
    assert "How is condition X treated?" in prompt
    assert "Do NOT answer" in prompt
    assert "qrels" not in prompt.lower()
    assert COMPACT_JSON_SCHEMA["additionalProperties"] is False


def test_generation_stage_has_no_qrels_reader_or_evaluator_import() -> None:
    source = Path("tools/generate_r2med_crb_compact_repair.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any("evaluator" in module or "qrels" in module.lower() for module in imported_modules)
    assert "qrels.jsonl" not in source
