import json
from pathlib import Path

import pytest

from health_ai_copilot.knowledge.loader import KnowledgeCardLoadError, load_knowledge_cards

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "knowledge_cards"


def test_valid_cards_load_in_deterministic_id_order() -> None:
    cards = load_knowledge_cards(FIXTURE_DIR)

    assert [card.id for card in cards] == [
        "fixture-hypertension",
        "fixture-sleep",
        "fixture-unrelated",
    ]


def test_duplicate_id_fails(tmp_path: Path) -> None:
    source = json.loads((FIXTURE_DIR / "fixture-sleep.json").read_text(encoding="utf-8"))
    source["id"] = "fixture-hypertension"
    (tmp_path / "a.json").write_text(json.dumps(source), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(KnowledgeCardLoadError, match="duplicate"):
        load_knowledge_cards(tmp_path)


def test_missing_field_fails(tmp_path: Path) -> None:
    source = json.loads((FIXTURE_DIR / "fixture-sleep.json").read_text(encoding="utf-8"))
    del source["publisher"]
    (tmp_path / "invalid.json").write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(KnowledgeCardLoadError, match="publisher"):
        load_knowledge_cards(tmp_path)


def test_malformed_json_fails(tmp_path: Path) -> None:
    (tmp_path / "invalid.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(KnowledgeCardLoadError, match="invalid JSON"):
        load_knowledge_cards(tmp_path)


def test_non_http_source_url_fails(tmp_path: Path) -> None:
    source = json.loads((FIXTURE_DIR / "fixture-sleep.json").read_text(encoding="utf-8"))
    source["source_url"] = "ftp://example.org/source"
    (tmp_path / "invalid.json").write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(KnowledgeCardLoadError, match=r"http\(s\)"):
        load_knowledge_cards(tmp_path)


def test_empty_content_fails(tmp_path: Path) -> None:
    source = json.loads((FIXTURE_DIR / "fixture-sleep.json").read_text(encoding="utf-8"))
    source["content"] = "  "
    (tmp_path / "invalid.json").write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(KnowledgeCardLoadError, match="content"):
        load_knowledge_cards(tmp_path)
