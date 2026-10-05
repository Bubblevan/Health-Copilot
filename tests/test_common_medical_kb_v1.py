from __future__ import annotations

import hashlib
from collections import Counter

import pytest

from health_ai_copilot.providers.common_medical_kb_v1 import (
    _check_build_environment,
    _index_hash,
    common_medical_kb_v1_provider_factory,
)
from tools.eval.build_common_medical_kb_v1 import ROOT, _approved_card_inventory


def test_common_kb_source_inventory_matches_u2d_scope() -> None:
    sources, cards = _approved_card_inventory(ROOT)
    assert len(sources) == len(cards) == 26
    assert Counter(row["publisher"] for row in sources) == {"CDC": 21, "WHO": 5}
    assert {card.id for card in cards} == {row["id"] for row in sources}
    assert not any("nhc" in card.id for card in cards)
    for row in sources:
        path = ROOT / "data/knowledge_cards" / row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


def test_common_kb_provider_rejects_unqualified_config() -> None:
    with pytest.raises(ValueError, match="not marked READY"):
        common_medical_kb_v1_provider_factory({"status": "BLOCKED", "readiness": "NO"})


def test_common_kb_runtime_refuses_python_version_drift() -> None:
    with pytest.raises(ValueError, match="Python version differs"):
        _check_build_environment({"build_environment": {"python": "0.0.0", "packages": {}}})


def test_common_kb_identity_hash_is_canonical() -> None:
    left = {"corpus_sha256": "a", "index": {"top_k": 5, "method": "rrf"}}
    right = {"index": {"method": "rrf", "top_k": 5}, "corpus_sha256": "a"}
    assert _index_hash(left) == _index_hash(right)
    assert _index_hash(left) != _index_hash({**left, "top_k": 10})
