import importlib.util
import sys
from pathlib import Path

import pytest


_module_path = Path(__file__).parents[1] / "tools" / "research" / "memory" / "context_bundle.py"
_spec = importlib.util.spec_from_file_location("mem1_context_bundle_test", _module_path)
context_bundle = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = context_bundle
_spec.loader.exec_module(context_bundle)


def test_context_bundle_serialization_and_hash_are_deterministic():
    args = {
        "system": "openclaw",
        "question_id": "dev-1",
        "items": [
            {"text": "Swimming is preferred.", "kind": "chunk", "source_session_ids": ["s2", "s2"]},
            {"text": "The old preference was running.", "kind": "chunk", "source_session_ids": ["s1"]},
        ],
        "token_counter": lambda values: sum(len(value.split()) for value in values),
        "token_counter_name": "test-whitespace-v1",
        "provenance_available": True,
        "retrieval_latency_ms": 2.5,
        "ingestion_latency_ms": 10.0,
    }
    first = context_bundle.build_context_bundle(**args).to_dict()
    second = context_bundle.build_context_bundle(**args).to_dict()

    assert first == second
    assert first["items"][0]["rank"] == 1
    assert first["items"][0]["source_session_ids"] == ["s2"]
    assert first["context_token_count"] == len(first["serialized_context"].split())
    assert context_bundle.verify_context_bundle(first)
    first["serialized_context"] += " changed"
    assert not context_bundle.verify_context_bundle(first)


@pytest.mark.parametrize(
    "items",
    [
        [{"text": 3, "kind": "chunk"}],
        [{"text": "x", "kind": "chunk", "rank": 0}],
        [{"text": "x", "kind": "chunk", "source_session_ids": "s1"}],
    ],
)
def test_context_bundle_rejects_malformed_items(items):
    with pytest.raises((TypeError, ValueError)):
        context_bundle.build_context_bundle(
            system="test",
            question_id="q1",
            items=items,
            token_counter=lambda values: 0,
            token_counter_name="test",
            provenance_available=False,
            retrieval_latency_ms=None,
            ingestion_latency_ms=None,
        )
