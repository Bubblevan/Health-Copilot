import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

from health_ai_copilot.providers.memory import FinalMemoryProvider


class Store:
    def query(self, query):
        assert query.scope_id == "subject-1"
        assert query.now == datetime(2026, 1, 1, tzinfo=UTC)
        return [SimpleNamespace(memory_id="mem-1", value="observed", valid_from=None,
                                created_at="2025-12-01T00:00:00+00:00")]


def test_final_memory_adapter_maps_policy_store_rows_to_memory_facts() -> None:
    result = asyncio.run(FinalMemoryProvider(Store()).read(
        subject_id="subject-1", query="query", as_of_time=datetime(2026, 1, 1, tzinfo=UTC),
    ))
    assert result.facts[0].fact_id == "mem-1"
    assert result.facts[0].text == "observed"
