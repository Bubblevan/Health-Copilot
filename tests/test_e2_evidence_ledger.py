from e2_support import evidence, source_catalog

from health_ai_copilot.capabilities import E2WorkerRole, SourceMetadata
from health_ai_copilot.e2_team import EvidenceLedgerV2


def _ledger_run():
    catalog = source_catalog(SourceMetadata("source-a", "public_health", ("facts",)))
    ledger = EvidenceLedgerV2()
    ledger.add_initial([evidence("source-a", title="first title")], source_catalog=catalog)
    ledger.add_worker(
        [evidence("source-a", title="duplicate title")],
        worker_id="worker-a",
        role=E2WorkerRole.PUBLIC_HEALTH,
        capability_id="cap-public",
        task_id="task-a",
        allowed=lambda source_id: source_id == "source-a",
        retrieval_tool_origin="search_knowledge",
    )
    ledger.add_worker(
        [evidence("source-a", title="later duplicate")],
        worker_id="worker-b",
        role=E2WorkerRole.GUIDELINE,
        capability_id="cap-guideline-synthetic",
        task_id="task-b",
        allowed=lambda source_id: source_id == "source-a",
        retrieval_tool_origin="search_knowledge",
    )
    return ledger


def test_ledger_deduplicates_by_source_and_preserves_all_first_seen_provenance():
    ledger = _ledger_run()

    assert ledger.source_ids == ("source-a",)
    record = ledger.get("source-a")
    assert record.evidence.title == "first title"
    assert record.first_observation.first_seen_order == 1
    assert [item.worker_id for item in record.observations] == [None, "worker-a", "worker-b"]
    assert [item.capability_id for item in record.observations] == [
        None,
        "cap-public",
        "cap-guideline-synthetic",
    ]


def test_ledger_dedup_order_and_provenance_are_deterministic():
    assert [item.to_dict() for item in _ledger_run().records] == [
        item.to_dict() for item in _ledger_run().records
    ]
