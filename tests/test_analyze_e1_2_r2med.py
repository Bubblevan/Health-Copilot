import pytest

from tools.analyze_e1_2_r2med import paired_query_maps


def test_paired_query_maps_namespace_reused_ids_by_subset():
    rows = [
        {"subset": "MedQA-Diag", "query_id": "q1", "ndcg@10": 0.2},
        {"subset": "Medical-Sciences", "query_id": "q1", "ndcg@10": 0.8},
    ]

    keyed_rows, subset_by_key = paired_query_maps(rows)

    assert len(keyed_rows) == 2
    assert set(subset_by_key.values()) == {"MedQA-Diag", "Medical-Sciences"}


def test_paired_query_maps_reject_duplicate_ids_within_subset():
    rows = [
        {"subset": "MedQA-Diag", "query_id": "q1"},
        {"subset": "MedQA-Diag", "query_id": "q1"},
    ]

    with pytest.raises(ValueError, match="duplicate query ID within MedQA-Diag"):
        paired_query_maps(rows)
