from tools.research.memory.run_mem3b1_b0q_reviewed_slot_sample_v2 import (
    PER_STRATUM,
    STRATA,
    _select_sample,
)


def test_v2_sample_is_disjoint_and_keeps_review_metadata_out_of_model_rows():
    records_a, evaluation_a, hashes_a = _select_sample()
    records_b, evaluation_b, hashes_b = _select_sample()

    assert records_a == records_b
    assert evaluation_a == evaluation_b
    assert hashes_a == hashes_b
    assert len(evaluation_a["groups"]) == PER_STRATUM * len(STRATA)
    assert {
        label: sum(group["review_label"] == label for group in evaluation_a["groups"])
        for label in STRATA
    } == {label: PER_STRATUM for label in STRATA}
    assert set(evaluation_a["excluded_v1_slot_ids"]).isdisjoint(
        {group["revision_slot_id"] for group in evaluation_a["groups"]}
    )
    assert all(set(row) == {"record_id", "proposition"} for row in records_a)
