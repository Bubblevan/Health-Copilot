from tools.research.memory.run_mem3b1_b0q_reviewed_slot_sample_v1 import (
    PER_STRATUM,
    STRATA,
    _load_sample,
)


def test_reviewed_sample_is_stable_and_stratified_without_exposing_labels():
    records_a, evaluation_a, hashes_a = _load_sample()
    records_b, evaluation_b, hashes_b = _load_sample()

    assert records_a == records_b
    assert evaluation_a == evaluation_b
    assert hashes_a == hashes_b
    assert len(evaluation_a["groups"]) == PER_STRATUM * len(STRATA)
    assert {
        label: sum(group["review_label"] == label for group in evaluation_a["groups"])
        for label in STRATA
    } == {label: PER_STRATUM for label in STRATA}
    assert len(records_a) == len(evaluation_a["record_map"])
    assert all(set(row) == {"record_id", "proposition"} for row in records_a)
    assert all(row["record_id"].startswith("r") for row in records_a)
