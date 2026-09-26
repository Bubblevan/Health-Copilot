from __future__ import annotations

import pytest

from eval.r2med_candidate_union import fuse_dual_source_rrf, source_agreement


def test_candidate_union_deduplicates_and_preserves_source_rank() -> None:
    fused = fuse_dual_source_rrf(["a", "shared"], ["shared", "b"], k=60)

    assert len(fused) == 3
    by_id = {candidate.doc_id: candidate for candidate in fused}
    assert by_id["shared"].lamer_rank == 2
    assert by_id["shared"].crb_rank == 1
    assert by_id["a"].lamer_rank == 1
    assert by_id["a"].crb_rank is None
    assert source_agreement(by_id["shared"]) == 1
    assert source_agreement(by_id["a"]) == 0


def test_dualsource_rrf_is_deterministic_and_weighted() -> None:
    first = fuse_dual_source_rrf(["a", "shared"], ["shared", "b"], crb_weight=2.0)
    second = fuse_dual_source_rrf(["a", "shared"], ["shared", "b"], crb_weight=2.0)

    assert first == second
    assert first[0].doc_id == "shared"
    assert first[0].score == pytest.approx(1 / 62 + 2 / 61)


@pytest.mark.parametrize(
    ("lamer", "crb", "kwargs"),
    [
        (["a", "a"], ["b"], {}),
        (["a"], ["b"], {"k": 0}),
        (["a"], ["b"], {"lamer_weight": 0, "crb_weight": 0}),
    ],
)
def test_dualsource_rrf_rejects_invalid_input(lamer, crb, kwargs) -> None:
    with pytest.raises(ValueError):
        fuse_dual_source_rrf(lamer, crb, **kwargs)
