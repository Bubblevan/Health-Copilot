from __future__ import annotations

import pytest

from tools.score_r2med_frozen_test_artifacts import validate_subset_scoped_generation_budget


def _rows(subset: str, ids: list[str]) -> list[dict[str, str]]:
    return [{"subset": subset, "query_id": query_id} for query_id in ids]


def test_generation_identity_is_subset_scoped() -> None:
    # R2MED uses numeric IDs that legitimately repeat across TEST subsets.
    rows = (
        _rows("MedQA-Diag", [str(index) for index in range(118)])
        + _rows("MedXpertQA-Exam", [str(index) for index in range(97)])
        + _rows("Medical-Sciences", [str(index) for index in range(88)])
    )
    validate_subset_scoped_generation_budget(rows, rows)


def test_generation_identity_rejects_duplicate_within_subset() -> None:
    rows = (
        _rows("MedQA-Diag", [str(index) for index in range(117)] + ["0"])
        + _rows("MedXpertQA-Exam", [str(index) for index in range(97)])
        + _rows("Medical-Sciences", [str(index) for index in range(88)])
    )
    with pytest.raises(ValueError, match=r"duplicate \(subset, query_id\)"):
        validate_subset_scoped_generation_budget(rows, rows)


def test_generation_identity_rejects_wrong_subset_counts() -> None:
    rows = (
        _rows("MedQA-Diag", [str(index) for index in range(117)])
        + _rows("MedXpertQA-Exam", [str(index) for index in range(98)])
        + _rows("Medical-Sciences", [str(index) for index in range(88)])
    )
    with pytest.raises(ValueError, match="subset counts differ"):
        validate_subset_scoped_generation_budget(rows, rows)
