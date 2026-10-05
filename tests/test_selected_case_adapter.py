from __future__ import annotations

import json

import pytest

from health_ai_copilot.evaluation.contracts import CaseScore, EvalCase
from health_ai_copilot.evaluation.datasets import SelectedCaseAdapter, case_ids_sha256
from health_ai_copilot.harness.contracts import AnswerSchema, HarnessResponse
from tools.eval.extract_failed_case_ids import extract_case_ids


class _FixtureAdapter:
    dataset_id = "fixture"
    source_revision = "fixture-revision"
    snapshot_sha256 = "a" * 64
    subset_sha256 = "b" * 64

    def __init__(self) -> None:
        self._cases = tuple(
            EvalCase(
                case_id=f"case-{index}",
                query=f"question {index}",
                answer_schema=AnswerSchema.SINGLE_CHOICE,
                gold="A",
            )
            for index in range(3)
        )

    def cases(self) -> tuple[EvalCase, ...]:
        return self._cases

    def score(self, case: EvalCase, response: HarnessResponse) -> CaseScore:
        return CaseScore(case.case_id, response.parsed_answer == case.gold, True, "exact")


def test_selected_case_adapter_keeps_frozen_identity_and_hashes_ordered_ids() -> None:
    base = _FixtureAdapter()
    selected_ids = ("case-2", "case-0")

    adapter = SelectedCaseAdapter(base, selected_ids)

    assert [case.case_id for case in adapter.cases()] == list(selected_ids)
    assert adapter.dataset_id == base.dataset_id
    assert adapter.source_revision == base.source_revision
    assert adapter.snapshot_sha256 == base.snapshot_sha256
    assert adapter.selected_case_ids_sha256 == case_ids_sha256(selected_ids)
    assert adapter.subset_sha256 == adapter.selected_case_ids_sha256
    assert adapter.frozen_subset_sha256 == base.subset_sha256


@pytest.mark.parametrize(
    ("case_ids", "message"),
    [
        (("case-0", "case-0"), "must be unique"),
        (("missing",), "absent from frozen dataset"),
        ((), "non-empty tuple"),
    ],
)
def test_selected_case_adapter_rejects_invalid_case_lists(
    case_ids: tuple[str, ...], message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SelectedCaseAdapter(_FixtureAdapter(), case_ids)


def test_failure_extractor_selects_transport_errors_only(tmp_path) -> None:
    run_dir = tmp_path / "prior-run"
    run_dir.mkdir()
    cases = [
        {
            "case_id": "case-api",
            "dataset_id": "fixture",
            "profile_id": "B2",
            "response": {"safety_flags": ["reasoning_failure:APIConnectionError"]},
        },
        {
            "case_id": "case-method",
            "dataset_id": "fixture",
            "profile_id": "B2",
            "response": {"safety_flags": ["reasoning_failure:ValueError"]},
        },
    ]
    (run_dir / "cases.jsonl").write_text(
        "".join(f"{json.dumps(row)}\n" for row in cases), encoding="utf-8",
    )
    (run_dir / "invalidation.json").write_text(
        json.dumps({
            "status": "INVALID_INFRASTRUCTURE_FAILURE",
            "profile_id": "B2",
            "dataset_id": "fixture",
            "connection_failures": {"APIConnectionError": 1},
        }),
        encoding="utf-8",
    )

    selection = extract_case_ids(run_dir)

    assert selection["case_ids"] == ["case-api"]
    assert selection["case_ids_sha256"] == case_ids_sha256(("case-api",))
    assert selection["retryable_failure_counts"] == {"APIConnectionError": 1}
