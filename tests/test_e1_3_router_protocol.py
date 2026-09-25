import numpy as np

from eval.e1_3_learned_router import outer_fold_indices
from eval.e1_3_router_dataset import (
    cost_oracle_v2_action,
    historical_row_is_correct,
    load_cases,
)
from tools.prepare_e1_3_router_protocol import build_fold_manifest


def test_cost_oracle_v2_closed_book_success_is_cheapest() -> None:
    assert cost_oracle_v2_action(
        {"closed_book": True, "rag_bm25": True, "rag_medcpt": True}
    ) == ("closed_book", "CLOSED_SUFFICIENT", False)


def test_cost_oracle_v2_bm25_is_cheapest_success() -> None:
    assert cost_oracle_v2_action(
        {"closed_book": False, "rag_bm25": True, "rag_medcpt": True}
    ) == ("rag_bm25", "BM25_RESCUE", True)


def test_cost_oracle_v2_medcpt_only_rescue() -> None:
    assert cost_oracle_v2_action(
        {"closed_book": False, "rag_bm25": False, "rag_medcpt": True}
    ) == ("rag_medcpt", "MEDCPT_RESCUE", True)


def test_all_wrong_routes_closed() -> None:
    assert cost_oracle_v2_action(
        {"closed_book": False, "rag_bm25": False, "rag_medcpt": False}
    ) == ("closed_book", "UNRESOLVED", False)


def test_question_only_case_loader_omits_options(tmp_path) -> None:
    source = tmp_path / "cases.json"
    source.write_text(
        '{"medqa":{"sample-id":{"question":"Synthetic question",'
        '"options":{"A":"not a feature"}}}}',
        encoding="utf-8",
    )
    case = load_cases(source)[0]
    assert case.question == "Synthetic question"
    assert case.case_id == "medqa:sample-id"
    assert not hasattr(case, "options")


def test_provider_failure_counts_incorrect() -> None:
    assert not historical_row_is_correct({"status": "failed", "is_correct": True})
    assert not historical_row_is_correct({"status": "completed", "is_correct": False})
    assert historical_row_is_correct({"status": "completed", "is_correct": True})


def test_fold_no_overlap_and_one_heldout_assignment_per_case() -> None:
    rows = []
    case_ids = []
    for subdataset in ("medqa", "medmcqa", "mmlu"):
        for label, action, benefit in (
            ("CLOSED_SUFFICIENT", "closed_book", False),
            ("BM25_RESCUE", "rag_bm25", True),
            ("MEDCPT_RESCUE", "rag_medcpt", True),
        ):
            for index in range(5):
                case_id = f"{subdataset}:{label}:{index}"
                case_ids.append(case_id)
                rows.append(
                    {
                        "case_id": case_id,
                        "subdataset": subdataset,
                        "cost_oracle_class": label,
                        "cost_oracle_action": action,
                        "retrieval_benefit": benefit,
                    }
                )
    manifest = build_fold_manifest(rows, case_ids, "fixture-sha")
    assignment = manifest["case_id_to_fold"]
    assert len(assignment) == len(case_ids)
    assert set(assignment) == set(case_ids)
    assert len(set(assignment.values())) == 5
    fold_array = [assignment[case_id] for case_id in case_ids]
    for fold in range(5):
        train_indices, eval_indices = outer_fold_indices(
            np.asarray(fold_array), fold
        )
        train = {case_ids[index] for index in train_indices}
        heldout = {case_ids[index] for index in eval_indices}
        assert train.isdisjoint(heldout)
        assert len(heldout) == 9


def test_outer_case_never_in_its_training_partition() -> None:
    case_ids = [f"case-{index}" for index in range(50)]
    folds = np.asarray([index % 5 for index in range(50)])
    for heldout_case_index, heldout_fold in enumerate(folds):
        train, evaluation = outer_fold_indices(folds, int(heldout_fold))
        assert heldout_case_index in evaluation
        assert heldout_case_index not in train
        assert {case_ids[index] for index in train}.isdisjoint(
            {case_ids[index] for index in evaluation}
        )
