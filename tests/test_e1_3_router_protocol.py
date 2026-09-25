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
    for fold in range(5):
        heldout = {case_id for case_id, value in assignment.items() if value == fold}
        train = set(case_ids) - heldout
        assert not (train & heldout)
        assert heldout


def test_outer_case_never_in_its_training_partition() -> None:
    case_ids = [f"case-{index}" for index in range(50)]
    folds = {case_id: index % 5 for index, case_id in enumerate(case_ids)}
    for heldout_case, heldout_fold in folds.items():
        train_ids = {case_id for case_id, fold in folds.items() if fold != heldout_fold}
        eval_ids = {case_id for case_id, fold in folds.items() if fold == heldout_fold}
        assert heldout_case in eval_ids
        assert heldout_case not in train_ids
        assert train_ids.isdisjoint(eval_ids)
