from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from eval.r2med_multiview import RankedDocument
from eval.rag_e5.counterfactual import (
    canonical_sha256,
    expected_arm_specs,
    parse_reader_output,
    verify_complete_arm,
    write_complete_arm,
)
from eval.rag_e5.e5b2_evaluator import _bootstrap, analyze_frozen_run
from eval.rag_e5.retrieval import E5B1LuceneBM25, weighted_rrf
from tools.research.rag_e5.run_e5b2_counterfactual import (
    _verify_resume_call_budget,
    load_runtime_cases,
)
from tools.research.rag_e5.run_e5b2_recovery import (
    _NullableBridgePayload,
    normalize_arm_bridge,
)


def test_frozen_arm_order_is_60_cases_by_exact_action_order() -> None:
    specs = expected_arm_specs([f"case-{number:02d}" for number in range(60)], "lock")

    assert len(specs) == 180
    assert [(row["case_id"], row["action"]) for row in specs[:6]] == [
        ("case-00", "OFF"),
        ("case-00", "STANDARD"),
        ("case-00", "STRONG"),
        ("case-01", "OFF"),
        ("case-01", "STANDARD"),
        ("case-01", "STRONG"),
    ]


def test_runtime_loader_never_opens_teacher_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_path = tmp_path / "runtime_cases.jsonl"
    rows = [
        {
            "case_id": f"case-{index:02d}",
            "user_id": f"user-{index // 3:02d}",
            "question": "Describe the recorded trend.",
            "decision_boundary": None,
            "state_packet_ref": None,
            "runtime_capability_context_ref": "capability_context.json",
        }
        for index in range(60)
    ]
    runtime_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        assert path.name != "teacher_cases.jsonl"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    cases = load_runtime_cases(runtime_path)

    assert len(cases) == 60
    assert cases[0].question == "Describe the recorded trend."


def test_arm_artifact_is_hash_checked_and_immutable(tmp_path: Path) -> None:
    artifact = tmp_path / "arm.json"
    payload = {"case_id": "case-01", "action": "OFF", "run_id": "run-1", "lock_sha256": "lock"}

    digest = write_complete_arm(artifact, payload)
    verified = verify_complete_arm(artifact, expected_run_id="run-1", expected_lock_sha256="lock")

    assert verified["completion_sha256"] == digest
    with pytest.raises(FileExistsError, match="immutable"):
        write_complete_arm(artifact, payload)


def test_nullable_bridge_compatibility_preserves_raw_arm_and_normalizes_verified_copy() -> None:
    payload = _NullableBridgePayload({"case_id": "c", "bridge": None})
    verified_copy = normalize_arm_bridge({"case_id": "c", "bridge": None})

    assert payload.get("bridge", {}).get("fallback_original_query") is None
    assert dict(payload)["bridge"] is None
    assert verified_copy["bridge"] == {}


def test_reader_parser_fails_closed_without_repair_or_retry() -> None:
    assert parse_reader_output("not json") == (None, False)
    assert parse_reader_output(json.dumps({"answer": "extra fields are invalid"})) == (None, False)


def test_rrf_is_deterministic_and_rejects_duplicates() -> None:
    channels = [
        [RankedDocument("b", 2.0), RankedDocument("a", 1.0)],
        [RankedDocument("a", 2.0), RankedDocument("c", 1.0)],
    ]
    first = weighted_rrf(channels, weights=[1, 1], k=60, top_k=3)
    second = weighted_rrf(channels, weights=[1, 1], k=60, top_k=3)

    assert first == second
    assert [row.doc_id for row in first] == ["a", "b", "c"]
    with pytest.raises(ValueError, match="duplicate"):
        weighted_rrf(
            [[RankedDocument("a", 1.0), RankedDocument("a", 0.5)]],
            weights=[1],
            k=60,
            top_k=2,
        )


def test_lucene_bm25_query_weights_match_frozen_lucene_formula() -> None:
    class Analyzer:
        @staticmethod
        def analyze(text: str) -> list[str]:
            return text.split()

    retriever = E5B1LuceneBM25.__new__(E5B1LuceneBM25)
    retriever.analyzer = Analyzer()
    retriever.doc_ids = ("doc-a", "doc-b")
    retriever.term_to_id = {"term": 0}
    retriever.idfs = np.asarray([1.2], dtype=np.float64)
    retriever.avgdl = 10.0
    retriever.k1 = 0.9
    retriever.b = 0.4
    retriever.matrix = sparse.csr_matrix(np.asarray([[2.0], [1.0]]))

    result = retriever.search("term", top_k=2)
    expected_query_weight = 1.2 / (1 + 0.9 * (1 - 0.4 + 0.4 * 1 / 10))

    assert [row.doc_id for row in result] == ["doc-a", "doc-b"]
    assert result[0].score == pytest.approx(2 * expected_query_weight)
    assert result[1].score == pytest.approx(expected_query_weight)


def test_resume_refuses_to_repeat_an_attempted_inference(tmp_path: Path) -> None:
    ledger = tmp_path / "call_ledger.jsonl"
    specs = [
        {"case_id": "case-1", "action": "OFF", "run_id": "r1"},
        {"case_id": "case-2", "action": "STRONG", "run_id": "r2"},
    ]
    _verify_resume_call_budget(ledger_path=ledger, specs=specs, completed=set())
    ledger.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "action": "OFF",
                "phase": "reader",
                "status": "STARTED",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="refusing to retry"):
        _verify_resume_call_budget(ledger_path=ledger, specs=specs, completed=set())


def test_user_bootstrap_resamples_three_tasks_together() -> None:
    rows = []
    action_scores = {"OFF": 0.2, "STANDARD": 0.4, "STRONG": 0.6}
    for user in range(20):
        for action, base in action_scores.items():
            for family_index, family in enumerate(("T0", "T1", "T2")):
                rows.append(
                    {
                        "user_id": f"user-{user:02d}",
                        "action": action,
                        "task_family": family,
                        "end_to_end_quality": min(base + 0.01 * family_index, 1.0),
                    }
                )

    first = _bootstrap(rows, seed=20260930, samples=100)
    second = _bootstrap(rows, seed=20260930, samples=100)

    assert first == second
    assert first["resampling_unit"] == "user_id; three related tasks sampled together"
    assert first["comparisons"]["STRONG_minus_OFF"]["mean_delta"] == pytest.approx(0.4)


def test_evaluator_refuses_teacher_read_until_all_arms_are_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = {"status": "FROZEN_PROTOCOL_NO_OUTCOMES"}
    lock["counterfactual_lock_sha256"] = canonical_sha256(lock)
    lock_path = tmp_path / "lock.json"
    execution_path = tmp_path / "execution.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    execution_path.write_text(json.dumps({"status": "PARTIAL"}), encoding="utf-8")
    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if path.name == "teacher_cases.jsonl":
            raise AssertionError("teacher opened before the execution freeze")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError, match="until every arm is frozen"):
        analyze_frozen_run(
            lock_path=lock_path,
            execution_manifest_path=execution_path,
            private_root=tmp_path / "private",
            corpus_root=tmp_path / "corpus",
            artifact_root=tmp_path / "artifacts",
            private_output_path=tmp_path / "scores.json",
            report_json_path=tmp_path / "report.json",
            matrix_json_path=tmp_path / "matrix.json",
            report_markdown_path=tmp_path / "report.md",
            bootstrap_samples=10,
        )
