import hashlib
import json

import pytest

import tools.run_e1_2_r2med_retrieval as retrieval


def _write_source_dataset(tmp_path, duplicate_text="identical passage"):
    rows = {
        "corpus.jsonl": [
            {"id": "doc-a", "text": "identical passage"},
            {"id": "doc-a", "text": duplicate_text},
            {"id": "doc-b", "text": "another passage"},
        ],
        "query.jsonl": [{"id": "query-1", "text": "question"}],
        "qrels.jsonl": [{"q_id": "query-1", "p_id": "doc-a", "score": 1}],
    }
    source_root = tmp_path / "sources"
    dataset_dir = source_root / "demo"
    dataset_dir.mkdir(parents=True)
    file_info = {}
    for filename, records in rows.items():
        path = dataset_dir / filename
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in records),
            encoding="utf-8",
        )
        payload = path.read_bytes()
        file_info[filename] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    entry = {
        "name": "demo",
        "directory": "demo",
        "corpus_document_count": len(rows["corpus.jsonl"]),
        "query_count": len(rows["query.jsonl"]),
        "qrels_record_count": len(rows["qrels.jsonl"]),
        "files": file_info,
    }
    return entry, source_root


def test_dataset_loader_removes_only_exact_duplicate_corpus_rows(tmp_path):
    entry, source_root = _write_source_dataset(tmp_path)

    data = retrieval.load_dataset(entry, source_root)

    assert [row["id"] for row in data["documents"]] == ["doc-a", "doc-b"]
    assert data["source_integrity"] == {
        "corpus_source_rows": 3,
        "unique_document_ids": 2,
        "identical_duplicate_rows_removed": 1,
    }


def test_dataset_loader_still_rejects_conflicting_duplicate_corpus_ids(tmp_path):
    entry, source_root = _write_source_dataset(tmp_path, duplicate_text="different passage")

    with pytest.raises(ValueError, match="conflicting duplicate corpus ID"):
        retrieval.load_dataset(entry, source_root)


def _write_fixed_arm(tmp_path, *, arm, status="COMPLETED", query_id="q1", model_hashes=None):
    identity = f"identity-{arm}"
    result_path = tmp_path / f"{arm}.jsonl"
    manifest_path = tmp_path / f"{arm}.manifest.json"
    ranking = [[f"doc-{index}", 1.0 / (index + 1)] for index in range(100)]
    result_path.write_text(
        json.dumps(
            {"query_id": query_id, "result_identity": identity, "ranked": ranking}
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "status": status,
                "partition": "DEV",
                "subset": "demo",
                "arm": arm,
                "query_count": 1,
                "config_sha256": "config-hash",
                "source_manifest_sha256": "source-hash",
                "model_hashes": model_hashes or {},
                "result_identity": identity,
            }
        ),
        encoding="utf-8",
    )


def test_fixed_dev_rankings_require_complete_compatible_manifests_and_rows(tmp_path, monkeypatch):
    medcpt_hashes = {"query_encoder_sha256": "query", "article_encoder_sha256": "article"}
    _write_fixed_arm(tmp_path, arm="bm25")
    _write_fixed_arm(tmp_path, arm="medcpt_dense", model_hashes=medcpt_hashes)
    monkeypatch.setattr(
        retrieval,
        "arm_paths",
        lambda _root, _partition, _entry, arm: (
            tmp_path / f"{arm}.jsonl",
            tmp_path / f"{arm}.manifest.json",
        ),
    )

    rankings = retrieval.load_dev_fixed_rankings(
        partition="DEV",
        entry={"name": "demo", "query_count": 1, "corpus_document_count": 120},
        config_hash="config-hash",
        source_hash="source-hash",
        scratch_root=tmp_path,
        expected_query_ids={"q1"},
        medcpt_hashes=medcpt_hashes,
    )

    assert set(rankings) == {"bm25", "medcpt_dense"}
    assert len(rankings["bm25"]["q1"]) == 100


def test_rrf_inputs_allow_other_fixed_arms_in_joint_test_run():
    retrieval.validate_rrf_inputs(
        {"bm25": {}, "medcpt_dense": {}, "bge_dense": {}}
    )


def test_rrf_inputs_reject_missing_base_ranking():
    with pytest.raises(ValueError, match="missing: medcpt_dense"):
        retrieval.validate_rrf_inputs({"bm25": {}, "bge_dense": {}})


@pytest.mark.parametrize(
    ("status", "query_id"),
    [("RUNNING", "q1"), ("COMPLETED", "wrong-query")],
)
def test_fixed_dev_rankings_reject_incomplete_or_unpaired_rows(
    tmp_path, monkeypatch, status, query_id
):
    medcpt_hashes = {"query_encoder_sha256": "query", "article_encoder_sha256": "article"}
    _write_fixed_arm(tmp_path, arm="bm25", status=status, query_id=query_id)
    _write_fixed_arm(tmp_path, arm="medcpt_dense", model_hashes=medcpt_hashes)
    monkeypatch.setattr(
        retrieval,
        "arm_paths",
        lambda _root, _partition, _entry, arm: (
            tmp_path / f"{arm}.jsonl",
            tmp_path / f"{arm}.manifest.json",
        ),
    )

    with pytest.raises(ValueError, match="fixed result identity|query IDs"):
        retrieval.load_dev_fixed_rankings(
            partition="DEV",
            entry={"name": "demo", "query_count": 1, "corpus_document_count": 120},
            config_hash="config-hash",
            source_hash="source-hash",
            scratch_root=tmp_path,
            expected_query_ids={"q1"},
            medcpt_hashes=medcpt_hashes,
        )
