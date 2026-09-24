import json

import pytest

import tools.run_e1_2_r2med_retrieval as retrieval


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
