from __future__ import annotations

import hashlib
import json

import pytest

from tools.build_pubhealthbench_validation_corpus import (
    ValidationCorpusError,
    build_validation_corpus,
)


class _Column:
    def to_pylist(self):
        return ["public guidance chunk", "public guidance chunk", None, "another chunk"]


class _Table:
    num_rows = 4

    def __getitem__(self, key):
        assert key == "source_chunk_text"
        return _Column()


class _Parquet:
    def __init__(self):
        self.columns = None

    def read_table(self, path, *, columns):
        self.columns = columns
        assert path.name == "validation-00000-of-00001.parquet"
        return _Table()


def test_only_the_validation_filename_is_accepted_before_parquet_read(tmp_path) -> None:
    source = tmp_path / "reviewed-00000-of-00001.parquet"
    source.write_bytes(b"not parquet")
    fake_parquet = _Parquet()
    with pytest.raises(ValidationCorpusError, match="Only the pinned Validation"):
        build_validation_corpus(
            source,
            tmp_path / "corpus.jsonl",
            tmp_path / "manifest.json",
            parquet_module=fake_parquet,
            expected_sha256=hashlib.sha256(b"not parquet").hexdigest(),
            expected_bytes=len(b"not parquet"),
            permitted_output_root=tmp_path,
        )
    assert fake_parquet.columns is None


def test_corpus_reads_only_source_chunk_text_and_deduplicates(tmp_path) -> None:
    source = tmp_path / "validation-00000-of-00001.parquet"
    source.write_bytes(b"validation fixture")
    fake_parquet = _Parquet()
    manifest = build_validation_corpus(
        source,
        tmp_path / "corpus.jsonl",
        tmp_path / "manifest.json",
        parquet_module=fake_parquet,
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        expected_bytes=source.stat().st_size,
        permitted_output_root=tmp_path,
    )
    assert fake_parquet.columns == ["source_chunk_text"]
    rows = [json.loads(line) for line in (tmp_path / "corpus.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(set(row) == {"doc_id", "text"} for row in rows)
    assert manifest["corpus"]["qa_columns_read"] == []
    assert manifest["corpus"]["question_id_to_chunk_mapping_written"] is False
    assert manifest["data_boundary"]["reviewed_materialized"] is False


def test_source_text_output_is_refused_inside_repository_tree(tmp_path) -> None:
    source = tmp_path / "validation-00000-of-00001.parquet"
    source.write_bytes(b"validation fixture")
    fake_parquet = _Parquet()
    with pytest.raises(ValidationCorpusError, match="only be written under the E1.4"):
        build_validation_corpus(
            source,
            tmp_path / "repo" / "source_text.jsonl",
            tmp_path / "manifest.json",
            parquet_module=fake_parquet,
            expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            expected_bytes=source.stat().st_size,
            permitted_output_root=tmp_path / "external",
        )
    assert fake_parquet.columns is None
