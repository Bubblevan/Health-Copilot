"""Build an E1.4 retrieval-only corpus from the pinned Validation parquet."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HF_REVISION = "e061be207a5aaa834ed2b5d493090fafac392fd3"
VALIDATION_FILENAME = "validation-00000-of-00001.parquet"
VALIDATION_SHA256 = "027b84d9fdfb37afdfb3c035c58aaaeeb4c04377466ba5c06b773aee5d5afdac"
VALIDATION_BYTES = 930_422
SOURCE_COLUMN = "source_chunk_text"
DEFAULT_CORPUS_ROOT = Path("E:/Health-Copilot-E1.4/corpus")


class ValidationCorpusError(RuntimeError):
    """Raised when corpus input is not the exact permitted Validation artifact."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_validation_corpus(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    parquet_module: Any | None = None,
    expected_sha256: str = VALIDATION_SHA256,
    expected_bytes: int = VALIDATION_BYTES,
    permitted_output_root: Path | None = None,
) -> dict[str, Any]:
    """Read only ``source_chunk_text`` from the hash-pinned Validation artifact.

    The exact filename, byte count, and SHA are checked before importing a parquet
    reader. The function never reads question, options, answer, or question ID columns.
    """
    source = input_path.resolve()
    if source.name != VALIDATION_FILENAME:
        raise ValidationCorpusError("Only the pinned Validation parquet is permitted.")
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.stat().st_size != expected_bytes:
        raise ValidationCorpusError("Validation parquet size differs from the pinned artifact.")
    source_sha256 = _sha256_file(source)
    if source_sha256 != expected_sha256:
        raise ValidationCorpusError("Validation parquet SHA-256 differs from the pinned artifact.")
    safe_output_root = (permitted_output_root or DEFAULT_CORPUS_ROOT).resolve()
    resolved_output = output_path.resolve()
    if not resolved_output.is_relative_to(safe_output_root):
        raise ValidationCorpusError("Raw source text may only be written under the E1.4 corpus directory.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing retrieval corpus: {output_path}")

    parquet = parquet_module or importlib.import_module("pyarrow.parquet")
    table = parquet.read_table(source, columns=[SOURCE_COLUMN])
    raw_values = table[SOURCE_COLUMN].to_pylist()
    unique: dict[str, str] = {}
    for value in raw_values:
        if not isinstance(value, str) or not value.strip():
            continue
        text = value.strip()
        chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        unique.setdefault(chunk_hash, text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        for chunk_hash, text in sorted(unique.items()):
            handle.write(
                json.dumps(
                    {"doc_id": f"doc-{chunk_hash[:20]}", "text": text},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    corpus_sha256 = _sha256_file(output_path)
    manifest = {
        "schema_version": "pubhealthbench-validation-retrieval-corpus-v1",
        "benchmark": "PubHealthBench",
        "partition": "validation",
        "partition_rows": int(table.num_rows),
        "source": {
            "dataset_id": "Joshua-Harris/PubHealthBench",
            "revision": HF_REVISION,
            "filename": VALIDATION_FILENAME,
            "size_bytes": source.stat().st_size,
            "sha256": source_sha256,
        },
        "corpus": {
        "path": str(resolved_output),
            "format": "jsonl; one opaque doc_id and text field per unique chunk",
            "source_column": SOURCE_COLUMN,
            "unique_chunks": len(unique),
            "size_bytes": output_path.stat().st_size,
            "sha256": corpus_sha256,
            "qa_columns_read": [],
            "question_id_to_chunk_mapping_written": False,
        },
        "data_boundary": {
            "validation_only": True,
            "reviewed_materialized": False,
            "test_materialized": False,
            "raw_corpus_text_stays_outside_git": True,
        },
        "license": {
            "dataset": "CC-BY-4.0",
            "source_chunk_text": "Open Government Licence v3.0",
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=("validation",), default="validation")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("E:/Health-Copilot-E1.4/data") / VALIDATION_FILENAME,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("E:/Health-Copilot-E1.4/corpus/validation_corpus.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "runs/e1_4/pubhealthbench_corpus_manifest.json",
    )
    args = parser.parse_args()
    manifest = build_validation_corpus(args.input, args.output, args.manifest)
    print(
        json.dumps(
            {
                "partition": manifest["partition"],
                "partition_rows": manifest["partition_rows"],
                "unique_chunks": manifest["corpus"]["unique_chunks"],
                "corpus_sha256": manifest["corpus"]["sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
