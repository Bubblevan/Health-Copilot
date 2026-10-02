"""Normalize raw snapshots and build deterministic canonical documents/chunks."""

from __future__ import annotations

import json
import sys

from _common import DATA_ROOT, REPO_ROOT, RUN_ROOT, copy_if_exists, update_summary, write_json

sys.path.insert(0, str(REPO_ROOT / "src"))

from health_ai_copilot.huiyi.chunk import chunk_documents
from health_ai_copilot.huiyi.normalize import normalize_raw_corpus
from health_ai_copilot.huiyi.schema import CORPUS_SCHEMA_VERSION
from health_ai_copilot.huiyi.validation import validate_corpus


def main() -> int:
    reports = normalize_raw_corpus(DATA_ROOT)
    chunk_report = chunk_documents(DATA_ROOT)
    corpus = validate_corpus(DATA_ROOT)
    index_root = DATA_ROOT / "index"
    index_root.mkdir(parents=True, exist_ok=True)
    corpus_manifest = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        **corpus,
        "embedding_manifest_sha256": None,
        "corpus_identity_sha256": None,
    }
    write_json(index_root / "corpus_manifest.json", corpus_manifest)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    copy_if_exists(DATA_ROOT / "source_report.json", RUN_ROOT / "source_report.json")
    write_json(RUN_ROOT / "normalization_report.json", reports["normalization"])
    write_json(RUN_ROOT / "dedup_report.json", reports["dedup"])
    write_json(RUN_ROOT / "chunk_report.json", chunk_report)
    build_report = {
        "pipeline": "HY-DATA-0",
        "status": "corpus_built",
        "corpus": corpus,
        "normalization": reports["normalization"],
        "chunking": chunk_report,
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "index_status": "pending",
    }
    write_json(RUN_ROOT / "build_report.json", build_report)
    update_summary()
    print(json.dumps(build_report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
