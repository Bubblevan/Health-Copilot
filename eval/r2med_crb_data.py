"""Gold-blind R2MED input loading and frozen split identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PINNED_UPSTREAM_COMMIT = "11244a4925a39082967a6c9d38ef01f279c316a5"
SOURCE_MANIFEST_PATH = Path("runs/rag_r2med_crb/source_manifest.json")
DEFAULT_SOURCE_ROOT = Path(r"E:\Health-Copilot-E1.2\sources")
UPSTREAM_PROMPT_FAMILY = {
    "PMC-Treatment": "PMC-Treat",
    "PMC-Clinical": "PMCPatients",
    "IIYi-Clinical": "IIYiPatients-EN",
    "MedQA-Diag": "MedQA-Diag",
    "MedXpertQA-Exam": "MedXpertQA-Exam",
    "Medical-Sciences": "Stack-Medical",
}
PARTITIONS = {
    "DEV": ("PMC-Treatment", "PMC-Clinical", "IIYi-Clinical"),
    "TEST": ("MedQA-Diag", "MedXpertQA-Exam", "Medical-Sciences"),
}
SPRINT_LOCKED_PATHS = (
    "eval/r2med_crb.py",
    "eval/r2med_crb_data.py",
    "eval/r2med_crb_evaluator.py",
    "eval/r2med_gar_generation.py",
    "eval/r2med_multiview.py",
    "tools/prepare_r2med_crb_source_manifest.py",
    "tools/verify_r2med_models.py",
    "tools/generate_r2med_gar.py",
    "tools/run_r2med_baselines.py",
    "tools/run_r2med_crb_dev.py",
    "tools/freeze_r2med_crb.py",
    "tools/run_r2med_crb_test.py",
    "tests/test_r2med_gar_sprint.py",
    "docs/research/r2med_gar_prompt_mapping.md",
    "docs/research/r2med_crb.md",
    "runs/rag_r2med_crb/source_manifest.json",
)


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


@dataclass(frozen=True)
class R2MedSubset:
    name: str
    upstream_prompt_family: str
    directory: str
    queries: tuple[Query, ...]
    documents: tuple[Document, ...]

    @property
    def dense_documents(self) -> tuple[Document, ...]:
        """Match upstream dense loading, which keys the corpus dict by document ID."""
        seen: set[str] = set()
        unique: list[Document] = []
        for document in self.documents:
            if document.doc_id not in seen:
                seen.add(document.doc_id)
                unique.append(document)
        return tuple(unique)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path.name}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def load_source_manifest(path: Path = SOURCE_MANIFEST_PATH) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("upstream", {}).get("commit") != PINNED_UPSTREAM_COMMIT:
        raise ValueError("R2MED source manifest is not pinned to the required upstream commit")
    if manifest.get("test_status") != "PUBLIC_BENCHMARK_REUSED":
        raise ValueError("R2MED TEST status must remain PUBLIC_BENCHMARK_REUSED")
    return manifest


def _subset_manifest(manifest: dict[str, Any], partition: str, subset: str) -> dict[str, Any]:
    if partition not in PARTITIONS or subset not in PARTITIONS[partition]:
        raise ValueError(f"subset {subset!r} is not part of partition {partition!r}")
    entries = manifest["datasets"][partition]
    matches = [entry for entry in entries if entry["name"] == subset]
    if len(matches) != 1:
        raise ValueError(f"expected one source-manifest entry for {partition}/{subset}")
    return matches[0]


def load_subset_inputs(
    partition: str,
    subset: str,
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    source_manifest_path: Path = SOURCE_MANIFEST_PATH,
) -> R2MedSubset:
    """Load only corpus and native query text; this function has no qrels path."""
    manifest = load_source_manifest(source_manifest_path)
    entry = _subset_manifest(manifest, partition, subset)
    directory = entry["directory"]
    data_dir = source_root / directory
    files = entry["files"]
    loaded: dict[str, list[dict[str, Any]]] = {}
    for name in ("corpus.jsonl", "query.jsonl"):
        path = data_dir / name
        identity = files[name]
        if not path.is_file() or path.stat().st_size != identity["bytes"]:
            raise ValueError(f"R2MED input missing or wrong size: {subset}/{name}")
        if sha256_file(path) != identity["sha256"]:
            raise ValueError(f"R2MED input hash mismatch: {subset}/{name}")
        loaded[name] = read_jsonl(path)

    documents: list[Document] = []
    document_text_by_id: dict[str, str] = {}
    for row in loaded["corpus.jsonl"]:
        doc_id, text = row.get("id"), row.get("text")
        if not isinstance(doc_id, str) or not doc_id or not isinstance(text, str) or not text:
            raise ValueError(f"invalid R2MED corpus row in {subset}")
        previous_text = document_text_by_id.get(doc_id)
        if previous_text is not None and previous_text != text:
            raise ValueError(f"conflicting duplicate R2MED document ID in {subset}: {doc_id}")
        document_text_by_id[doc_id] = text
        documents.append(Document(doc_id, text))

    queries: list[Query] = []
    query_ids: set[str] = set()
    for row in loaded["query.jsonl"]:
        query_id, text = row.get("id"), row.get("text")
        if not isinstance(query_id, str) or not query_id or not isinstance(text, str) or not text:
            raise ValueError(f"invalid R2MED query row in {subset}")
        if query_id in query_ids:
            raise ValueError(f"duplicate R2MED query ID in {subset}: {query_id}")
        query_ids.add(query_id)
        queries.append(Query(query_id, text))

    if len(queries) != entry["query_count"] or len(documents) != entry["corpus_document_count"]:
        raise ValueError(f"R2MED row count mismatch in {subset}")
    return R2MedSubset(
        name=subset,
        upstream_prompt_family=UPSTREAM_PROMPT_FAMILY[subset],
        directory=directory,
        queries=tuple(queries),
        documents=tuple(documents),
    )


def load_partition_inputs(
    partition: str,
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    source_manifest_path: Path = SOURCE_MANIFEST_PATH,
) -> tuple[R2MedSubset, ...]:
    if partition not in PARTITIONS:
        raise ValueError(f"unsupported R2MED partition: {partition}")
    return tuple(
        load_subset_inputs(
            partition,
            subset,
            source_root=source_root,
            source_manifest_path=source_manifest_path,
        )
        for subset in PARTITIONS[partition]
    )
