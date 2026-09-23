"""Offline-only adapters for E0 benchmark normalization."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .contracts import BenchmarkCase, NormalizedDatasetIdentity, canonical_json, sha256_file


class BenchmarkAdapterError(ValueError):
    """Raised when a local benchmark artifact does not match its adapter."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkAdapterError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise BenchmarkAdapterError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkAdapterError(f"invalid JSON at {path}") from exc
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return list(value)
    if isinstance(value, dict):
        return [value]
    raise BenchmarkAdapterError(f"expected JSON object/list at {path}")


def _find_file(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    joined = ", ".join(names)
    raise FileNotFoundError(f"could not find one of {joined} below {root}")


def _raw_file_hashes(root: Path) -> tuple[str, ...]:
    identity_path = root / "raw_identity.json"
    if identity_path.exists():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        hashes = tuple(
            item.get("sha256") for item in identity.get("artifacts", []) if item.get("sha256")
        )
        if hashes:
            return hashes
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"raw benchmark directory is empty: {root}")
    return tuple(sha256_file(path) for path in files)


def _load_extraction_provenance(root: Path) -> dict[str, Any] | None:
    path = root / "extraction_manifest.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value.get("archive_sha256"):
        raise BenchmarkAdapterError("extraction manifest lacks archive SHA-256")
    return value


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(canonical_json(row) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
    return sha256_file(path)


def _write_normalized_identity(
    output_root: Path,
    *,
    benchmark_id: str,
    raw_hashes: tuple[str, ...],
    adapter_version: str,
    schema_version: str,
    cases: list[BenchmarkCase],
    normalized_hash: str,
    split_manifest: dict[str, Any],
    extraction_provenance: dict[str, Any] | None = None,
) -> NormalizedDatasetIdentity:
    split_path = output_root / "split_manifest.json"
    split_path.write_text(canonical_json(split_manifest) + "\n", encoding="utf-8")
    identity = NormalizedDatasetIdentity(
        benchmark_id=benchmark_id,
        raw_artifact_sha256=raw_hashes,
        adapter_version=adapter_version,
        schema_version=schema_version,
        case_count=len(cases),
        normalized_sha256=normalized_hash,
        split_manifest_sha256=sha256_file(split_path),
        extraction_provenance=extraction_provenance,
    )
    (output_root / "identity.json").write_text(
        canonical_json(identity.to_dict()) + "\n", encoding="utf-8"
    )
    return identity


class BenchmarkAdapter(ABC):
    """Pure-ish local adapter boundary used by E0."""

    adapter_id: str
    adapter_version: str
    schema_version: str

    @abstractmethod
    def inspect(self, raw_root: Path) -> dict[str, Any]:
        """Read local metadata only; never download or normalize."""

    @abstractmethod
    def normalize(self, raw_root: Path, output_root: Path) -> NormalizedDatasetIdentity:
        """Normalize an already-present local raw artifact without network calls."""

    @abstractmethod
    def validate(self, normalized_root: Path) -> list[str]:
        """Return deterministic validation errors for normalized local data."""


class NFCorpusAdapter(BenchmarkAdapter):
    adapter_id = "nfcorpus-beir"
    adapter_version = "nfcorpus-adapter-v1"
    schema_version = "benchmark-case-v1"

    def __init__(self, split: str = "test") -> None:
        self.split = split

    def _paths(self, raw_root: Path) -> tuple[Path, Path, Path]:
        corpus = _find_file(raw_root, ("corpus.jsonl",))
        queries = _find_file(raw_root, ("queries.jsonl",))
        qrels = _find_file(raw_root, (f"{self.split}.tsv",))
        return corpus, queries, qrels

    def inspect(self, raw_root: Path) -> dict[str, Any]:
        corpus, queries, qrels = self._paths(raw_root)
        return {
            "corpus_path": str(corpus),
            "queries_path": str(queries),
            "qrels_path": str(qrels),
            "split": self.split,
            "corpus_records": len(_read_jsonl(corpus)),
            "query_records": len(_read_jsonl(queries)),
        }

    @staticmethod
    def _qrels(path: Path) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split("\t")
            if fields[0].lower() in {"query-id", "query_id"}:
                continue
            if len(fields) != 3:
                raise BenchmarkAdapterError(f"expected 3 tab-separated qrel fields at {path}:{line_number}")
            query_id, document_id, score = fields
            try:
                relevance = int(score)
            except ValueError as exc:
                raise BenchmarkAdapterError(f"qrel score is not an integer at {path}:{line_number}") from exc
            result.setdefault(query_id, {})[document_id] = relevance
        return result

    def normalize(self, raw_root: Path, output_root: Path) -> NormalizedDatasetIdentity:
        corpus_path, query_path, qrels_path = self._paths(raw_root)
        corpus = _read_jsonl(corpus_path)
        queries = _read_jsonl(query_path)
        qrels = self._qrels(qrels_path)
        documents = []
        for record in sorted(corpus, key=lambda item: str(item.get("_id", ""))):
            document_id = record.get("_id")
            if not isinstance(document_id, str) or not document_id:
                raise BenchmarkAdapterError("NFCorpus documents need non-empty string _id")
            documents.append(
                {
                    "document_id": document_id,
                    "title": record.get("title", ""),
                    "text": record.get("text", ""),
                }
            )
        cases: list[BenchmarkCase] = []
        for record in sorted(queries, key=lambda item: str(item.get("_id", ""))):
            query_id = record.get("_id")
            query = record.get("text")
            if not isinstance(query_id, str) or not query_id or not isinstance(query, str):
                raise BenchmarkAdapterError("NFCorpus queries need string _id and text")
            if query_id not in qrels:
                continue
            cases.append(
                BenchmarkCase(
                    case_id=query_id,
                    benchmark_id="nfcorpus-v1",
                    payload={"query_id": query_id, "query": query, "split": self.split},
                    gold={"qrels": qrels[query_id]},
                    metadata={"split": self.split, "retrieval_only": True},
                    source_provenance=({"kind": "BEIR", "artifact": "corpus.jsonl"},),
                )
            )
        output_root.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_root / "corpus.jsonl", documents)
        normalized_hash = _write_jsonl(
            output_root / "cases.jsonl", [case.to_dict() for case in cases]
        )
        return _write_normalized_identity(
            output_root,
            benchmark_id="nfcorpus-v1",
            raw_hashes=_raw_file_hashes(raw_root),
            adapter_version=self.adapter_version,
            schema_version=self.schema_version,
            cases=cases,
            normalized_hash=normalized_hash,
            split_manifest={"split": self.split, "case_ids": [case.case_id for case in cases]},
            extraction_provenance=_load_extraction_provenance(raw_root),
        )

    def validate(self, normalized_root: Path) -> list[str]:
        errors: list[str] = []
        try:
            rows = _read_jsonl(normalized_root / "cases.jsonl")
        except (FileNotFoundError, BenchmarkAdapterError) as exc:
            return [str(exc)]
        ids = [row.get("case_id") for row in rows]
        if len(ids) != len(set(ids)):
            errors.append("duplicate case ID")
        for row in rows:
            if not isinstance(row.get("payload", {}).get("query_id"), str):
                errors.append(f"missing query ID: {row.get('case_id')}")
            qrels = row.get("gold", {}).get("qrels")
            if not isinstance(qrels, dict) or not qrels:
                errors.append(f"missing graded qrels: {row.get('case_id')}")
        return errors


class MedicalMirageAdapter(BenchmarkAdapter):
    adapter_id = "medical-mirage"
    adapter_version = "medical-mirage-adapter-v1"
    schema_version = "benchmark-case-v1"

    def _source(self, raw_root: Path) -> Path:
        return _find_file(raw_root, ("benchmark.json", "benchmark.jsonl"))

    def inspect(self, raw_root: Path) -> dict[str, Any]:
        source = self._source(raw_root)
        return {"source_path": str(source), "bytes": source.stat().st_size}

    @staticmethod
    def _records(source: Path) -> list[tuple[str, int, dict[str, Any]]]:
        if source.suffix == ".jsonl":
            return [("unknown", index, row) for index, row in enumerate(_read_jsonl(source))]
        value = json.loads(source.read_text(encoding="utf-8"))
        records: list[tuple[str, int, dict[str, Any]]] = []
        if isinstance(value, list):
            return [("unknown", index, row) for index, row in enumerate(value) if isinstance(row, dict)]
        if not isinstance(value, dict):
            raise BenchmarkAdapterError("MIRAGE benchmark.json must be an object or list")
        datasets = value.get("datasets", value)
        if isinstance(datasets, list):
            return [("unknown", index, row) for index, row in enumerate(datasets) if isinstance(row, dict)]
        if not isinstance(datasets, dict):
            raise BenchmarkAdapterError("MIRAGE dataset container must be a mapping")
        for dataset_name in sorted(datasets):
            dataset_rows = datasets[dataset_name]
            if isinstance(dataset_rows, dict):
                nested_rows = dataset_rows.get("data", dataset_rows.get("questions"))
                if nested_rows is None:
                    nested_rows = []
                    for record_id, record in sorted(dataset_rows.items(), key=lambda item: str(item[0])):
                        if isinstance(record, dict):
                            record = dict(record)
                            record.setdefault("id", str(record_id))
                            nested_rows.append(record)
                dataset_rows = nested_rows
            if not isinstance(dataset_rows, list):
                continue
            for index, row in enumerate(dataset_rows):
                if isinstance(row, dict):
                    records.append((str(dataset_name), index, row))
        return records

    def normalize(self, raw_root: Path, output_root: Path) -> NormalizedDatasetIdentity:
        source = self._source(raw_root)
        cases: list[BenchmarkCase] = []
        for dataset_name, index, record in self._records(source):
            question = record.get("question", record.get("query"))
            options = record.get("options", record.get("choices", {}))
            answer = record.get("answer", record.get("label"))
            if not isinstance(question, str) or not question:
                raise BenchmarkAdapterError(f"MIRAGE record {dataset_name}:{index} has no question")
            case_id = str(record.get("id", f"{dataset_name}:{index}"))
            cases.append(
                BenchmarkCase(
                    case_id=f"{dataset_name}:{case_id}",
                    benchmark_id="medical-mirage-v1",
                    payload={
                        "question": question,
                        "options": options,
                        "retrieval_protocol": {"question_only": True},
                    },
                    gold={"answer": answer, "answer_format": "multiple_choice"},
                    metadata={"subdataset": dataset_name, "zero_shot": True, "rag": True},
                    source_provenance=({"kind": "MIRAGE", "subdataset": dataset_name},),
                )
            )
        output_root.mkdir(parents=True, exist_ok=True)
        normalized_hash = _write_jsonl(
            output_root / "cases.jsonl", [case.to_dict() for case in cases]
        )
        return _write_normalized_identity(
            output_root,
            benchmark_id="medical-mirage-v1",
            raw_hashes=_raw_file_hashes(raw_root),
            adapter_version=self.adapter_version,
            schema_version=self.schema_version,
            cases=cases,
            normalized_hash=normalized_hash,
            split_manifest={"split": "official_benchmark_json", "case_ids": [case.case_id for case in cases]},
        )

    def validate(self, normalized_root: Path) -> list[str]:
        try:
            rows = _read_jsonl(normalized_root / "cases.jsonl")
        except (FileNotFoundError, BenchmarkAdapterError) as exc:
            return [str(exc)]
        errors: list[str] = []
        ids = [row.get("case_id") for row in rows]
        if len(ids) != len(set(ids)):
            errors.append("duplicate case ID")
        for row in rows:
            payload = row.get("payload", {})
            if not isinstance(payload.get("question"), str):
                errors.append(f"missing MIRAGE question: {row.get('case_id')}")
            if not isinstance(row.get("metadata", {}).get("subdataset"), str):
                errors.append(f"missing MIRAGE subdataset: {row.get('case_id')}")
            if payload.get("retrieval_protocol", {}).get("question_only") is not True:
                errors.append(f"question-only retrieval protocol missing: {row.get('case_id')}")
        return errors


class HealthBenchAdapter(BenchmarkAdapter):
    adapter_id = "healthbench-official"
    adapter_version = "healthbench-adapter-v1"
    schema_version = "benchmark-case-v1"

    def _source(self, raw_root: Path) -> Path:
        candidates = sorted(
            path
            for path in raw_root.rglob("*")
            if path.is_file() and path.suffix in {".json", ".jsonl"}
        )
        if not candidates:
            raise FileNotFoundError(f"could not find HealthBench JSON artifact below {raw_root}")
        return candidates[0]

    def inspect(self, raw_root: Path) -> dict[str, Any]:
        source = self._source(raw_root)
        return {"source_path": str(source), "bytes": source.stat().st_size}

    @staticmethod
    def _records(source: Path) -> list[dict[str, Any]]:
        if source.suffix == ".jsonl":
            return _read_jsonl(source)
        value = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return list(value)
        if isinstance(value, dict):
            records = value.get("data", value.get("examples", []))
            if isinstance(records, list) and all(isinstance(item, dict) for item in records):
                return list(records)
        raise BenchmarkAdapterError("HealthBench artifact must contain a list of records")

    def normalize(self, raw_root: Path, output_root: Path) -> NormalizedDatasetIdentity:
        source = self._source(raw_root)
        cases: list[BenchmarkCase] = []
        for index, record in enumerate(self._records(source)):
            case_id = record.get("prompt_id", record.get("id", f"healthbench:{index}"))
            conversation = record.get("prompt", record.get("messages", record.get("conversation")))
            rubrics = record.get("rubrics", record.get("rubric_items"))
            if not isinstance(case_id, str) or not case_id:
                raise BenchmarkAdapterError(f"HealthBench record {index} has no prompt ID")
            if conversation is None or not isinstance(rubrics, list):
                raise BenchmarkAdapterError(f"HealthBench record {case_id} lacks prompt/rubrics")
            cases.append(
                BenchmarkCase(
                    case_id=case_id,
                    benchmark_id="healthbench-v1",
                    payload={"conversation": conversation},
                    gold={"rubrics": rubrics},
                    metadata={"subset": record.get("subset", "main")},
                    source_provenance=({"kind": "HealthBench", "artifact": source.name},),
                )
            )
        output_root.mkdir(parents=True, exist_ok=True)
        normalized_hash = _write_jsonl(
            output_root / "cases.jsonl", [case.to_dict() for case in cases]
        )
        return _write_normalized_identity(
            output_root,
            benchmark_id="healthbench-v1",
            raw_hashes=_raw_file_hashes(raw_root),
            adapter_version=self.adapter_version,
            schema_version=self.schema_version,
            cases=cases,
            normalized_hash=normalized_hash,
            split_manifest={"split": "official_artifact", "case_ids": [case.case_id for case in cases]},
        )

    def validate(self, normalized_root: Path) -> list[str]:
        try:
            rows = _read_jsonl(normalized_root / "cases.jsonl")
        except (FileNotFoundError, BenchmarkAdapterError) as exc:
            return [str(exc)]
        errors: list[str] = []
        ids = [row.get("case_id") for row in rows]
        if len(ids) != len(set(ids)):
            errors.append("duplicate case ID")
        for row in rows:
            if "conversation" not in row.get("payload", {}):
                errors.append(f"missing conversation: {row.get('case_id')}")
            if not isinstance(row.get("gold", {}).get("rubrics"), list):
                errors.append(f"missing rubric structure: {row.get('case_id')}")
        return errors


def adapter_for(benchmark_id: str) -> BenchmarkAdapter:
    if benchmark_id == "nfcorpus-v1":
        return NFCorpusAdapter()
    if benchmark_id == "medical-mirage-v1":
        return MedicalMirageAdapter()
    if benchmark_id == "healthbench-v1":
        return HealthBenchAdapter()
    raise KeyError(f"no E0 adapter for {benchmark_id}")
