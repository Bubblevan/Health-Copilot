"""Gold-blind U3-R retrieval capability execution for the owned DEV universe.

The runtime side consumes only U2-F runtime episodes and an allowlisted,
episode-visible external corpus. Evaluator truth is loaded by a separate
post-freeze scorer; no LLM output has authority over actions, access, or answers.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from eval.r2med_crb import BGE_QUERY_PREFIX
from eval.r2med_gar_generation import load_upstream_prompt_catalog
from eval.r2med_multiview import (
    LuceneBM25Index,
    RankedDocument,
    dense_search,
    encode_bge,
    ensure_java_home,
    ranked_ids,
)
from health_ai_copilot.research.integration.actions import CapabilityAction
from health_ai_copilot.research.integration.contracts import (
    ArchitectureMode,
    EpisodeBudget,
    EvaluatorRef,
    ExternalEvidenceWorldRef,
    ExternalRetrievalLevel,
    IntegrationEpisode,
    ObservableState,
    PatientStateRef,
    ToolSurfaceRef,
    WorkerManifest,
)
from health_ai_copilot.research.integration.evidence_world import (
    ExternalEvidenceRecord,
    ExternalEvidenceWorld,
)
from health_ai_copilot.research.integration.executor import (
    DeterministicIntegrationExecutor,
    ExecutionResources,
)

U2F_DATASET_ID = "health-copilot-owned-longitudinal-v1"
U2F_DATASET_ROOT_SHA256 = "e28ea9ef9ecae47d3f27f28c68042066e9af297fe808cafebf1d3c8c80fb2134"
U2F_MANIFEST_SHA256 = "34e6d1a8123ee63eea220c1a1b9b0b9b5f2e3349aeeec05a4504a508d4ca814e"
U2F_ROOT = Path("runs/integration/u2f-owned-v1-55955b2eff38")
U3R_RUN_ROOT = Path("runs/integration/u3r-rag-transfer-v1-e28ea9ef-final")
DEV_SPLITS = ("DEV_IID", "DEV_STRUCTURAL")
ACTION_ORDER = ("OFF", "STANDARD", "STRONG")
TOP_K = 100
ANSWER_CONTEXT_K = 10
STANDARD_RRF_K = 60
STANDARD_RRF_WEIGHTS = (1, 1)
STRONG_RRF_K = 20
STRONG_RRF_WEIGHTS = (1, 2, 1, 2)
BM25_K1 = 0.9
BM25_B = 0.4
LAMER_PROMPT_FAMILY = "Stack-Medical"
LLAMA_MODEL_NAME = "local-qwen3-8b"
LLAMA_SYSTEM_PROMPT = "You are a helpful assistant."
LLAMA_MAX_OUTPUT_TOKENS = 8192
LLAMA_TEMPERATURE = 0.0
LLAMA_REASONING_ENABLED = False
DEFAULT_LLAMA_URL = "http://127.0.0.1:8091/v1"
DEFAULT_QWEN_PATH = Path(
    r"E:\Health-Copilot-Models\models\qwen3-8b\Qwen3-8B-Q4_K_M.gguf"
)
DEFAULT_BGE_PATH = Path(r"E:\Health-Copilot-Models\models\bge-large-en-v1.5")
DEFAULT_UPSTREAM_ROOT = Path(r"D:\MyLab\Jianli\external\rag\R2MED")
LATENT_ID_PATTERN = re.compile(r'"latent_world_id"\s*:\s*"LW-([^"\\]+)"')

_EXTERNAL_RECORD_FIELDS = (
    "source_id",
    "source_family",
    "publication_time",
    "effective_time",
    "effective_until",
    "natural_language_content",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"expected object at {path}:{line_number}")
            yield value


def _skip_json_value(text: str, position: int) -> int:
    """Advance over one JSON value without decoding or retaining its contents."""
    while position < len(text) and text[position].isspace():
        position += 1
    if position >= len(text):
        raise ValueError("unexpected end while skipping a JSON value")
    first = text[position]
    if first in "{[":
        closing = {"{": "}", "[": "]"}
        stack = [closing[first]]
        position += 1
        in_string = False
        escaped = False
        while position < len(text) and stack:
            char = text[position]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char in "{[":
                stack.append(closing[char])
            elif char in "}]" and (not stack or stack.pop() != char):
                raise ValueError("malformed nested JSON while skipping a value")
            position += 1
        if stack or in_string:
            raise ValueError("unterminated JSON value while skipping")
        return position
    if first == '"':
        position += 1
        escaped = False
        while position < len(text):
            char = text[position]
            position += 1
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return position
        raise ValueError("unterminated JSON string while skipping")
    while position < len(text) and text[position] not in ",]}" and not text[position].isspace():
        position += 1
    return position


_JSON_DECODER = json.JSONDecoder()
_WORLD_FIELDS = frozenset({"latent_world_id", "split_role", "external_evidence"})
_EVIDENCE_FIELDS = frozenset({
    "source_id", "source_family", "publication_time", "effective_time",
    "effective_until", "natural_language_content",
})


def _selected_json_object(
    text: str,
    position: int,
    *,
    allowed_fields: frozenset[str],
    nested_evidence_array: bool = False,
) -> tuple[dict[str, Any], int]:
    while position < len(text) and text[position].isspace():
        position += 1
    if position >= len(text) or text[position] != "{":
        raise ValueError("expected JSON object")
    position += 1
    selected: dict[str, Any] = {}
    seen_fields: set[str] = set()
    first_member = True
    while True:
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            raise ValueError("unterminated JSON object")
        if text[position] == "}":
            return selected, position + 1
        if not first_member:
            if text[position] != ",":
                raise ValueError("JSON object members must be comma-separated")
            position += 1
            while position < len(text) and text[position].isspace():
                position += 1
            if position >= len(text) or text[position] == "}":
                raise ValueError("trailing comma in JSON object")
        key, end = _JSON_DECODER.raw_decode(text, position)
        if not isinstance(key, str):
            raise TypeError("JSON object key must be a string")
        if key in seen_fields:
            raise ValueError("duplicate JSON object keys are not allowed")
        seen_fields.add(key)
        position = end
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text) or text[position] != ":":
            raise ValueError("malformed JSON object member")
        position += 1
        while position < len(text) and text[position].isspace():
            position += 1
        if key not in allowed_fields:
            position = _skip_json_value(text, position)
            first_member = False
            continue
        if nested_evidence_array and key == "external_evidence":
            selected[key], position = _selected_evidence_array(text, position)
        else:
            selected[key], position = _JSON_DECODER.raw_decode(text, position)
        first_member = False


def _selected_evidence_array(text: str, position: int) -> tuple[list[dict[str, Any]], int]:
    if position >= len(text) or text[position] != "[":
        raise ValueError("external_evidence must be a JSON array")
    position += 1
    records: list[dict[str, Any]] = []
    first_item = True
    while True:
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            raise ValueError("unterminated external_evidence array")
        if text[position] == "]":
            return records, position + 1
        if not first_item:
            if text[position] != ",":
                raise ValueError("external evidence rows must be comma-separated")
            position += 1
            while position < len(text) and text[position].isspace():
                position += 1
            if position >= len(text) or text[position] == "]":
                raise ValueError("trailing comma in external_evidence array")
        record, position = _selected_json_object(
            text, position, allowed_fields=_EVIDENCE_FIELDS
        )
        records.append(record)
        first_item = False


def select_runtime_world_fields(line: str) -> dict[str, Any]:
    """Decode only the runtime world ID, split, and allowlisted evidence projection."""
    selected, position = _selected_json_object(
        line, 0, allowed_fields=_WORLD_FIELDS, nested_evidence_array=True
    )
    if line[position:].strip():
        raise ValueError("unexpected content after latent-world JSON object")
    if set(selected) != _WORLD_FIELDS:
        raise ValueError("latent-world runtime projection is missing an allowlisted field")
    return selected


def _parse_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("evidence timestamps must be ISO strings or null")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("evidence timestamps must be timezone-aware")
    return result


@dataclass(frozen=True)
class RuntimeEpisode:
    episode_id: str
    split: str
    query: str
    decision_time: datetime
    subject_id: str | None
    available_source_families: tuple[str, ...]
    external_world_id: str | None
    external_world_version: str | None
    budget_class: str
    deadline_class: str
    contract: IntegrationEpisode

    @classmethod
    def from_runtime_row(cls, row: Mapping[str, Any], split: str) -> RuntimeEpisode:
        required = {
            "episode_id",
            "environment_version",
            "source_provenance",
            "decision_time",
            "subject_id",
            "query",
            "observable_state",
            "patient_state_ref",
            "external_world_ref",
            "tool_surface_ref",
            "budget",
            "evaluator_ref",
        }
        if set(row) != required:
            raise ValueError("U2-F runtime episode does not match the frozen runtime contract")
        if split not in DEV_SPLITS:
            raise ValueError("U3-R may materialize DEV_IID and DEV_STRUCTURAL only")
        observable = row["observable_state"]
        budget = row["budget"]
        world_ref_raw = row["external_world_ref"]
        if not isinstance(observable, Mapping) or not isinstance(budget, Mapping):
            raise TypeError("runtime observable state and budget must be objects")
        if not isinstance(row["query"], str) or not row["query"].strip():
            raise ValueError("runtime query must be non-empty")
        decision_time = _parse_time(row["decision_time"])
        if decision_time is None:
            raise ValueError("runtime episode decision_time is required")
        families = observable.get("available_external_source_families")
        if not isinstance(families, list) or any(not isinstance(item, str) for item in families):
            raise TypeError("runtime source-family scope is malformed")
        if world_ref_raw is not None and not isinstance(world_ref_raw, Mapping):
            raise TypeError("runtime external-world reference is malformed")
        observable_state = ObservableState(
            history_exists=bool(observable["history_exists"]),
            history_length_bucket=str(observable["history_length_bucket"]),
            history_time_span=observable["history_time_span"],
            available_personal_state_types=tuple(observable["available_personal_state_types"]),
            available_external_source_families=tuple(families),
            available_tool_ids=tuple(observable["available_tool_ids"]),
            available_worker_capabilities=tuple(observable["available_worker_capabilities"]),
            budget_class=str(observable["budget_class"]),
            deadline_class=str(observable["deadline_class"]),
            task_intent_metadata=tuple(
                sorted((str(key), str(value))
                       for key, value in observable["task_intent_metadata"].items())
            ),
        )
        patient_raw = row["patient_state_ref"]
        patient_ref = (
            PatientStateRef(
                str(patient_raw["subject_id"]),
                str(patient_raw["snapshot_id"]),
                tuple(patient_raw["record_types"]),
            )
            if patient_raw is not None else None
        )
        world_ref = (
            ExternalEvidenceWorldRef(
                str(world_ref_raw["world_id"]),
                str(world_ref_raw["version"]),
                tuple(world_ref_raw["source_families"]),
            )
            if world_ref_raw is not None else None
        )
        surface_raw = row["tool_surface_ref"]
        surface = None
        if surface_raw is not None:
            workers = tuple(
                WorkerManifest(
                    worker_id=str(worker["worker_id"]),
                    personal_state_scopes=tuple(worker["personal_state_scopes"]),
                    external_source_families=tuple(worker["external_source_families"]),
                    tool_ids=tuple(worker["tool_ids"]),
                    capability_domains=tuple(worker["capability_domains"]),
                )
                for worker in surface_raw["workers"]
            )
            surface = ToolSurfaceRef(
                str(surface_raw["surface_id"]),
                str(surface_raw["version"]),
                tuple(surface_raw["tool_ids"]),
                workers,
                str(surface_raw["model_identity"]),
            )
        budget_raw = row["budget"]
        budget_contract = EpisodeBudget(**budget_raw)
        evaluator_raw = row["evaluator_ref"]
        contract = IntegrationEpisode(
            episode_id=str(row["episode_id"]),
            environment_version=str(row["environment_version"]),
            source_provenance=str(row["source_provenance"]),
            decision_time=decision_time,
            subject_id=str(row["subject_id"]) if row["subject_id"] is not None else None,
            query=row["query"],
            observable_state=observable_state,
            patient_state_ref=patient_ref,
            external_world_ref=world_ref,
            tool_surface_ref=surface,
            budget=budget_contract,
            evaluator_ref=EvaluatorRef(
                str(evaluator_raw["evaluator_id"]), str(evaluator_raw["version"])
            ),
        )
        return cls(
            episode_id=str(row["episode_id"]),
            split=split,
            query=row["query"],
            decision_time=decision_time,
            subject_id=(str(row["subject_id"]) if row["subject_id"] is not None else None),
            available_source_families=tuple(sorted(set(families))),
            external_world_id=world_ref.world_id if world_ref else None,
            external_world_version=world_ref.version if world_ref else None,
            budget_class=str(budget["budget_class"]),
            deadline_class=str(budget["deadline_class"]),
            contract=contract,
        )

    @property
    def query_sha256(self) -> str:
        return sha256_bytes(self.query.encode("utf-8"))


@dataclass(frozen=True)
class U3RDocument:
    doc_id: str
    source_family: str
    publication_time: datetime
    effective_time: datetime | None
    effective_until: datetime | None
    text: str

    def __post_init__(self) -> None:
        for name in ("doc_id", "source_family", "text"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.publication_time.tzinfo is None or self.publication_time.utcoffset() is None:
            raise ValueError("publication_time must be timezone-aware")
        for name in ("effective_time", "effective_until"):
            value = getattr(self, name)
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError(f"{name} must be timezone-aware")

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_family": self.source_family,
            "publication_time": self.publication_time.isoformat(),
            "effective_time": self.effective_time.isoformat() if self.effective_time else None,
            "effective_until": self.effective_until.isoformat() if self.effective_until else None,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> U3RDocument:
        publication = _parse_time(row.get("publication_time"))
        if publication is None:
            raise ValueError("document publication_time is required")
        return cls(
            doc_id=str(row["doc_id"]),
            source_family=str(row["source_family"]),
            publication_time=publication,
            effective_time=_parse_time(row.get("effective_time")),
            effective_until=_parse_time(row.get("effective_until")),
            text=str(row["text"]),
        )


class OwnedExternalCorpusAdapter:
    """Projects only visible evidence fields into a runtime-only document view."""

    @staticmethod
    def documents_for_episode(
        episode: RuntimeEpisode, raw_world_projection: Mapping[str, Any]
    ) -> tuple[U3RDocument, ...]:
        expected_world_id = f"LW-{episode.episode_id}"
        if raw_world_projection.get("latent_world_id") != expected_world_id:
            raise ValueError("runtime episode does not match its external evidence world")
        if raw_world_projection.get("split_role") != episode.split:
            raise ValueError("runtime episode split differs from external evidence world")
        raw_records = raw_world_projection.get("external_evidence")
        if not isinstance(raw_records, list):
            raise TypeError("external evidence records must be a list")

        visible: dict[str, U3RDocument] = {}
        allowed_families = set(episode.available_source_families)
        for raw_record in raw_records:
            if not isinstance(raw_record, Mapping):
                raise TypeError("external evidence record must be an object")
            # Deliberately project an allowlist. No latent, answer, or evaluator key is read.
            fields = {name: raw_record[name] for name in _EXTERNAL_RECORD_FIELDS if name in raw_record}
            if set(fields) != set(_EXTERNAL_RECORD_FIELDS):
                raise ValueError("external evidence record is missing a runtime field")
            publication = _parse_time(fields["publication_time"])
            effective = _parse_time(fields["effective_time"])
            effective_until = _parse_time(fields["effective_until"])
            if publication is None:
                raise ValueError("external evidence publication time is required")
            family = fields["source_family"]
            if family not in allowed_families:
                continue
            if publication > episode.decision_time:
                continue
            if effective is not None and effective > episode.decision_time:
                continue
            if effective_until is not None and episode.decision_time >= effective_until:
                continue
            document = U3RDocument(
                doc_id=str(fields["source_id"]),
                source_family=str(family),
                publication_time=publication,
                effective_time=effective,
                effective_until=effective_until,
                text=str(fields["natural_language_content"]),
            )
            prior = visible.get(document.doc_id)
            if prior is not None and prior != document:
                raise ValueError("duplicate visible document ID has conflicting content")
            visible[document.doc_id] = document
        return tuple(visible[key] for key in sorted(visible))


def load_dev_runtime_episodes(u2f_root: Path = U2F_ROOT) -> tuple[RuntimeEpisode, ...]:
    manifest_path = u2f_root / "manifest.json"
    if sha256_file(manifest_path) != U2F_MANIFEST_SHA256:
        raise ValueError("frozen U2-F manifest SHA-256 mismatch")
    manifest = _read_json(manifest_path)
    if manifest.get("dataset_id") != U2F_DATASET_ID:
        raise ValueError("unexpected U2-F dataset identity")
    if manifest.get("dataset_root_hash") != U2F_DATASET_ROOT_SHA256:
        raise ValueError("unexpected U2-F dataset root hash")
    if manifest.get("reserved_evaluation_rows_materialized") is not False:
        raise ValueError("reserved TEST/OOD rows must remain unmaterialized")

    episodes: list[RuntimeEpisode] = []
    for split, folder in (("DEV_IID", "dev_iid"), ("DEV_STRUCTURAL", "dev_structural")):
        path = u2f_root / folder / "episodes.jsonl"
        for row in _read_jsonl(path):
            episodes.append(RuntimeEpisode.from_runtime_row(row, split))
    episodes.sort(key=lambda item: (item.split, item.episode_id))
    if len(episodes) != 1024 or len({item.episode_id for item in episodes}) != 1024:
        raise ValueError("U3-R runtime scope must contain exactly 1024 unique DEV episodes")
    return tuple(episodes)


def materialize_runtime_corpus(
    *,
    u2f_root: Path = U2F_ROOT,
    run_root: Path = U3R_RUN_ROOT,
) -> dict[str, Any]:
    """Build the allowlisted DEV corpus; never opens evaluator truth or TRAIN rows."""
    episodes = load_dev_runtime_episodes(u2f_root)
    episode_by_id = {item.episode_id: item for item in episodes}
    worlds_path = u2f_root / "latent_worlds.jsonl"
    corpora: dict[str, tuple[U3RDocument, ...]] = {}
    matched: set[str] = set()
    for line_number, line in enumerate(worlds_path.open(encoding="utf-8"), start=1):
        match = LATENT_ID_PATTERN.search(line)
        if match is None:
            continue
        episode_id = match.group(1)
        episode = episode_by_id.get(episode_id)
        if episode is None:
            # TRAIN, and any non-DEV world, are skipped before JSON parsing.
            continue
        projection = select_runtime_world_fields(line)
        corpora[episode_id] = OwnedExternalCorpusAdapter.documents_for_episode(episode, projection)
        matched.add(episode_id)
    if matched != set(episode_by_id):
        raise ValueError("a DEV episode has no matching frozen external evidence world")

    run_root.mkdir(parents=True, exist_ok=True)
    corpus_path = run_root / "u3r_runtime_visible_corpus.jsonl"
    rows = [
        {
            "episode_id": episode.episode_id,
            "split": episode.split,
            "documents": [item.to_dict() for item in corpora[episode.episode_id]],
        }
        for episode in episodes
    ]
    content = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    if corpus_path.exists():
        if corpus_path.read_bytes() != content:
            raise FileExistsError("existing U3-R corpus differs; refusing to overwrite")
    else:
        with corpus_path.open("xb") as handle:
            handle.write(content)
            handle.flush()

    manifest = {
        "schema_version": "u3r-runtime-corpus-v1",
        "dataset_id": U2F_DATASET_ID,
        "dataset_root_sha256": U2F_DATASET_ROOT_SHA256,
        "u2f_manifest_sha256": U2F_MANIFEST_SHA256,
        "split_counts": {split: sum(item.split == split for item in episodes) for split in DEV_SPLITS},
        "episode_count": len(episodes),
        "visible_document_count": sum(len(value) for value in corpora.values()),
        "latent_world_rows_parsed": len(matched),
        "train_world_rows_parsed": 0,
        "evaluator_truth_opened": False,
        "reserved_test_ood_opened": False,
        "reserved_test_ood_materialized": False,
        "runtime_corpus_sha256": sha256_bytes(content),
    }
    manifest_path = run_root / "u3r_runtime_corpus_manifest.json"
    manifest_bytes = canonical_json_bytes(manifest) + b"\n"
    if manifest_path.exists():
        if manifest_path.read_bytes() != manifest_bytes:
            raise FileExistsError("existing U3-R corpus manifest differs")
    else:
        with manifest_path.open("xb") as handle:
            handle.write(manifest_bytes)
            handle.flush()
    return manifest


def load_runtime_corpora(path: Path) -> dict[str, tuple[U3RDocument, ...]]:
    result: dict[str, tuple[U3RDocument, ...]] = {}
    for row in _read_jsonl(path):
        episode_id = row.get("episode_id")
        documents = row.get("documents")
        if not isinstance(episode_id, str) or not isinstance(documents, list):
            raise TypeError("runtime corpus row is malformed")
        if episode_id in result:
            raise ValueError("duplicate episode in runtime corpus")
        result[episode_id] = tuple(U3RDocument.from_dict(item) for item in documents)
    return result


def weighted_rrf(
    channels: Sequence[Sequence[RankedDocument]],
    *,
    rrf_k: int,
    weights: Sequence[int],
    top_k: int = TOP_K,
) -> list[RankedDocument]:
    if not channels or len(channels) != len(weights):
        raise ValueError("RRF channels and weights must have the same nonzero length")
    if rrf_k <= 0 or top_k <= 0 or any(weight <= 0 for weight in weights):
        raise ValueError("RRF k, top_k, and weights must be positive")
    scores: dict[str, float] = {}
    first_rank: dict[str, int] = {}
    for channel, weight in zip(channels, weights, strict=True):
        ids = ranked_ids(channel)
        if len(ids) != len(set(ids)):
            raise ValueError("an RRF channel contains duplicate document IDs")
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (rrf_k + rank)
            first_rank[doc_id] = min(first_rank.get(doc_id, rank), rank)
    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], first_rank[doc_id], doc_id))[:top_k]
    return [RankedDocument(doc_id, scores[doc_id]) for doc_id in ordered]


def _text_sha256(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


class DenseDocumentCache:
    """CPU BGE encoder with content-addressed in-process document vectors."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.vectors: dict[str, np.ndarray] = {}
        self.query_vectors: dict[str, np.ndarray] = {}

    def encode_documents(self, documents: Sequence[U3RDocument]) -> np.ndarray:
        hashes = [_text_sha256(item.text) for item in documents]
        missing_text: dict[str, str] = {}
        for item, text_hash in zip(documents, hashes, strict=True):
            if text_hash not in self.vectors:
                missing_text[text_hash] = item.text
        missing_hashes = sorted(missing_text)
        for offset in range(0, len(missing_hashes), 256):
            current = missing_hashes[offset : offset + 256]
            embeddings = encode_bge(
                self.model,
                [missing_text[key] for key in current],
                batch_size=32,
                show_progress_bar=False,
            )
            if len(embeddings) != len(current):
                raise ValueError("BGE returned a misaligned document embedding batch")
            self.vectors.update(zip(current, embeddings, strict=True))
        if not documents:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([self.vectors[key] for key in hashes]).astype(np.float32, copy=False)

    def encode_query(self, text: str) -> np.ndarray:
        key = _text_sha256(text)
        if key not in self.query_vectors:
            self.query_vectors[key] = encode_bge(
                self.model, [BGE_QUERY_PREFIX + text], batch_size=1
            )[0]
        return self.query_vectors[key]


@dataclass(frozen=True)
class RetrievalView:
    final_ids: tuple[str, ...]
    candidate_union_ids: tuple[str, ...]
    channel_ids: tuple[tuple[str, ...], ...]
    search_invocations: int


class U3RRetriever:
    """Frozen BM25/BGE and LameR-MV views; retrieval output carries no authority."""

    def __init__(self, dense_cache: DenseDocumentCache) -> None:
        ensure_java_home()
        self.dense_cache = dense_cache
        self.bm25_indexes: dict[str, LuceneBM25Index] = {}

    @staticmethod
    def _corpus_key(documents: Sequence[U3RDocument]) -> str:
        payload = [(item.doc_id, item.text) for item in documents]
        return sha256_bytes(canonical_json_bytes(payload))

    def _bm25(self, documents: Sequence[U3RDocument], query: str) -> list[RankedDocument]:
        if not documents:
            return []
        key = self._corpus_key(documents)
        index = self.bm25_indexes.get(key)
        if index is None:
            index = LuceneBM25Index([(item.doc_id, item.text) for item in documents])
            self.bm25_indexes[key] = index
        return index.search(query, top_k=TOP_K)

    def _dense(self, documents: Sequence[U3RDocument], query: str) -> list[RankedDocument]:
        if not documents:
            return []
        vectors = self.dense_cache.encode_documents(documents)
        query_vector = self.dense_cache.encode_query(query)
        return dense_search(query_vector, vectors, [item.doc_id for item in documents], top_k=TOP_K)

    def standard(self, episode: RuntimeEpisode, documents: Sequence[U3RDocument]) -> RetrievalView:
        channels = (
            self._bm25(documents, episode.query),
            self._dense(documents, episode.query),
        )
        fused = weighted_rrf(
            channels, rrf_k=STANDARD_RRF_K, weights=STANDARD_RRF_WEIGHTS, top_k=TOP_K
        )
        channel_ids = tuple(tuple(ranked_ids(channel)) for channel in channels)
        union = tuple(sorted({doc_id for values in channel_ids for doc_id in values}))
        return RetrievalView(
            tuple(ranked_ids(fused[:ANSWER_CONTEXT_K])), union, channel_ids, len(channels)
        )

    def strong(
        self,
        episode: RuntimeEpisode,
        documents: Sequence[U3RDocument],
        bridge: str,
        original_bm25: Sequence[RankedDocument],
    ) -> RetrievalView:
        bridge_query = f"{episode.query} {bridge}".strip()
        channels = (
            list(original_bm25),
            self._bm25(documents, bridge_query),
            self._dense(documents, episode.query),
            self._dense(documents, bridge),
        )
        fused = weighted_rrf(
            channels, rrf_k=STRONG_RRF_K, weights=STRONG_RRF_WEIGHTS, top_k=TOP_K
        )
        channel_ids = tuple(tuple(ranked_ids(channel)) for channel in channels)
        union = tuple(sorted({doc_id for values in channel_ids for doc_id in values}))
        return RetrievalView(
            tuple(ranked_ids(fused[:ANSWER_CONTEXT_K])), union, channel_ids, len(channels),
        )


@lru_cache(maxsize=4)
def _cached_lamer_template(upstream_root: str) -> tuple[str, str]:
    catalog = load_upstream_prompt_catalog(Path(upstream_root))
    template = catalog["lamer"].get(LAMER_PROMPT_FAMILY)
    if not isinstance(template, str):
        raise TypeError("pinned LameR prompt family is missing")
    return template, sha256_bytes(template.encode("utf-8"))


def lamer_prompt(
    query: str,
    feedback_passages: Sequence[str],
    upstream_root: Path = DEFAULT_UPSTREAM_ROOT,
) -> tuple[str, str]:
    template, _template_sha = _cached_lamer_template(str(upstream_root.resolve()))
    passage_text = "\n".join(
        f"[{index}]. {passage.replace(chr(10), ' ').strip()}"
        for index, passage in enumerate(feedback_passages[:10], start=1)
    )
    prompt = template.format(TEXT=query, PASSAGE=passage_text)
    prompt_identity = canonical_json_bytes({"system": LLAMA_SYSTEM_PROMPT, "user": prompt})
    return prompt, sha256_bytes(prompt_identity)


class LocalCpuLlamaClient:
    """One-attempt, loopback-only llama.cpp client with a frozen CPU budget."""

    def __init__(
        self,
        base_url: str = DEFAULT_LLAMA_URL,
        *,
        model_name: str = LLAMA_MODEL_NAME,
        timeout_seconds: int = 1800,
        max_output_tokens: int = LLAMA_MAX_OUTPUT_TOKENS,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
            raise ValueError("U3-R generator must use plain HTTP on 127.0.0.1")
        if max_output_tokens != LLAMA_MAX_OUTPUT_TOKENS:
            raise ValueError("U3-R LameR completion budget is frozen at 8192 tokens")
        if not model_name.strip():
            raise ValueError("local llama.cpp model name must be non-empty")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

    def complete(self, prompt: str) -> dict[str, Any]:
        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": LLAMA_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": LLAMA_TEMPERATURE,
            "top_p": 1,
            "max_tokens": LLAMA_MAX_OUTPUT_TOKENS,
            "chat_template_kwargs": {"enable_thinking": LLAMA_REASONING_ENABLED},
            "stream": False,
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"local llama.cpp completion failed: {type(exc).__name__}") from exc
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise RuntimeError("local llama.cpp returned an invalid choices payload")
        choice = choices[0]
        message = choice.get("message") or {}
        text = message.get("content")
        if not isinstance(text, str):
            raise TypeError("local llama.cpp returned no assistant text")
        usage = payload.get("usage") or {}
        return {
            "text": text.strip(),
            "finish_reason": choice.get("finish_reason"),
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
        }


def _external_world_for_view(
    episode: RuntimeEpisode,
    documents: Sequence[U3RDocument],
    view: RetrievalView | None,
) -> ExternalEvidenceWorld | None:
    if view is None:
        return None
    reference = episode.contract.external_world_ref
    if reference is None:
        raise ValueError("retrieval action requires the episode's external-world reference")
    by_id = {item.doc_id: item for item in documents}
    rows = []
    for doc_id in view.final_ids:
        item = by_id[doc_id]
        rows.append(ExternalEvidenceRecord(
            source_id=item.doc_id,
            source_family=item.source_family,
            publication_time=item.publication_time,
            effective_time=item.effective_time,
            effective_until=item.effective_until,
            authority_metadata=(("authority", "project-owned synthetic namespace"),),
            content=item.text,
        ))
    return ExternalEvidenceWorld.from_records(reference.world_id, reference.version, rows)


def build_lamer_generation_record(
    episode: RuntimeEpisode,
    documents: Sequence[U3RDocument],
    retriever: U3RRetriever,
    client: LocalCpuLlamaClient,
    *,
    original_bm25: Sequence[RankedDocument] | None = None,
    upstream_root: Path = DEFAULT_UPSTREAM_ROOT,
) -> dict[str, Any]:
    original = list(original_bm25) if original_bm25 is not None else retriever._bm25(
        documents, episode.query
    )
    feedback = original[:10]
    prompt, prompt_sha = lamer_prompt(
        episode.query, [next(item.text for item in documents if item.doc_id == row.doc_id)
                        for row in feedback], upstream_root
    )
    base = {
        "episode_id": episode.episode_id,
        "query_sha256": episode.query_sha256,
        "feedback_doc_ids": [item.doc_id for item in feedback],
        "prompt_sha256": prompt_sha,
        "generator_model": LLAMA_MODEL_NAME,
        "generator_model_api_id": client.model_name,
        "generator_max_output_tokens": LLAMA_MAX_OUTPUT_TOKENS,
        "temperature": LLAMA_TEMPERATURE,
        "reasoning": "disabled",
        "provider_calls": 1,
    }
    started = time.perf_counter()
    try:
        response = client.complete(prompt)
    except (OSError, RuntimeError, TimeoutError, ValueError, TypeError) as exc:
        return {
            **base,
            "generated_text": episode.query,
            "valid": False,
            "completed": False,
            "truncated": False,
            "fallback_original": True,
            "error": type(exc).__name__,
            "input_tokens": 0,
            "output_tokens": 0,
            "generation_latency_ms": round((time.perf_counter() - started) * 1000),
        }
    text = str(response.get("text", "")).strip()
    valid = bool(text)
    truncated = response.get("finish_reason") == "length"
    return {
        **base,
        "generated_text": text if valid else episode.query,
        "valid": valid,
        "completed": not truncated,
        "truncated": truncated,
        "fallback_original": not valid,
        "error": None if valid else "empty_generation",
        "input_tokens": int(response.get("input_tokens", 0)),
        "output_tokens": int(response.get("output_tokens", 0)),
        "generation_latency_ms": round((time.perf_counter() - started) * 1000),
    }


def fixed_arm_runtime_row(
    episode: RuntimeEpisode,
    documents: Sequence[U3RDocument],
    *,
    standard: RetrievalView,
    strong: RetrievalView,
    bridge: Mapping[str, Any],
    executor: DeterministicIntegrationExecutor | None = None,
    retrieval_latency_ms: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    deterministic_executor = executor or DeterministicIntegrationExecutor()
    latency = retrieval_latency_ms or {}

    def arm_row(name: str, view: RetrievalView | None) -> dict[str, Any]:
        resource = _external_world_for_view(episode, documents, view)
        action = CapabilityAction(
            memory_read=False,
            external_retrieval=(ExternalRetrievalLevel.STANDARD if view else ExternalRetrievalLevel.OFF),
            architecture=ArchitectureMode.SINGLE,
        )
        execution = deterministic_executor.execute(
            episode.contract, action, ExecutionResources(external_evidence_world=resource)
        )
        outcome = execution.outcome
        has_retrieval = view is not None
        strong_action = name == "STRONG"
        return {
            "action": name,
            "answer": outcome.answer,
            "answer_sha256": sha256_bytes(outcome.answer.encode("utf-8")),
            "used_evidence_ids": list(execution.observed_evidence_ids),
            "ranked_evidence_ids": list(view.final_ids) if view else [],
            "candidate_union_ids": list(view.candidate_union_ids) if view else [],
            "channel_ids": [list(ids) for ids in view.channel_ids] if view else [],
            "retrieval_activations": int(has_retrieval),
            "retrieval_search_invocations": view.search_invocations if view else 0,
            "retrieved_document_count": len(view.final_ids) if view else 0,
            "used_document_count": len(execution.observed_evidence_ids),
            "bridge_activations": int(strong_action),
            "model_calls": int(strong_action and bool(bridge.get("provider_calls"))),
            "input_tokens": int(bridge.get("input_tokens", 0)) if strong_action else 0,
            "output_tokens": int(bridge.get("output_tokens", 0)) if strong_action else 0,
            "retrieval_latency_ms": int(latency.get(name, 0)),
            "bridge_sha256": (
                sha256_bytes(str(bridge.get("generated_text", "")).encode("utf-8"))
                if strong_action else None
            ),
            "activated_capabilities": list(outcome.activated_capabilities),
            "deterministic_execution": execution.to_dict(),
        }

    return {
        "episode_id": episode.episode_id,
        "split": episode.split,
        "subject_id": episode.subject_id,
        "query_sha256": episode.query_sha256,
        "bridge_sha256": sha256_bytes(
            str(bridge.get("generated_text", "")).encode("utf-8")
        ),
        "actions": {
            "OFF": arm_row("OFF", None),
            "STANDARD": arm_row("STANDARD", standard),
            "STRONG": arm_row("STRONG", strong),
        },
    }


def load_upstream_lamer_prompt(upstream_root: Path = DEFAULT_UPSTREAM_ROOT) -> dict[str, Any]:
    """Return the pinned source identity and selected upstream prompt hash."""
    catalog = load_upstream_prompt_catalog(upstream_root)
    prompt = catalog["lamer"].get(LAMER_PROMPT_FAMILY)
    if not isinstance(prompt, str):
        raise TypeError("pinned upstream LameR prompt family is missing")
    return {
        "upstream_root": str(upstream_root.resolve()),
        "upstream_prompt_family": LAMER_PROMPT_FAMILY,
        "prompt_template_sha256": sha256_bytes(prompt.encode("utf-8")),
    }
