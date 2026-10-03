"""Local-only Qwen3-Embedding-0.6B adapter for document and query encoding."""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_MODEL_ROOT = Path("E:/Health-Copilot-Models/models/Qwen3-Embedding-0.6B")
EXPECTED_DIMENSION = 1024
MAX_LENGTH = 8192
INFERENCE_FILES = (
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "merges.txt",
    "model.safetensors",
    "modules.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
QUERY_INSTRUCTION = (
    "Given a Chinese patient or hospital-service query, retrieve relevant passages from an "
    "ophthalmology hospital knowledge base that answer the query."
)


class LocalModelUnavailable(RuntimeError):
    """The configured local model is missing or cannot be loaded offline."""


def _file_hashes(model_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    missing = [name for name in INFERENCE_FILES if not (model_root / name).is_file()]
    if missing:
        raise LocalModelUnavailable(
            f"Qwen3 embedding model is missing inference-critical files: {', '.join(missing)}"
        )
    for name in INFERENCE_FILES:
        path = model_root / name
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        result[name] = digest.hexdigest()
    return result


class Qwen3LocalEmbedder:
    """Sentence Transformers wrapper that never downloads or substitutes a model."""

    def __init__(self, model_root: str | Path = DEFAULT_MODEL_ROOT, *, device: str | None = None) -> None:
        self.model_root = Path(model_root).expanduser().resolve()
        if not self.model_root.is_dir():
            raise LocalModelUnavailable(
                f"Qwen3 embedding model not found at {self.model_root}; "
                f"download {MODEL_ID} into this directory before building the index."
            )
        self.file_hashes = _file_hashes(self.model_root)
        # The explicit local path plus offline flags prevent Hub resolution or
        # hidden network fallback during load and inference.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise LocalModelUnavailable(
                "Install the Huiyi extra with `uv pip install -e \".[huiyi]\"` to load Qwen3 locally."
            ) from exc
        try:
            self.model = SentenceTransformer(
                str(self.model_root),
                device=device,
                local_files_only=True,
            )
        except Exception as exc:
            raise LocalModelUnavailable(
                f"Could not load local {MODEL_ID} from {self.model_root} with network access disabled: {exc}"
            ) from exc
        self.model.max_seq_length = MAX_LENGTH
        self.dimension = int(self.model.get_embedding_dimension())
        if self.dimension != EXPECTED_DIMENSION:
            raise ValueError(
                f"{MODEL_ID} embedding dimension is {self.dimension}, expected {EXPECTED_DIMENSION}"
            )
        self.device = str(self.model.device)
        self.dtype = str(next(self.model.parameters()).dtype).replace("torch.", "")
        self.weights_identity = hashlib.sha256(
            "\n".join(f"{name}:{digest}" for name, digest in sorted(self.file_hashes.items())).encode("utf-8")
        ).hexdigest()
        self.revision = f"local-files-sha256:{self.weights_identity}"
        self.library_versions = {
            "sentence_transformers": _version("sentence_transformers"),
            "transformers": _version("transformers"),
            "torch": torch.__version__,
            "huggingface_hub": _version("huggingface_hub"),
            "numpy": _version("numpy"),
        }
        self.pooling = self._pooling_description()

    def _pooling_description(self) -> str:
        descriptions: list[str] = []
        for module in self.model:
            if module.__class__.__name__ == "Pooling":
                modes = [
                    name.removeprefix("pooling_mode_")
                    for name in vars(module)
                    if name.startswith("pooling_mode_") and vars(module)[name] is True
                ]
                descriptions.append(f"Pooling({','.join(sorted(modes))})")
            elif module.__class__.__name__ == "Normalize":
                descriptions.append("L2Normalize")
        return "+".join(descriptions) if descriptions else "+".join(
            module.__class__.__name__ for module in self.model
        )

    def document_input(self, chunk: dict[str, Any]) -> str:
        context = [chunk["title"], *chunk.get("section_path", [])]
        return "\n".join(part for part in context if part) + "\n" + chunk["text"]

    def encode_documents(self, texts: Sequence[str], *, batch_size: int = 16) -> list[list[float]]:
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return self._validate_vectors(vectors, len(texts))

    def encode_query(self, query: str) -> list[float]:
        formatted = f"Instruct: {QUERY_INSTRUCTION}\nQuery:{query}"
        vectors = self.model.encode(
            [formatted],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return self._validate_vectors(vectors, 1)[0]

    def manifest(self, *, batch_size: int = 16) -> dict[str, Any]:
        return {
            "model_id": MODEL_ID,
            "resolved_revision": self.revision,
            "model_root": str(self.model_root),
            "model_file_hashes": self.file_hashes,
            "embedding_dimension": self.dimension,
            "dtype": self.dtype,
            "pooling": self.pooling,
            "normalized_embeddings": True,
            "document_instruction": None,
            "query_instruction": QUERY_INSTRUCTION,
            "query_format": "Instruct: <instruction>\\nQuery:<query>",
            "max_length": MAX_LENGTH,
            "batch_size": batch_size,
            "library_versions": self.library_versions,
            "device": self.device,
            "network_fallback": False,
        }

    @staticmethod
    def _validate_vectors(vectors: Any, expected_count: int) -> list[list[float]]:
        rows = [[float(item) for item in row] for row in vectors]
        if len(rows) != expected_count:
            raise ValueError(f"embedding count mismatch: expected {expected_count}, got {len(rows)}")
        for row in rows:
            if len(row) != EXPECTED_DIMENSION:
                raise ValueError(f"embedding dimension mismatch: expected 1024, got {len(row)}")
            if not all(math.isfinite(value) for value in row):
                raise ValueError("embedding contains a non-finite value")
            norm = math.sqrt(math.fsum(value * value for value in row))
            # The model's Normalize module runs in bfloat16 on CUDA, so the
            # float32 numpy output can drift a few thousandths from unit norm.
            # Correct that rounding drift in Python float precision, while
            # still rejecting outputs that were not normalized by the model.
            if norm == 0.0 or abs(norm - 1.0) > 1e-2:
                raise ValueError(f"embedding is not approximately L2-normalized (norm={norm:.6f})")
            row[:] = [value / norm for value in row]
        return rows


def _version(module_name: str) -> str:
    module = __import__(module_name)
    return str(getattr(module, "__version__", "unknown"))
