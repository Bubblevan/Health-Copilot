"""Project-native adapter for the frozen local MEM-1 Qwen3 embedding model."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
MODEL_TREE_SHA256 = "9d2d790d6448ef2c0911ffeb03f959d035c71ac3d2b14b7d586f2d2b39fb0efa"
WEIGHTS_SHA256 = "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
DIMENSIONS = 1024
BATCH_SIZE = 8
MAX_LENGTH = 8192
MAX_BATCH_TOKENS = 8192
DEVICE = "cuda:0"
DTYPE = "float16"
QUERY_INSTRUCTION = (
    "Given a conversation-history question, retrieve memory passages that provide evidence "
    "needed to answer the question."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_tree_sha256(root: Path) -> str:
    root = root.resolve()
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file() and ".hfd" not in item.parts)
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative + b"\0" + bytes.fromhex(sha256_file(path)) + b"\n")
    return digest.hexdigest()


def verify_local_model(
    model_path: str | Path,
    *,
    expected_tree_sha256: str = MODEL_TREE_SHA256,
    expected_weights_sha256: str = WEIGHTS_SHA256,
    expected_revision: str = MODEL_REVISION,
) -> dict[str, Any]:
    path = Path(model_path).expanduser().resolve()
    required = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"Frozen local embedding files missing from {path}: {missing}")

    weights_hash = sha256_file(path / "model.safetensors")
    if weights_hash != expected_weights_sha256:
        raise RuntimeError("Qwen3 embedding weights do not match the frozen MEM-1 artifact")
    metadata_path = path / ".hfd" / "repo_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("id") != MODEL_ID or metadata.get("sha") != expected_revision:
            raise RuntimeError("Local Qwen3 embedding repository revision is not frozen")

    tree_hash = model_tree_sha256(path)
    if tree_hash != expected_tree_sha256:
        raise RuntimeError("Local Qwen3 embedding model-tree hash differs from frozen MEM-1")
    return {
        "model_id": MODEL_ID,
        "revision": expected_revision,
        "model_path": str(path),
        "model_tree_sha256": tree_hash,
        "weights_sha256": weights_hash,
    }


def render_query(text: str) -> str:
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery:{text}"


def retrieval_document_text(key: str, value: Any) -> str:
    value_text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return f"{key} {value_text}"


@dataclass(frozen=True)
class EncodedBatch:
    start: int
    end: int
    vectors: Any
    full_input_tokens: int
    retained_input_tokens: int
    truncated_count: int
    latency_ms: float


class LocalQwen3Embedding:
    def __init__(self, model_path: str | Path, *, device: str = DEVICE, dtype: str = DTYPE) -> None:
        if device != DEVICE or dtype != DTYPE:
            raise RuntimeError("MEM-2D requires the frozen MEM-1 cuda:0 / float16 runtime")
        self.identity = verify_local_model(model_path)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        import numpy as np
        import torch
        from torch.nn import functional
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils import logging as transformers_logging

        if torch.__version__ != "2.10.0+cu128" or torch.version.cuda != "12.8":
            raise RuntimeError("MEM-2D requires the frozen PyTorch 2.10.0+cu128 CUDA runtime")
        if not torch.cuda.is_available():
            raise RuntimeError("MEM-2D local embedding requires CUDA; CPU fallback is forbidden")
        device_name = torch.cuda.get_device_name(DEVICE)
        if device_name != "NVIDIA GeForce RTX 4090 Laptop GPU":
            raise RuntimeError(f"Unexpected frozen MEM-1 embedding device: {device_name}")

        self.torch = torch
        self.functional = functional
        self.np = np
        self.device = DEVICE
        self.dtype = DTYPE
        self.device_name = device_name
        self.batch_size = BATCH_SIZE
        self.max_length = MAX_LENGTH
        self.max_batch_tokens = MAX_BATCH_TOKENS
        transformers_logging.disable_progress_bar()
        transformers_logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.identity["model_path"],
            local_files_only=True,
            padding_side="left",
            trust_remote_code=False,
        )
        self.tokenizer.padding_side = "left"
        self.model = AutoModel.from_pretrained(
            self.identity["model_path"],
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float16,
        ).to(self.device)
        self.model.eval()
        if int(self.model.config.hidden_size) != DIMENSIONS:
            raise RuntimeError("Frozen Qwen3 embedding output dimension is not 1024")
        self.torch_version = str(torch.__version__)
        self.cuda_version = str(torch.version.cuda)

    def prepare_texts(
        self, texts: list[str], input_type: Literal["query", "document"]
    ) -> list[str]:
        if input_type == "document":
            return list(texts)
        if input_type == "query":
            return [render_query(text) for text in texts]
        raise ValueError(f"Unknown embedding input type: {input_type}")

    def count_tokens(self, texts: list[str], input_type: Literal["query", "document"]) -> list[int]:
        prepared = self.prepare_texts(texts, input_type)
        counts: list[int] = []
        for start in range(0, len(prepared), 128):
            encoded = self.tokenizer(
                prepared[start : start + 128],
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )
            counts.extend(len(input_ids) for input_ids in encoded["input_ids"])
        if len(counts) != len(texts):
            raise RuntimeError("Qwen3 tokenizer did not return one length per input")
        return counts

    def _batch_slices(self, token_counts: list[int]) -> list[tuple[int, int]]:
        slices = []
        start = 0
        token_total = 0
        for index, token_count in enumerate(token_counts):
            if index > start and (
                index - start >= self.batch_size
                or token_total + token_count > self.max_batch_tokens
            ):
                slices.append((start, index))
                start = index
                token_total = 0
            token_total += min(token_count, self.max_length)
        if start < len(token_counts):
            slices.append((start, len(token_counts)))
        return slices

    def encode_batches(
        self,
        texts: list[str],
        input_type: Literal["query", "document"],
        *,
        token_counts: list[int] | None = None,
    ) -> Iterator[EncodedBatch]:
        prepared = self.prepare_texts(texts, input_type)
        counts = token_counts if token_counts is not None else self.count_tokens(texts, input_type)
        if len(prepared) != len(counts):
            raise ValueError("Embedding text and token-count lengths differ")
        for start, end in self._batch_slices(counts):
            batch_started = time.perf_counter()
            tokenized = self.tokenizer(
                prepared[start:end],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            attention = tokenized["attention_mask"]
            full_count = sum(counts[start:end])
            retained_count = int(attention.sum().item())
            truncated_count = sum(count > self.max_length for count in counts[start:end])
            tokenized = {name: tensor.to(self.device) for name, tensor in tokenized.items()}
            with self.torch.inference_mode():
                hidden = self.model(**tokenized).last_hidden_state
            pooled = pool_final_nonpadding(hidden, tokenized["attention_mask"], self.torch)
            normalized = normalize_l2_float32(pooled, self.functional)
            if normalized.shape != (end - start, DIMENSIONS):
                raise RuntimeError(
                    f"Invalid Qwen3 embedding batch shape: {tuple(normalized.shape)}"
                )
            vectors = normalized.detach().cpu().numpy().astype(self.np.float32, copy=False)
            norms = self.np.linalg.norm(vectors, axis=1)
            if not self.np.isfinite(vectors).all() or not self.np.allclose(norms, 1.0, atol=1e-5):
                raise RuntimeError("Qwen3 adapter returned non-finite or non-normalized vectors")
            yield EncodedBatch(
                start=start,
                end=end,
                vectors=vectors,
                full_input_tokens=full_count,
                retained_input_tokens=retained_count,
                truncated_count=truncated_count,
                latency_ms=(time.perf_counter() - batch_started) * 1000,
            )

    def close(self) -> None:
        model = getattr(self, "model", None)
        if model is None:
            return
        self.torch.cuda.synchronize(self.device)
        self.model = None
        del model
        gc.collect()
        self.torch.cuda.empty_cache()


def pool_final_nonpadding(hidden: Any, attention_mask: Any, torch: Any) -> Any:
    position_grid = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    positions = (attention_mask * position_grid).max(dim=1).values
    batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_indices, positions]


def normalize_l2_float32(pooled: Any, functional: Any) -> Any:
    return functional.normalize(pooled.float(), p=2, dim=1)
