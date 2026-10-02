"""MilvusClient-only adapter for the version-pinned Huiyi collection."""

from __future__ import annotations

from typing import Any

COLLECTION_NAME = "huiyi_knowledge_v0"
DIMENSION = 1024
INDEX_TYPE = "HNSW"
METRIC_TYPE = "COSINE"
INDEX_PARAMS = {"M": 16, "efConstruction": 128}
SEARCH_PARAMS = {"ef": 64}


class MilvusUnavailable(RuntimeError):
    """Milvus client or local server is unavailable."""


class HuiyiMilvusStore:
    def __init__(self, uri: str = "http://127.0.0.1:19530", *, collection_name: str = COLLECTION_NAME) -> None:
        try:
            import pymilvus
            from pymilvus import DataType, MilvusClient
        except ImportError as exc:
            raise MilvusUnavailable("Install the Huiyi extra: `uv pip install -e \".[huiyi]\"`.") from exc
        self.pymilvus_version = str(getattr(pymilvus, "__version__", "unknown"))
        if self.pymilvus_version != "3.0.2":
            raise MilvusUnavailable(
                f"pymilvus==3.0.2 is required for the Milvus 3.0.2 contract; found {self.pymilvus_version}."
            )
        self.uri = uri
        self.collection_name = collection_name
        try:
            self.client = MilvusClient(uri=uri, timeout=10)
            self._DataType = DataType
            self.server_version = str(self.client.get_server_version())
        except Exception as exc:
            raise MilvusUnavailable(
                f"Milvus is unavailable at {uri}. Start it with `docker compose -f "
                "infra/milvus/docker-compose.yml up -d`, then check `docker compose -f "
                f"infra/milvus/docker-compose.yml ps`. Details: {exc}"
            ) from exc

    @staticmethod
    def schema_manifest() -> list[dict[str, Any]]:
        return [
            {"name": "chunk_id", "type": "VARCHAR", "primary": True, "max_length": 128},
            {"name": "document_id", "type": "VARCHAR", "max_length": 128},
            {"name": "source_id", "type": "VARCHAR", "max_length": 160},
            {"name": "dense_vector", "type": "FLOAT_VECTOR", "dimension": DIMENSION},
            {"name": "text", "type": "VARCHAR", "max_length": 8192},
            {"name": "title", "type": "VARCHAR", "max_length": 1024},
            {"name": "document_type", "type": "VARCHAR", "max_length": 64},
            {"name": "department", "type": "VARCHAR", "max_length": 256},
            {"name": "topic", "type": "VARCHAR", "max_length": 128},
            {"name": "review_status", "type": "VARCHAR", "max_length": 64},
            {"name": "freshness_class", "type": "VARCHAR", "max_length": 64},
            {"name": "source_url", "type": "VARCHAR", "max_length": 2048},
            {"name": "content_sha256", "type": "VARCHAR", "max_length": 64},
        ]

    def create_collection(self) -> None:
        if self.client.has_collection(self.collection_name):
            raise MilvusUnavailable(
                f"Collection {self.collection_name!r} already exists; refusing to replace existing Milvus data."
            )
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        data_type = self._DataType
        schema.add_field(field_name="chunk_id", datatype=data_type.VARCHAR, is_primary=True, max_length=128)
        schema.add_field(field_name="document_id", datatype=data_type.VARCHAR, max_length=128)
        schema.add_field(field_name="source_id", datatype=data_type.VARCHAR, max_length=160)
        schema.add_field(field_name="dense_vector", datatype=data_type.FLOAT_VECTOR, dim=DIMENSION)
        schema.add_field(field_name="text", datatype=data_type.VARCHAR, max_length=8192)
        schema.add_field(field_name="title", datatype=data_type.VARCHAR, max_length=1024)
        schema.add_field(field_name="document_type", datatype=data_type.VARCHAR, max_length=64)
        schema.add_field(field_name="department", datatype=data_type.VARCHAR, max_length=256)
        schema.add_field(field_name="topic", datatype=data_type.VARCHAR, max_length=128)
        schema.add_field(field_name="review_status", datatype=data_type.VARCHAR, max_length=64)
        schema.add_field(field_name="freshness_class", datatype=data_type.VARCHAR, max_length=64)
        schema.add_field(field_name="source_url", datatype=data_type.VARCHAR, max_length=2048)
        schema.add_field(field_name="content_sha256", datatype=data_type.VARCHAR, max_length=64)
        indexes = self.client.prepare_index_params()
        indexes.add_index(
            field_name="dense_vector",
            index_type=INDEX_TYPE,
            metric_type=METRIC_TYPE,
            params=dict(INDEX_PARAMS),
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=indexes,
        )

    def insert(self, chunks: list[dict[str, Any]], vectors: list[list[float]], *, batch_size: int = 256) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(f"embedding count {len(vectors)} != chunk count {len(chunks)}")
        inserted = 0
        for start in range(0, len(chunks), batch_size):
            batch: list[dict[str, Any]] = []
            for chunk, vector in zip(chunks[start:start + batch_size], vectors[start:start + batch_size], strict=True):
                batch.append({
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "source_id": chunk["source_id"],
                    "dense_vector": vector,
                    "text": chunk["text"],
                    "title": chunk["title"],
                    "document_type": chunk["document_type"],
                    "department": chunk.get("department") or "",
                    "topic": chunk.get("topic") or "",
                    "review_status": chunk["review_status"],
                    "freshness_class": chunk["freshness_class"],
                    "source_url": chunk["source_url"],
                    "content_sha256": chunk["content_sha256"],
                })
            result = self.client.insert(collection_name=self.collection_name, data=batch)
            count = int(result.get("insert_count", 0))
            if count != len(batch):
                raise MilvusUnavailable(f"Milvus inserted {count} rows from a batch of {len(batch)}")
            inserted += count
        self.client.flush(collection_name=self.collection_name)
        if inserted != len(chunks):
            raise MilvusUnavailable(f"Milvus insert count mismatch: expected {len(chunks)}, got {inserted}")
        return inserted

    def entity_count(self) -> int:
        rows = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["chunk_id"],
            limit=16384,
        )
        return len(rows)

    def search(self, vector: list[float], *, top_k: int = 5, filter_expression: str | None = None) -> list[dict[str, Any]]:
        result = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            anns_field="dense_vector",
            search_params={"metric_type": METRIC_TYPE, "params": dict(SEARCH_PARAMS)},
            limit=top_k,
            filter=filter_expression,
            output_fields=[
                "chunk_id", "document_id", "source_id", "text", "title", "document_type",
                "department", "topic", "review_status", "freshness_class", "source_url", "content_sha256",
            ],
        )
        return [dict(hit["entity"], score=float(hit["distance"])) for hit in (result[0] if result else [])]

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close:
            close()
