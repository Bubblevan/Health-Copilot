# Milvus Standalone 3.0.2

`docker-compose.yml` is the official Milvus v3.0.2 standalone Compose asset,
downloaded from the [v3.0.2 release](https://github.com/milvus-io/milvus/releases/tag/v3.0.2).
The local change is limited to keeping the upstream file at this repository
path; service definitions and image tags remain upstream. Persistent data uses
`infra/milvus/volumes/`, which is Git-ignored.

```powershell
docker compose -f infra/milvus/docker-compose.yml up -d
docker compose -f infra/milvus/docker-compose.yml ps
docker compose -f infra/milvus/docker-compose.yml down
```

The deployment includes `milvus-standalone`, `milvus-etcd`, and `milvus-minio`.
Milvus is exposed on 19530 and its WebUI on 9091; MinIO uses 9000/9001. Check
that all three service health checks report healthy before building the Huiyi
index. `docker compose down` keeps the volumes. To permanently clear this local
development index, stop the services and remove `infra/milvus/volumes/`.

The Python integration is pinned to `pymilvus==3.0.2` and uses `MilvusClient`.
The collection and HNSW/COSINE parameters are recorded in
`data/huiyi/index/milvus_manifest.json`.
