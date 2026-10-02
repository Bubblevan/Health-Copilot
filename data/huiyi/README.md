# Huiyi Knowledge Corpus v0

This namespace contains a reproducible public-source pipeline for the official
Enshi Huiyi Eye Hospital site at `https://yk.huiyi9e.com/`. It is separate from
the hypertension KnowledgeCard pack and all frozen research artifacts.

The corpus may contain public hospital facts, department profiles, doctor
profiles, FAQs, patient education, and perioperative instructions. It contains
no patient records. Public medical education defaults to
`NEEDS_HUMAN_REVIEW`; `AUTO_ACCEPTED_PUBLIC_INFO` only denotes static public
hospital facts and profiles. Nothing here is clinically reviewed or clinically
validated.

## Provenance and freshness

`source_catalog.json` pins the URLs, source IDs, page categories, and admission
policy. `raw/manifest.jsonl` points to content-addressed, byte-exact HTML under
`raw/pages/`; each canonical document retains its source ID, original URL, raw
SHA256, and fetch time. A chunk therefore traces back through its document and
snapshot to the official URL.

The crawler only follows HTML links on `yk.huiyi9e.com`, has a page and depth
limit, uses a named user agent, a timeout, bounded deterministic retries, and a
polite delay. It does not send or store cookies, submit forms, access accounts,
fetch off-site links, or run OCR. Dynamic schedules, registration/insurance
pages, news/events, and notices remain raw-only. The homepage and category
indexes are stored for discovery but are not normalized as documents. Raw HTML
is committed when small enough to review; all snapshots are content addressed.

Normalization preserves page text and punctuation without model rewriting.
Stale-looking or dynamic text is excluded or flagged for human review. Exact
duplicate text retains each source's provenance. Near duplicates are only
reported; they are not merged.

## Build and retrieve

Install the isolated dependencies with `uv pip install -e ".[huiyi]"`.

```powershell
python tools/huiyi/crawl.py
python tools/huiyi/build_corpus.py
docker compose -f infra/milvus/docker-compose.yml up -d
python tools/huiyi/build_index.py `
  --model-root "E:/Health-Copilot-Models/models/Qwen3-Embedding-0.6B" `
  --milvus-uri "http://127.0.0.1:19530"
python tools/huiyi/smoke_retrieval.py --milvus-uri "http://127.0.0.1:19530"
python tools/huiyi/inspect_corpus.py
```

The embedding loader requires local `Qwen/Qwen3-Embedding-0.6B` files, runs in
offline mode, uses no hosted embedding API, and never falls back to another
model. To re-index with a different local model, retain `raw/`, `normalized/`,
and `chunks/`, pass its local directory to `build_index.py`, and review the new
embedding manifest. The corpus snapshots and chunks do not depend on the model.

`data/huiyi/index/` holds small manifests and corpus identity. Rebuildable local
vectors and Milvus volumes are under ignored `artifacts/` and
`infra/milvus/volumes/`. Removing a collection is a separate operator action;
the build command refuses to replace a collection with the same name. To stop
Milvus, use `docker compose -f infra/milvus/docker-compose.yml down`. To clear
the local database after stopping it, remove `infra/milvus/volumes/` from the
repository's `infra/milvus` directory.

The reported retrieval numbers are labeled **HY-DATA-0 SMOKE**, **NOT FROZEN
BENCHMARK**, and **NOT CLINICAL ACCURACY**. Do not add patient information to
this corpus or its evaluation queries.
