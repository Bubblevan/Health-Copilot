# Huiyi Knowledge Corpus v0.1

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

Topic annotations use an ordered `primary_topic` and a deduplicated `topics[]`
list. Titles and department names determine the primary label; doctor profiles
can add tags from an explicit major-specialty section. Incidental mentions in a
profile's biography do not become topic tags. Hospital-information documents
remain untagged. Doctor profiles group short facts into identity, experience,
and specialty chunks; patient-education paragraphs and list items remain
atomic.

## Build and retrieve

Install the isolated dependencies with `uv pip install -e ".[huiyi]"`.

For an initial, intentional acquisition only:

```powershell
python tools/huiyi/crawl.py
```

To rebuild from the already captured snapshots, do not rerun the crawler:

```powershell
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

`corpus_manifest.json` identifies the source catalog, raw manifest, normalized
documents, chunks, and transformation versions. `index_manifest.json` has a
separate identity for the inference-critical local model files, vectors, BM25
configuration, and Milvus schema/index settings. Model identity hashes the
weights, model and pooling configs, and tokenizer files; it excludes download
logs, README files, and other repository metadata. Milvus stores both topic
fields, with `topics[]` as an `ARRAY<VARCHAR>` that supports topic filters.

Rebuildable local vectors and Milvus volumes are under ignored `artifacts/` and
`infra/milvus/volumes/`. `build_index.py` refuses to overwrite an existing
collection by default. Add `--replace-collection` to explicitly replace only
`huiyi_knowledge_v0`. To stop Milvus, use `docker compose -f
infra/milvus/docker-compose.yml down`. To clear the local database after
stopping it, remove `infra/milvus/volumes/` from the repository's `infra/milvus`
directory.

The reported retrieval numbers are labeled **HY-DATA-0 SMOKE**, **NOT FROZEN
BENCHMARK**, and **NOT CLINICAL ACCURACY**. Do not add patient information to
this corpus or its evaluation queries.
