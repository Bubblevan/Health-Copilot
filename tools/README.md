# M0 data tools

`fetch_m0_data.py` has two intentionally separate workflows:

```powershell
# Fetch allow-listed official pages into short, review-required candidates.
python tools/fetch_m0_data.py crawl

# Retry selected sources only, for example after a source site changes its HTTP policy.
python tools/fetch_m0_data.py crawl --only who-hypertension-fact-sheet

# Download NFCorpus, HealthBench and MIRAGE into ignored local artifacts.
python tools/fetch_m0_data.py benchmarks
```

The crawler does not save raw HTML and does not create approved
`KnowledgeCard` files. It records source metadata, a content hash for each short
candidate, and a bounded excerpt so a human can verify and rewrite atomic cards.
The benchmark command stores upstream files and SHA-256 manifests under
`artifacts/benchmarks/`; these files are intentionally ignored by Git.

The source allow-list is [`data/source_catalog.json`](../data/source_catalog.json).
Adding a URL is an explicit review decision; this is not an unrestricted web
crawler. Check each upstream site's terms before redistributing downloaded data.
