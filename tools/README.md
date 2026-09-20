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

## M7 evaluation CLI

`health-eval` is the primary unified evaluation command. It uses a source-defined
registry and never scans or dynamically imports plugins:

```powershell
python -m health_ai_copilot.eval.cli list
python -m health_ai_copilot.eval.cli run --suite m0-regression-v1 --execution offline
python -m health_ai_copilot.eval.cli run --suite m4-replay-v1 --execution replay `
  --run-dir runs/m4/20260920T132709+0800 --public-eval-content
python -m health_ai_copilot.eval.cli compare runs/m7/<left> runs/m7/<right>
```

The historical `run_m*_eval.py` tools remain for M0–M5 parity and are not silently
redirected. No live provider call can occur without `--allow-live-provider`; CI uses
offline/replay paths and does not download models.
