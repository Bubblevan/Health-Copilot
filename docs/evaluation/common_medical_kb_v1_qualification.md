# Common Medical KB V1 qualification (H1-C)

## Status

`COMMON_KB_V1_READY=YES` for the explicitly limited hypertension patient-education scope below. This means the corpus, index identity, local provider, and use controls are bound and reproducible on the L40 workspace. It is not a claim that the corpus covers general medicine or that retrieval improves benchmark accuracy.

## Admitted source scope

The KB contains exactly **26 existing Health-Copilot cards**:

- 21 CDC cards
- 5 WHO cards

This is the source set explicitly qualified in [`u2d_external_evidence_qualification.json`](../../docs/research/integration/u2d_external_evidence_qualification.json). Its allowed role is non-commercial external retrieval, with publisher attribution and source links preserved. Source documents cannot be redistributed or used for training. The 4 NHC cards are excluded. No full WHO guideline documents, benchmark items, R2MED corpus, or synthetic U2-F evidence are included.

The older E5 combined index was not reused: its 30-card public-health view includes the later-excluded NHC cards, and its added guideline documents are outside U2-D's selected source scope. The eligible CDC/WHO cards were rebuilt as a separate frozen view.

## Frozen identity

| Field | Value |
|---|---|
| Corpus ID | `COMMON_MEDICAL_KB_V1_CDC_WHO_BP_26_399c528c7784` |
| Corpus SHA-256 | `399c528c7784b1a722e20c181d6889f2c4a043d56240bf6e5dd1d4824da5f4e0` |
| Index SHA-256 | `e9ad9ff82073f63867487c118602d5da1e7c4f156dec6ea31d4a3005b8f3834c` |
| Dense-vector artifact SHA-256 | `595559b8d3a5b3c56bd8b5599872221c2ba6a9241ae28418d9a9019b69846760` |
| Dense model | `BAAI/bge-large-en-v1.5`, revision `d4aa6901d3a41ba39fb536a557fa166f842b0e09` |
| Dense weight SHA-256 | `45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7` |
| BM25 | Health-Copilot `BM25Retriever`, Jieba NFKC/lowercase tokenizer, `k1=1.5`, `b=0.75`, title/content/tags |
| Fusion | RRF `k=60`, channel weights `[1,1]`, candidate depth 5 |
| Harness evidence top-k | 5 |
| Runtime provider | `health_ai_copilot.providers.common_medical_kb_v1:common_medical_kb_v1_provider_factory` |
| External artifact root | `/root/gpufree-share/data/health-copilot/common-medical-kb-v1` |

Embedding/index build environment: Python `3.11.17`, sentence-transformers `6.1.0`, Transformers `5.17.0`, Torch `2.13.0+cu130` (CPU execution), NumPy `2.3.5`, Jieba `0.42.1`, safetensors `0.8.0`. The provider refuses a runtime with a different Python or package version.

The full source-card hashes, relevant BGE configuration hashes, tokenizer/index configuration, implementation file hashes, and artifact manifest are recorded in [`common_medical_kb_v1.json`](../../configs/eval/common_medical_kb_v1.json) and the external `manifest.json`. The source text and derived dense vectors remain outside Git.

## Qualification checks

- The builder admitted exactly 21 CDC and 5 WHO cards from the U2-D-qualified scope, checked each card hash, and rejected any inventory-count drift.
- The exact BGE weight hash matched the existing frozen BGE identity. The local provider validated the model configuration files, corpus cards, dense-index manifest, dense-vector hash, implementation hashes, corpus ID, and index ID before returning evidence.
- A source-title wiring smoke retrieved the expected card in the top 5 for **26/26** titles. This is a diagnostic only; it is not a benchmark or relevance-quality result.
- A separate one-query provider smoke returned five evidence items with stable corpus/index identity and an evidence hash.
- A gold-blind overlap audit checked the 1,024 CMB and 915 DiagnosisArena candidate views against the 26 card texts. It opened neither scorer view nor gold. It found **0 exact full-prompt overlaps** and **0 five-gram containment candidates at 0.95**. Details and input hashes are in [`overlap_audit.json`](../../runs/common_eval/harness-v1-base-20261005/common-kb-v1-qualification/overlap_audit.json).
- Retrieval smoke output is in [`title_smoke.json`](../../runs/common_eval/harness-v1-base-20261005/common-kb-v1-qualification/title_smoke.json).

## Limits and use in the factorial

This 26-card collection covers hypertension patient education, not the full CMB or DiagnosisArena medical subject range. Questions outside that scope may retrieve no useful evidence. The BGE encoder is English-oriented; earlier RAG work recorded weaker dense-only ranking for some Chinese public-health title queries. BM25 and dense fusion are frozen here; no query or scoring threshold was changed in response to the candidate benchmark cases.

The retrieval provider is top-level Harness-owned, shared by B1 and B3, and receives each question once before reasoning. Adaptive MDT does not issue independent retrieval calls. The run manifest records the exact KB config, corpus ID/hash, index hash, dense-vector hash, and artifact-manifest hash; resume refuses a changed retrieval identity.

This qualification removes the infrastructure and source-scope blocker for B1/B3. It does not itself authorize a causal claim. Final B0/B1/B2/B3 results still require the same frozen IDs, parser, Qwen3-8B serving process/config, thinking mode, seed, token limits, and capture settings across all four arms.
