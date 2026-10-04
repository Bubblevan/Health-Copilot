# Memora on LoCoMo with Local Qwen

Status: `complete`

This is a controlled architecture transfer of the pinned Microsoft Memora implementation to a fully local Qwen stack. It is not an exact numerical reproduction of the paper's GPT reader/judge results.

Questions answered: 1986 expected.

## Main Results

| Strategy | N | Official token F1 | Official EM | Normalized EM | Local Qwen judge accuracy | Judge N | Judge format fallbacks | Mean memory-context tokens | Retrieval sec | Answer sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| semantic | 1986 | 0.3186 | 0.0403 | 0.0826 | 0.8195 | 1540 | 177 | 19542.1616 | 0.6429 | 26.9251 |
| prompt | 1986 | 0.3181 | 0.0468 | 0.0811 | 0.8026 | 1540 | 157 | 20373.2034 | 31.6205 | 15.0351 |

## Category Results

| Strategy | Category | N | Official token F1 | Official EM | Normalized EM | Local Qwen judge accuracy | Mean memory-context tokens | Retrieval sec | Answer sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| semantic | single-hop | 282 | 0.3437 | 0.0071 | 0.0355 | 0.8227 | 20416.4858 | 0.6637 | 27.3873 |
| semantic | temporal | 321 | 0.3646 | 0.0187 | 0.0374 | 0.6947 | 19473.4361 | 0.6972 | 28.6177 |
| semantic | multi-hop | 96 | 0.1529 | 0.0312 | 0.0417 | 0.6354 | 19512.3750 | 0.6805 | 26.4994 |
| semantic | open-domain | 841 | 0.4782 | 0.0809 | 0.1617 | 0.8870 | 19582.2390 | 0.7027 | 25.0184 |
| semantic | adversarial | 446 | 0.0045 | 0.0022 | 0.0045 | n/a | 18969.6413 | 0.4696 | 29.1016 |
| prompt | single-hop | 282 | 0.3445 | 0.0213 | 0.0461 | 0.7943 | 21130.2660 | 38.6640 | 16.4638 |
| prompt | temporal | 321 | 0.3529 | 0.0156 | 0.0343 | 0.6449 | 20231.2368 | 31.4277 | 16.5307 |
| prompt | multi-hop | 96 | 0.1566 | 0.0312 | 0.0417 | 0.5729 | 20487.1042 | 40.1708 | 17.1323 |
| prompt | open-domain | 841 | 0.4818 | 0.0939 | 0.1570 | 0.8918 | 20407.0012 | 31.3506 | 14.5125 |
| prompt | adversarial | 446 | 0.0022 | 0.0000 | 0.0022 | n/a | 19908.4529 | 25.9744 | 13.5896 |

## Paired Strategy Contrast

Prompted-policy minus semantic official F1: `-0.0006` over 1986 paired questions; stratified paired-bootstrap 95% CI `-0.0100 to 0.0087` (10,000 resamples, seed 42).

Prompted-policy minus semantic local-Qwen judge accuracy: `-0.0169` over 1540 paired, judge-eligible questions; stratified paired-bootstrap 95% CI `-0.0364 to 0.0026` (10,000 resamples, seed 42; category 5 excluded). Both intervals include zero, so this run does not establish a quality difference between the two retrieval strategies.

## Interpretation

- Under this local Qwen stack, the prompted policy shows no statistically resolved quality gain over semantic retrieval on either deterministic F1 or local-Qwen judge accuracy.
- Mean retrieval-plus-answer latency is about `27.57s` for semantic and `46.66s` for prompted policy; prompted retrieval also uses about `831` more reader-context tokens on average. Semantic is therefore the practical local default from this run, not a claim of general superiority.
- The semantic token F1 is `0.3186`, and this experiment contains no same-run FullContext comparator. It supports successful method transfer and a measured policy ablation, but does not support a claim that Memora improves answer quality or reduces context versus FullContext under Qwen.

## Shared Memory Ingestion Usage

| Role | Calls | Prompt tokens known | Completion tokens known | Service sec |
|---|---:|---:|---:|---:|
| memory_ingest | 12011 | 9404943 / 12011 calls | 2246284 / 12011 calls | 105075.9250 |
| embedding | 31931 | 272543 / 31931 calls | 0 / 0 calls | 6373.1320 |

## Local Model Usage

Role-wise request counts, local-server prompt/completion usage where available, and local service latency:

| Strategy | Role | Calls | Prompt tokens known | Completion tokens known | Service sec |
|---|---|---:|---:|---:|---:|
| semantic | memory_reasoning | 0 | 0 / 0 calls | 0 / 0 calls | 0.0000 |
| semantic | reader_answer | 1986 | 41487294 / 1986 calls | 23119 / 1986 calls | 53464.0790 |
| semantic | judge_local | 1971 | 819545 / 1971 calls | 26110 / 1971 calls | 18879.0220 |
| semantic | embedding | 3979 | 144880 / 3979 calls | 0 / 0 calls | 851.7850 |
| prompt | memory_reasoning | 5155 | 14022751 / 5155 calls | 359851 / 5155 calls | 61599.2620 |
| prompt | reader_answer | 1992 | 43204261 / 1986 calls | 22585 / 1986 calls | 29837.8270 |
| prompt | judge_local | 1540 | 638767 / 1540 calls | 19149 / 1540 calls | 7260.2450 |
| prompt | embedding | 5216 | 192634 / 5216 calls | 0 / 0 calls | 790.1770 |

## Storage Diagnostics

Captured Chroma query retries: `0`; terminal query failures: `0`. Terminal failures prevent a complete status. Raw events: `D:\MyLab\Jianli\Health-Copilot-main-integration-mem3b0r\runs\memory\memora-locomo-qwen-v7-isolated-output\storage_events.jsonl`.

## Protocol Notes

- Reader, memory-internal LLM, and judge: the same frozen Qwen3-8B Q4_K_M through the loopback llama.cpp service.
- Embedding: frozen Qwen3-Embedding-0.6B, local CUDA FP16, 1024 dimensions, normalized vectors.
- Memory method: Memora's LLM segmentation, primary abstraction plus specific memory values, cue anchors, episodic links, update decisions, per-conversation local Chroma storage, BM25 hybrid retrieval, and the official semantic/prompted-policy retrieval paths.
- No hosted model or embedding APIs; no API keys; no GRPO training. Category 5 is omitted only from the Memora-style judge accuracy, matching its official LoCoMo evaluation code; deterministic metrics include it.
- Memory-context token counts use the pinned Qwen reader's local llama.cpp tokenizer on formatted memory lines joined by newlines; they are not full reader-prompt token counts. Model prompt/completion token totals are reported from the local server only when it supplies usage fields.
- The llama.cpp server was configured with total `131072` context and four parallel slots, so the effective maximum was `32768` tokens per sequence. One prompted-policy answer request reached `32891` tokens and received the server's explicit context-overflow error; the adapter retried that question's full Memora call, which then succeeded. Three such explicit local-reader retry events were recorded in `provider_retry_events.jsonl`; all final predictions and eligible judgments are present. Do not interpret total server context as per-request context.
- This run has no same-run FullContext baseline. The reported memory-context tokens alone do not establish a token-reduction percentage or a quality improvement over FullContext.
- The semantic judge call total includes retired 64/128-token format attempts from earlier recovery passes; only the 1540 final judgment rows contribute to judge accuracy. Historical failed attempts remain in `failures/*.jsonl`; `summary.json` reports zero unresolved failures after resume.
- Predictions, judge rows, call ledger, and persisted memories are resumable local artifacts under the run directory.
- Conversations have disjoint users and no cross-conversation retrieval, so Chroma persistence is partitioned by conversation and upstream trace JSON by strategy/conversation; QA within each conversation is sequential while conversations run concurrently.
- The paper's published GPT-based results are external historical coordinates, not a directly comparable baseline for this Qwen run.

Run artifacts: `D:\MyLab\Jianli\Health-Copilot-main-integration-mem3b0r\runs\memory\memora-locomo-qwen-v7-isolated-output`

## Provenance and Integrity

- Run identity: `c46462b0d06c7ab1803b72bf9f9e4e22f27dabf4e65e7c7fd515cc78e2c7c694`
- Microsoft Memora source: commit `dec3f8f2444eace7004fc084abe1be9f3d88270e`
- LoCoMo source: commit `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`; dataset SHA256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
- Local paper copy: `D:\MyLab\Jianli\external\memory\Memora\papers\memora_harmonic_memory_2602.03315.pdf`; SHA256 `c6f1b9381bff11a05b24a5dadd6d3a88d200c85e564c364e5813aa1493c1ad760`
- Reader: Qwen3-8B Q4_K_M GGUF SHA256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`; llama.cpp `10068 / 571d0d540`, binary SHA256 `3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb`.
- Embedding: `Qwen/Qwen3-Embedding-0.6B`, revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, weights SHA256 `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`.
- Frozen upstream runner SHA256: `8d48d6719cb7d911b4514c0eabf9a7deea906f0e4b4eb8995d1c1a6cf31d645e`; amended local wrapper SHA256 is recorded in `judge_format_amendment.json`.

| Artifact | SHA256 |
|---|---|
| `protocol_manifest.json` | `a4737490e5570ba33b0e4a01309535b4ce8aa2814064ff610328fd5fcafe4956` |
| `judge_format_amendment.json` | `294ca81f25940668fe6bc138a4295ee0b39b38be9fd6c63bfe6793c214d4e921` |
| `summary.json` | `380f337cd4437a3b5296fc02ab0693222bb9c85f32c7360e962249eb0b6a239a` |
| `predictions/semantic.jsonl` | `f09090f3f883a4beee948564b2d12cd53651f73ec3a829104604d5ac8aed92e1` |
| `judgments/semantic.jsonl` | `78820907ff4ce43597c1d941db40638826fd51b55db14e5e5d6f69ef7e3fd7b6` |
| `predictions/prompt.jsonl` | `e108c23bbb6aaa49227680a82043255febd15face0bc52079ceb0b7ea430b928` |
| `judgments/prompt.jsonl` | `b5978b51a8ea027bffed62636de34f537d444c479300d6fa67752e132364fd73` |
| `call_ledger.jsonl` | `9877b1b9792a98ed6cc4fd148abad3808ed21298ad445cf1f20ae7ad6964be80` |
| `provider_retry_events.jsonl` | `cfa9917c55e4ddc7615a3a68d4900cfd821401ab89270acbebfedcab578e7beb` |
