# MEM-1D0.5 SimpleMem Provenance Correction

**Base:** `b9b7047279ec1c069abad3281c9615b481ad1c61`  
**Gate:** `MEM1D_BASELINE_FIDELITY_READY=YES`  
**Primary state:** `STATE_A_OFFICIAL_SIMPLEMEM_V0_1_0_TAG`  
**SimpleMem:** `KEEP`  
**Frozen 10-case diagnostic:** `NOT_RUN_BY_GATE`  
**LongMemEval TEST:** untouched

## Provenance Decision

The earlier finding applies to the MemEval-pinned PyPI arm only: `simplemem==0.1.0` had an effective semantic-only planning execution path in the audited adapter. That is not a description of SimpleMem as an architecture. The arm is now identified as `SimpleMem-PyPI-0.1.0-MemEval`, retained as a provenance diagnostic and excluded from the primary comparator matrix.

The primary source is the separate, clean official `aiming-lab/SimpleMem` checkout at tag `v0.1.0`, commit `7da777f56a15db81bb261d296c89cad5915e8d67`, source-tree SHA256 `e84e01b775296db1fcde799fac1a5ec2ffafe4a00de4a9f2000eb885aa5c1191`. The machine-readable source audits are [PyPI fidelity](simplemem_pypi_010_fidelity.json) and [official-tag fidelity](simplemem_official_v010_fidelity.json). Both use a synthetic fixture and no LongMemEval data.

The official source's planner calls semantic, keyword and structured retrieval, then merge/deduplication; optional reflection remains upstream-controlled. Its official-tag synthetic fidelity gate passed. No ranking, top-k, prompts, query analysis, compression, synthesis or merge behavior was changed by the adapter.

## Compatibility Boundary

The Health-Copilot adapter imports the official source tree behind a scoped module boundary and validates its remote, exact tag, commit, source-tree hash and clean worktree. It routes the source's LLM client to the frozen loopback Qwen3-8B and its embedding interface to the unified local Qwen3-Embedding-0.6B adapter. Execution is serialized to one worker for the single-slot local reader; the local provider uses non-streaming transport for usage capture. The official native answer head is not invoked on the context-controlled track.

The adapter uses the already isolated MemEval environment's shared dependencies only; no official full requirements set was installed into that environment and `uv.lock` was not changed. The PyPI package remains installed as a pinned diagnostic dependency but is not imported by the official source adapter. MemEval's adapter patch SHA256 is `c7e44015527a17d43cf9bbd2e2a6e50d007943ee2efaa244899d3b048fd80eab`; patched-tree SHA-1 is `f9e1e89e58c5e0792017dfcd40cb8c9c28fb6a83`. The patch applied and reverse-applied cleanly; the external MemEval and official SimpleMem checkouts both ended clean.

## One-Case Adapter Smoke

Only DEV question `1cea1afa` and only official SimpleMem were run, using `context_controlled`. This is runtime/adapter evidence, not a performance comparison. The run used the frozen shared reader template and local model stack; the native SimpleMem answer head was not called.

| Diagnostic | Observed |
|---|---:|
| Context reader tokens (llama.cpp Qwen3-8B tokenizer) | 2,066 |
| Context embedding tokens (Qwen3-Embedding tokenizer) | 2,067 |
| Shared reader prompt tokens | 2,139 |
| Retrieval latency | 17,499.520 ms |
| Ingestion latency | 1,580,903.119 ms |
| Answer-session Recall@5 / Recall@10 / MRR | null / null / null; upstream source-session provenance unavailable |
| Semantic / keyword / structured / merge-deduplicate / reflection calls | 7 / 1 / 1 / 3 / 1 |
| Provider calls | 33 total: 25 local Qwen, 8 local embedding |
| Hosted calls / judge calls / infrastructure failures | 0 / 0 / 0 |
| Deterministic token F1 / precision / recall / normalized EM | 0.0 / 0.0 / 0.0 / 0.0 |
| Baseline-internal recoveries / infrastructure warnings | 4 / 0 |

The answer F1 is recorded as a single-case diagnostic only. It is not evidence of a quality regression or ranking against other systems; no other system was run in this stage. The recovery warnings are retained in `runs/memory/mem1/official-simplemem-one-case-20260927/baseline_warnings.jsonl`, classified as `BASELINE_INTERNAL_WARNING`, with no raw logs or prompts copied into the warning artifact.

Local compute accounting from the call ledger: Qwen used 1,590,737.549 ms summed request latency (192,959 input tokens and 78,720 output tokens); local embedding used 3,608.201 ms summed request latency (10,608 input tokens). These are local runtime measurements, not hosted API spend. Hosted reader, embedding and judge cost was `$0` because no hosted calls occurred.

The ContextBundle canonical SHA256 is `5843a78e34a33fca83723474468079d1980e54cb97be5e964d473d8136ffd65d`; the prediction row references that same bundle SHA. `native_answer_head_invoked=false`. The prediction artifact is frozen at SHA256 `8e916a90a8b0f4694eb1a65a42d9e8569941673e58e09d4a0d0cc88d850fbe4b`.

An unchanged resume preserved the prediction SHA, call-ledger SHA `a0177244a75688776de409dcee9ede738ac5e2a35a3ea960e8413c43437f9b24`, and ContextBundle file SHA `deb115bef8fba84373eb245f7dbf7da705e85298a9e34e3cfe1bd00bddd0e161`; call count remained 33 and no model calls were added.

## Runtime Revalidation

| Role/runtime | Revalidated configuration |
|---|---|
| Reader / memory-internal LLM | Qwen3-8B Q4_K_M SHA256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`; llama.cpp `10068 / 571d0d540`, binary SHA256 `3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb` |
| Reader runtime | Loopback `127.0.0.1:8081`; 99 GPU layers; Flash Attention on; Q4_0 K/V cache on GPU; observed slot `131072`; no CPU-KV fallback |
| Embedding | `Qwen/Qwen3-Embedding-0.6B`, revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, tree SHA256 `9d2d790d6448ef2c0911ffeb03f959d035c71ac3d2b14b7d586f2d2b39fb0efa`, weights SHA256 `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`, 1024 dimensions, normalized, CUDA FP16 |
| Hardware | NVIDIA GeForce RTX 4090 Laptop GPU; CUDA `12.8`; Torch `2.10.0+cu128` |
| Hosted runtime / API key | None / none required; process ran without `OPENAI_API_KEY`; embedding loader uses local files only; all ledger providers were local |

The exact llama-server version/build/hash and GGUF hash passed before launch. The server was stopped after the run and cache verification; GPU use returned to the observed desktop baseline of approximately 244 MiB. This one-case run does not exercise FullContext, so it makes no per-question `truncated=false` claim for that baseline; the independent 131,072-token slot was verified.

## Verification And Stop

- Official SimpleMem synthetic hybrid fidelity: `PASS`.
- MemEval adapter/provider source-boundary tests: `22 passed`.
- Health-Copilot MEM-1 runner tests: `16 passed`.
- Official source tree and MemEval patch applied/reversed cleanly; both upstream checkouts clean afterward.
- Local-only call audit: 33/33 provider calls local; 0 hosted calls; 0 infrastructure failures.
- Cache/resume: unchanged hashes and no additional model calls.
- DEV questions: one (`1cea1afa`); primary systems run: official SimpleMem only.
- Frozen 10-case, remaining 102 DEV, and TEST: not run/accessed.

`MEM1D_BASELINE_FIDELITY_READY=YES`. Keep the official v0.1.0-tag SimpleMem arm in State A. Stop here pending the next reviewed instruction; do not automatically start the frozen 10-case diagnostic.
