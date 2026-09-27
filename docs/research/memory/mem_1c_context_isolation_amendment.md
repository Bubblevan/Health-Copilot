# MEM-1C: Context Isolation and Metric Hardening

## Protocol Amendment

MEM-1C retains the approved fully local model topology and adds a primary
`context_controlled` track. It is a controlled re-evaluation of public memory
architectures under a unified fully-local model stack, not an exact MemEval
numerical reproduction or the official LongMemEval GPT-4o accuracy setting.
The earlier native-system smoke remains an adapter/runtime audit.

All five systems produce a benchmark-side `ContextBundle` from their native
retrieval path. The bundle is frozen before the shared answer call. One fixed
Qwen3-8B reader prompt, temperature 0, seed 42, thinking disabled, and a
256-token output cap consume every bundle. Only the context contents differ.
System-specific final answer prompts, JSON schemas, CoT answer heads, and
answer parsers are excluded from this controlled track. Memory-internal
extraction/classification remains part of the compared system and uses the
same local Qwen3-8B artifact.

The bundle artifact contains the system and question ID, ordered items with
text/rank/kind/source session IDs, serialized context, Qwen3 tokenizer context
count, provenance availability, retrieval/ingestion latency, and a canonical
SHA256. The bundle JSONL and frozen prediction JSONL are retained independently.
The prediction row embeds the bundle hash so a reader output cannot be replayed
against a different context unnoticed.

## Correctness and Metrics

FullContext preflight uses the same configured 256-token reserve as the actual
reader request. It still requires a 131072-token llama.cpp slot, exact template
tokenization, and successful `truncated=false` telemetry.

Abstention is selected only by `question_id.endswith("_abs")`. These cases
retain their original one of six LongMemEval question types and are additionally
reported as `abstention_n` and deterministic `abstention_accuracy`.

Primary controlled-track metrics are answer-session Recall@5, Recall@10, MRR,
context token count, retrieval/ingestion latency, and provenance coverage.
Downstream metrics include deterministic token precision, recall, F1, and
normalized exact match. Metrics remain broken out by all six question types.
When a baseline does not expose source-session provenance, session retrieval
metrics are `null`, not zero.

SimpleMem's current SSE path does not reliably expose prompt/completion usage.
Missing usage is recorded as `null` / `NOT_CAPTURED`, never synthesized as zero.
Context token counts and latency remain measurable. No total-token Pareto claim
may include SimpleMem unless accounting becomes reliable without changing its
semantics.

An optional local Qwen judge is not run during this amendment. If enabled in a
later reviewed stage, it must run only after prediction freeze, use a separate
cache/trace, and be named `local_qwen_judge_accuracy`. It is not official
GPT-4o accuracy; headline use requires manual disagreement calibration on the
frozen 10-case diagnostic. The opt-in command is
`python tools/research/memory/run_mem1_local_judge.py --run-dir <frozen-run-dir>`;
its result manifest marks the metric ineligible for headline use until that
calibration is completed.

Failure attribution is limited to observable causes from the frozen taxonomy:
`INFRA_FAILURE`, `FALSE_ABSTENTION`, `SHOULD_ABSTAIN`,
`CONTEXT_HAS_ANSWER_READER_MISSED`, and `CONTEXT_MISSING_ANSWER` in this stage.
Revision/stale-state labels remain future RevMem diagnostics. No failures are
described as SFT/RL-related.

## Local CUDA Runtime and Reader Acceleration

The shared embedding model remains `Qwen/Qwen3-Embedding-0.6B` at the frozen
local revision and hash, now executed on `cuda:0` in float16 with L2-normalized
float32 vectors. Every dense-compatible baseline uses this adapter. The
PyTorch `2.10.0+cu128` wheel is installed as an ignored local runtime overlay;
the pinned MemEval `uv.lock` is not changed. Recreate it with
`tools/research/memory/install_mem1_cuda_torch.ps1`.

The recommended llama.cpp launch keeps all model layers on the RTX 4090 Laptop
GPU, enables Flash Attention, and uses Q4_0 GPU KV cache so the CUDA embedding
model can remain resident. The frozen runtime passed repeated 4K synthetic
queries and a synthetic 115000-token prompt plus a 256-token output reserve
inside its 131072-token slot; server prompt-token counts matched tokenizer
counts and llama.cpp logged `truncated=0`. Peak sampled VRAM was 12682/16376
MiB. In the 4K diagnostic, both with the same Q4_0 KV precision and CUDA
embedding resident, GPU KV achieved 56.98 generated tokens/second median,
1408 ms median TTFT, and 3095 ms total versus CPU KV at 26.26 tokens/second,
2132 ms TTFT, and 5787 ms total. GPU KV was 2.17x faster on decode, with
sampled peak VRAM 12682 MiB versus 7546 MiB. The 115K synthetic prompt prefill took about 153
seconds (110932 tokens evaluated after a cached prefix) and decoded at 17.59
tokens/second; that long-context cost remains explicit. A synthetic Q8_0 GPU KV
co-residency attempt exhausted VRAM and disconnected the reader, so Q8_0 GPU KV
is not used. The launcher is
`tools/research/memory/start_mem1_local_reader.ps1`; `-CpuKv` is a fallback.
These are hardware diagnostics, not benchmark quality results.

The system Anaconda installation has CUDA-enabled PyTorch `2.2.2+cu118`. The
initial MEM-1 isolated environment instead resolved a CPU-only PyTorch wheel
from MemEval's generic dependency range; this was an environment-installation
mistake, not a machine limitation. The isolated venv now uses the official
`2.10.0+cu128` wheel and reports CUDA available on the RTX 4090. The upstream
`uv.lock` and system Anaconda environment are unchanged.

## Stage Gate

This amendment does not authorize a new 10-case run, a 102-question DEV run,
TEST, M10-Flat, or RevMem. Run offline tests, parser/telemetry tests, and a
small local hardware smoke only. The next 10-case diagnostic requires human
review and compares both native-system audit outputs and the shared-reader
`context_controlled` outputs. Stop after those 10 cases.
