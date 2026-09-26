# MEM-0 Protocol Freeze

**Status:** `PROTOCOL_FROZEN`
**Gate:** `MEM_PROTOCOL_LOCK=YES`; split parity and Main Track long-context feasibility passed. MEM-1 has not started.
**Audit date:** 2026-09-26
**Health-Copilot base commit:** `76836cf162a9b3adcc0f52c4ee5edcf1dfd1f4fe`

## Research question

Can Health-Copilot's existing revisioned, temporal, provenance-aware memory substrate improve structured temporal proposition memory on public long-term-memory benchmarks, especially by reducing stale or invalid memory reuse while retaining valid historical recall?

The working adaptation name is **RevMem (Revision-Aware Temporal Proposition Memory)**. It is not a new-algorithm claim before results exist.

## Positioning constraint

Health-Copilot already has medical RAG for externally sourced, citable clinical evidence. Memory addresses a different problem: continuity of user-specific state across sessions, including what is current, what changed, and what was true at an earlier time. A remembered user statement remains personalization/context, never clinical evidence or citation authority.

The coherent resume story is therefore: **a safety-first medical assistant separates evidence retrieval from longitudinal user state, then evaluates whether explicit revisions and validity intervals reduce stale-state errors.** Multi-Agent and RL are later research stages, only justified by measured remaining failures. They are not extra methods to list in this Memory result. No training or RL is allowed in this stage.

## Frozen benchmark plan

### Track A: LongMemEval-S

- Use the canonical `xiaowu0162/longmemeval-cleaned` dataset at revision `98d7416c24c778c2fee6e6f3006e7a073259d48f`; use `longmemeval_s_cleaned.json` only.
- DEV is the 102-question, six-category, 17-per-category sample produced by the pinned MemEval sampler at commit `807ae6d7d8a5b76f6fe964d5a581d96c036e2ac4` with seed `42`, applied to the canonical S file in source order. Active IDs and hashes are in `split_manifest.json` v2.
- TEST is the exact complement of those DEV IDs in the canonical 500-question S inventory: 398 cases. It is a public, process-level held-out split, not hidden or unseen.
- S file SHA/size, unique ID inventory, all-ID digest, category counts, exact DEV sampler reproduction, TEST complement, disjointness and union have been recomputed and pass. DEV SHA256: `c6e0b423f720bcb06e1c571a8f6b0d018e0c1707d21fe17108349962d30f739f`; TEST SHA256: `506b025d1ec4a3b18737863fa62ac12f7b272631fcb748d0cc2f466cd6bf424b`.
- Preserve the Oracle-derived v1 file and `split_parity_audit.json` as audit records. The old split is explicitly invalid and must never be used for tuning or evaluation.
- Tune only on DEV. Do not run the TEST gold scorer until the method/configuration lock. Do not describe a public complement as unseen or hidden data.

### Track B: Genies Memora / FAMA

Use the distinct `geniesinc/Memora` benchmark repository, not the checked-out Microsoft memory system with the same name. This is the forgetting-aware highlight track and remains out of scope until the LongMemEval baseline/ablation gates are reviewed. At MEM-4, pin the benchmark commit, data hashes and evaluator before running it.

## LongMemEval systems and controls

MEM-1 systems are `Full Context`, `OpenClaw`, `SimpleMem`, `PropMem` and `Mem0 OSS`, all run in the same Main Track reader configuration where compatible. PropMem is the strong primary comparator. Mem0 managed-platform numbers are excluded. M10-Flat and RevMem are later Health-Copilot adapters. MemEval's README reports four LongMemEval systems; its GPT-4.1 scores are historical upstream coordinates, never a comparator for our Qwen Main Track claims.

Keep these four roles separate in every config, run manifest, cost report and resume claim:

| Role | Frozen identity | What it means |
|---|---|---|
| Reader / answer model | Main: local Qwen3-8B Q4_K_M GGUF, SHA in `model_protocol.json`; upstream sanity only: GPT-4.1 | Reads a system's supplied context and produces the answer. The Main Track uses the same model artifact/runtime/settings for all compatible systems. |
| Memory system | Full Context, OpenClaw, SimpleMem, PropMem, Mem0 OSS; later M10-Flat/RevMem | The method being compared: write/materialize, store, revise, retrieve and project context. System-specific prompts/configs are hashed and logged. |
| Embedding model | OpenAI `text-embedding-3-small`, 1536 dimensions | Dense-vector representation used by compatible memory systems. Input-token usage is accounted separately from reader and judge. |
| Judge model | OpenAI `gpt-4o`, temperature `0`, `max_tokens=10` | LongMemEval native binary-accuracy judge only. It does not answer benchmark questions. |

The deterministic MemEval token-F1 implementation is a local metric, not another model. Reader, memory system, embedding and judge must never be conflated. Any system-internal model calls during ingestion are logged separately; no hidden/proprietary model advantage or unreported call is allowed.

### Main Track: controlled Qwen reader

- Run every compatible baseline and Health-Copilot adapter with the exact local Qwen3-8B Q4_K_M GGUF, llama.cpp build, generation settings and context configuration in `model_protocol.json`.
- Reader settings: temperature `0`, seed `42`, `enable_thinking=false`, maximum output `64` tokens. System-specific memory and answer prompts may differ where upstream defines them; hash and report them rather than claiming prompt identity.
- Main Track scores are Qwen3-8B results. Do not compare them as outperforming or matching GPT-4.1 numbers in the MemEval README.
- Use `text-embedding-3-small` wherever dense embeddings are required and compatible. Record exact embedding input tokens and API receipts per system.
- Run the five MEM-1 systems on 1-case, 10-case, then all 102 DEV cases. Keep F1 and GPT-4o judge accuracy.

### Upstream-parity sanity track

- Before the main reproduction, run only six DEV questions, one per category, on the five MEM-1 systems using the pinned upstream GPT-4.1 reader configuration and GPT-4o judge. The deterministic question IDs are frozen in `model_protocol.json`.
- Purpose: adapter wiring and broad direction/sanity checks only. It is not the primary benchmark, cannot tune the Main Track, and cannot support performance claims against upstream results.
- The same reader is used across all five systems within this separate track. Report it separately from Qwen3-8B Main Track results.

### Full Context long-context gate

- Qwen3-8B natively supports 32,768 tokens; YaRN can extend it to 131,072. Our local GGUF reports training-context metadata `40960`; the installed llama.cpp server otherwise caps a requested 131,072-token slot at 40,960. The tested configuration applies YaRN factor `4`, original context `32768`, and explicit `qwen3.context_length=int:131072` metadata override. Do not omit or silently change this override.
- The tested reader uses a Q4_K_M model on GPU and Q8_0 K/V cache on CPU (`--no-kv-offload`) with one slot. Full settings and measured evidence are in `long_context_preflight.json`.
- The exact MemEval FullContext prompt was tokenized for all 102 DEV cases with the Qwen GGUF tokenizer and chat template. Maximum input is `110289`; with the configured 64-token output reserve it fits in `131072`. The longest case completed with `truncated=false`.
- Every scored FullContext answer must record tokenizer prompt tokens, `max_model_length=131072`, output budget and server `truncated=false`. Any truncated query invalidates that run item; no silent context shifting/truncation is allowed. Before TEST scoring, repeat the all-case token-length preflight on all 398 TEST cases without invoking the gold scorer.

## Metrics and analysis

LongMemEval primary results: token F1 and native judge accuracy, overall and by the six `question_type` categories. Report both numerator and denominator. Retrieval diagnostics for Health-Copilot adapters: answer-session Recall@5, Recall@10 and MRR. Memory diagnostics: obsolete-context rate, current-state conflict rate, entity contamination, revision resolution, historical recall, abstention accuracy. Efficiency: ingestion tokens/session, retrieval tokens/query, reader-context tokens/query, total tokens/query, p50 and p95 query latency.

For each paired RevMem-minus-comparator result, use 10,000 paired bootstrap resamples stratified by question type and report absolute delta with 95% CI. If the interval crosses zero, label the result inconclusive. Keep quality, forgetting and cost as separate metrics; do not invent a combined MemoryScore.

Memora primary metric at MEM-4: FAMA, broken down by Remembering/Reasoning/Recommending and weekly/monthly/quarterly, plus memory-presence accuracy, forgetting-absence accuracy and obsolete-memory exposure. Check that any FAMA gain does not come from suppressing valid memories.

## Ablation sequence after MEM-1

1. MEM-2 M10-Flat: atomic propositions, flat storage, hybrid retrieval and RRF only.
2. MEM-3A: entity-scoped retrieval.
3. MEM-3B: revision chain and ADD/UPDATE/DELETE materialization.
4. MEM-3C: CURRENT/AS_OF/CHANGE query mode and hard temporal validity filtering.
5. MEM-3D: deterministic, budgeted context projection.

Do not implement these together. Do not change the frozen M10/M10.1 schemas or profiles. Keep Memory separate from Evidence, citations, clinical authority, safety policy, tool permissions and side effects.

## Cost estimate for MEM-1 DEV

The old estimate based on 55.2M GPT-4.1 reader tokens and `$110–$442` is withdrawn; it is not applicable to this local-Qwen Main Track.

| Cost bucket | DEV planning basis | Estimate/accounting |
|---|---|---|
| Local Qwen reader compute | 5 systems × 102 questions = 510 answer calls; no hosted reader API charge | Report wall time, GPU time and, where available, energy. FullContext preflight measured 110,289 prompt tokens in 280.016 s (393.87 tok/s). The DEV mean is 106,597 prompt tokens; a simple throughput extrapolation puts FullContext alone around 7.7 GPU-hours for 102 prompts. This is a planning extrapolation, not total Main Track runtime or a utility-bill estimate. |
| GPT-4o native judge | Up to 510 Main Track calls; `max_tokens=10` (5,100 output tokens maximum) | Standard rate: `$2.50/M` input, `$10/M` output. At an explicit 250–600 input-token planning range per judge call, estimate `$0.37–$0.82` for Main Track DEV; replace with actual usage receipts. Six-case sanity adds at most 30 judge calls, approximately `$0.02–$0.05` under the same assumption. |
| Embedding API | `text-embedding-3-small`, 1536 dimensions; token volume depends on each system's embedded memory representation and any deduplication | Standard rate: `$0.02/M` input tokens. No fixed total is claimed before system adapters produce usage receipts. Record embedded input tokens and dollars per system; do not fold them into reader cost. |
| GPT-4.1 sanity reader | 5 systems × 6 fixed questions = 30 answer calls, only in the separate upstream-parity sanity track | Six FullContext prompts contain 638,418 Qwen-tokenizer tokens; at GPT-4.1's `$2/M` input rate this is approximately `$1.28` input for those six calls alone, before output and the other 24 calls. This is an approximate budget proxy, not a Main Track cost or performance comparison. |

These are DEV-only planning quantities. Internal memory-system model calls, retries, caching, and API usage for embeddings are separate line items and must be captured in run manifests. Recheck official prices at run time. Do not run the 10/102-case API stages until the 1-case receipt has been reviewed; do not run TEST until method/configuration lock.

## Baseline reproduction and closure gates

MEM-1 smoke order: 1 case, then 10 cases, then 102 DEV. Save per-question raw predictions, retrieved memories, judge outputs, F1, category, token usage, call count, ingestion usage, and latency. Failures must go to `failures.jsonl`; provider/judge failures cannot silently become valid zero scores.

No resume headline is authorized until the relevant result has been reproduced with the pinned data, scorer, models and configuration. The defensible target is a measured improvement in temporal/knowledge-update or forgetting-aware quality without a serious loss of valid-memory recall, not a claim to beat every memory system. No Memory result authorizes a clinical efficacy or safety claim.

## MEM-0 review checklist

- [x] Identify the four existing external repositories and distinguish the two Memora projects.
- [x] Pin LongMemEval-S source revision and expected file digest from upstream metadata.
- [x] Archive Oracle-derived v1 as invalid candidate; freeze S-derived DEV v2 and public process-level TEST v2.
- [x] Recompute dataset SHA/size, all-ID inventory, sampler parity, category counts, TEST complement, disjointness and union.
- [x] Record distinct reader, memory system, embedding and judge roles, both reader tracks, metrics and upstream caveats.
- [x] Verify local Qwen3-8B YaRN 131K slot and all 102 DEV FullContext prompt lengths; longest request returned `truncated=false`.
- [x] Freeze Main Track reader, six-case upstream sanity selection, long-context configuration and separate cost buckets.
- [x] Human review completed and subsequent requested protocol revision incorporated.

`MEM_PROTOCOL_LOCK=YES`. MEM-0 is complete. Stop here: MEM-1 has not started, no GPT-4.1/GPT-4o/embedding API calls have been made, and no benchmark score exists.
