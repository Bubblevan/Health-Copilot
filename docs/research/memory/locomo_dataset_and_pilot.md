# LoCoMo Dataset and Local Pilot

## Decision

The four architecture baselines are available as public source and can be run through pinned implementations/adapters. That does not make their published scores exact reproductions under Health-Copilot's local Qwen stack. The LoCoMo table in the pinned MemEval README reports PropMem at token F1 `0.605`, ahead of OpenClaw `0.557` and FullContext `0.542`; those are external historical coordinates, not Health-Copilot results. The table's model stack uses GPT-4.1-mini, `text-embedding-3-small`, and GPT-5.2 judge, unlike our local reader and no-judge protocol.

There is no metric-independent "best" architecture. For this project's no-judge, deterministic-token-F1 track, choose **PropMem** as the single primary architecture baseline/adaptation starting point: it has the strongest score in the pinned MemEval LoCoMo F1 table (`0.605`). A newer open implementation, Microsoft's **Memora-System** (MIT), reports the highest semantic judge score in its ICML 2026 comparison: `0.863` versus FullContext `0.825`; its reported overall token F1 is `0.553` versus FullContext `0.565`. These are a different paper protocol and are not comparable to MemEval's PropMem `0.605` F1 coordinate. Because the Main Track has no judge, the judge-leading score is not the selection criterion here.

Memora-P remains a strong 2026 architecture reference, not the primary baseline for this local run. The existing Microsoft/Memora checkout is the **system**, not the separately named Genies/Memora ACL benchmark. Its published runner defaults to GPT-4.1-mini and hosted embedding configuration, so a local-only controlled re-evaluation still needs a transparent Qwen/Qwen3-Embedding adapter. Its experimental GRPO path remains out of scope.

Do not rename or present Memora/PropMem mechanisms as ours. A defensible Health-Copilot contribution would be one separately implemented and measured extension: a deterministic scope + temporal-validity/revision gate before retrieval context is admitted, plus the existing medical boundary that keeps memory from becoming evidence or safety authority. This specifically targets stale-state reuse rather than adding another retrieval trick. Until that extension is evaluated, call it a proposed adaptation, not a result or novel architecture. The optional Memora GRPO path remains out of scope.

## Source And Reproduction Status

| System | Code source and license | Status |
|---|---|---|
| OpenClaw | Original project MIT; benchmark adapter in `ProsusAI/MemEval`, Apache-2.0 | Public implementation; local Qwen/local-embedding adapter exists. Exact README numbers use a different model stack. |
| PropMem | `ProsusAI/MemEval`, Apache-2.0 | Public implementation; strongest listed LoCoMo F1 comparator. Its local Qwen write path is unusually slow on this hardware for long conversations. |
| SimpleMem | `aiming-lab/SimpleMem` v0.1.0, MIT | Official source is pinned and the adapter routes model calls locally. Do not substitute the separate PyPI artifact for the official source. |
| Mem0 OSS | `mem0ai/mem0`, Apache-2.0 | Public OSS implementation; managed-service scores are not OSS reproduction evidence. |
| Microsoft Memora-System | `microsoft/Memora` at `dec3f8f2444eace7004fc084abe1be9f3d88270e`, MIT | Public official code and LoCoMo runner; published configuration uses GPT-4.1-mini and hosted embeddings, so local-only reproduction requires a provider adapter. Keep separate from `geniesinc/Memora` benchmark. |
| FullContext | `ProsusAI/MemEval`, Apache-2.0 | A useful brute-force control, not a memory architecture. |

“Reproducible” here means source/config/data can be pinned and the method can be executed. It does not mean proprietary provider-backed scores can be recreated exactly with different readers, embeddings, prompts, or judges.

## 2026 Architecture Check

- The [Microsoft Memora paper](https://proceedings.mlr.press/v306/xia26k.html) and [official implementation](https://github.com/microsoft/Memora) are a strong recent architecture reference. Its Memora-P result uses prompted retrieval; Microsoft reports LoCoMo LLM-judge `0.863` versus FullContext `0.825` and up to 98% fewer context tokens. Those are author-reported external results, not local measurements, and its primary score advantage is on a judge metric absent from this Main Track.
- [APEX-MEM](https://arxiv.org/abs/2604.14362) is closely related to temporal/property-value memory and reports high LoCoMo/LongMemEval accuracy, but the official paper page does not link an implementation; it is related work, not the chosen reproducible base.
- [LoCoMo-Plus](https://aclanthology.org/2026.acl-long.1150/) evaluates implicit user constraints under cue/trigger mismatch. It is a useful future transfer benchmark, not a substitute for the already downloaded original LoCoMo run.
- The Memora-System and Memora benchmark are different: the first is Microsoft's MIT-licensed memory code; the second is Genies' ACL 2026 forgetting benchmark/FAMA.

## Dataset Materialization

- Official repository: [snap-research/LoCoMo](https://github.com/snap-research/locomo)
- Revision: `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`
- Data: `external/memory/LoCoMo/data/locomo10.json`
- SHA256: `79FA87E90F04081343B8C8DEBECB80A9A6842B76A7AA537DC9FDF651EA698FF4`
- Contents: 10 conversations and 1,986 QA pairs; conversation `conv-26` has 199 QA pairs.
- License: CC BY-NC 4.0. Keep the dataset in the external checkout; review the non-commercial restriction before any commercial redistribution/use.

## Runs

### PropMem Attempt

`runs/memory/locomo/propmem-conv26-category-balanced-20-20261002T051217Z`

This was an exploratory 20-question plan on `conv-26`. It was stopped during ingestion, before any QA answers or metric calculation. The call ledger contains 9 successful local embedding calls and 2 completed local-Qwen ingestion calls; those two calls took `123.207s` and `149.913s`. A third generation was still in progress at about `4.36 tokens/s`. This is an ingestion-latency/capacity observation only: **no PropMem quality score exists for this attempt**. The run status is recorded beside its partial ledger.

### FullContext Smoke

`runs/memory/locomo/fullcontext-conv26-five-category-smoke-20261002T053213Z`

Five questions were selected, one from each LoCoMo category, from `conv-26`. The exact MemEval FullContext answer prompt was used with frozen local Qwen3-8B Q4_K_M at `127.0.0.1:8081`, `enable_thinking=false`, temperature `0.1`, seed `42`, and 50 generated-token cap. There was no embedding, judge, hosted call, or infrastructure failure. Prediction SHA256:

`03f27e239efd19d5d4fe2bcd3467f6cc281d4d88c06f2affbf848568094e6108`

| Category | n | Mean token F1 |
|---|---:|---:|
| Factual | 1 | 0.800 |
| Temporal | 1 | 0.667 |
| Inferential | 1 | 0.286 |
| Multi-hop | 1 | 1.000 |
| Adversarial | 1 | 0.000 |

Mean deterministic token F1 is `0.5505` over only five hand-selected questions. This is a smoke/diagnostic, **not a benchmark score or ranking claim**. In particular, it must not be compared as a result against MemEval's 1,986-question GPT-backed LoCoMo table. It does show concrete local reader limitations: a one-day temporal error and an adversarial answer where abstention was expected.

The first FullContext script attempt is separately marked invalid: Qwen thinking was not disabled, so its 50-token allowance was consumed before a visible answer; it produced no scored artifact. A runner fix and the rerun above disabled thinking and added the category-5 empty-gold contract.

### OpenClaw Diagnostic

The local 20-question OpenClaw run is recorded at `runs/memory/locomo/openclaw-conv26-category-balanced-20-20261002T054418Z`. It uses the pinned MemEval OpenClaw chunk + BM25/vector implementation with the shared local embedding and Qwen answer model.

- 20/20 predictions; deterministic mean token F1 `0.4044`.
- Per-category F1: factual `0.625`, temporal `0.333`, inferential `0.139`, multi-hop `0.425`, adversarial `0.500`.
- The same five qids used in the FullContext smoke yield OpenClaw F1 `0.600` and FullContext F1 `0.5505`; n=5 only, so this is a paired diagnostic, not evidence of an advantage.
- 49 successful local calls: 29 CUDA embedding calls and 20 local-Qwen reader calls; hosted calls and infrastructure failures are zero. Mean retrieval latency is `297ms`, mean reader latency `20.57s`, and mean reader prompt is `9,376` tokens.
- Prediction SHA256: `07d89d3fb8d9d5362e47b92d02439b7783b265d55f9330c8ea94492fb1014199`.

Session-provenance diagnostics are **not available** for this run. The temporary LoCoMo adapter expected a `session_id` field that the raw `locomo10.json` turns do not contain, so all 400 retrieved chunk/session groups were empty despite the adapter's provenance flag. This does not alter chunk text, retrieval ranking, reader input, or token F1, but it invalidates any answer-session Recall claim for this run. The defect is captured in `run_status.json`; fix the LoCoMo session-ID sidecar before using retrieval-recall metrics.

## Runtime Cleanup

The temporary local reader on `127.0.0.1:8081` was stopped after the pilot. The pre-existing `127.0.0.1:8092` llama.cpp service was left untouched. MemEval's temporary compatibility patch from the PropMem attempt was reversed; checkout remains at the pinned commit and clean. No API key was required; hosted calls were zero.

## Next Honest Step

Use PropMem as the named strong OSS comparator, but first make its local ingestion budget practical without changing its algorithm. The proposed Health-Copilot extension is a deterministic temporal revision/validity gate plus the medical memory/evidence boundary, not a renamed proposition-memory algorithm. Then evaluate the full frozen LoCoMo set and the forgetting-aware track; until those runs complete, the present diagnostics are not résumé evidence and do not show that any memory architecture beats another.
