# Memory Track: Baseline Sources and Upstream Audit

Audit date: 2026-09-26. All listed local checkouts were clean. Health-Copilot itself was already dirty before these documents were added; existing work was preserved.

## Repository identity and role

| Local directory | Remote / upstream | Checked-out SHA | Observed remote HEAD | Frozen role |
|---|---|---|---|---|
| `D:\MyLab\Jianli\external\memory\MemEval` | `https://github.com/ProsusAI/MemEval.git` | `807ae6d7d8a5b76f6fe964d5a581d96c036e2ac4` | same | Main baseline/evaluation harness |
| `D:\MyLab\Jianli\external\memory\mem0` | `https://github.com/mem0ai/mem0.git` | `8d6c001966573786d36908bfe4dfd52935749200` | `94c3fe9f238f3dbf29c9ce98643bd71eb13077cd` | Mem0 OSS baseline; use the checked-out SHA until an intentional pin change is reviewed |
| `D:\MyLab\Jianli\external\memory\Memora` | `https://github.com/microsoft/Memora.git` | `dec3f8f2444eace7004fc084abe1be9f3d88270e` | same | Microsoft memory-system architecture reference; optional, not the benchmark |
| `D:\MyLab\Jianli\external\memory\LongMemEval-V2` | `https://github.com/xiaowu0162/LongMemEval-V2.git` | `2cc8c540bdb87fe6761629b585e727e1c4704520` | same | Stretch only; excluded before public-memory closeout |
| Not checked out | `https://github.com/geniesinc/Memora.git` | not present | `a6493188efc836d6511ed5e4163fe3ba87da30ff` | ACL 2026 Memora benchmark/FAMA; separate from Microsoft Memora |

The exact required command `git -C .\Memora remote -v` fails from the Health-Copilot root because `Memora` is not a child of that repository. The actual checkout is the sibling path in the table. It is Microsoft/Memora, confirmed by its remote and README. The Genies benchmark is not present locally. Keep these projects in separately named directories when it is time to organize the external checkouts; do not mix their code or datasets.

## MemEval capability snapshot

The pinned registry contains nine systems: Full Context, OpenClaw, SimpleMem, PropMem, Mem0, Graphiti, Hindsight, Memory-R1 and MemU. The LongMemEval README table currently reports four systems: PropMem, SimpleMem, OpenClaw and Full Context. Mem0 is implemented but not in that published LongMemEval result table. The five-system MEM-1 matrix adds a same-harness Mem0 OSS reproduction. Memory-R1's local trained model is excluded from the first baseline matrix because it does not share the same reader model.

LongMemEval reader/model coordinates in the pinned README are `gpt-4.1`, `text-embedding-3-small`, and the native binary judge `gpt-4o`. The code's generic retrieval judge defaults to `gpt-5.2`; do not accidentally use it in place of the LongMemEval native judge. Native accuracy and token F1 are distinct metrics. The pinned implementation computes token F1 from unique lowercase word tokens and applies a refusal rule for empty gold answers; the native judge is category-aware and uses a 10-token response cap.

The harness tracks prompt/completion tokens for LLM calls, but its README's displayed system token totals exclude embedding and judge calls. The implementations have per-system prompts, temperatures and retrieval depths. For example, OpenClaw uses top-20 chunks; PropMem defaults to 30 propositions plus 3 chunks. Preserve the pinned upstream configs, record prompt/config hashes and actual token counts, and disclose differences. Do not describe these as one identical answer prompt or identical top-k.

The 102-question DEV set is not committed under `MemEval/data` in this checkout. The pinned `scripts/stratified_sample.py --split s --total 102 --seed 42` samples 17 records from each of six sorted categories using the input record order. Active v2 IDs and fingerprints are in `split_manifest.json`. The original Oracle-derived proposal is archived as `split_manifest_v1_oracle_invalid.json`, explicitly invalid and never used for tuning or evaluation; `split_parity_audit.json` preserves the 89-ID overlap and 13-per-side membership delta.

## Reader and long-context controls

Main Track uses the locally pinned `Qwen/Qwen3-8B-GGUF` Q4_K_M artifact at revision `6a569868d07d3bd59e8b97fb001bf8c0b254bb20`, SHA256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`. The runtime is local llama.cpp build `b10068-571d0d540` (version 10068). This exact reader is shared across compatible systems. GPT-4.1 is restricted to the six-case upstream-parity sanity track; GPT-4o is the judge, not the reader. OpenAI `text-embedding-3-small` is the separate embedding model. The full role and generation settings are in `model_protocol.json`.

The official [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B) specifies 32,768 native context and validates YaRN extension to 131,072; it documents `rope_scaling` factor 4 and original context 32,768, including local llama.cpp flags. The installed GGUF reports a 40,960 training-context field, and the installed llama.cpp server initially capped a 131,072 request at 40,960. The audited configuration uses `qwen3.context_length=int:131072` metadata override, YaRN factor 4, and Q8_0 CPU KV cache. Server slot was observed at 131,072. Using the exact MemEval FullContext prompt and Qwen GGUF tokenizer/template, all 102 DEV prompts fit with a 64-token output reserve: max input 110,289, mean 106,597.4. A one-token local completion on the longest prompt evaluated all 110,289 input tokens with `truncated=false` in 280.016 seconds. Full measurements and the required TEST-side pre-score gate are in `long_context_preflight.json`.

The public LongMemEval README coordinates (PropMem F1 0.550, SimpleMem 0.480, OpenClaw 0.244, Full Context 0.222) are not Health-Copilot results and are not resume evidence. Mem0's managed-service scores are not an OSS comparator; this track runs the OSS package in the common harness.

## Memora identity

Microsoft/Memora is a memory system with harmonic semantic/episodic representations and experimental GRPO retrieval. Genies/Memora is the ACL 2026 benchmark repository for “From Recall to Forgetting,” with weekly, monthly and quarterly histories, Remembering/Reasoning/Recommending tasks, memory-presence and forgetting-absence labels, and FAMA. The latter is the MEM-4 benchmark; the former is only an optional architecture reference. No GRPO training is in scope.

## Dependency audit

Health-Copilot and MemEval both declare Python `>=3.11`; both accept `openai>=1.0`, so no direct Python-floor or OpenAI-SDK conflict was found by metadata inspection. MemEval's optional benchmark extras pull a broad mix including SimpleMem, Mem0, Graphiti/Kuzu and Hindsight; MemU parity requires Python `>=3.13`. The Microsoft Memora checkout has a separate broad requirements list including ChromaDB, Torch, Transformers, Azure identity and platform-conditional `uvloop`. No integration environment was installed or resolved during MEM-0.

Use an isolated MemEval environment for MEM-1 and a separate adapter boundary. Do not merge optional benchmark or Microsoft Memora dependencies into Health-Copilot's locked runtime. This is an isolation recommendation, not a reported dependency-resolution failure.

## API rates used only for the estimate

Prices were checked against official OpenAI model pages on 2026-09-26:

- `gpt-4.1`: $2.00/M input tokens, $8.00/M output tokens: [official model page](https://developers.openai.com/api/docs/models/gpt-4.1)
- `gpt-4o`: $2.50/M input tokens, $10.00/M output tokens: [official model page](https://developers.openai.com/api/docs/models/gpt-4o)
- `text-embedding-3-small`: $0.02/M input tokens: [official model page](https://developers.openai.com/api/docs/models/text-embedding-3-small)

These are planning inputs, not fixed cost claims. Recheck and record effective prices at run time; actual receipts override the rough estimate.
