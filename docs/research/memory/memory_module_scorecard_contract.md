# Memory Module Scorecard Contract

**Contract ID:** `health-copilot-memory-module-scorecard-v1`  
**Status:** pre-registered; freeze before MEM-3B0P model calls  
**Locked:** 2026-09-30

## Research Claim

The final memory comparison asks whether revision-aware state handling improves revision-sensitive queries without materially degrading general memory performance. The scorecard is fixed before further method work; targets below are interpretation thresholds, not stage-completion gates or promised results.

## Controlled Local Stack

- Answer reader: the same frozen local Qwen3-8B Q4_K_M for every compatible system.
- Memory-internal LLM: the same frozen local Qwen3-8B Q4_K_M when a memory method needs one.
- Embedding: local `Qwen/Qwen3-Embedding-0.6B` for every compatible dense-retrieval system.
- Judge: none. Use deterministic metrics only.
- Hosted APIs: none.
- Official provider-specific numbers remain `OFFICIAL_HISTORICAL`; they are never mixed into controlled-local claims.

## LongMemEval One-Shot Closeout

Track: `ONE_SHOT_GENERAL_MEMORY_CLOSEOUT` on the frozen 102-case LongMemEval-S DEV. Run only after architecture and method freeze. Do not tune from the result.

Required comparators:

- FullContext
- M10-Base
- RawSpan-Dense
- PropMem
- Final Health-Copilot Memory

Secondary/historical comparators:

- OpenClaw
- Mem0 OSS
- SimpleMem official

Primary metrics:

- Overall token F1 and normalized EM.
- Knowledge-update, temporal-reasoning, and multi-session token F1.
- `CURRENT`, `CHANGE`, and `AS_OF` accuracy.
- Answer-session Recall@8, MRR, and gold-token coverage.
- Reader-visible context tokens, ingestion/read latency, memory size, and LLM-call count.

Use 10,000 paired bootstrap replicates, stratified by LongMemEval question type, and report 95% intervals. Strong evidence requires positive paired revision-sensitive improvement with the 95% CI lower bound above zero. A practical target is approximately +0.05 absolute on knowledge-update/revision-sensitive quality over the strongest required controlled non-revision comparator. Overall token F1 should not regress by more than approximately 0.02 absolute against that comparator. These are claim-interpretation thresholds, not infrastructure gates.

## MedMemoryBench Controlled Local Transfer

Track: `CONTROLLED_LOCAL_TRANSFER`. Required comparators:

- `MEMORY_READ_BASELINE::FULL_CONTEXT`
- `MEMORY_READ_BASELINE::BM25`
- `MEMORY_READ_BASELINE::DENSE`
- `MEMORY_READ_BASELINE::MEM0`
- `MEMORY_READ_BASELINE::AMEM`
- `MEMORY_READ_BASELINE::MEMOS`
- Health-Copilot Memory

Secondary where executable fairly: MEMRL, Letta, LightMem, ReMem, and Graph/Hippo methods. Official upstream results remain historical and are not pooled with controlled-local results.

Primary endpoint: `state_update`. Also report entity exact match, temporal localization, inference generation, multiple choice, and multi-hop clinical deduction. Retrieval/provenance diagnostics are source-session Recall@5, source-session MRR, multi-source session coverage, stale-context exposure, and revision-resolution accuracy. Efficiency includes write LLM calls/tokens, retrieval latency, reader-visible tokens, and memory-store size.

Minimum useful transfer evidence is `state_update` above both BM25 and Dense. Strong target: at least approximately +0.05 absolute over the best of Mem0, A-MEM, and MemOS, with paired-bootstrap 95% CI lower bound above zero. Report overall performance alongside the state-update result; a slice gain does not imply general superiority.

## Claim Guardrails

- Do not claim general superiority from a revision-sensitive slice alone.
- Report overall performance alongside revision-sensitive gains.
- Do not compare controlled-local values directly with official provider-specific results.
- Do not tune methods, prompts, thresholds, or splits after the one-shot closeout result.

The machine-readable contract is in the adjacent `memory_module_scorecard_contract.json` and is copied byte-for-byte into the MEM-3B0P run artifacts.
