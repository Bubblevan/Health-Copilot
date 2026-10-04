# PT-E0 Frozen Evaluation and Decontamination Foundation

**Status: IN PROGRESS.** The evaluation, source, and judge protocols are frozen. Full Base predictions and scores remain pending because the last L40 check showed 39,827 MiB allocated (about 6 GiB free), 0% utilization, and no visible GPU process; the evaluation runner refuses to load Qwen3 below its 20 GiB free-memory guard. PT-E0 has not trained or updated any model weights.

## 1. Purpose

PT-E0 freezes external evaluation, checks the planned training sources for contamination, assigns disjoint SFT/RL prompt families, and establishes the Qwen3-8B Base baseline. It does not perform SFT, LoRA, QLoRA, RL, reward tuning, or LiveMedBench scoring.

## 2. Frozen model

| Field | Value |
|---|---|
| Model | `Qwen/Qwen3-8B` |
| Revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| Source | Original Hugging Face BF16 safetensors; no adapter or medical fine-tune |
| Local source | `/root/gpufree-share/data/Qwen3-8B` |
| Chat template SHA256 | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |

Config, tokenizer, index, and five weight shard hashes are recorded in [`qwen3-8b-base.json`](../../../training/posttrain/manifests/model/qwen3-8b-base.json).

## 3. Frozen evaluation suite

| Benchmark | Frozen source/revision | N | Source file SHA256 | Candidate/scorer split |
|---|---|---:|---|---|
| DiagnosisArena test | `SII-SPIRAL-MED/DiagnosisArena` / `64bb873fe651e1c71cd8b0958104dd911f042872` | 915 | `de980033fa681d9cb5bf4138f04dea204c1326d18017a752a137458195c36520` | Case, exam, tests, options / right option |
| CMB-Exam test | `FreedomIntelligence/CMB` / `935fbc09edf1303d89872b21265ff597f426ac0d` | 11,200 | questions `99e6e858dd1955fe2b27fd1da75a7be8fbf6d27059fdb8cf61bc28383a68129c`; answer key `42130b20f7660a83dd92f3c005eb6a8ef1221c98d9b27d3351204143944b532f` | Question/options/type / answer and categories |
| HealthBench Professional test | `openai/healthbench-professional` / `349962fd46dd02343a0d8a606491baf59154ea1a` | 525 | `d44b08e6e952e04c945e2c406f02533d9e7a989a84e35820ee7efdff20c9e4e2` | Conversation / rubric and analysis metadata |
| LiveMedBench reserved | `JuelieYann/LiveMedBench` / `db7eb218959aeac83fc9d9353efd7d909b6616f0` | snapshot 5,286; reserved 1,024 | `8b0fffadd77f45ff6e5a54c8f88b886ff6830d7ad49b8f4eb7759e4e35c48651` | Reserved IDs only; no Base answers or scores |

The complete file list, prepared-view hashes, answer-key revision, and LiveMedBench reserved ID hash are in [`eval_dataset_manifest.json`](../../../training/posttrain/manifests/eval/eval_dataset_manifest.json). LiveMedBench reserved ID SHA256 is `686577f315ac523f56244b1f93e0d22c2aed52d8fdf4c1b25276bb104be56ea0`; `SCORE=UNOPENED`.

DiagnosisArena has no specialty field in its frozen test file, so specialty scores will be reported as unavailable. CMB contains 9,999 single-choice rows, 1,190 multiple-answer rows, and 11 C-type rows. HealthBench Professional contains 1,135 rubric criteria; subgroup metadata includes consult 236, writing 142, research 147, good-faith 334, red-teaming 191, typical 256, and difficult 269.

## 4. Candidate prompt and inference contract

The candidate model sees no custom system prompt and receives no RAG, memory, tools, retrieved evidence, answer repair, or Health-Copilot runtime safety post-processing. The empty system prompt SHA256 is `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

DiagnosisArena candidates receive only the case, exam, diagnostic tests, and options. CMB candidates receive only the question and options. HealthBench generation sees only the conversation. Gold answers, rubrics, physician responses, and grader metadata are loaded separately by scorer code. LiveMedBench candidates are prepared for decontamination and reservation only.

All Qwen checkpoints use the same Qwen chat template, thinking enabled, `temperature=0`, `do_sample=false`, `top_p=1`, and `max_new_tokens=2048`. Both raw generation and extracted final answer are retained. Scoring starts only after the complete prediction file and its SHA256 sidecar are written.

The frozen prompt builder, parser, candidate generator, and MCQ scorer hashes are in [`eval_protocol.json`](../../../training/posttrain/manifests/eval/eval_protocol.json). The same runner accepts `--checkpoint-name`, `--model-path`, and `--model-revision` for later merged SFT/RL HF checkpoints, records all local checkpoint file hashes, and rejects a changed Qwen chat template or evaluation protocol. The scorer reports the checkpoint identity from its frozen prediction manifest.

## 5. HealthBench local judge contract

The metric is named **HB-Pro Local-Rubric Score** and is not an official HealthBench leaderboard score. The local judge is Mistral Small 3.1 24B Instruct, pinned to a Q4_K_M GGUF from revision `f73dfd9e812922fb503a993e3fa5671424f486d3`, SHA256 `c5743c1bf39db0ae8a5ade5df0374b8e9e492754a199cfdad7ef393c1590f7c0` (14,333,910,496 bytes).

The local OpenAI-compatible server is built from `ggml-org/llama.cpp` commit `836d57176dc699a726c55418e4f96b8ca628e1bf`; `llama-server` SHA256 is `e85c1074f278fdd776ee224fd952b11593de38a49913388e7ef152bc0b29302b`. It uses CUDA 13.0, L40 compute capability 8.9, Q4 weights, 32,768 context, all layers on GPU, one request slot, Flash Attention, F16 K/V cache, temperature 0, top-p 1, 512 max output tokens, and seed 20261004. The server binds only to `127.0.0.1`; the grader rejects non-loopback URLs.

Each grader request contains one candidate conversation, the Qwen final answer, and one criterion with its points. It does not contain `physician_response`, difficulty, checkpoint identity, or other rubric items. Criterion-level JSON, explanation, raw judge output, prompt/output hashes, and latency are retained outside Git. The local judge protocol and its prompt/runtime hashes are in [`healthbench_local_judge.json`](../../../training/posttrain/manifests/eval/healthbench_local_judge.json).

A CPU-only format/generation smoke passed on a synthetic arithmetic prompt (no medical evaluation data): the Q4 GGUF loaded and returned parseable JSON with `criteria_met=true` in 70.708 seconds. The prompt and output hashes plus CPU smoke settings are recorded in [`healthbench_judge_cpu_smoke.json`](../../../training/posttrain/manifests/eval/healthbench_judge_cpu_smoke.json). The official GPU judge runtime has not yet run because the L40 is occupied.

The score follows the published per-case calculation: sum signed points for every criterion marked met, divided by the sum of positive rubric points. Aggregate means are clipped to [0, 1]. Length adjustment is `score - 0.0147 * ((answer_characters - 2000) / 500)`. Raw and length-adjusted means use 10,000 bootstrap resamples with seed 20261004. See the [HealthBench reference scoring implementation](https://github.com/openai/simple-evals/blob/main/healthbench_eval.py). Hosted API calls: 0. OpenAI API key: not used.

## 6. Training-source decontamination

Canonicalization is frozen as NFKC, line-ending and repeated-whitespace normalization, safe punctuation normalization, Latin lowercase, and option-label normalization. Exact prompts use SHA256 of the question-only canonical text. Near-duplicate detection uses character 5-gram MinHash/LSH candidate generation, RapidFuzz `token_set_ratio >= 95`, and a normalized 64-character substring detector. The canonicalizer and decontamination implementation hashes are captured in the source-count manifest.

| Source | Raw rows | Unique question families | Within-source duplicate rows | Duplicate rate | Clean unique families |
|---|---:|---:|---:|---:|---:|
| medical-o1 SFT English | 19,704 | 19,677 | 27 | 0.1370% | 19,664 |
| medical-o1 SFT Chinese | 20,171 | 17,172 | 2,999 | 14.8679% | 17,163 |
| MedReason | 32,682 | 32,624 | 58 | 0.1775% | 31,938 |
| Huatuo26M-Lite | 177,703 | 177,703 | 0 | 0.0000% | 177,703 |
| medical-o1 verifiable RL | 40,644 | 40,310 | 334 | 0.8218% | 40,297 |

Exact cross-source overlaps include o1-English ↔ MedReason: 5,796 row pairs / 5,771 families; o1-English ↔ RL: 13,704 pairs / 13,540 families; MedReason ↔ RL: 4,501 pairs / 4,446 families. Across the four SFT sources, 5,771 / 241,405 union-unique prompt families are duplicated across sources (2.3906%); the rate counts each shared family once. The corrected stage split applies to every exact SFT/RL shared family, not just the medical-o1 files.

External contamination removal counts by source and benchmark are in [`pt_e0_contamination_audit.md`](pt_e0_contamination_audit.md). Exact external prompt matches were 0. Near/64-character removals included: o1-English/DiagnosisArena 13 substring matches; o1-Chinese/CMB 4 near and 6 substring matches; MedReason/DiagnosisArena 17 substring, CMB 1 near, HB-Pro 2 substring, LiveMedBench 4 substring; RL/DiagnosisArena 12 substring and LiveMedBench 1 substring. Huatuo had no matches under the frozen rules. Counts are row-level removals, and HealthBench/LiveMedBench raw prompts are not printed in public reports.

MedReason `dataset_name=MedXpertQA` rows are excluded unconditionally: 666. Recorded MedReason provenance counts are MedQA 8,016; MedMCQA 6,197; PubMedQA variants 10,444; MMLU 827.

## 7. SFT/RL prompt-family separation and internal DEV

Every clean exact family shared between any SFT source and RL is assigned from the first 16 SHA256 hex characters modulo 100: buckets 0–49 SFT, 50–96 RL train, 97–99 RL DEV. This 3% RL DEV allocation was fixed before model scoring/training to target about 1,000 clean DEV prompts.

| Relationship / candidate pool | Count |
|---|---:|
| Raw shared families across SFT and RL | 13,620 |
| Clean shared families assigned once | 13,612 |
| Clean medical-o1 shared families | 13,532 |
| Raw unmatched medical-o1 SFT-only families | 23,309 |
| SFT candidates before internal DEV | 233,909 |
| SFT internal DEV | 2,000 |
| SFT train candidates | 231,909 |
| RL internal DEV | 1,179 |
| RL train candidates | 32,296 |
| SFT/RL candidate family intersection | **0** |

SFT DEV is deterministically stratified by source and language. These counts are clean candidate-pool ceilings, not a final quality-filtered PT-1 mixture. No mixture weights were selected from external test performance.

## 8. Qwen3-8B Base results

**Pending.** No baseline predictions have been started because the L40 currently has only about 6 GiB free. The runner's safety guard requires 20 GiB free. No metrics below are inferred from other papers or model cards.

| Checkpoint | DiagnosisArena | CMB-Exam | HB-Pro Local-Rubric | LiveMedBench reserved |
|---|---:|---:|---:|---:|
| Qwen3-8B Base | PENDING | PENDING | PENDING | UNOPENED |
| Medical SFT | NOT RUN | NOT RUN | NOT RUN | UNOPENED |
| SFT + GSPO/GDPO | NOT RUN | NOT RUN | NOT RUN | UNOPENED |

After Base generation, the report will include DiagnosisArena accuracy, Wilson 95% CI, invalid rate, mean tokens and latency; CMB exact accuracy, macro-28, major-category, single/multiple scores and Wilson intervals; HB-Pro Local raw and adjusted scores, requested subgroups and paired/bootstrap-ready CIs. No weighted overall medical score will be created.

## 9. Known limitations

- Baseline model generation and actual Q4 judge GPU inference have not run yet. The GPU currently belongs to an unseen process; PT-E0 does not kill it. A CPU-only synthetic judge smoke passed, but it is not a benchmark score.
- DiagnosisArena provides no specialty metadata in the pinned test file.
- The local Q4 judge is an independent grader and may disagree with the official GPT-5.4 HealthBench grader; its score is explicitly labeled local.
- Near-duplicate filters are deterministic string methods and do not detect all semantic paraphrases.

## 10. LiveMedBench reserved status

The N=1,024 ID list is frozen with SHA256 `686577f315ac523f56244b1f93e0d22c2aed52d8fdf4c1b25276bb104be56ea0`. Its language × theme allocation preserves all five themes. No Qwen predictions or rubric scores were generated or inspected for tuning. `SCORE=UNOPENED`.

## 11. Reproduction commands

Run from `training/posttrain` with the existing isolated environment and data paths:

```bash
export PT_E0_DATA_ROOT=/root/gpufree-data/Health-Copilot-PT-E0-data
export PYTHONPATH=/root/gpufree-data/Health-Copilot-PT-E0/training/posttrain
PYTHONDONTWRITEBYTECODE=1 /root/gpufree-data/Health-Copilot/training/posttrain/.venv/bin/python scripts/prepare_external_eval.py
PYTHONDONTWRITEBYTECODE=1 /root/gpufree-data/Health-Copilot/training/posttrain/.venv/bin/python scripts/audit_training_sources.py
PYTHONDONTWRITEBYTECODE=1 /root/gpufree-data/Health-Copilot/training/posttrain/.venv/bin/python scripts/freeze_pt_e0_manifests.py
PYTHONDONTWRITEBYTECODE=1 /root/gpufree-data/Health-Copilot/training/posttrain/.venv/bin/python scripts/run_base_eval.py --benchmark diagnosisarena
PYTHONDONTWRITEBYTECODE=1 /root/gpufree-data/Health-Copilot/training/posttrain/.venv/bin/python scripts/run_base_eval.py --benchmark cmb
PYTHONDONTWRITEBYTECODE=1 /root/gpufree-data/Health-Copilot/training/posttrain/.venv/bin/python scripts/run_base_eval.py --benchmark hbpro
```

For later merged HF checkpoints, use the same runner and frozen settings, for example:

```bash
python scripts/run_base_eval.py --benchmark diagnosisarena --checkpoint-name Medical-SFT --model-path /path/to/merged-sft --model-revision local-sft-sha
```

After all 525 HB-Pro predictions are frozen, start the local judge in one terminal with `scripts/start_healthbench_judge_server.sh`, then in another run:

```bash
PYTHONDONTWRITEBYTECODE=1 /root/gpufree-data/Health-Copilot/training/posttrain/.venv/bin/python scripts/run_healthbench_local_judge.py --base-url http://127.0.0.1:8080
PYTHONDONTWRITEBYTECODE=1 /root/gpufree-data/Health-Copilot/training/posttrain/.venv/bin/python scripts/score_base_eval.py --benchmark diagnosisarena
PYTHONDONTWRITEBYTECODE=1 /root/gpufree-data/Health-Copilot/training/posttrain/.venv/bin/python scripts/score_base_eval.py --benchmark cmb
PYTHONDONTWRITEBYTECODE=1 /root/gpufree-data/Health-Copilot/training/posttrain/.venv/bin/python scripts/score_healthbench_local.py
```

The contamination scan is CPU-only. Model generation and judge inference require the L40 to have enough free VRAM.

## 12. PT-1 readiness decision

**NO.** PT-1 remains blocked until full E1/E2/E3 Base predictions are frozen and scored, all final reports and hashes are committed, and the remaining PT-E0 gates pass. The LiveMedBench reserved score must remain unopened.
