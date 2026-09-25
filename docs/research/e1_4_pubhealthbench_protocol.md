# E1.4 — PubHealthBench source and holdout protocol

## Status and decision

- Validation-161 is the only permitted development partition.
- Reviewed-760 remains the intended confirmatory candidate; Full-7,929 is post-confirmatory secondary and overlaps Reviewed.
- No parquet file was downloaded locally and no Reviewed inference was run.
- The Reviewed untouched claim is **not certified**: a Hugging Face Dataset Viewer search preview returned example-row text without an unambiguous split label. Treat potential non-Validation preview exposure conservatively; do not run the confirmatory evaluation under this protocol.
- `runs/e1_4/final_method_lock.json` is intentionally absent. The guard refuses Reviewed without a valid committed lock and always blocks the Full/Test partition.

## Pinned upstream metadata

`runs/e1_4/pubhealthbench_source_manifest.json` pins GitHub commit `1d0139a83ae27b41096856fbf45ba5943db90702` and Hugging Face revision `e061be207a5aaa834ed2b5d493090fafac392fd3`. The upstream LFS metadata reports 930,422 bytes for Validation, 3,881,076 for Reviewed, and 42,985,147 for Full/Test. SHA-256 values are recorded there. Hugging Face was unreachable from this host during the Validation download attempt, so hashes are source-side LFS metadata and have not been verified against downloaded bytes.

The dataset is published under CC BY 4.0; the `source_chunk_text` and `retrieved_context_for_judge` columns contain Crown-copyright text under OGL v3. Keep all raw rows and derived source text outside Git. See the [official dataset card](https://huggingface.co/datasets/Joshua-Harris/PubHealthBench) and [upstream evaluation repository](https://github.com/ukhsa-collaboration/UKHSA-pubhealthbench).

## Corpus availability and leakage boundary

The 2026 RAG paper describes Markdown-header chunking and a 5,358-chunk corpus, but the pinned public UKHSA repository lists evaluation scripts rather than a released chunk corpus, qrels, or a RAG runner. No standalone official retrieval corpus or query-to-chunk artifact was found in the audited releases. This is a scoped availability finding, not proof that no private or separately hosted artifact exists. The paper's corpus method is described in [Healthier LLMs (arXiv:2607.06641)](https://arxiv.org/html/2607.06641).

If development continues, build only a Validation-derived prototype using `tools/build_pubhealthbench_validation_corpus.py`. It checks the exact Validation filename, byte count, and SHA before reading; the Parquet reader requests only `source_chunk_text`; output rows contain only opaque `doc_id` and chunk `text`. It writes no question IDs or mappings. This script was tested with synthetic fixtures only because the pinned data download timed out. It has not materialized a corpus.

Before any later split-specific materialization, preserve this rule:

```text
question / options / answer / answer_index / question_id → retriever input: never
retrieval corpus: document text and opaque document IDs only
Reviewed/Test rows: do not materialize before final method lock
```

## Frozen development and final evaluation contract

- DEV: Validation-161. Allowed for adapter, formatting, prompt, retrieval and threshold development only if its exact file is obtained and verified.
- Final candidate: Reviewed-760; fixed denominator 760; provider failures and invalid answers count incorrect; completion and invalid rates also reported.
- Full-7,929: no run in this sprint; if used after Reviewed, label secondary and exposed.
- No random re-split and no post-hoc category headline.
- Generator for this series: Qwen3-8B-Q4_K_M, SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`. Do not swap in Qwen3.5, Llama, DeepSeek, or a cloud model.
- Planned retrieval arms for later Validation development: R0 BM25; R1 BGE-base-en-v1.5; R2 canonical MedCPT query/article encoders + cross-encoder; R3 BM25+MedCPT RRF followed by MedCPT cross-encoder.
- Confirmatory comparison: final method versus the strongest fixed baseline selected on DEV; paired case bootstrap, 10,000 resamples, stratified by topic if topic stratification is retained. Report accuracy difference and 95% interval.
- Positive quality headline only if final accuracy exceeds the DEV-selected baseline and the paired 95% CI lower bound is above zero. A router may instead qualify as a quality-cost tradeoff if it is within 0.5 percentage points and meets one frozen cost reduction: retrieval calls ≥25%, answer input tokens ≥20%, or component p95 ≥15%.

## Test firewall

`eval/pubhealthbench_test_guard.py` defaults to `--partition validation`. `--partition reviewed` requires `runs/e1_4/final_method_lock.json` to be valid, Git-tracked, committed, and unchanged, and prints `THIS WILL CONSUME THE CONFIRMATORY HOLDOUT`. The lock must bind the repo commit, Qwen SHA, prompt SHA, retriever/router identities, thresholds, cost rules, metric protocol, DEV results hash, and `test_opened=false`. Full/Test is always refused by this guard.

Because the Viewer preview split could not be established, the sprint gate is **PUBHEALTHBENCH_TEST_UNTOUCHED = NO (not certifiable)**. Stop before confirmatory inference. A future clean confirmatory claim requires an owner-approved replacement holdout or documented resolution of the preview's split provenance; do not silently proceed with Reviewed-760 as if untouched.
