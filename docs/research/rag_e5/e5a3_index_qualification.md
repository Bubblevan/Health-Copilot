# E5-A3 corpus and index qualification

## Outcome

Owner review and deterministic corpus preparation passed. The frozen index
qualification did **not** pass: dense BGE missed two source-title checks in the
`PUBLIC_HEALTH_ONLY` view (8/10). Therefore the candidate corpus is not active,
STANDARD/STRONG remain unbound, `E5A_READY = NO`, and E5-B did not start.

No model, query, chunking, or retrieval setting was changed to repair the smoke
score. The failure is recorded as a model/input-language compatibility concern
for the frozen English BGE profile; it is not claimed as a proven extraction or
source-ID defect.

## Owner decision and scope

The owner decision record is
[`e5a3_owner_decision.json`](e5a3_owner_decision.json), SHA-256
`f563d8b9a837f4627c750874a030f3beb5e8728c91638568dd74eab241d12ac7`.

All three reviewed WHO sources are approved for candidate-corpus retrieval.
Only two are eligible for future task authoring: physical-activity guidance and
total-fat guidance. The hypertension pharmacological guideline is retrieval
context/distractor only; E5-v1 must not author medication-initiation, target,
drug-class, diagnosis, or individualized-prescription gold tasks from it.

The profile temporal contract excludes untimestamped health-profile prose and
derives condition categories only from structured pre-cutoff timeline/exam
labels. State packet version is `e5-longitudinal-state-v2`.

## Corpus construction

Three separately scoped views were built outside Git under
`D:/MyLab/Jianli/external/rag_e5/e5a3/`:

| View | Sources | Chunks | Corpus SHA-256 |
|---|---:|---:|---|
| `PUBLIC_HEALTH_ONLY` | 30 | 30 | `3f8da93a349d199ce2b78a91c7ef3483d6b3f298b714315f33ab2a19c92ed2a7` |
| `GUIDELINE_ONLY` | 3 | 26 | `19576a23b1c6afb55c5af3d7eb48610445910875816dd73d1db6b51d7dbb5ce4` |
| `PUBLIC_HEALTH_PLUS_GUIDELINE` | 33 | 56 | `ea713d948b2b5bfde0dcb5b0b2f22f6e37cba4bc56ab61971516b5a7135e7ccc` |

The public-health identity matches the A2 frozen card-file-set identity. Each
card maps one-to-one to an unchanged text chunk. WHO extraction is deterministic
and source-section scoped; no generated summaries or paraphrases are used.
Recommendation retention matched every canonical manifest section ID:

| Source | Declared sections | Retained sections | Blocks/chunks |
|---|---:|---:|---:|
| WHO hypertension pharmacological guideline | 8 | 8 | 8 |
| WHO physical activity and sedentary behaviour | 12 | 12 | 14 |
| WHO total-fat guideline | 2 | 2 | 2 |

The 14 activity blocks keep child/adult disability recommendations separate
while preserving the 12 source section IDs. Missing, unexpected, and duplicate
canonical section IDs are all zero. Third-party figures, attributed assets,
rationale/evidence tables, and bibliography material are excluded. The
candidate combined external corpus identity is
`9b19ad467f47641032277707cb1cfdb1d05c2fb39c558039b180efc7394692bd`; it is
recorded as a candidate only, not an active runtime identity.

## Frozen index profiles and smoke results

BM25 uses Gensim `LuceneBM25Model`, the Anserini/Lucene English analyzer with
Porter stemming, `k1=0.9`, `b=0.4`, and top-100 retrieval. BGE uses the pinned
`BAAI/bge-large-en-v1.5` revision
`d4aa6901d3a41ba39fb536a557fa166f842b0e09`, weight SHA-256
`45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7`, normalized
1024-dimensional float32 CLS vectors, max length 512, frozen query instruction,
and exact-flat cosine. The embedding build ran on the RTX 4090 Laptop GPU.

| View | BM25 source-title hits@10 | BGE source-title hits@10 | Result |
|---|---:|---:|---|
| `PUBLIC_HEALTH_ONLY` | 10/10 | **8/10** | **Failed** |
| `GUIDELINE_ONLY` | 10/10 | 10/10 | Pass |
| `PUBLIC_HEALTH_PLUS_GUIDELINE` | 10/10 | 10/10 | Pass |

The two dense misses are:

- `public-health-08`: “CDC：家庭自测血压应与医疗团队支持结合”; expected source
  `cdc-high-blood-pressure-measuring-01-home` ranked 18th.
- `public-health-10`: “CDC：测量前应避免影响读数的准备因素”; expected source
  `cdc-high-blood-pressure-measuring-03-before-reading` ranked 15th.

Both chunks are present, their section titles are included in the indexed
document text, and the expected source IDs are bound to the correct chunks.
This rules out a missing-document and observed source-ID mismatch. The frozen
English BGE model's performance on these Chinese title/body pairs remains the
unresolved qualification concern. The top-10 threshold was not relaxed.

The first dense build encountered a duplicate Intel OpenMP runtime only during
CPU NumPy/MKL scoring after repeated GPU inference. Scoring was moved to the
same CUDA/PyTorch runtime; no unsafe duplicate-runtime override was used. The
resulting smoke failure is semantic ranking behavior, not a failed model load
or incomplete embedding build.

## Runtime binding and stop boundary

The frozen STANDARD and STRONG method config hashes remain unchanged:

- STANDARD: `6ddb91bb0c31f5bc5b69372df6a3bdd3b4f71d70cdbcece8ef22c5bfd9a94330`
- STRONG: `d9e3de9bf986a1f08b1c217153422d6e86ad0a84bc1982d90bade897c9274a8b`

Their `external_corpus_identity` remains `null`; neither method is bound to the
candidate corpus. No real T0/T1/T2 tasks, OFF/STANDARD/STRONG runs, answer-model
calls, counterfactual outcomes, oracle, or policy training were performed.
E5-A3 stops at index qualification. Resolving the frozen BGE smoke failure
requires a separately approved next decision; this stage does not silently
substitute another embedding model.

Reproducible outputs and per-query smoke rankings are external artifacts. The
repo-tracked summary is
[`runs/rag_e5/e5a3_activation_report.json`](../../../runs/rag_e5/e5a3_activation_report.json).
