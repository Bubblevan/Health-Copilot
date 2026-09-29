# E5-A2 external evidence corpus qualification

> Historical snapshot: this document records the E5-A2 state before owner
> review. E5-A3 later recorded approval, built the candidate corpus and indexes,
> and stopped activation because the frozen dense smoke was 8/10 for
> `PUBLIC_HEALTH_ONLY`. See
> [`E5-A3 qualification`](e5a3_index_qualification.md) for current status.

## Result

Three official WHO guideline candidates are hash-verified and ready for owner
review. The raw candidate PDFs and deterministic `pdftotext -layout` outputs
are stored outside Git under
`D:/MyLab/Jianli/external/rag_e5/guidelines/`. The candidate set SHA-256 is
`01136e09675780a3e658d1fb5abd4a231740aead6571334b8644ae8bdc7c1023`.

There are **0 owner-approved guideline sources**, **0 active guideline chunks**,
and **0 formal E5 indexes**. Therefore no chunk corpus was frozen, no smoke
retrieval was run, and no external corpus identity was bound to STANDARD or
STRONG. This is an intentional human-review stop, not a successful retrieval
index qualification.

The pre-existing public-health family remains reference-only: its 30 reviewed
knowledge-card files were hashed in place and not copied or reclassified. The
E5-A2 report records that family hash separately. Neither the 30 cards nor the
candidate WHO PDFs were indexed by this stage.

## Selection rationale

The source selection follows the permitted 202607 domain inventory, not
question/answer inspection. The three candidates cover BP/hypertension,
physical activity, and nutrition/weight signals. For each selected domain the
coarse audit records at least 5 users (in fact 20/20 indicator-name signals for
these broad measurements). These counts are only schema/indicator-name
coverage; they do not establish diagnosis prevalence or future task solvability.

The diabetes pharmacotherapy candidate was excluded because no medication
management task contract exists. The 2026 dementia source was excluded because
the allowed 202607 audit surfaced no direct cognitive domain. This avoids
selecting sources just to reach the three-document minimum.

## Corpus and chunk contract

After owner approval, `eval/rag_e5/corpus.py` admits only sources whose
`review_status` and `owner_review_status` are both `APPROVED`; it verifies raw
bytes against the manifest before chunking. Recommendation blocks remain
separate, paragraph packing is deterministic, target size is 384 tokenizer
tokens, and hard maximum is 480. Chunk identity binds source ID, raw-source
hash, heading path, recommendation ID, part number, and text hash. The
tokenizer is supplied by the caller so token counts must use the frozen BGE
tokenizer when corpus construction is authorized.

Because owner review is pending and the PDFs flag third-party material, this
stage deliberately has not extracted active chunks or indexed any material.
The manifest's recommendation-section counts are inventory only; the
recommendation-retention audit remains `NOT_RUN_PENDING_OWNER_REVIEW`.

## Frozen retrieval identities

The BM25/BGE/RRF and Qwen/LameR action-profile hashes remain unchanged. The
STANDARD and STRONG algorithm identities are preserved, but their
`external_corpus_identity` remains `null`. The index manifest declares
`PUBLIC_HEALTH_ONLY`, `GUIDELINE_ONLY`, and
`PUBLIC_HEALTH_PLUS_GUIDELINE` as pre-scoped views; all E5-specific indexes are
`NOT_BUILT`. A future runner must retrieve inside one frozen view and reject
out-of-view result IDs; it must not retrieve globally and post-filter families.

## Reproduction

```powershell
$env:PYTHONPATH = ".;src"
D:\Anaconda\python.exe tools\research\rag_e5\qualify_external_corpus.py
D:\Anaconda\python.exe -m pytest -q tests\test_rag_e5_guideline_corpus.py
```

The qualifier reads only the source manifest, the 202607 aggregate inventory,
the frozen action profile, the 30 existing knowledge-card files, and the three
hash-pinned WHO PDFs/extracted files. It does not read ESL questions, answers,
the 202608 state batch, MIRAGE labels, or R2MED qrels.
