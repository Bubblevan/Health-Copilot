# RSEL-v1: Relation-Structured Evidence Ledger

## Stage and claim boundary

RSEL-v1 is a new, narrow reader-side method family selected after two completed
negative BUILD readers (CFEC-v1.4 and CAV-v1). Its target is the explicit
identifier-to-value evidence format used by the owned longitudinal benchmark;
it is not a general clinical-prose extraction claim. The evidence is synthetic
and does not represent real patient observations. Any positive result must be
described as a structured-evidence result on this benchmark.

The first RSEL development signal was obtained after BUILD truth had already
been opened for earlier method selection. It is therefore explicitly
in-sample development, not independent evidence. Only the frozen subject-level
FROZEN_DEV partition can test whether this method generalizes. Reserved TEST,
OOD, and FUTURE_TRAIN remain closed.

## Method

For each query and frozen top-10 evidence set, Harness code:

1. extracts exact `SYNKEY-XXXXXXXX` tokens from the query;
2. scans only the visible evidence for exact `SYNKEY-XXXXXXXX to
   SYNVAL-XXXXXXXXXX` relations;
3. retains relations whose key occurs in the query, preserving the issued
   evidence alias and document ID;
4. deduplicates values and emits them with only the aliases of supporting
   visible documents;
5. if no exact relation matches, preserves the paired Vanilla answer,
   citations, and provenance unchanged.

No answer labels, qrels, gold document IDs, or truth-derived IDs are read by
this transform. It cannot invent a value or source ID: every emitted value
must be present in a visible relation, and every cited alias was issued for
the same frozen top-10. It does not search again, reorder the top-10, or expand
the evidence budget.

RSEL is a deterministic extractive postprocessor with a Vanilla fallback. It
reuses the same Vanilla Qwen3-8B draft and call record as its paired arm; on a
matched structured relation, the final value string is derived by Harness
code rather than by Qwen. Thus the comparison holds query, retrieval, top-10,
and generator invocation constant, but it is not a claim that the LLM itself
became more capable. The mechanism being tested is structured evidence
normalization and Harness-owned composition.

## Development-signal provenance

An exploratory diagnostic on the already-scored 156-query BUILD RAG slice
found explicit query-key relations in top-10 for 122 queries. Across those
rows, the extracted candidate values contained all BUILD gold values and no
additional value; overall extracted gold-value recall was 194/194. A
deterministic offline simulation with unchanged Vanilla fallback yielded
155/156 grounded successes versus 119/156 for Vanilla. These are
post-selection BUILD diagnostics only. They must not be presented as a
confirmatory result or resume headline; a frozen-dev result is required.

### RSEL BUILD execution (development-only)

The frozen gold-blind BUILD runtime was materialized from the paired Vanilla
outputs and call journal. It covers all 818 episodes (156 in the RAG slice,
49 subjects), reuses 3,272 completed model-call records, and adds zero model
calls. The RSEL transform produced 622 structured-relation actions and 1,014
no-match Vanilla fallbacks across the two RAG reader arms. BUILD evaluator
truth was opened only after the output and call-journal hashes were verified.

| BUILD RAG metric | Vanilla STRONG | RSEL STRONG |
|---|---:|---:|
| Grounded task success | 119/156 (76.28%) | 155/156 (99.36%) |
| Grounding pass | 129/156 (82.69%) | 155/156 (99.36%) |
| Full required evidence in top-10 | 155/156 (99.36%) | 155/156 (99.36%) |

Paired subject-cluster bootstrap (10,000 resamples) gives grounded-success
delta **+23.08 percentage points**, 95% CI **[+15.19, +31.13] pp**. Among the
155 cases with full required evidence in both arms, task-success delta is
**+23.23 pp**, 95% CI **[+14.97, +31.61] pp**. The pre-registered mechanical
BUILD gates pass. This result is still **in-sample**: earlier CFEC/CAV
diagnostics had already exposed BUILD truth, and the RSEL family was selected
using that BUILD signal. The BUILD result only authorizes the one frozen DEV
test; it is not confirmatory evidence, not a held-out gain, and not a resume
headline.

### FROZEN_DEV execution and internal confirmation

The locked runner completed all 1,628 subject-split FROZEN_DEV episodes and
all 8,140 arm executions before scoring. It made 6,512 completed model calls
(one LameR bridge plus three paired Vanilla reader calls per episode); RSEL
adds no calls. Across the 3,256 RSEL arm executions, the transform emitted
1,336 structured relation ledgers and used 1,920 no-match Vanilla fallbacks.
The call journal contains 6,512 matched started/completed pairs,
with zero failed or truncated calls. The reader-output and call-journal hashes
matched the frozen runtime manifest before the scorer opened truth. The scorer
decoded only the 1,628 FROZEN_DEV truth rows; it skipped the other 2,468 rows
without JSON decoding. FUTURE_TRAIN outcomes and reserved TEST/OOD remain
unopened and unmaterialized.

On the RAG-required slice (289 episodes across 97 subjects), RSEL-STRONG is
compared with VANILLA-STRONG. Both arms use the same query, LameR bridge,
retriever, and byte-identical top-10 evidence; RSEL transforms the paired
Vanilla answer using exact visible key/value relations and harness-owned
provenance. All 289 rows have the complete required external evidence in the
shared top-10, so this comparison isolates answer/evidence utilization rather
than retrieval recall.

| FROZEN_DEV RAG metric (n=289) | Vanilla STRONG | RSEL STRONG | Delta |
|---|---:|---:|---:|
| Grounded task success | 203/289 (70.24%) | 289/289 (100.00%) | **+29.76 pp** |
| Grounding pass | 237/289 (82.01%) | 289/289 (100.00%) | **+17.99 pp** |
| Required external evidence used coverage | 91.29% | 100.00% | +8.71 pp |

Paired subject-cluster bootstrap (10,000 resamples; 97 subjects) gives the
primary grounded-task-success delta a 95% CI of **[+22.88, +36.84] pp**. The
grounding-pass delta is **+17.99 pp**, 95% CI **[+12.99, +23.13] pp**. The
full-evidence utilization comparison has the same 289 eligible rows and the
same **+29.76 pp** task-success delta. All pre-registered FROZEN_DEV gates
pass: primary point delta >= +10 pp, CI lower bound > 0, no grounding
degradation, and utilization delta >= +10 pp.

This is the first independent subject-level internal confirmation of the
BUILD-selected RSEL-v1 method. It supports a narrow claim: on the owned
synthetic longitudinal structured-evidence RAG slice, exact relation
normalization plus deterministic, provenance-preserving composition improved
grounded success over the same strong Vanilla reader, with no extra model
calls. It is **not** a reserved TEST result, a public-benchmark result, or
evidence of generalization to clinical prose. The 100% candidate score also
makes the format-specific nature of this benchmark especially important to
disclose. Do not claim a retrieval improvement: retrieval and top-10 evidence
were held identical.

## Cost and parity

RSEL adds zero generator calls relative to the paired Vanilla reader. The
materialized BUILD output reuses only the Vanilla and LameR call records from
the completed gold-blind baseline run; the unrelated CAV verifier calls are
excluded from the RSEL journal. The FROZEN_DEV execution uses one Vanilla
draft per existing reader arm, then deterministic RSEL transformation. The
candidate and baseline retain the same top-10 evidence IDs and byte-level
evidence identity. Parser/fallback counts and relation-ledger hashes are
recorded.

## Frozen gates and stop rule

RSEL uses the existing BUILD gate unchanged:

- `RSEL_STRONG - VANILLA_STRONG` grounded-task-success point delta at least
  +10 percentage points;
- paired subject-clustered 95% CI lower bound above zero;
- grounding degradation no worse than -1 percentage point;
- utilization-given-full-evidence point delta at least +10 percentage points;
- generation health at least 99% nonempty calls.

The registered procedure was followed: after BUILD passed, the method lock
was committed and all 1,628 FROZEN_DEV episodes were executed before opening
FROZEN_DEV truth. The preceding section records the resulting independent
internal gate pass. No regex/prompt tuning was performed against FROZEN_DEV;
reserved TEST/OOD remains closed and must not be used to rescue or extend this
result without a separately approved protocol.

## Limits

RSEL intentionally recognizes one explicit relation grammar and exact query
keys. It falls back on prose or unrecognized formats. Its performance does
not establish superiority on open-ended medical text, clinical safety, or
general RAG. A future generalization claim requires a separate benchmark with
ordinary prose, paraphrased entities, ambiguous/conflicting relations, and
independently curated temporal/source relevance.
