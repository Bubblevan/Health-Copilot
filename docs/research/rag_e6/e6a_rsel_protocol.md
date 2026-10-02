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

If all gates pass, commit the method lock and execute all 1,628 FROZEN_DEV
episodes before opening FROZEN_DEV truth. A FROZEN_DEV score is the first
independent RSEL result; a failed gate closes RSEL-v1 with no regex/prompt
tuning against FROZEN_DEV. Do not open reserved TEST/OOD as a way to rescue a
failed result.

## Limits

RSEL intentionally recognizes one explicit relation grammar and exact query
keys. It falls back on prose or unrecognized formats. Its performance does
not establish superiority on open-ended medical text, clinical safety, or
general RAG. A future generalization claim requires a separate benchmark with
ordinary prose, paraphrased entities, ambiguous/conflicting relations, and
independently curated temporal/source relevance.
