# RAG-E6A protocol

Status: implementation in progress; no BUILD or FROZEN_DEV model execution has started.

## Question and scope

E6A asks whether a claim-first reader can improve evidence use when the retrieval
result is already fixed. It is not a retriever, reranker, router, memory, agent-team,
or post-training experiment. The only primary contrast is `CFEC_STRONG` against
`VANILLA_STRONG` on the evaluator-derived `RAG` slice. STANDARD is the secondary
contrast; `VANILLA_OFF` is the parametric-only control. NONE and INSUFFICIENT are
diagnostics, while MEMORY and MEMORY+RAG are reported separately and excluded from
the primary comparison.

## Source data and subject split

The source is the frozen U2-F TRAIN runtime universe:

- 4,096 runtime episodes, 320 subjects, 12–14 episodes per subject;
- source manifest SHA-256 `34e6d1a8123ee63eea220c1a1b9b0b9b5f2e3349aeeec05a4504a508d4ca814e`;
- dataset root SHA-256 `e28ea9ef9ecae47d3f27f28c68042066e9af297fe808cafebf1d3c8c80fb2134`.

Before any TRAIN evaluator truth is opened, subjects are sorted by
`SHA256(UTF8("rag-e6a-subject-split-v1" + NUL + subject_id))` and allocated in
fixed blocks: BUILD 64, FROZEN_DEV 128, FUTURE_TRAIN 128. The assignment unit is
the subject. U2-F keeps counterfactual siblings within a subject, so disjoint
subject assignment also prevents sibling leakage across E6A partitions. The
split manifest records source/runtime hashes and counts, not evaluator labels.

BUILD truth may be opened only after every BUILD episode has executed all five
arms and the execution artifacts are frozen. FROZEN_DEV truth stays unopened
until the final method/protocol lock is committed, all five arms have run on all
FROZEN_DEV episodes, and the resulting output hashes are frozen. FUTURE_TRAIN
outcomes and reserved TEST/OOD remain unopened and unmaterialized.

All episodes for BUILD and FROZEN_DEV subjects are run before any evaluator-based
slice is selected. No runtime query is filtered by a capability class, scenario
family, fact ID, answerability, or other teacher label.

## Frozen retrieval and generator

The U3-R retrieval code/configuration is reused without tuning:

- STANDARD: Lucene BM25 (`k1=0.9`, `b=0.4`) and pinned BGE-large, equal RRF
  (`k=60`, weights `[1,1]`), fused top 10.
- STRONG: pinned LameR-MV (BM25 query, BM25 query+bridge, BGE query, BGE bridge),
  RRF (`k=20`, weights `[1,2,1,2]`), fused top 10.
- BGE: `BAAI/bge-large-en-v1.5`, revision
  `d4aa6901d3a41ba39fb536a557fa166f842b0e09`, weights SHA-256
  `45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7`.
- LameR and every reader call use Qwen3-8B-Q4_K_M, SHA-256
  `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.

All generator requests use temperature 0, top-p 1, reasoning disabled, one attempt,
8,192 completion tokens and the common 65,536-token context ceiling. Runtime
backend/device details are recorded but are not method variables. No automatic
retry is allowed. Retrieval output, query text, evidence aliases, requirement IDs,
and evidence provenance are harness-owned.

## Arms and execution order

Every episode receives, in a fixed order:

1. `VANILLA_OFF`: question-only reader.
2. `VANILLA_STANDARD`: question plus the frozen STANDARD top-10 evidence.
3. `VANILLA_STRONG`: question plus the frozen STRONG top-10 evidence.
4. `CFEC_STANDARD`: requirements, per-requirement claims from the same STANDARD
   evidence, then final composition from validated claims only.
5. `CFEC_STRONG`: the same graph using the frozen STRONG evidence.

No CFEC_OFF arm is defined. The question-only decomposition is shared between
the two CFEC retrieval conditions within one episode because it is independent
of evidence/action; the harness then separately runs claim extraction and final
composition for each condition. Both CFEC arms use the same requirement strings
and harness-assigned `req_1`…`req_n` identities.

Vanilla output parsing uses the last literal `FINAL:` marker and parses citation
aliases only after that marker. Missing/empty FINAL is an output-contract failure;
there is no retry. Issued aliases are `[E1]`…`[E10]` in frozen rank order. The
model never sees underlying document IDs.

## CFEC authority boundary

The decomposer receives only the original question. It emits at most four
non-empty requirement lines; the harness keeps the first four and assigns stable
IDs. Each claimant receives one requirement and the exact same ranked top-10
passages shown to the corresponding Vanilla arm. Claimants return short claims
with issued aliases or `UNSUPPORTED`. Unknown aliases are recorded and ignored;
claims without any valid issued alias are rejected. The final composer receives
the original question plus validated claim text and aliases, never raw passages
or document IDs. Final used evidence is the harness union of validated claim
provenance; model text cannot alter that set.

All model prompts exclude evaluator truth, answer values, required fact/evidence
IDs, derived capability classes, scenario family, oracle action, sibling outcomes,
and evaluator labels. LLM output cannot select/modify an action, query, retrieval
result, evidence scope, requirement identity, provenance, or scoring slice.

## Evaluation order and gates

For BUILD: execute all episodes/all arms, freeze and hash execution artifacts,
then open BUILD truth and score. BUILD is method development only and never the
headline result. Changes to prompts/parser/composition are allowed only here;
retrieval tuning, gold-aware features, case-specific rules and hard-coded answers
are prohibited.

Before FROZEN_DEV truth is opened: freeze the selected CFEC-v1 prompts, parser,
model, retrieval identities/configuration, top-k, budgets, execution graph, metric
definitions, code commit and BUILD-derived method choice. Then execute every
FROZEN_DEV episode under all five arms, freeze all artifacts/hashes, and only then
open FROZEN_DEV truth for scoring.

Primary: `CFEC_STRONG - VANILLA_STRONG` grounded task success on the RAG slice.
The internal positive gate is at least +10 percentage points, subject-clustered
95% CI lower bound greater than zero, and no more than 1 pp degradation in
grounding pass. The pre-registered utilization contrast is task success among
episodes for which all required external evidence is in the supplied top-10; its
positive threshold is also +10 pp. These are project gates, not literature
standards. A failed gate authorizes no reserved-test materialization and no
prompt iteration on FROZEN_DEV.

Subject-clustered bootstrap: 10,000 resamples, seed `20260930`. Report task
success, grounded task success, grounding/provenance pass, external-evidence
coverage, answer-value correctness, valid citation/provenance, and
utilization-given-full-evidence. Report predeclared NONE, INSUFFICIENT,
multi-source, dependency-width/depth, scenario-family and versioned-evidence
diagnostics where their post-freeze labels are available.

## Expected artifacts

See `runs/rag_e6/` for the immutable subject split, BUILD run/freeze/report,
protocol lock, FROZEN_DEV run/freeze/report and failure attribution. Reserved
TEST/OOD files must not be copied, decoded or materialized in E6A.
