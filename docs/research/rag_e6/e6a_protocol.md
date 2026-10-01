# RAG-E6A protocol

Status: the initial BUILD remains diagnostic after a baseline prompt/parser alignment defect. `build_v2` was stopped at 232/818 after an output-contract/scoring defect; `build_v3` was stopped at 8/818 because the 256-token cap truncated ordinary claim outputs. Neither attempt was scored or opened evaluator truth. The next complete candidate is strict-output `CFEC-v1.3` in `build_v4`. FROZEN_DEV and reserved TEST/OOD truth remain unopened; reserved TEST/OOD remains unmaterialized.

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
and a uniform 512-token per-call output cap; the llama.cpp server itself retains
its separately recorded 8,192-token ceiling. Runtime backend/device details are
recorded but are not method variables. The pinned GGUF
has a 40,960-token native/effective context despite the requested 65,536 setting;
the client therefore applies a conservative UTF-8 prompt-byte guard before each
generation request (reserving the full completion ceiling plus 1,024 bytes), then
checks returned token usage against the effective context. This guard fails closed
and never truncates or retries a request. No automatic retry is allowed. Retrieval
output, query text, evidence aliases, requirement IDs, and evidence provenance are
harness-owned.

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

Final-answer parsing accepts exactly one single-line response beginning with
`FINAL:`; preambles, multiple markers, continuations, missing output, unknown
aliases, and any truncated model call are output-contract failures. There is no
retry. Issued aliases are `[E1]`…`[E10]` in frozen rank order. The model never
sees underlying document IDs. Contract failures cannot count as task success.

## CFEC authority boundary

The decomposer receives only the original question. It emits at most four
non-empty requirement lines; the harness keeps the first four and assigns stable
IDs. Each claimant receives one requirement and the exact same ranked top-10
passages shown to the corresponding Vanilla arm. Claimants return short claims
with issued aliases or `UNSUPPORTED`. Unknown aliases are recorded and invalidate
that output, even if a line also contains a valid alias; claims without any valid
issued alias are rejected. The final composer receives the original question
plus validated claim text and aliases, never raw passages or document IDs. Final
used evidence is the harness union of validated claim provenance; model text
cannot alter that set. Final citations must be a subset of aliases attached to
those validated claims.

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

Before FROZEN_DEV truth is opened: freeze the selected CFEC-v1.3 prompts, parser,
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
standards. A protocol lock additionally requires zero truncated model calls.
A failed gate authorizes no reserved-test materialization and no prompt iteration
on FROZEN_DEV.

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

## Initial BUILD audit (diagnostic only; excluded from method selection)

The initial BUILD executed all 818 selected episodes under all five arms (4,090
arm executions) before evaluator truth was opened. Its primary RAG slice contains
156 episodes from 49 subjects. One interrupted generation request was not
retried; total truncations were zero. These immutable artifacts are retained at
`runs/rag_e6/build/` as diagnostics.

The first score appeared to show `CFEC_STRONG` versus `VANILLA_STRONG` grounded
task success moving from 0/156 to 138/156 (+88.46 pp). That result is **not
accepted for method selection or as a headline**. Output audit found that the
Vanilla prompt asked for exact aliases but did not say they had to appear after
`FINAL:`, while the scorer intentionally counts only bracketed aliases in the
returned FINAL segment. Among 156 Vanilla STRONG RAG completions, 59 contained
bracketed aliases somewhere before/around the final segment, 37 used bare `E#`
references, and zero had a bracketed alias after `FINAL:`; these categories are
not mutually exclusive. This is a prompt/parser contract mismatch, so the
apparent grounded-success delta is confounded by citation formatting. The
answer-value correctness delta (136/156 to 138/156, +1.28 pp) is retained only
as a diagnostic, not evidence of a confirmed CFEC uplift.

The citation-parity `build_v2` attempt was stopped after 232 completed episodes
because a CFEC compose call hit the old 8,192-token cap and emitted over 1,300
unissued aliases. The runtime correctly kept claim provenance harness-owned, but
the scorer only recorded `output_contract_failure` and did not require it to be
false for task success. That partial run is retained as diagnostic data only;
it has no completion manifest, was not scored, and did not open evaluator truth.

The `build_v3` attempt used `CFEC-v1.2` with a 256-token cap. It was stopped after
8 completed episodes: eight CFEC claim calls across those episodes reached the
cap, although earlier valid claim calls commonly used about 325 tokens. This was
too small for the normal claim contract and would force the pre-registered zero-
truncation lock to fail; this partial run is retained but not scored and did not
open evaluator truth.

`build_v4` uses `CFEC-v1.3`: every call, across every arm, is capped at 512 output
tokens; the parser accepts exactly one `FINAL:` line; unknown aliases invalidate
the output; CFEC final citations must come from validated claims; and truncated,
failed, or malformed calls fail closed in task-success scoring. The protocol lock
also refuses any BUILD with a truncated generation call. This version executes
all five arms on the same frozen BUILD subjects and evidence. Only after its
complete run passes the pre-registered gate will the code/protocol be committed
and locked, followed by FROZEN_DEV execution and scoring. A failed FROZEN_DEV gate
means no reserved TEST/OOD materialization.

## Method positioning

The diagnostic separation follows the motivation of
[RAGChecker](https://arxiv.org/abs/2408.08067): retrieval coverage and generation/use
must be measured separately. Claim-first decomposition is related in spirit to
the retrieve-per-subquestion and evidence-pooling pattern in
[Question Decomposition for RAG](https://arxiv.org/abs/2507.00355), but CFEC-v1.3
does not reproduce that pipeline: retrieval candidates are fixed, and the
Harness validates claim aliases and owns provenance. It also is not
[RankRAG](https://arxiv.org/abs/2407.02485), which instruction-tunes one model
for ranking and answer generation. E6A isolates evidence utilization with the
same frozen retrieval and generator across Vanilla and CFEC arms; it does not
train a ranker or alter the retriever.
