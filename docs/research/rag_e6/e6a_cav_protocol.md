# CAV-v1: Conservative Answer Verification

## Status and scope

CAV-v1 is a new, bounded BUILD-only method family after the negative CFEC-v1.4
result. It does not reopen CFEC, reuse its claims, or authorize FROZEN_DEV or
reserved TEST access. The full RAG objective remains open: demonstrate a real
reader-side improvement over a strong, same-retrieval Vanilla baseline on the
owned longitudinal environment, with evidence identity and provenance still
controlled by the Harness.

The primary contrast is `CAV_STRONG` versus `VANILLA_STRONG` on the evaluator-
defined RAG-required slice. `CAV_STANDARD` versus `VANILLA_STANDARD` is
secondary; `VANILLA_OFF` remains the parametric-only control. All 818 BUILD
episodes execute before BUILD truth is opened.

## Method

For each STANDARD/STRONG arm, CAV reuses the exact Vanilla draft and the exact
ranked top-10 evidence already supplied to that baseline:

```text
same query + same frozen top-10 + same Qwen3-8B
                         |
                 Vanilla draft
                         |
           one conservative verifier call
                   /           \
                KEEP           FINAL repair
                  |                |
         unchanged draft     strict alias validation
                                  /       \
                              valid       invalid
                                |            |
                         accepted repair  unchanged draft
```

The verifier is instructed to prefer `KEEP` unless evidence clearly identifies
a concrete omission or factual error. A repair must be a single `FINAL:` line,
answer the whole question, and cite only aliases issued for that exact evidence
list. The Harness resolves aliases to document IDs and records provenance.
Unknown aliases, malformed responses, truncation, model-call failure, and
context-guard rejection all select the unchanged Vanilla answer. The model
cannot request retrieval, alter the evidence list, create evidence IDs, or
change Harness state.

This is a second-pass verifier/edit baseline, not a claim of a novel algorithm.
It tests whether conservative evidence-grounded self-correction can recover
some of the utilization headroom while avoiding CFEC's multi-call decomposition
and claim-interface failure mode.

## Cost and parity

Retrieval, query, evidence order/content, generator weights, temperature,
reasoning setting, and per-call token budget are fixed against the paired
Vanilla arm. CAV adds exactly one verifier request per STANDARD and STRONG
episode (up to two additional requests per episode total); those calls,
tokens,
latency, truncations, and fallback reasons are recorded. No reranker,
additional retrieval, answer key, evaluator label, or gold document enters the
generation path. Cost is therefore not matched: CAV is an explicitly higher-
generation-cost reader treatment, and any uplift must be reported with that
cost.

## Gold-blind preflight

Before full BUILD, `tools/research/rag_e6/smoke_cav.py` selects 24 episodes from
the frozen BUILD runtime/output set by a fixed SHA-256 ordering independent of
evaluator labels. It reconstructs evidence from the gold-blind runtime corpus,
verifies byte-level evidence identity against the Vanilla output, and audits
only verifier contract health. It requires at least 23/24 responses to be
valid `KEEP` or valid `FINAL` repairs and zero truncations. This preflight is
not a quality score and cannot establish expected uplift; a pass authorizes
only the preregistered full BUILD experiment.

## BUILD gates and stop rule

The frozen scoring implementation defines the RAG slice and computes
grounded-task-success and subject-clustered 10,000-resample paired intervals.
Promotion beyond BUILD requires all existing locked gates:

- `CAV_STRONG - VANILLA_STRONG` grounded-task-success point delta at least
  +10 percentage points;
- paired subject-clustered 95% CI lower bound above zero;
- grounding degradation no worse than -1 percentage point;
- utilization-given-full-evidence point delta at least +10 percentage points;
- generation health at least 99% nonempty calls.

A failed gate closes CAV-v1. No prompt tuning against BUILD outcomes, no
FROZEN_DEV scoring, and no reserved TEST/OOD access follows a failed gate. A
pass permits an immutable method lock and then FROZEN_DEV only; it does not by
itself establish the final headline claim. Even a successful future holdout
must state the added verifier cost and compare against the frozen strong
baseline.

## Interpretation boundary

CAV can produce a positive outcome only by changing answer content through a
valid evidence-citing repair; `KEEP` and every invalid path preserve the exact
Vanilla answer. This asymmetric fallback intentionally protects baseline
quality from malformed model output, but does not guarantee a positive result:
the verifier can make valid yet incorrect edits, or may fail to identify the
actual omission. BUILD will measure that risk rather than assume it away.
