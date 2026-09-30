# U3-R: Cost-Aware Retrieval Capability Transfer

## Research question

U3-R does not tune or claim a new retriever. It binds the already-frozen Health-Copilot retrieval capability to the owned longitudinal universe and asks whether an execution policy can preserve task quality while avoiding unnecessary retrieval strength.

The experiment is a fixed counterfactual comparison, not a learned router: every eligible DEV episode is executed under OFF, STANDARD, and STRONG. The LLM is used only to produce the STRONG retrieval bridge. The harness owns action execution, evidence access, answer production, and evaluation boundaries.

## Data and split

The substrate is `health-copilot-owned-longitudinal-v1`, pinned by U2-F root SHA-256 `e28ea9ef9ecae47d3f27f28c68042066e9af297fe808cafebf1d3c8c80fb2134`. Runtime input consists of 512 `DEV_IID` and 512 `DEV_STRUCTURAL` episodes. TRAIN outcomes and reserved TEST/OOD rows are not used; the frozen dataset manifest states that reserved evaluation rows are unmaterialized.

All 1,024 DEV episodes receive all three runtime arms before evaluator truth is read. The primary scoring slice is derived only after execution and includes `NONE`, `RAG`, and `INSUFFICIENT`; `MEMORY` and `MEMORY+RAG` remain present in execution artifacts but are marked `NOT_PRIMARY_U3R` because the Memory capability is outside this stage.

For the retrieval corpus, the runtime materializer scans only the DEV worlds and selectively decodes an allowlist: source identity/family, publication and effective-time metadata, and visible document text. It skips latent dependency graphs, required-fact IDs, retrieval-term annotations, answers, and evaluator labels. Documents are filtered by episode-visible source families and decision time. The serialized runtime episode schema is exact-key checked as a second boundary.

## Fixed retrieval arms

| Arm | Retrieval pipeline | Reader / authority |
| --- | --- | --- |
| OFF | No external retrieval | U1.1 deterministic executor |
| STANDARD | Lucene BM25 (`k1=0.9`, `b=0.4`) + BGE-large; equal-weight RRF (`k=60`, `[1,1]`) | Same deterministic executor |
| STRONG | Pinned R2MED LameR prompt receives BM25 top-10 feedback and emits a bridge; four views are BM25(query), BM25(query + bridge), BGE(query), BGE(bridge), fused with weighted RRF (`k=20`, `[1,2,1,2]`) | Same deterministic executor |

Both retrieval arms expose at most the fused top 10 documents to the executor. BGE is `BAAI/bge-large-en-v1.5`, pinned revision `d4aa6901d3a41ba39fb536a557fa166f842b0e09`, with the model weights hash recorded in the run manifest. STRONG uses the pinned R2MED repository/prompt identity and the local Qwen3-8B Q4_K_M artifact whose SHA-256 is `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.

The generator runs through llama.cpp on CPU, bound to `127.0.0.1`, with zero GPU layers. Sampling is deterministic (`temperature=0`, `top_p=1`), reasoning is explicitly disabled, and the completion ceiling is 8,192 tokens. The lower token cap used in the preflight smoke was not used for dataset generation. There is one generation attempt per episode; errors fall back to the original query and are recorded, never retried silently.

The generated bridge may affect retrieval ranking only. It cannot choose an action, write the answer, select a source scope, access gold/evaluator data, or alter harness policy. The deterministic executor alone produces the answer from the evidence made observable by the fixed arm; it reports zero model/provider calls. The scorer independently verifies this contract before reading evaluator truth.

## Outcome and cost definitions

- **Task success**: the deterministic answer matches the evaluator's answer semantics and all required resources were actually observed.
- **Grounding pass**: every used evidence ID is among the arm's ranked documents and in the episode-visible corpus, and every required resource is observed. `Grounded success = task success AND grounding pass`.
- **Required external fact coverage**: required external evidence IDs used divided by required external evidence IDs; undefined when none are required.
- **Abstention correctness**: for unanswerable cases, the executor returns `INSUFFICIENT_EVIDENCE` and uses no evidence.
- **Minimal sufficient action (`A*`)**: cheapest successful arm in the fixed order OFF < STANDARD < STRONG. If no arm has grounded success, the label is `UNRESOLVED`, not STRONG.
- **Retrieval search invocation**: one actual channel search. STANDARD uses two; STRONG uses four (the original-query BM25 result is reused from LameR feedback rather than searched twice).
- **Bridge activation / model call**: one STRONG bridge generation. Tokens and retrieved/used document counts are also reported as structural cost; latency is diagnostic only.

The quality oracle is the fraction of primary episodes where at least one arm achieves grounded success. Quality headroom is the oracle minus the best fixed-action grounded-success rate. Cost headroom compares the minimal-cost oracle with Always-STRONG while preserving oracle quality. The cost-aware signal requires either at least 15% fewer retrieval search invocations or at least 20% fewer bridge activations, with no quality loss versus Always-STRONG. Action diversity requires at least two minimal-action classes, each representing 10% or more of primary episodes and at least 10 subjects.

After counterfactual outcomes are frozen, one diagnostic probe compares TF-IDF + logistic regression on query text alone versus query plus observable runtime metadata, using subject-grouped cross-validation. It is a predictability diagnostic, not a trained production router and not SFT/OPD/GRPO.

## Artifact and evaluation boundary

The runtime freeze records dataset, prompt, model, retrieval, corpus, code, and output hashes before scoring. The post-freeze scorer then opens only DEV evaluator-truth files and writes fixed-action metrics, minimal-action labels, the cost frontier, failure attribution, and the predictability probe. TEST/OOD and TRAIN evaluator outcomes remain unopened. Frozen outputs are immutable; rerunning the scorer validates every output hash.

## Interpretation limits

This is an owned synthetic DEV characterization, not a public-clinical benchmark result and not evidence of clinical safety. The external evidence is synthetic, and the primary slice excludes Memory-dependent tasks. The quality oracle and minimal-action labels are privileged counterfactual targets, not runtime inputs. CPU latency is hardware-specific and excluded from policy labels. Any positive cost headroom justifies considering U3-R2 training-view materialization; it does not itself establish that a learned policy generalizes. No SFT, OPD, or GRPO is run in U3-R.

## Results

### Execution integrity

The frozen run executed all `1,024` DEV episodes under all three arms (`3,072` counterfactual arms): `512 DEV_IID` and `512 DEV_STRUCTURAL`. The primary RAG-isolatable slice contains `582` episodes (`NONE=164`, `RAG=241`, `INSUFFICIENT=177`); the other `442` Memory-dependent episodes were executed but excluded from primary U3-R scoring.

The CPU-only Qwen3-8B bridge produced `1,024/1,024` valid completions with one call per episode, zero truncations, and zero original-query fallbacks. The run freeze records `evaluator_truth_opened=false`; the scoring manifest confirms evaluator truth was opened only after the freeze, for the two DEV truth files (`1,024` rows total). TRAIN outcomes and reserved TEST/OOD were not opened; TEST/OOD rows remain unmaterialized. The bridge and counterfactual hashes match the frozen manifest and recovery checkpoints.

### Fixed-action quality

Rates below are over the `582` primary episodes. Grounded success requires both task success and grounding pass; it is not just evidence recall.

| Fixed action | Grounded/task success | Answer-value coverage | Required external-fact coverage | Grounding pass | Correct abstention |
| --- | ---: | ---: | ---: | ---: | ---: |
| OFF | **35.40%** | 28.18% | 0% by design | 58.59% | 23.73% |
| STANDARD | 9.45% | 44.56% | 99.31% | 99.48% | 0% |
| STRONG | 9.62% | 44.85% | 100.00% | 100.00% | 0% |

The fixed-action winner is **OFF** on this synthetic primary slice. That aggregate hides the requirement-class split:

| Derived class | n | OFF success / abstention | STANDARD success | STRONG success | External-fact coverage (STANDARD / STRONG) |
| --- | ---: | ---: | ---: | ---: | ---: |
| NONE | 164 | **100.00%** | 3.66% | 3.66% | n/a |
| RAG | 241 | 0% | 20.33% | 20.75% | 99.31% / 100.00% |
| INSUFFICIENT | 177 | 23.73% correct abstention | 0% | 0% | n/a |

The same overall pattern appears in both development partitions. On `DEV_IID` (308 primary cases), grounded success is OFF `39.61%`, STANDARD `6.82%`, STRONG `7.14%`; on `DEV_STRUCTURAL` (274), it is OFF `30.66%`, STANDARD `12.41%`, STRONG `12.41%`. These are development characterization results, not a held-out public benchmark claim.

### Quality and cost frontier

The privileged quality oracle succeeds on `256/582 = 43.99%`; the best fixed action, OFF, succeeds on `206/582 = 35.40%`. Thus quality headroom over the best fixed action is **+8.59 percentage points**. STRONG has only **one unique success** beyond OFF and STANDARD.

For the primary slice, Always-STRONG costs `582` retrieval activations, `2,328` actual channel-search invocations, `3,009` retrieved-document instances, `582` bridge activations/model calls, `171,457` input tokens, and `24,708` output tokens. The minimum-cost oracle, constrained to preserve per-case maximal grounded success, costs `50` retrieval activations, `102` search invocations, `224` retrieved-document instances, and `1` bridge/model call (`430` input / `20` output tokens). This corresponds to **91.41% fewer retrieval activations**, **95.62% fewer actual retrieval searches**, and **99.83% fewer bridge calls** than Always-STRONG. The protocol's cost-headroom threshold passes.

These oracle reductions are **upper-bound headroom, not measured savings from a deployed or learned policy**. The oracle sees sibling counterfactual outcomes and is privileged. `UNRESOLVED` cases are assigned OFF for cost accounting only; their training label remains `UNRESOLVED`.

### Minimal sufficient actions and predictability

| Minimal-action label | Count | Share of primary slice | Distinct subjects |
| --- | ---: | ---: | ---: |
| OFF | 206 | 35.40% | 71 |
| STANDARD | 49 | 8.42% | 30 |
| STRONG | 1 | 0.17% | 1 |
| UNRESOLVED (all arms fail) | 326 | 56.01% | — |

The action-diversity gate **fails**: only OFF meets both the `>=10%` and `>=10 subjects` criteria. STANDARD has enough subjects but is below the 10% episode threshold; STRONG is rare. This is why the final status is `COST_AWARE_POLICY_SIGNAL=YES` but `POST_TRAINING_RETRIEVAL_POLICY=NOT_JUSTIFIED`: cost headroom exists, while the observed target distribution is not sufficiently diverse to justify policy training.

The single subject-grouped, five-fold TF-IDF/logistic diagnostic yields query-only accuracy `92.44%` / macro-F1 `60.59%`; query plus runtime metadata yields `90.55%` / `59.76%`. The metadata variant does not improve this probe. Accuracy is inflated by the imbalanced action labels (especially `UNRESOLVED`) and synthetic task structure; these numbers are not a production-router result and do not override the failed action-diversity gate.

### Failure attribution and interpretation

- All `241` RAG-required cases are correctly marked retrieval-required; there are **zero candidate retrieval misses**. Ranking misses fall from `3` under STANDARD to `0` under STRONG.
- Yet `189/241` STANDARD and `191/241` STRONG RAG cases are attributed `RETRIEVAL_NOT_SUFFICIENT`: required evidence was present in the candidate set and used, but the task still did not meet the success contract. The gap is therefore not explained by recall alone; evidence sufficiency/composition or downstream task semantics remains limiting.
- Retrieval causes `CONTEXT_INTERFERENCE` on `200` cases for each retrieval arm. On NONE cases, OFF succeeds `164/164` while each retrieval arm succeeds only `6/164`. For INSUFFICIENT, OFF abstains correctly on `42/177`, while STANDARD and STRONG do not abstain correctly on any.
- External-fact coverage is near-perfect when retrieval is required, but this does not translate into broad task success. Conversely, retrieval is harmful on many cases where no external evidence is needed or evidence is insufficient. This supports studying *when to spend retrieval capability*, not further tuning the already-frozen retriever.

### Gate and next step

```text
REAL_RAG_BINDING_COMPLETE = YES
DEV_EXECUTION_COMPLETE = YES
QUALITY_HEADROOM = +0.08591 (8.59 percentage points)
COST_AWARE_HEADROOM = 95.62% fewer channel searches; 99.83% fewer bridge calls (oracle)
OFF_MINIMAL_COUNT = 206
STANDARD_MINIMAL_COUNT = 49
STRONG_MINIMAL_COUNT = 1
UNRESOLVED_COUNT = 326
QUERY_ONLY_PREDICTABILITY = 92.44% accuracy / 60.59% macro-F1
CONTEXT_METADATA_PREDICTABILITY = 90.55% accuracy / 59.76% macro-F1
ACTION_DIVERSITY = FAIL
COST_AWARE_POLICY_SIGNAL = YES
POST_TRAINING_RETRIEVAL_POLICY = NOT_JUSTIFIED
RESERVED_TEST_OOD_MATERIALIZED = NO
RESERVED_TEST_OOD_OPENED = NO
TRAINING_STARTED = NO
```

Per the frozen gate, do not proceed to SFT/OPD/GRPO or claim a learned adaptive router from these labels. The best fixed action on this U3-R synthetic primary slice is OFF; this is scoped to this owned DEV characterization and does **not** invalidate the separate public R2MED retrieval results or prescribe disabling RAG in the product. Preserve these artifacts and wait for the next combined-capability decision after Memory is ready. No retriever, reranker, prompt, corpus, or retrieval configuration was tuned in U3-R.
