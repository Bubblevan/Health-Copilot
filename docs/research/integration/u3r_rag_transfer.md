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

To be completed only from the frozen run artifacts. No numeric outcome should be inferred from protocol expectations or from B4's smaller mechanism-only sample.
