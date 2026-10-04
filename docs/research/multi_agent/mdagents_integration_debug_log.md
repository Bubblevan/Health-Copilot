# MDAgents integration issues and fixes

Date: 2026-10-04

This log records the implementation and parity-debugging decisions that led to the final 128-case sample. Intermediate v1/v2/v3 sample directories were diagnostic runs and are removed after this log is committed. The frozen upstream full reproduction and final parity artifacts are retained.

## 1. Upstream code could not be copied into the product

**Problem.** The pinned MDAgents checkout has no root LICENSE file. Copying its implementation into Health-Copilot would create unclear redistribution rights.

**Resolution.** Kept the upstream checkout and its minimal local-vLLM compatibility patch external. Implemented the complexity triage, basic/intermediate/advanced branches, dynamic specialist selection, team reasoning, and moderation as a separate Health-Copilot module. The product code does not import the reference checkout.

## 2. Reusing the post-training vLLM environment

**Problem.** The adjacent post-training task was already preparing a uv environment. A second large vLLM/Torch install risked duplicating storage, and the stable binary installation could not be completed within the shared quota.

**Resolution.** Reused the available isolated Python 3.11 vLLM environment for the local model server. The frozen manifest records vLLM 0.30.1rc1.dev622+gf03026a54, Torch 2.13.0+cu130, and CUDA runtime 13.0. Health-Copilot itself used its separate Python 3.13 environment. No global Python packages were changed. The endpoint stayed on loopback.

## 3. Local OpenAI-compatible calls could inherit proxy settings

**Problem.** The OpenAI client can honor environment proxy configuration, which could route a nominally local model request through a proxy.

**Resolution.** The local provider constructs an httpx.AsyncClient(trust_env=False), checks that the configured HTTP host is loopback, and uses only the configured local OpenAI-compatible endpoint. The parity run used http://127.0.0.1:8000/v1.

## 4. The first graft classifier prompt drifted from the reference

**Problem.** The first diagnostic version changed the classifier wording and constrained it to a JSON-shaped response. This changed route distribution substantially: basic 9, intermediate 108, advanced 11 versus the sampled reference's basic 82, intermediate 33, advanced 13. Accuracy was 63.28% versus reference 60.16%, while exact route agreement was only 33.59%.

**Resolution.** Removed the prompt rewrite and JSON requirement. Restored the pinned classifier's wording and the reference-compatible substring route parser. The diagnostic score was not used to tune the method.

## 5. A stateless classifier call lost the reference conversation

**Problem.** After restoring the wording, the next version still sent only the final difficulty question. Upstream first sends an initial classifier setup turn and then asks for a decision in the same conversation. Without those turns, the classifier routes differed: the v2 sample had 6 parse failures, no advanced decisions, and 49.22% route agreement.

**Resolution.** Added complete_messages() to the local provider and preserved system, user, and assistant role history for the classifier setup and decision turns. Other provider implementations retain a fallback adapter path.

## 6. Classifier token limit truncated the setup response

**Problem.** The first history-preserving probe retained a 96-token classifier cap. The setup response was truncated and the classifier did not produce a valid route on the initial v3 cases, so that diagnostic run was stopped after seven completed rows and was not scored as a parity result.

**Resolution.** Restored the protocol's 1,024-token output cap for classifier calls, consistent with the frozen reference configuration. The final 128-case run completed without a parse failure or runtime error.

## 7. Accuracy closeness did not imply case-level parity

**Observation.** The final fixed sample scored 77/128 (60.16%) for the frozen reference and 76/128 (59.38%) for Health-Copilot, a -0.78 pp delta. Exact answer agreement was 59.38% over all rows and 68.47% when both parsers succeeded. Route agreement was 58.59%.

**Interpretation.** Aggregate accuracy can be close while systems disagree on which cases they answer correctly and which routes they choose. The sample meets only the accuracy-delta threshold; it misses the answer and route agreement thresholds. It is not evidence that a full rerun would be identical, and we did not claim that it is.

## 8. Parse failures need explicit accounting

**Problem.** A parser that drops nonconforming answers would inflate accuracy by changing the denominator.

**Resolution.** The runner stores raw output and parsed option for every case, treats parse failures as incorrect in accuracy, and reports parse success separately. Every completed case is checkpointed in JSONL. The final sample parsed 127/128 Health-Copilot answers; the reference parsed 112/128.

## Verification performed

- The deterministic 128-case sample completed with zero runtime failures.
- Final source and runner passed Ruff and Python compilation checks in the recorded environments.
- Import and deterministic A-D parser smoke checks passed.
- No full Health-Copilot MedQA run was performed; the frozen reference full run was not modified.

- The repository-wide pytest run reached roughly 35% and then produced no progress for about 2.5 minutes. It was interrupted without a failure report. The focused agent/runtime suite completed: 30 passed. Repository-wide pytest therefore remains unverified in this handoff.
