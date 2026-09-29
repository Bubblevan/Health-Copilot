# MedMemoryBench Compatibility Audit

Audit date: 2026-09-29. This is a source/data compatibility audit only: no benchmark method ran, no answer was scored, and no prompt or method was tuned.

## Executive disposition

MEDMEMORYBENCH_COMPATIBILITY_AUDITED=YES
MEDMEMORYBENCH_SOURCE_PINNED=YES
MEDMEMORY_STREAMING_COMPATIBLE=CONDITIONAL
MEDMEMORY_LABEL_EXPOSURE_OCCURRED=true
LONGMEM_MEMORY_MECHANISM_FROZEN=NO

The source checkout and local LFS data are pinned and structurally audited. The official independent streaming path is compatible in principle with per-persona session/memory APIs, subject to an adapter contract for session ordering, checkpoint time, scope/reset, and evaluator-only labels. Public answers and provenance mean future work is PUBLIC_EXTERNAL_TRANSFER, not BLIND_TEST. This gate means the compatibility questions are resolved enough to plan a transfer; it does not mean transfer has run or succeeded.

### Label exposure disclosure

The public Hugging Face dataset page unexpectedly rendered a sample dialogue and one associated knowledge_points value while its structure was being inspected. No answers field was opened, the displayed content was not used for method/prompt choice or scoring, and no literal sample content is copied into this packet. Because benchmark sample text was actually visible, the exposure flag is conservatively true. Future evaluation remains public-label transfer rather than a blind test.

## Source and data identity

The local read-only checkout at external/MedMemoryBench is AQ-MedAI/MedMemoryBench, branch main, HEAD 7227bc105b84a1a9f7a75861eb9e1be3ea502882, matching the public upstream commit observed for this audit. Worktree is clean; HEAD...origin/main is 0/0. Source tree hash is 0433ceb8502784d29004c57e254ca5dbdb6e813f. No pull/fetch or mutation was performed. Four declared submodules are not initialized; their pinned gitlinks are recorded in the run manifest, so those methods cannot be assumed locally executable from this checkout. The source commit includes LFS pointer objects; all audited local dataset files are hydrated and each file SHA256 matches its LFS OID.

Per-language clean data: 20 personas, 101 regular sessions/persona, 2,020 sessions, 31,976 turns, and 1,986 queries. The noisy dialogue file has 301 sessions/persona: those 101 regular sessions plus 100 health-noise and 100 family-health-noise sessions. Therefore there are 2,000 injected noise sessions per language and 6,020 rows in the noisy dialogue file. The README's approximate sizes (about 598 MB Chinese and 443 MB English) differ from audited decimal bytes (616,424,610 and 464,444,729). The public per-language counts match the local files; summing bilingual files gives 40 language-personas, 4,040 regular sessions, 63,952 turns, and 3,972 queries. The repository default config selects personas 1-10 and inject_noise=false, while the local dataset contains 20 personas and a separate noisy arm. This is a default configuration, not a train/dev/test split.

The canonical artifact and schema hashes, per-file hashes, and source identities are in medmemorybench_dataset_manifest.json. The Hugging Face Cyan27 mirror is not identical in representation to the repository-native file layout: its viewer flattens dialogue rows and exposes annotation columns. Its displayed CC-BY-NC-SA-4.0 label differs from the repository README's CC-BY-4.0. Resolve licensing and redistribution with the authoritative source before reuse; this discrepancy is a transfer blocker for redistribution, not for this local audit.

## Splits and public-label status

At the pinned source/config audit, no train/dev/test split, persona-level held-out split, hidden evaluator, or official hidden leaderboard protocol was found. The config's 1-10 persona selection is a run default only. Answers and source_key_points provenance are present in public Git-LFS data, so no new split over these files is blind. Do not invent or name a split TEST. Future status: PUBLIC_EXTERNAL_TRANSFER. An external leaderboard claim was not verified.

## Runtime visibility, gold leakage, and schema

Runtime-visible content is each session's ordered messages (turn, role, content, agent_type), exposed only as sessions arrive. Session and event IDs may be used as adapter provenance identifiers, but event_info and knowledge_points are not ordinary conversational input. Evaluator-only or generation-only fields include answers, source_key_points, metadata labels/reasoning structures, knowledge_points, event_info, trap_score, trap_events, and generated event/persona metadata. Embedding or storing those annotations in runtime memory would leak gold or construction labels. The field-by-field classification and schema union are in medmemorybench_schema_audit.json.

The public dataset schema is sufficient to support session-level answer Recall@K/MRR for most queries through source_key_points session IDs, but it does not provide direct source turn IDs or direct source event IDs. Four queries have no resolvable source session in the audit join. Evidence Recall@K must therefore be defined at session or knowledge-point proxy level and disclosed; it is not source-turn evidence recall. Provenance coverage and limitations are quantified in medmemorybench_provenance_compatibility.json.

## Official streaming and isolation semantics

The audited default evaluator mode is independent. For each persona, the dataset unit builder walks regular sessions in file order, accumulates pending sessions, counts only regular sessions toward evaluation_interval=10, and yields the pending batch after the tenth regular session together with queries keyed to that session. The evaluator retains one agent/memory across units for a persona and resets/recreates it for the next persona. It memorizes the current unit before asking its queries; it does not pass future sessions into the active unit. The 101st regular session is after the last query checkpoint. Thus the operational shape is session 1..10 ingested, then checkpoint queries, followed by 11..20, etc.; it is not static all-history QA.

Noise sessions are interleaved in the noisy file and do not increment the regular-session counter. They are part of the same default persona stream. The generator preserves regular order and inserts noise using a weighted early-position distribution of 3:3:2:1:1. Noise therefore creates a wrong-person/identity contamination risk when family health information is written into the patient's own memory scope. The official merged mode is unsafe for person isolation (shared/global context); future adapter must use independent mode only unless a separately audited scope fix is made.

Target Health-Copilot scope: medmemory:<language>:persona:<persona_id>. Reset at persona boundary. Checkpoint time, stable session identity, monotonic stream order, noise provenance, and inclusion of only arrived messages must be deterministic. Conditional compatibility is recorded because timestamp interpretation and revision/query-time semantics still require protocol decisions, and Health-Copilot adapter code has not been implemented.

## Query families and relation to LongMemEval

The six exact categories and counts are in medmemorybench_query_taxonomy.json. Broadly, entity_exact_match (400) and multiple_choice (398) have deterministic official metric families; temporal_localization (400), state_update (200), inference_generation (397), and multi_hop_clinical_deduction (191) use LLM judges. State update is the direct clinical state-replacement stressor; temporal localization is a partial temporal-reasoning analogue; inference and multi-hop are partial multi-session reasoning analogues but include clinical reasoning not present in generic memory recall. Do not infer equivalence by similar category names.

### LongMemEval field mapping

| LongMemEval concept | MedMemoryBench concept | Relation | Limitation |
| --- | --- | --- | --- |
| Session history | Consultation/session messages | APPROXIMATE | Both are ordered dialogue units; generated clinical scenarios and turn roles differ. |
| question_date | Checkpoint/query time or source time | APPROXIMATE | Queries occur at stream checkpoints, but there is no single universal question_date contract. |
| answer_session_ids | source_key_points session IDs | APPROXIMATE | 1,982/1,986 queries have resolvable session provenance; no source-turn ID and four unresolved joins. |
| knowledge-update | state_update | APPROXIMATE | Both test replacing stale with current state; medical state and official judge semantics add domain-specific requirements. |
| temporal-reasoning | temporal_localization | APPROXIMATE | Overlapping temporal target, not full equivalence to all temporal QA or AS_OF queries. |
| multi-session reasoning | inference_generation / multi_hop_clinical_deduction | APPROXIMATE | Both may compose history, but clinical causal reasoning is an additional task. |

No one-to-one equivalence is claimed for the other MedMemoryBench categories.

## Noise and trap events

There are separate clean and noisy dialogue arms; the repository default is clean. Do not select the noise arm during this audit. Health and family noise are distinct inserted session families. No direct mapping from injected noise IDs to query gold source_key_points was found. Future noise-exposure metrics must use run-time read/write provenance, not answer labels alone.

Trap-event annotations cover high-consequence longitudinal categories including allergy, medication history, disease history, medication preference, diet preference, and lifestyle/economic context. The annotations are generation/evaluation metadata, not runtime facts or permission to elevate a memory to clinical evidence. A future evaluator may use them to analyze critical-memory retention, stale-state exposure, wrong-patient contamination, and unsafe forgetting, but only after an explicit evaluator protocol.

## Official evaluator

The six category mappings are string_contain, llm_judge, option_match, and llm_judge_mcd. string_contain lowercases and strips punctuation/whitespace, then requires all correct normalized answer strings to occur as substrings in the output and at least one expected answer. option_match parses A-F selections and requires exact selected-set equality with the gold set. The temporal judge asks whether the response identifies the right time point or event, allowing equivalent date formats. The state-update judge requires evidence of patient-specific memory use, rejects lucky guesses and generic knowledge, and rejects outdated states or abstention. The inference judge requires patient-specific facts, correct reasoning and conclusion, and uses evaluator metadata such as required patient information/common wrong-answer patterns. The MCD judge receives evaluator reasoning-chain nodes and required-memory nodes, checks node and causal-link coverage, and returns structured scores. These metadata/answer explanations are judge-only inputs and must never enter the memory runtime.

The LLM judge is configurable; the example environment names gpt-4o, but code falls back to gpt-4o-mini if not configured, so the judge is not source-pinned by implementation alone. Temperature is 1.0. The client is initialized with a 10,000-token setting, while actual standard judge calls pass a 500-token output limit and MCD passes 2,000. Retry policy can retry transient errors up to ten attempts with randomized 10-20 second delays; malformed JSON and non-retryable/auth/request errors are not retried. Failed or malformed judge responses can become false/zero, conflating infrastructure failure with quality. MCD combines NCR/CRC/CC with 0.35/0.35/0.30 weights, applies a 0.5 penalty when patient-specific information is absent, and multiplies by a retrieval-quality factor. These are source-audit findings only; no judge ran.

string_contain and option_match can be reproduced locally and deterministically if the pinned evaluator code and output normalization are retained. LLM-judged categories are not deterministic and depend on a model/provider/config; judge failure must be separated from quality in any future adapter.

## Baseline and fairness disposition

The official README claims 14 systems across long-context, classic retrieval, agentic memory, and graph methods. The source/config audit is summarized in medmemorybench_baseline_protocol_matrix.json. The BM25, dense, and graph methods index personal histories and are named MEMORY_READ_BASELINE::BM25, ::DENSE, and ::GRAPH.

Official configurations vary reader model/version, internal LLM, prompt, top-k, chunk size, context budget, embedding model, and hosted dependencies. Examples include Qwen3-235B-A22B versus Qwen3-235B-A22B-Instruct-2507, GPT-5.1-only configurations, BGE-small local embeddings, and Zep Cloud. Therefore official scores are historical coordinates only; they are not controlled comparisons against Health-Copilot's local Qwen3-8B protocol.

Future reporting must separate:

- OFFICIAL_HISTORICAL: preserve the upstream protocol and report as published; no parity claim.
- CONTROLLED_LOCAL_TRANSFER: one frozen reader, evaluator, answer prompt, and where substitution is allowed one embedding contract, while preserving each system's native write/read semantics. Disclose systems that cannot accept the shared stack.

Never mix the two tables. In particular, do not imply Health-Copilot reproduction of official MedMemoryBench scores.

## Health-Copilot API compatibility

Existing SessionStore/MemoryStore primitives can represent session events, per-scope writes, source_session_id, validity intervals, source event IDs, sensitivity, supersession, snapshots, and replay identity. A future adapter can map ordered conversation messages and stable session IDs to these primitives, with checkpoint-level evaluation and reset per persona. Adapter-only work includes data normalization, scope/reset, timestamp/checkpoint mapping, noise provenance, and evaluator-side source-session mapping.

Actual mechanism decisions still open include what query-time "now" means, event-time versus observation-time semantics, state identity/materialization, and whether a medical state replacement is supported by the frozen LongMem mechanism. No schema extension is authorized until the remaining LongMem frozen-ten mechanism chain is complete. Memory remains personal context only: never Evidence, citation authority, clinical fact authority, safety-policy override, or tool permission.

## E2-A compatibility

E2 workers currently have Memory OFF. Source-family isolation is external evidence retrieval. Capability contracts are not personal-memory contracts. Optional ComponentManifest capability fields are omitted when absent to preserve legacy canonical hashes. No historical Memory run identity changed. This audit adds only a routing cross-reference; no E2 runtime changed.

## Medical transfer blockers

| Class | Severity | Evidence | Next action | Blocks transfer? |
| --- | --- | --- | --- | --- |
| MEMORY_ARCHITECTURE_GAP | High | LongMem mechanism gate remains NO; revision identity/materialization and CURRENT/AS_OF/CHANGE are not frozen. | Finish reviewed frozen-ten diagnostic chain; freeze deterministic contracts. | Yes |
| PROTOCOL_DECISION | High | Timestamp authority, question-time now, language arm, clean/noisy arm, and controlled evaluator need explicit choices. | Freeze MED-M0 protocol after mechanism gate; do not tune on labels. | Yes |
| MISSING_PROVENANCE | Medium | No direct source-turn/event IDs; four unresolved source-session joins; no direct noise-to-query gold mapping. | Use disclosed session-level proxies; define evaluator provenance and runtime exposure telemetry. | Blocks evidence-level claims, not all transfer |
| MODEL_FAIRNESS | High | Official baselines use heterogeneous readers, embeddings, prompts, context and hosted services. | Separate OFFICIAL_HISTORICAL from CONTROLLED_LOCAL_TRANSFER; adapter compatibility audit. | Yes for controlled comparison |
| PUBLIC_LABEL_EXPOSURE | Medium | Answers and provenance are public; HF viewer rendered sample dialogue/annotation. | Mark PUBLIC_EXTERNAL_TRANSFER; freeze methods before scoring; no blind-test claim. | No, but limits claim |
| INFRASTRUCTURE | Medium | Four submodules uninitialized; judge retry/failure may become quality zero; mirror licensing mismatch. | Pin/init only in a later approved environment; separate judge infra failures; resolve licensing before redistribution. | Blocks affected baselines/redistribution |
| ADAPTER_ONLY | Medium | Stable session mapping, order, per-persona reset, checkpoint stream, noise provenance not implemented. | Implement and test in MED-M0 only after both gates. | Yes |

## Non-goals and gate meaning

No baseline, adapter, Memory write, query scoring, LLM/API/embedding call, LongMemEval 102 DEV run, MEM-3B work, or ESL dataset download occurred in MEM-B0. MEDMEMORYBENCH_COMPATIBILITY_AUDITED=YES means the source, dataset, evaluator, streaming, provenance, safety boundary, and known blockers have been audited sufficiently to hand off for Reflection. It is not a compatibility pass for every baseline, a transfer result, or permission to start MED-M0.

## Source files inspected

The pinned source audit covered README and repository metadata (.gitmodules, LFS attributes/pointers), configs/dataset_config/medmemorybench.yaml, configs/method_config, benchmarks/medmemorybench/dataset.py and evaluator.py, metrics/__init__.py, metrics/string_match.py, metrics/llm_judge.py, utils/prompts_judge.py, utils/llm_client.py, and generation/augmentation noise injectors. Dataset payloads were traversed only for structural schema, IDs, counts, and hashes; no answer text was emitted or used for method selection.
