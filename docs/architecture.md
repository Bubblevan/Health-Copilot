# Architecture

## M0 vertical slice

```text
User question
  -> basic input validation
  -> deterministic safety gate
       -> urgent care / human review (terminal)
       -> retrieval-eligible question
  -> KnowledgeCard loader
  -> BM25 retrieval
  -> EvidenceBundle (Evidence[])
  -> injected Generator
  -> citation-ID verifier
       -> answer with stored Citation metadata
       -> abstain on insufficient evidence or invalid citation
  -> AssistantResponse
```

安全路由先于检索和模型调用。Generator 只接收本次检索到的 Evidence，模型不能选择任意
URL 或伪造来源；Citation 的 metadata 从 Evidence 复制，不信任模型生成的 metadata。

## M1 bounded recovery slice

M1 保留上述安全门和初始检索，并只在初始证据非空后进入 Agent：

```text
question
  -> input validation
  -> deterministic safety gate
       -> urgent care / human review (terminal, no Agent call)
       -> retrieval-eligible question
  -> initial BM25 retrieval
       -> empty (terminal ABSTAIN, no Agent call)
  -> AgentSession + initial Evidence
  -> AgentLoop, max_model_turns=2, max_tool_calls=1
       -> structured final turn
            -> verify citations against initial + successful tool Evidence
       -> search_knowledge(query)
            -> ToolResult observation
            -> second model turn
       -> budget/model/tool failure (ABSTAIN)
```

`AgentSession` 只保存这一次运行的 typed transcript；`AgentState` 保存当前 turn/tool 计数、
停止原因、`initial_ranked_evidence`、`recovery_ranked_evidence` 和去重后的 observed evidence。
`ToolRegistry` 明确注册 `search_knowledge`，不做
目录扫描、插件发现或 YAML 自动加载。工具失败是结构化 observation，允许第二个 model turn
看到失败并 abstain；如果最终没有可靠的结构化 final turn，运行时 fail closed。

## Project structure

| Directory | Responsibility |
| --- | --- |
| `src/health_ai_copilot/contracts.py` | `Route`、KnowledgeCard、Evidence、Generator draft 和响应契约 |
| `src/health_ai_copilot/knowledge/` | JSON knowledge-card 校验和确定性加载 |
| `src/health_ai_copilot/retrieval/` | 中文分词和直接实现的 BM25 baseline |
| `src/health_ai_copilot/generation/` | 最小 Generator Protocol 和 OpenAI-compatible provider |
| `src/health_ai_copilot/agent/` | typed message、session/state、受限 AgentLoop、事件和模型协议 |
| `src/health_ai_copilot/tools/` | M1 唯一的只读 `search_knowledge` 工具 |
| `src/health_ai_copilot/verification/` | citation-ID 完整性校验 |
| `src/health_ai_copilot/pipeline.py` | M0/M1 安全、检索、生成与 Agent 组件编排 |
| `tests/` | 不联网的 synthetic 单元/集成测试 |
| `evals/` | M0 评测 schema 和离线评测入口 |

## M0 invariants

1. Safety routing happens before retrieval or model calls.
2. The model only sees the retrieved Evidence bundle.
3. A normal answer needs at least one citation ID verified against that bundle.
4. Empty evidence, generator abstention and invalid citations fail closed to `ABSTAIN`.
5. Source metadata is separate from model logic and is copied from stored objects.
6. Tests do not require an API key, network or real patient data.

## M1 invariants

1. Safety routing happens before Agent model or tool calls.
2. Initial lexical retrieval remains deterministic and empty evidence still abstains before Agent.
3. Tool choice and arguments are structural provider tool calls, never prose parsing.
4. Product limits are deterministic: at most two model turns and one retrieval tool call.
5. Final citation verification receives the union of initial and successful recovery evidence,
   deduplicated by `source_id`.
6. Budget exhaustion and provider/tool failures never force a medical answer.

`HealthCopilotPipeline` 要求 `generator` 与 `agent_model` 二选一；两者同时传入会抛出配置
错误。Agent tool arguments 使用 M1 所需的有限 object/string schema 校验子集，不宣称支持
完整 JSON Schema。

M0 为保持现有 `AssistantResponse` contract，字段名仍是 `safety_reasons`；但 pipeline
目前也会在其中记录 `insufficient_evidence`、`retrieval_error`、`generation_error` 和
`invalid_citation` 等 status reason。M1 保持该兼容字段；后续里程碑再演进为 `reasons` 或
`status_reasons`，按 safety 与 runtime failure domain 分离，避免字段名产生误导。

## M4 budgeted and replayable runtime

```text
domain adapter -> ProviderExecutor -> RunBudget guard -> provider / recorded response
AgentLoop tool proposal -> ToolRunner -> RunBudget guard -> registry / recorded tool result
                                       \-> metadata trace events
```

`RunContext` 是 execution-local control-plane state，而非 memory。默认 trace 不持久化用户内容；
public evaluation recording 必须显式 opt-in。Replay 对 canonical request fingerprint 做严格匹配，并以
recorded provider/tool outputs 重跑相同 pipeline branch；mismatch、缺失 exchange 或 budget denial 都 fail closed。
M4 没有增加 product tool，也没有修改 M0–M3 wire contracts。

## Later milestones

后续再考虑 permission、production persistence、dense/hybrid retrieval、multimodal evidence 与 post-training；
它们都必须先有 baseline、
失败案例、ablation 和可复现评测，再增加复杂度。

## M2 evidence policy and grounding

```text
M1 tool proposal -> EvidencePolicy
  recoverable -> execute search_knowledge -> second model turn
  sufficient  -> policy_denied observation -> second model turn
  insufficient/conflicting/error -> ABSTAIN
grounded final -> claim citation integrity -> coverage + grounding verifier -> response / ABSTAIN
```

The policy is called only for a proposed `search_knowledge`, never for direct finals, safety routes,
or empty retrieval. M2 non-abstaining finals contain `GroundedClaim(text, citation_ids)` entries. The
runtime verifies every claim citation against observed evidence before the semantic verifier, requires
coverage of substantive answer claims, and materializes citation metadata solely from observed Evidence.
Unsupported, contradicted, uncovered, malformed, or verifier-error results fail closed. The generator
and verifier may use the same configured model; that is recorded as same-model verification, not an
independent judge.

## M3 capability-aware, claim-first path

```text
initial evidence -> Agent proposal -> EvidencePolicy + reviewed KnowledgeScope
  recoverable + valid topic -> same search_knowledge -> second Agent turn
  all other decisions / invalid topic -> deny or abstain
claim-first final -> citation integrity -> claim support verifier -> deterministic materializer -> response
```

`data/knowledge_scope.json` is the explicit reviewed source of truth for the closed corpus capability.
It is not inferred from tags at runtime. `SearchKnowledgeTool` carries only narrow metadata tying its
single name to scope ID/version/domain/topic IDs; this is not a plugin system or a second tool. M1 and
M2 instantiate the tool without a scope and retain their existing tool description and semantics.

M3 `EvidenceAssessment` adds optional `matched_topic_ids`. The runtime validates every ID against the
active scope and requires a non-empty valid set for `RECOVERABLE`, including fake policies in tests.
The Agent event stop reason remains `FINAL` after a final turn; a later claim-support/materialization
failure is stored separately as the harness disposition.

M3 has no semantic coverage verifier in its critical path. The model may internally carry an `answer`
field for compatibility, but the pipeline ignores it: only verified, whitespace-normalized, exact-deduped
claim text is rendered as ordered bullets, and citations are reconstructed from observed Evidence.

Claim support uses **single-call batched cited-evidence binding**. The payload explicitly associates each
`claim_index` with `cited_evidence`; observed evidence that no claim cites is excluded. This narrows the
verifier's evidence set without claiming cryptographic or physical per-claim isolation: a shared verifier
context can still contain another claim's cited evidence. A physically isolated design would require one
verifier call per claim and therefore additional model round trips; M3 keeps one bounded verifier call.
