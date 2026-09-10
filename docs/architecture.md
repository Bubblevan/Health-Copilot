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

## Project structure

| Directory | Responsibility |
| --- | --- |
| `src/health_ai_copilot/contracts.py` | `Route`、KnowledgeCard、Evidence、Generator draft 和响应契约 |
| `src/health_ai_copilot/knowledge/` | JSON knowledge-card 校验和确定性加载 |
| `src/health_ai_copilot/retrieval/` | 中文分词和直接实现的 BM25 baseline |
| `src/health_ai_copilot/generation/` | 最小 Generator Protocol 和 OpenAI-compatible provider |
| `src/health_ai_copilot/verification/` | citation-ID 完整性校验 |
| `src/health_ai_copilot/pipeline.py` | M0 组件编排，不实现组件内部逻辑 |
| `tests/` | 不联网的 synthetic 单元/集成测试 |
| `evals/` | M0 评测 schema 和离线评测入口 |

## M0 invariants

1. Safety routing happens before retrieval or model calls.
2. The model only sees the retrieved Evidence bundle.
3. A normal answer needs at least one citation ID verified against that bundle.
4. Empty evidence, generator abstention and invalid citations fail closed to `ABSTAIN`.
5. Source metadata is separate from model logic and is copied from stored objects.
6. Tests do not require an API key, network or real patient data.

## Later milestones

M1 可以增加 typed Agent Core，但不是本阶段的一部分；后续再考虑 policy/budget/timeout/
recovery/trace/replay、可插拔 runtime、Agent Team、context/memory、dense/hybrid retrieval、
multimodal evidence 和 post-training。每个里程碑都应先有 baseline、失败案例、ablation 和
可复现评测，再增加复杂度。
