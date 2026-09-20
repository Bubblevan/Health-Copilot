# Health-Copilot

面向医疗健康场景的 **Safety-Gated Evidence RAG vertical slice**。医疗场景是验证
安全边界、证据追溯和拒答机制的测试环境；本项目不是诊断、处方或互联网诊疗系统。

> 不接入真实患者数据，不提供诊断、处方或治疗决策。知识卡必须来自可核验的公开来源，
> 评测数据不得包含可识别个人信息。

## M0 已实现

```text
User question
  -> deterministic safety gate
  -> reviewed JSON knowledge cards
  -> Chinese-first BM25 retrieval
  -> evidence-aware generator
  -> deterministic citation-ID verification
  -> AssistantResponse
```

- 高危问题在检索和模型调用前路由到 `URGENT_CARE`；
- 处方、剂量、停药和加药请求路由到 `HUMAN_REVIEW`；
- 无证据、模型主动 abstain 或伪造引用时 fail closed 到 `ABSTAIN`；
- KnowledgeCard loader 严格校验来源、版本、审核字段、重复 ID 和 JSON 格式；
- BM25 直接实现，使用 `k1=1.5`、`b=0.75`，不把分数当作概率；
- 生成器通过最小 Protocol 注入，测试使用 FakeGenerator，不需要网络或 API Key；
- OpenAI-compatible generator 只从环境变量读取配置，默认温度为 `0.1`；
- CLI 支持真实本地知识卡和可选的 OpenAI-compatible live demo；
- 离线评测器只计算 `safety_route_accuracy` 和有标注来源时的 retrieval Hit@K；前者
  只评估 safety gate，不是 end-to-end route accuracy。

## M1 已实现：受限 Agent Recovery

M1 在冻结的 M0 安全边界之内增加一个小型、单 Agent 运行时：

- `AgentState`、`AgentSession`、typed message 和 `StopReason`；
- 显式 `ToolSpec`、参数校验、`ToolRegistry` 和结构化 `ToolResult`；
- 最多 2 次模型 turn、最多 1 次工具调用的确定性 `AgentLoop`；
- 唯一产品工具 `search_knowledge(query)`，只读包装现有 BM25；
- 初始证据与恢复检索证据按 `source_id` 去重后统一做 citation verification；
- 运行结果同时保存初始 ranked evidence、恢复 ranked evidence 和最终 evidence union；
- `agent_start`、`turn_start`、`model_response`、`tool_start`、`tool_end`、`turn_end`、
  `agent_end` 内存事件；事件只携带元数据，不携带问题、工具参数或回答内容。

控制流是：安全门 → 初始 BM25 → 空证据则 `ABSTAIN` → 第一次 Agent turn → 可选的一次
`search_knowledge` → 工具结果作为 observation → 第二次 Agent turn → 引用校验。任何预算
耗尽、模型失败或未经观察的引用都会 fail closed；预算耗尽不会强行让模型回答。

M1 Agent autonomy 只用于已观察到的词汇/表达不匹配恢复，不是自主医疗诊断 Agent，也不
声称临床验证。`AgentSession` 只是一次运行的内存 transcript，不是长期记忆。

## 快速开始

在仓库根目录建立并使用 uv 环境：

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
& .\.venv\Scripts\Activate.ps1
pytest -q
ruff check .
```

运行确定性 M0 离线评测（`m0.jsonl` 包含 80 条 reviewed cases）：

```powershell
python -m health_ai_copilot.eval.runner `
  --knowledge-dir data/knowledge_cards `
  --dataset evals/m0.jsonl
```

运行 live CLI 前设置：

```powershell
$env:HEALTH_COPILOT_API_KEY = "..."
$env:HEALTH_COPILOT_BASE_URL = "https://your-openai-compatible-endpoint/v1"
$env:HEALTH_COPILOT_MODEL = "your-model"
python -m health_ai_copilot.cli `
  --mode m0 `
  --knowledge-dir data/knowledge_cards `
  --question "你的患者教育问题"
```

将 `--mode m0` 改为 `--mode m1` 可演示受限 Agent recovery；`--mode m2` 额外启用
EvidencePolicy 与 claim grounding verifier。M2 的 policy/verifier 使用同一套 API key/base URL，
模型可分别由 `HEALTH_COPILOT_POLICY_MODEL`、`HEALTH_COPILOT_VERIFIER_MODEL` 指定，缺省时
回退到 `HEALTH_COPILOT_MODEL`；两者均为 temperature 0。M1/M2 离线 mechanics 测试不需要 API Key。
`--mode m3` 加载人工审核的 `data/knowledge_scope.json`，让同一个 `search_knowledge`
工具显式暴露 closed-corpus capability，并使用 claim-first final contract。

没有 API 配置时，确定性测试仍可完整运行；live demo 会给出配置错误，不会伪装成离线成功。

## 数据边界

`data/knowledge_cards/` 中的 JSON 是 source-versioned knowledge unit。一份文件对应一张
卡片，格式见 [`_schema.example.json`](data/knowledge_cards/_schema.example.json)。该示例
不是临床证据。真实卡片必须保留可核验 URL、发布/采集/审核时间、审核人、适用人群和版本。

测试用 synthetic cards 位于 `tests/fixtures/knowledge_cards/`，不能当作真实医学资料。

## M0.2 / M0.3 baseline

当前 Knowledge Pack 有 30 张中文优先的高血压患者教育卡，Eval Pack 有 80 条手工构造
案例。BM25 v0 在这次固定运行中的检索结果为：`Hit@1=0.9032`、`Hit@3=0.9516`、
`MRR=0.9274`（62 条带 `expected_source_ids` 的 patient-education cases）。这不是
临床准确率，也不是泛化能力声明；完整失败快照见
[`m0_failure_table.json`](evals/m0_failure_table.json)。

当前真正观察到的失败主要是 3 条 query-expression mismatch 和 4 条 OOD false
retrieval；segmentation、overly generic、source overlap、source conflict 在这个小样本
中暂未形成实际 miss。M1 的恢复回归集见
[`m1_recovery.jsonl`](evals/m1_recovery.jsonl)；它保留了 3 条已观察到的 paraphrase miss、
直接命中控制、4 条 OOD false-retrieval 控制以及 safety controls。恢复查询由模型的
`search_knowledge(query=...)` 参数产生，生产代码不硬编码 case ID 或 rewrite 字符串。

官方资料采集工具位于 `tools/fetch_m0_data.py`：`crawl` 按
`data/source_catalog.json` 抓取短候选片段和 provenance，供人工改写成 atomic
KnowledgeCard；`benchmarks` 将 HealthBench 与 MIRAGE 下载到 Git 忽略的
`artifacts/benchmarks/`。工具不会把整页 HTML 或未经审核的内容写入
`data/knowledge_cards/`。

标准检索评测使用 BEIR NFCorpus：下载并解压后可运行
`python -m health_ai_copilot.eval.nfcorpus --data-dir artifacts/benchmarks/nfcorpus`，
输出 Recall@K、MRR 和 nDCG@K。NFCorpus 的 qrels 保留在独立的 retrieval-eval adapter
中，不会被伪装成产品 KnowledgeCard。

## 明确不在 M1 内

M1 明确不包含 Multi-Agent、Agent Swarm、Memory、MCP、Sandbox、Milvus、Qdrant、dense
retrieval、reranker、web search、VLM、SFT/DPO/RL、流式 UI、并行工具、队列、持久化 trace
或任何临床验证。这些能力不能被 M1 的 bounded single-agent runtime 暗示为已经具备。

## M2 已实现：Evidence Policy & Grounding Harness

M2 的 `search_knowledge` 仍是唯一 Agent tool。模型提出 tool call 后，外部 EvidencePolicy
只会允许 recoverable recovery，或以 structured `policy_denied` observation 拒绝已经足够的证据；
insufficient/conflicting/policy failure 都直接 abstain。工具 proposal 和真实 execution 分别记录。

非 abstain 的 M2 final 必须提交覆盖回答中实质事实的逐项 claims 及 citation IDs。运行时先做
deterministic citation integrity，再做 coverage/support verifier；任何 fabricated citation、coverage
缺口、unsupported、contradicted 或 verifier failure 都 fail closed。该机制只校验给定 reviewed
evidence 的关系，不声称 clinical validation 或 medical correctness。

## M3 已实现：Capability-Aware Policy & Claim-First Materialization

M3 不增加 Agent tool，`search_knowledge` 仍是唯一的只读 BM25 tool。M3 读取经过审核、版本化的
`data/knowledge_scope.json`，它明确将每张产品 KnowledgeCard 映射到 closed corpus 的 capability
topic。模型会看到简洁的 scope 描述；EvidencePolicy 同时判断问题、当前 evidence、提议查询和
scope。只有 `RECOVERABLE` 且返回至少一个有效 `matched_topic_ids` 时，才会执行唯一一次 recovery
search；无效或伪造 topic ID 会 fail closed 为 policy error。

M3 final 是 claim-first：模型提交原子 claims 与 citation IDs，而不是可作为第二事实来源的自由答案。
运行时顺序为 claim validation、deterministic citation integrity、claim support verification、deterministic
materialization。用户可见的事实文本只来自已验证 claim 的原文；M3 critical path 不再调用语义
`coverage_ok` 判断。这去除了自由 answer/claim coverage mismatch 的用户可见路径，但不声称消除
hallucination、clinical validation、独立验证或完美 claim support。

Claim support 使用 **single-call batched cited-evidence binding**：verifier payload 显式绑定
每个 `claim_index` 与其 cited evidence，且没有被任何 claim 引用的 observed evidence 不会进入 payload。
这不是 cryptographic 或 physical per-claim isolation：同一 context window 仍可能包含其他 claim 的
cited evidence。若需物理隔离，则必须每个 claim 单独 verifier call，代价是额外 model round trip；M3
明确选择了单次、有界调用的 trade-off。

## M4 已实现：Budgeted & Replayable Harness Runtime

M4 为既有 M0–M3 adapters 引入统一 `ProviderExecutor` 边界和 `RunContext`：provider/tool 副作用均在
执行前经过单调 deadline、provider/tool 次数及可选 hard token budget guard。默认 `metadata_only`
JSONL trace 不保存问题、回答、claims、evidence、provider messages 或工具 query；公开、reviewed eval
才可以显式选择 content recording。严格 fingerprint 的 provider/tool replay 会重新运行 pipeline control
flow，但不会创建 live provider client 或调用 live tool；mismatch fail closed。详见
[`docs/m4_runtime.md`](docs/m4_runtime.md)。这不是 clinical validation、隐私合规认证或泛化证明。

## 设计限制

M0/M1 的 citation verifier 只验证模型返回的 ID 是否属于本次实际观察到的 Evidence，并从
存储的 Evidence 复制标题、摘要和 URL；它不证明每个自然语言 claim 与引用之间存在语义
蕴含关系。M2 已补充 claim-level grounding；M4 已提供有界 budget、metadata trace 与 public-eval
replay，但没有添加生产 conversation memory、新 Agent tool、web search 或任何临床验证。

为保持 M0 的兼容性，`AssistantResponse.safety_reasons` 当前同时承载 safety reason 和
pipeline status reason（如 `retrieval_error`、`generation_error`、`invalid_citation`）。
这是已知技术债；M1 保持该兼容字段，后续里程碑再演进为更准确的 `reasons` 或
`status_reasons`，按 failure domain 分离。
