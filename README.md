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

## 快速开始

在仓库根目录建立并使用 uv 环境：

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
& .\.venv\Scripts\Activate.ps1
pytest -q
ruff check .
```

运行确定性离线评测（当前 `m0.jsonl` 仅包含 schema 示例）：

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
  --knowledge-dir data/knowledge_cards `
  --question "你的患者教育问题"
```

没有 API 配置时，确定性测试仍可完整运行；live demo 会给出配置错误，不会伪装成离线成功。

## 数据边界

`data/knowledge_cards/` 中的 JSON 是 source-versioned knowledge unit。一份文件对应一张
卡片，格式见 [`_schema.example.json`](data/knowledge_cards/_schema.example.json)。该示例
不是临床证据。真实卡片必须保留可核验 URL、发布/采集/审核时间、审核人、适用人群和版本。

测试用 synthetic cards 位于 `tests/fixtures/knowledge_cards/`，不能当作真实医学资料。

官方资料采集工具位于 `tools/fetch_m0_data.py`：`crawl` 按
`data/source_catalog.json` 抓取短候选片段和 provenance，供人工改写成 atomic
KnowledgeCard；`benchmarks` 将 HealthBench 与 MIRAGE 下载到 Git 忽略的
`artifacts/benchmarks/`。工具不会把整页 HTML 或未经审核的内容写入
`data/knowledge_cards/`。

标准检索评测使用 BEIR NFCorpus：下载并解压后可运行
`python -m health_ai_copilot.eval.nfcorpus --data-dir artifacts/benchmarks/nfcorpus`，
输出 Recall@K、MRR 和 nDCG@K。NFCorpus 的 qrels 保留在独立的 retrieval-eval adapter
中，不会被伪装成产品 KnowledgeCard。

## 尚未实现

M0 明确不包含 Agent Loop / ReAct、Agent Runtime、Multi-Agent、Memory、MCP、Sandbox、
Milvus、Qdrant、dense retrieval、reranker、web search、VLM、SFT/DPO/RL 和任何临床验证。
这些是后续里程碑，不能在简历或 README 中提前宣称已经具备。

## 设计限制

M0 的 citation verifier 只验证模型返回的 ID 是否属于本次检索到的 Evidence，并从存储的
Evidence 复制标题、摘要和 URL；它不证明每个自然语言 claim 与引用之间存在语义蕴含关系。
claim-level grounding、trace/replay 和完整 Harness Runtime 留到后续阶段。

为保持 M0 的兼容性，`AssistantResponse.safety_reasons` 当前同时承载 safety reason 和
pipeline status reason（如 `retrieval_error`、`generation_error`、`invalid_citation`）。
这是已知技术债，后续 M1 可演进为更准确的 `reasons` 或 `status_reasons`，再按 failure
domain 分离。
