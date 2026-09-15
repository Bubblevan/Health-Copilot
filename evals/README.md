# M0 Evaluation

`m0.jsonl` 是 M0.3 Eval Pack，当前包含 80 条人工构造案例。它们不含个人信息，也不
提供自动生成的“专家 gold answer”。患者教育案例的 `expected_source_ids` 是基于当前
M0.2 Knowledge Pack 的人工来源映射；urgent、prescription 和 unanswerable 案例用于
policy regression 或 OOD 检索诊断。

新增案例前应记录问题设计依据、来源卡片版本和数据许可情况。`status=reviewed` 表示
已检查字段、路由意图和 expected source mapping，不表示临床验证。

M0 的离线评测只覆盖可确定验证的能力：

- `safety_route_accuracy`；
- 当案例提供 `expected_source_ids` 时的 retrieval `Hit@1`、`Hit@3` 和 `MRR`。

`safety_route_accuracy` 只评估 deterministic safety gate：安全检查拦截的问题，
应得到对应的 `URGENT_CARE` 或 `HUMAN_REVIEW`；安全检查放行的问题在该指标中记为
`ANSWER`。它不是包含检索、生成和拒答的 end-to-end route accuracy，因此
`unanswerable` 案例不进入这个指标。

M0.3 Eval Pack 的检索指标只统计带有人工核对 `expected_source_ids` 的
`patient_education` 案例：`Hit@K` 判断前 K 名是否包含任一预期来源，`MRR` 取首个预期
来源的倒数排名。没有 `expected_source_ids` 的案例不会被伪造为“可评测答案”。

案例中的 `challenge_type` 用于 failure analysis，而不是直接作为失败标签：

- `segmentation_failure`
- `synonym_paraphrase`
- `overly_generic_term`
- `source_overlap`
- `source_conflict`
- `ood_false_retrieval`

只有在实际运行 BM25 并检查 ranked evidence 后，才把案例归入最终 failure table。

M0 不使用 LLM-as-Judge，也不声称测量诊断正确率、临床安全性或语义蕴含。未来的
grounding、abstention 和失败分类评测必须建立在可复现的 trace 和人工审核标准上。

## 外部 benchmark

`tools/fetch_m0_data.py benchmarks` 会把 HealthBench 和 MIRAGE 下载到被 Git 忽略的
`artifacts/benchmarks/`。它们不是 M0 自建 `expected_source_ids` 的替代品：HealthBench
更适合后续回答质量、安全与沟通评测；MIRAGE 更适合比较医学 RAG 的检索/语料组合。
在使用或再分发前，先阅读各自上游的数据和代码许可。

BEIR NFCorpus 是单独的标准检索轨道。运行
`python -m health_ai_copilot.eval.nfcorpus --data-dir artifacts/benchmarks/nfcorpus`
可在本地 corpus/queries/qrels 上测当前 BM25 的 Recall@K、MRR 和 nDCG@K。它与
HealthBench 的 rubric 以及产品 policy regression set 保持不同数据模型。
