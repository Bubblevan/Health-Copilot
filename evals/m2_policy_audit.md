# M2 EvidencePolicy 冻结输入与标签审计

`m2_policy.jsonl` 的每一行都是 Policy 的冻结实际输入：问题、已观察的
`evidence_source_ids` 和 Agent 提出的 `proposed_query`。standalone evaluator
仅将这些 ID materialize 为 Evidence，绝不在评测时重新运行 BM25。

以下是本次语义修正改变的四个 gold 标签。其余 20 个标签未因模型预测而修改。

| Case | Old label | New label | 理由 | 冻结 evidence IDs | proposed_query |
| --- | --- | --- | --- | --- | --- |
| m2p-005 | sufficient | recoverable | 冻结证据没有 `measuring-04-position`，也未陈述“坐着休息至少 5 分钟”；查询精确、同域且可检索到缺失的测量姿势资料。 | cdc-high-blood-pressure-measuring-01-home, cdc-high-blood-pressure-measuring-03-before-reading, cdc-high-blood-pressure-measuring-02-repeat, who-hypertension-05-lifestyle, cdc-high-blood-pressure-measuring-05-white-coat | 测量血压前 坐着休息 至少5分钟 |
| m2p-010 | recoverable | sufficient | 冻结初始证据已包含 `cdc-high-blood-pressure-measuring-05-white-coat`，可直接回答读数是否受影响；再次搜索不必要。 | cdc-high-blood-pressure-measuring-05-white-coat, cdc-high-blood-pressure-measuring-03-before-reading, cdc-high-blood-pressure-risk-04-tobacco, cdc-high-blood-pressure-risk-05-overweight, cdc-high-blood-pressure-managing-04-otc | 白大衣效应 血压读数 居家监测 |
| m2p-011 | recoverable | sufficient | 冻结初始证据已包含 `nhc-hypertension-day-03-home-monitoring`，直接包含居家监测频次信息。 | nhc-hypertension-day-03-home-monitoring, cdc-high-blood-pressure-measuring-01-home, cdc-high-blood-pressure-managing-01-monitoring, nhc-hypertension-day-04-adult-screening | 居家血压监测 每天 测量 几次 |
| m2p-012 | recoverable | sufficient | 冻结初始证据已包含 `who-hypertension-03-silent` 和症状卡，可直接回答通常是否无症状。 | cdc-high-blood-pressure-about-02-symptoms, who-hypertension-03-silent, cdc-high-blood-pressure-managing-01-monitoring, cdc-high-blood-pressure-about-01-us-threshold, nhc-hypertension-day-03-home-monitoring | 高血压 无症状 常见 |

三个 `recoverable_paraphrase` 案例（m2p-007 至 m2p-009）的查询来自已保存的
focused trajectory 中同类 Agent rewrite，并经人工复核为语义对齐的同域 recovery；
它们均不等于原问题。m2p-005 的 measurement rewrite 亦经人工复核。
