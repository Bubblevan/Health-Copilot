# Knowledge cards

M0 使用“一份 JSON 文件对应一个知识单元”的格式。字段示例见
[`_schema.example.json`](_schema.example.json)。该文件只是 schema 示例，不是
临床证据；不要把它当作真实医疗资料使用。

真实卡片进入目录前必须满足：

- 内容来自可核验的公开来源，并保留 `source_url`、发布/采集/审核时间和版本；
- 明确适用人群和“患者教育、不用于诊断或处方”的范围；
- 由人工审核后再设置 `reviewed_at` 和 `reviewer`；
- 不含真实患者信息、未经许可的抓取数据或密钥。

当前 M0.2 Knowledge Pack 包含 30 张中文优先的高血压患者教育卡。它们来自 WHO、美国
CDC 和国家卫生健康委公开页面，按一个可独立引用的事实拆分，并在
`2026-09-15` 做了人工来源核对。`manual-source-check-2026-09-15` 表示逐卡核对了来源
页面与改写内容，不表示临床专家背书或临床验证；WHO 的全球口径和 CDC 的美国指南口径
保留为不同卡片，不能不加范围地混合回答。

`_schema.example.json` 是目录内唯一的文档示例文件，不参与 loader 和 BM25 索引；其余
所有 `*.json` 都必须是可加载的知识卡，错误不会被静默跳过。

loader 会读取目录中的所有 `*.json`，遇到 malformed JSON、缺字段、重复 ID、空内容
或非 HTTP(S) 来源会直接失败，不会静默跳过。

## M3 reviewed capability manifest

`../knowledge_scope.json` 是 M3 的显式、版本化 capability source of truth。它将本目录全部产品
KnowledgeCard 的 source ID 映射到人工审核 topic；runtime 不从自由文本 tags 推导 topic membership。
scope loader 会拒绝空 scope/version、重复 topic、未知或重复 topic source ID、空 topic，以及未被任一
topic 覆盖的产品 card。它描述的是当前 reviewed closed KnowledgeCard corpus，不是完整高血压知识库。
`knowledge_scope.json` 的 `reviewed_at` 与 `reviewer` 记录 capability mapping review，
不表示临床专家验证。
