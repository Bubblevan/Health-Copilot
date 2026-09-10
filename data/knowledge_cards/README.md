# Knowledge cards

M0 使用“一份 JSON 文件对应一个知识单元”的格式。字段示例见
[`_schema.example.json`](_schema.example.json)。该文件只是 schema 示例，不是
临床证据；不要把它当作真实医疗资料使用。

真实卡片进入目录前必须满足：

- 内容来自可核验的公开来源，并保留 `source_url`、发布/采集/审核时间和版本；
- 明确适用人群和“患者教育、不用于诊断或处方”的范围；
- 由人工审核后再设置 `reviewed_at` 和 `reviewer`；
- 不含真实患者信息、未经许可的抓取数据或密钥。

loader 会读取目录中的所有 `*.json`，遇到 malformed JSON、缺字段、重复 ID、空内容
或非 HTTP(S) 来源会直接失败，不会静默跳过。
