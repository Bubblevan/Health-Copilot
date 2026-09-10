# M0 Evaluation

`m0.jsonl` 是评测数据的 schema 示例。当前文件中的样例用于说明格式，不是专家
标注，也不代表临床结论。新增案例前应记录来源、标注依据和数据许可情况，且不能
包含可识别个人信息。

M0 的离线评测只覆盖可确定验证的能力：

- safety route accuracy；
- 当案例提供 `expected_source_ids` 时的 retrieval Hit@K。

M0 不使用 LLM-as-Judge，也不声称测量诊断正确率、临床安全性或语义蕴含。未来的
grounding、abstention 和失败分类评测必须建立在可复现的 trace 和人工审核标准上。
