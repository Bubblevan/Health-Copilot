# U0 External Medical Corpus Gap Audit

## 结论

本阶段不下载、不构造、不索引 corpus。E2-A eligibility 仍为 PUBLIC_HEALTH=ELIGIBLE，GUIDELINE/LITERATURE=NOT_YET_ELIGIBLE。机器上有文件或检索结果不能自动证明可用于 unified evidence。ESL 的 longitudinal patient history 与 external evidence 必须存于不同 namespace。

## 本地资产

| 资产 | 可见 provenance / licence | U0 结论 |
|---|---|---|
| R2MED | D:/MyLab/Jianli/external/rag/R2MED；repo commit 11244a4925a39082967a6c9d38ef01f279c316a5；MIT code；working tree 有 untracked src/__pycache__/；本地可见 Biology-demo；HF Biology data card 声明 CC BY 4.0；data revision 未固定 | 可作 RAG subsystem/reference；尚无统一文档级许可 manifest，不能当现成 guideline corpus |
| MIRAGE | D:/MyLab/Jianli/external/rag/MIRAGE；repo commit b90fa616555199a0e4ed9b64d8c7224543d452a6，clean；code Apache-2.0；nlpai-lab/mirage HF 卡标 Apache-2.0，context pools 来自 Wikipedia QA sources；dataset revision 未单独锁定 | 可作 RAG benchmark characterization，不是医疗 guideline/literature corpus；底层 source terms 不能由代码 licence 推导 |
| NFCorpus | 本地 external/rag 顶层只有 FlashRAG、MIRAGE、R2MED；未发现独立 NFCorpus checkout 或文件 | 无本地 revision/许可清单，不能复用或下载 |
| MedRAG assets | 未发现独立 MedRAG repo/corpus；external/rag 当前只有 FlashRAG、MIRAGE、R2MED。已有 HC research result artifacts 不等于可复用 source corpus | 没有冻结的 document IDs、revision/hash、逐文许可或再索引授权；不可视为现成 evidence surface |
| 已批准 PUBLIC_HEALTH | E2-A 已标记 eligible，本轮未读改语料或实现 | 未来按既有 eligibility artifact、revision、更新策略和许可 manifest 引用，不扩写为 guideline/literature |

Primary source links: [R2MED code](https://github.com/R2MDE/R2MED), [R2MED Biology card](https://huggingface.co/datasets/R2MED/Biology), [MIRAGE code](https://github.com/JohnnyNLP/MIRAGE), [MIRAGE card](https://huggingface.co/datasets/nlpai-lab/mirage). Data-card license 是上游声明，并非 underlying sources 全部授权的法律意见。

## Guideline 最小需求

每份 guideline 必须有 publisher/issuing body、jurisdiction、标题、版本、发布日期、生效/撤回日期、stable identifier/URL 与明确许可。首个 prototype 应只选一个目标辖区，优先该辖区的卫生主管机关/国家 guideline body，再按任务加入少量专业学会；不能把不同辖区指南混成无标签的单一权威源。具体 publisher 需在目标部署辖区确认后冻结，本轮不假设用户的最终辖区。许可 review 必须区分存储、chunk/index、模型 API 片段传输、派生标注、benchmark 再分发、训练用途。新版本新增 snapshot，不能覆盖历史。

## Literature 最小需求

先评估可许可的 metadata/abstract surface；full text 仅在允许保存、索引、API 传输和目标用途时纳入。每文保存 PMID/DOI/PMCID、publication type、日期、retraction/update status、license、section/page offset。RCT/review/cohort 应作为 evidence metadata 分层管理。

## 进入 prototype 的门槛

1. 单独 namespace 与 document-level license/source record；
2. version、file hash、jurisdiction、日期、撤回状态可重放；
3. chunk 精确回溯 document/offset；
4. Single 与 Team 使用相同 evidence snapshot/预算；
5. gated/PHI/API transfer 限制审核完成；
6. U0 gap analysis 不等于 corpus eligible。

ScientificData2021_HealthGym 是旧 HealthGym provenance，不是 2026 Healthcare AI GYM 的官方实现；本轮未评估后者。
