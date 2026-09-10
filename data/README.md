# Data boundary

- `raw/`：本地临时资料，不提交到 Git。
- `processed/`：派生数据，不提交到 Git。
- `knowledge_cards/`：可提交的、带来源和版本号的公开资料卡。
- `source_catalog.json`：人工选择的官方页面白名单，供 `tools/fetch_m0_data.py` 使用。
- `evals/`：人工构造或经许可使用的评测样例；不得含可识别个人信息。

`artifacts/` 默认被 Git 忽略。抓取候选和外部 benchmark 只落在这里，不会因为运行
工具而进入提交历史。

任何材料进入知识库前都需要记录：来源 URL、发布日期、采集日期、适用人群、审核人、版本和失效日期。
