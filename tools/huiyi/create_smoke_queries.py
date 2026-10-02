"""Materialize the hand-curated, source-grounded HY-DATA-0 smoke query set."""

from __future__ import annotations

import json
import sys

from _common import DATA_ROOT, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "src"))

from health_ai_copilot.huiyi.schema import source_id_for_url

BASE = "https://yk.huiyi9e.com"

# Each query was written against a page included in the captured official-site
# corpus. Keep the source URLs here so expected IDs remain deterministic.
CASES: tuple[tuple[str, str, str], ...] = (
    ("恩施慧宜眼科医院属于什么类型的医院？", "hospital_info", "/about/yyjs.htm"),
    ("恩施慧宜眼科医院目前开放多少张床位？", "hospital_info", "/about/yyjs.htm"),
    ("医院目前有多少名职工？", "hospital_info", "/about/yyjs.htm"),
    ("恩施慧宜眼科医院金龙大道院区的地址是什么？", "hospital_info", "/Service/line.htm"),
    ("在恩施市内可以乘坐哪些公交线路到医院？", "hospital_info", "/Service/line.htm"),
    ("慧宜眼科医院的办院理念是什么？", "hospital_info", "/about/wenhua.htm"),
    ("医院公开介绍了哪些眼科手术和治疗项目？", "hospital_info", "/about/yyjs.htm"),
    ("白内障专科开展哪些手术？", "department", "/info/1022/1018.htm"),
    ("青光眼专科有哪些检查和手术项目？", "department", "/info/1022/1012.htm"),
    ("近视防控专科有哪些儿童验配项目？", "department", "/info/1022/1023.htm"),
    ("屈光诊疗中心常规开展哪些屈光手术？", "department", "/info/1022/1027.htm"),
    ("角膜病与眼表专科诊治哪些疾病？", "department", "/info/1022/1004.htm"),
    ("视光、斜弱视与小儿眼科提供哪些服务？", "department", "/info/1022/1008.htm"),
    ("神经眼科关注哪些视神经和视路疾病？", "department", "/info/1022/1003.htm"),
    ("泪道与整形专科能诊治哪些问题？", "department", "/info/1022/1006.htm"),
    ("综合眼病专科处理哪些常见眼病？", "department", "/info/1022/1002.htm"),
    ("眼底病专科开展哪些眼底病治疗？", "department", "/info/1022/1014.htm"),
    ("杨昊医生擅长诊治哪些眼科疾病？", "doctor_profile", "/info/1021/1530.htm"),
    ("包煜芝医生公开介绍的主要擅长是什么？", "doctor_profile", "/info/1021/1236.htm"),
    ("芦晓磊医生的主要擅长领域有哪些？", "doctor_profile", "/info/1021/1237.htm"),
    ("张晓峰医生主要擅长哪些眼科治疗？", "doctor_profile", "/info/1021/1527.htm"),
    ("向雪梅医生提供哪些视光相关服务？", "doctor_profile", "/info/1021/1534.htm"),
    ("施继光医生公开简介提到哪些专业方向？", "doctor_profile", "/info/1021/1531.htm"),
    ("RGP眼镜是什么，哪些人群适合配戴？", "patient_education", "/info/1024/1573.htm"),
    ("暑期护眼文章建议怎样安排用眼和户外活动？", "patient_education", "/info/1024/1568.htm"),
    ("宝宝歪头可能和哪些眼科视觉问题有关？", "patient_education", "/info/1024/1566.htm"),
    ("近视手术前需要做哪些准备？", "perioperative_instruction", "/info/1025/1102.htm"),
    ("近视手术前后大约需要安排几天？", "perioperative_instruction", "/info/1025/1102.htm"),
    ("近视手术后多久可以正常用眼？", "perioperative_instruction", "/info/1025/1102.htm"),
    ("近视手术后能否进行重体力训练？", "perioperative_instruction", "/info/1025/1101.htm"),
    ("血糖控制得好还需要做糖尿病视网膜眼底检查吗？", "faq", "/info/1025/1110.htm"),
    ("突然出现眼前闪光感可能是什么原因？", "faq", "/info/1025/1111.htm"),
    ("孩子患斜视后，页面建议先了解哪些信息？", "faq", "/info/1025/1121.htm"),
    ("怎样分辨真性近视与假性近视？", "faq", "/info/1025/1104.htm"),
    ("人工晶体是否需要清洗或者更换？", "faq", "/info/1025/1107.htm"),
    ("弱视治愈后会不会复发，页面提到怎样预防？", "faq", "/info/1025/1120.htm"),
    ("眼部受伤后没有明显视力下降和疼痛，还需要检查吗？", "faq", "/info/1025/1126.htm"),
    ("眼睑长了痒的小包块可以挤吗？", "faq", "/info/1025/1125.htm"),
    ("经常散瞳对眼睛有没有影响？", "faq", "/info/1025/1105.htm"),
    ("什么是人工晶体植入手术？", "faq", "/info/1025/1108.htm"),
    ("眼前总有蚊虫样黑影舞动可能是什么情况？", "faq", "/info/1025/1109.htm"),
    ("什么情况下可能诱发青光眼发作？", "faq", "/info/1025/1116.htm"),
    ("医院答疑页面如何描述白内障手术效果？", "faq", "/info/1025/1106.htm"),
    ("做完近视手术可以参军吗？", "faq", "/info/1025/1098.htm"),
    ("眼角长了息肉样组织是否正常，需要处理吗？", "faq", "/info/1025/1123.htm"),
    ("长时间使用电脑导致眼睛疲劳怎么办？", "faq", "/info/1025/1122.htm"),
    ("妊娠会对青光眼产生影响吗？", "faq", "/info/1025/1115.htm"),
    ("孩子弱视戴上眼镜后还能摘吗？", "faq", "/info/1025/1118.htm"),
    ("青光眼会遗传吗？", "faq", "/info/1025/1113.htm"),
    ("高度近视并伴有散光需要检查眼底吗？", "faq", "/info/1025/1112.htm"),
    ("孩子放学回家眼睛变红了怎么办？", "faq", "/info/1025/1124.htm"),
    ("青光眼可以预防吗？", "faq", "/info/1025/1114.htm"),
    ("什么是医学验光？", "faq", "/info/1025/1103.htm"),
    ("ICL和飞秒手术有什么区别？", "faq", "/info/1025/1099.htm"),
)


def main() -> int:
    catalog = json.loads((DATA_ROOT / "source_catalog.json").read_text(encoding="utf-8"))
    catalog_urls = {row["url"] for row in catalog}
    chunks = [json.loads(line) for line in (DATA_ROOT / "chunks" / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    source_types = {row["source_id"]: row["document_type"] for row in chunks}
    if len(CASES) < 30:
        raise ValueError("The smoke set must have at least 30 curated queries")
    output: list[dict[str, object]] = []
    for ordinal, (query, category, path) in enumerate(CASES, start=1):
        url = BASE + path
        source_id = source_id_for_url(url)
        if url not in catalog_urls or source_id not in source_types:
            raise ValueError(f"Smoke query source is not in the built corpus: {url}")
        actual_type = source_types[source_id]
        if actual_type != category:
            raise ValueError(f"Smoke category mismatch for {url}: expected {category}, got {actual_type}")
        output.append({
            "query_id": f"hy-smoke-{ordinal:03d}",
            "query": query,
            "expected_source_ids": [source_id],
            "category": category,
        })
    output_path = DATA_ROOT / "eval" / "smoke_queries.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in output),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"query_count": len(output), "categories": sorted({row["category"] for row in output}), "path": str(output_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
