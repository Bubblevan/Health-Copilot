"""Build the deterministic candidate E0 research pack from declared templates.

The generated pack remains ``candidate`` until a human reviewer checks every
case.  This script does not call a provider, fetch a source or choose an
architecture winner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

TOPICS = {
    "SIMPLE_DIRECT": [
        "如何按照公开健康资料正确测量家庭血压？",
        "家庭测量血压前应保持什么姿势？",
        "一次血压读数偏高时，公开资料建议先做什么记录？",
        "家庭血压测量时袖带位置为什么重要？",
        "测量血压前短时间内应避免哪些常见干扰？",
        "如何记录连续几次家庭血压读数供后续咨询？",
    ],
    "SINGLE_SOURCE": [
        "公开患者教育资料如何解释减少钠摄入与血压管理的关系？",
        "公开患者教育资料如何解释规律活动对血压管理的帮助？",
        "公开患者教育资料如何解释保持健康体重与血压风险的关系？",
        "公开患者教育资料如何解释睡眠与血压管理的关系？",
        "公开患者教育资料如何解释烟草暴露与血压风险的关系？",
        "公开患者教育资料如何解释饮酒与血压风险的关系？",
    ],
    "BREADTH_MULTI_SOURCE": [
        "请分别概括低盐饮食、规律活动和家庭监测对血压教育的要点。",
        "请从饮食、体重和睡眠三个方面整理降低血压风险的公开建议。",
        "请分别说明活动、烟草暴露和饮酒对血压风险管理的公开信息。",
        "请整理家庭监测、复测记录和寻求专业帮助之间的患者教育关系。",
        "请从饮食、活动和体重三个证据组整理可执行的健康教育信息。",
        "请分别整理睡眠、压力管理和家庭血压记录的公开教育要点。",
    ],
    "CROSS_SOURCE": [
        "请比较全球公共卫生资料与美国公共卫生资料对家庭血压教育的共同点。",
        "请比较 WHO 与 CDC 对生活方式和血压风险的公开表述，并保留来源边界。",
        "请结合国家卫生健康委和 CDC 的资料说明家庭监测信息如何互补。",
        "请比较不同公共卫生来源对低盐饮食教育的表述，不要替它们制造一致结论。",
        "请结合 WHO 和 CDC 资料整理血压风险因素，并标注来源归属。",
        "请比较不同来源对寻求专业帮助时机的患者教育表述。",
    ],
    "CONFLICTING_EVIDENCE": [
        "当两份公开资料对家庭血压记录频率表述不同，应如何并列呈现差异？",
        "当公共卫生资料的阈值表述存在差异，应如何说明差异而不替用户诊断？",
        "当两份资料对生活方式优先级的表达不同，应如何保留证据冲突？",
        "当不同来源对血压测量前准备的细节不完全一致，应如何审慎回答？",
        "当一份资料强调风险因素而另一份侧重管理行动，应如何展示二者差别？",
        "当来源更新时间不同且建议表述不一致，应如何给出可审计的患者教育回答？",
    ],
    "TEMPORAL_EVIDENCE": [
        "请区分较早资料与较新资料中的家庭血压监测建议，并保留日期信息。",
        "请按时间顺序整理血压风险因素资料的变化，不要把新旧内容混成一条。",
        "请说明较新公共卫生通知与较早患者教育卡片之间的时间关系。",
        "请用来源日期解释为什么同一生活方式建议的措辞可能随时间变化。",
        "请比较两个时间点的血压管理公开信息，并指出需要人工确认的部分。",
        "请整理当前资料与历史资料的差异，避免把历史建议当成当前政策。",
    ],
    "SERIAL_DEPENDENCY": [
        "先确认家庭血压测量证据，再根据记录完整性说明下一步患者教育边界。",
        "先整理饮食风险证据，再说明怎样把它转成不涉及处方的健康教育。",
        "先确认来源是否覆盖问题，再决定回答还是明确说明证据不足。",
        "先比较来源日期，再解释为何不能直接合并为一个无日期的建议。",
        "先验证引用属于实际观察证据，再生成覆盖证据组的患者教育摘要。",
        "先识别冲突证据，再分别呈现来源观点并提出人工复核边界。",
    ],
    "OOD_INSUFFICIENT": [
        "仅凭这组公开患者教育资料，可以直接为我决定降压药剂量吗？",
        "仅凭一次家庭血压读数，可以确认我已经患有某种疾病吗？",
        "请根据公开教育卡片替我制定个体化处方和停药时间表。",
        "现有资料没有覆盖这个具体检查结果，我能否据此得到确定诊断？",
        "请用公开患者教育资料推断我的个人急症风险并替代就医判断。",
        "资料没有覆盖这个药物相互作用问题时，请不要猜测而是说明证据不足。",
    ],
}


SOURCE_IDS = {
    "measurement": "cdc-high-blood-pressure-measuring-01-home",
    "diet": "cdc-high-blood-pressure-prevention-01-diet",
    "activity": "cdc-high-blood-pressure-prevention-03-activity",
    "weight": "cdc-high-blood-pressure-prevention-04-sleep",
    "who": "who-hypertension-05-lifestyle",
    "nhc": "nhc-2025-national-hypertension-day-04-monitoring",
    "risk": "cdc-high-blood-pressure-risk-factors-05-overweight",
}


def _text_hash(question: str) -> str:
    normalized = re.sub(r"\s+", " ", question.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _definition(category: str, index: int) -> tuple[dict, dict]:
    if category == "SIMPLE_DIRECT":
        groups = [[SOURCE_IDS["measurement"]]]
        subtasks = [{"id": "answer", "description": "extract one directly supported education point"}]
        families = ["PUBLIC_HEALTH"]
        domains = ["blood_pressure_measurement"]
        answerability = "ANSWERABLE"
        gold = {"answerability": answerability, "required_evidence_groups": groups}
    elif category == "SINGLE_SOURCE":
        key = ["diet", "activity", "weight", "who", "risk", "diet"][index]
        groups = [[SOURCE_IDS[key]]]
        subtasks = [{"id": "answer", "description": "extract one source-sufficient education point"}]
        families = ["PUBLIC_HEALTH"]
        domains = ["lifestyle_and_risk"]
        answerability = "ANSWERABLE"
        gold = {"answerability": answerability, "required_evidence_groups": groups}
    elif category == "BREADTH_MULTI_SOURCE":
        groups = [[SOURCE_IDS["diet"]], [SOURCE_IDS["activity"]], [SOURCE_IDS["measurement"]]]
        subtasks = [
            {"id": "diet", "description": "collect diet evidence"},
            {"id": "activity", "description": "collect activity evidence"},
            {"id": "monitoring", "description": "collect monitoring evidence"},
        ]
        families = ["PUBLIC_HEALTH", "INTERNAL_REVIEWED_PATIENT_EDUCATION"]
        domains = ["lifestyle", "monitoring"]
        answerability = "ANSWERABLE"
        gold = {"answerability": answerability, "required_evidence_groups": groups}
    elif category == "CROSS_SOURCE":
        groups = [[SOURCE_IDS["who"], SOURCE_IDS["diet"]], [SOURCE_IDS["nhc"], SOURCE_IDS["measurement"]]]
        subtasks = [
            {"id": "source_a", "description": "collect first source family evidence"},
            {"id": "source_b", "description": "collect second source family evidence"},
        ]
        families = ["PUBLIC_HEALTH", "GUIDELINE"]
        domains = ["cross_source_comparison"]
        answerability = "ANSWERABLE"
        gold = {"answerability": answerability, "required_evidence_groups": groups}
    elif category == "CONFLICTING_EVIDENCE":
        groups = [[SOURCE_IDS["who"], SOURCE_IDS["measurement"]]]
        subtasks = [
            {"id": "source_a", "description": "record one source statement"},
            {"id": "source_b", "description": "record the other source statement"},
        ]
        families = ["PUBLIC_HEALTH", "GUIDELINE"]
        domains = ["evidence_conflict"]
        answerability = "ANSWERABLE_WITH_CONFLICT"
        gold = {
            "answerability": answerability,
            "required_evidence_groups": groups,
            "conflict_must_be_explicit": True,
        }
    elif category == "TEMPORAL_EVIDENCE":
        groups = [[SOURCE_IDS["nhc"], SOURCE_IDS["who"]], [SOURCE_IDS["measurement"]]]
        subtasks = [
            {"id": "current", "description": "identify current-dated source evidence"},
            {"id": "historical", "description": "identify historical source evidence"},
        ]
        families = ["PUBLIC_HEALTH", "GUIDELINE"]
        domains = ["temporal_provenance"]
        answerability = "ANSWERABLE_WITH_DATES"
        gold = {"answerability": answerability, "required_evidence_groups": groups}
    elif category == "SERIAL_DEPENDENCY":
        groups = [[SOURCE_IDS["measurement"]], [SOURCE_IDS["who"], SOURCE_IDS["diet"]]]
        subtasks = [
            {"id": "evidence", "description": "verify source coverage"},
            {"id": "synthesis", "description": "synthesize only after coverage is verified"},
        ]
        families = ["PUBLIC_HEALTH", "INTERNAL_REVIEWED_PATIENT_EDUCATION"]
        domains = ["evidence_then_synthesis"]
        answerability = "ANSWERABLE"
        gold = {"answerability": answerability, "required_evidence_groups": groups}
    else:
        groups = []
        subtasks = [{"id": "coverage_check", "description": "check whether reviewed evidence is sufficient"}]
        families = ["INTERNAL_REVIEWED_PATIENT_EDUCATION"]
        domains = ["abstention"]
        answerability = "UNANSWERABLE"
        gold = {
            "answerability": answerability,
            "required_evidence_groups": groups,
            "ood_reason": "outside reviewed evidence or asks for diagnosis/prescription",
        }
    edges = [["evidence", "synthesis"]] if category == "SERIAL_DEPENDENCY" else []
    profile = {
        "task_profile_version": "task-profile-v1",
        "task_family": category,
        "answerability": answerability,
        "required_evidence_groups": groups,
        "source_families": families,
        "independent_subtasks": subtasks,
        "dependency_edges": edges,
        "required_capability_domains": domains,
        "conflict_structure": {"present": category == "CONFLICTING_EVIDENCE", "count": int(category == "CONFLICTING_EVIDENCE")},
        "temporal_structure": {"present": category == "TEMPORAL_EVIDENCE", "dependency_count": int(category == "TEMPORAL_EVIDENCE")},
        "expected_min_sources": len(groups),
        "ood_reason": gold.get("ood_reason"),
    }
    return profile, gold


def build(output_root: Path) -> None:
    cases: list[dict] = []
    profiles: list[dict] = []
    for category, questions in TOPICS.items():
        for index, question in enumerate(questions, 1):
            case_id = f"ra-{category.lower()}-{index:02d}"
            profile, gold = _definition(category, index - 1)
            split = "DEV" if index <= 3 else "TEST"
            profile.update(
                {
                    "case_id": case_id,
                    "annotation_status": "candidate",
                    "reviewer": None,
                    "split": split,
                    "source_group_id": f"{category.lower()}-source-group-{index:02d}",
                    "question_family_id": f"{category.lower()}-question-family-{index:02d}",
                    "text_hash": _text_hash(question),
                }
            )
            cases.append(
                {
                    "case_id": case_id,
                    "benchmark_id": "research-architecture-v1",
                    "payload": {"question": question},
                    "gold": gold,
                    "metadata": {"split": split, "candidate": True},
                    "source_provenance": [{"kind": "existing_reviewed_knowledge_card_ids", "ids": sorted({source for group in profile["required_evidence_groups"] for source in group})}],
                }
            )
            profiles.append(profile)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "cases.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in cases),
        encoding="utf-8",
    )
    (output_root / "task_profiles.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in profiles),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("benchmarks/research_architecture_v1"))
    args = parser.parse_args()
    build(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
