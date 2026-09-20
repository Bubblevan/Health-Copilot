"""Profile-driven CLI for the reproducible M0 through M6 runtime paths."""

import argparse
import sys

from .knowledge.loader import KnowledgeCardLoadError, load_knowledge_cards
from .knowledge.scope import KnowledgeScopeLoadError, load_knowledge_scope
from .runtime.builder import RuntimeBuilder, RuntimeBuildError, default_runtime_profiles
from .runtime.profile import RuntimeProfile
from .runtime.registry import ComponentRegistryError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Health-Copilot profile-driven M0 through M6 pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=("m0", "m1", "m2", "m3"),
        default=None,
        help="legacy execution mode; profile mode is preferred",
    )
    parser.add_argument(
        "--profile",
        help="explicit declarative RuntimeProfile ID, for example m3-bm25-default",
    )
    parser.add_argument(
        "--knowledge-dir", default="data/knowledge_cards", help="directory containing JSON cards"
    )
    parser.add_argument(
        "--knowledge-scope",
        default="data/knowledge_scope.json",
        help="reviewed M3 closed-corpus capability manifest",
    )
    parser.add_argument("--question", required=True, help="patient-education question")
    parser.add_argument(
        "--retriever",
        choices=("bm25", "dense", "hybrid", "hybrid_rerank"),
        help="deprecated legacy selector; maps explicitly to a demo profile component",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        profile = _select_profile(args)
        cards = load_knowledge_cards(args.knowledge_dir)
        scope = (
            load_knowledge_scope(args.knowledge_scope, cards) if profile.mode == "m3" else None
        )
        components = RuntimeBuilder().build(profile, cards=cards, knowledge_scope=scope)
        result = components.pipeline().answer(args.question)
    except (
        KnowledgeCardLoadError,
        KnowledgeScopeLoadError,
        RuntimeBuildError,
        ComponentRegistryError,
        ValueError,
        RuntimeError,
    ) as exc:
        print(f"配置或知识卡错误：{exc}", file=sys.stderr)
        return 2

    print(f"profile: {profile.profile_id}")
    print(f"component_manifest_hash: {components.manifest_hash}")
    print(f"route: {result.route.value}")
    print(f"answer: {result.message}")
    if result.safety_reasons:
        print(f"safety_reasons: {', '.join(result.safety_reasons)}")
    if result.citations:
        print("citations:")
        for citation in result.citations:
            print(f"- {citation.source_id}: {citation.title} ({citation.source_url})")
    return 0


def _select_profile(args: argparse.Namespace) -> RuntimeProfile:
    profiles = default_runtime_profiles()
    if args.profile and args.retriever:
        raise RuntimeBuildError("--profile and legacy --retriever cannot be used together")
    if args.profile:
        try:
            profile = profiles[args.profile]
        except KeyError as exc:
            known = ", ".join(sorted(profiles))
            raise RuntimeBuildError(
                f"unknown profile '{args.profile}'; registered profiles: {known}"
            ) from exc
        if args.mode and args.mode != profile.mode:
            raise RuntimeBuildError(
                f"--mode {args.mode} conflicts with profile mode {profile.mode}"
            )
        return profile

    mode = args.mode or "m0"
    if args.retriever:
        print(
            "警告：--retriever 已弃用；当前仅映射到显式 demo 组件，"
            "不会选择 M5 learned 模型。请改用 --profile。",
            file=sys.stderr,
        )
        return _legacy_profile(mode, args.retriever)
    return profiles[f"{mode}-bm25-default"]


def _legacy_profile(mode: str, retriever: str) -> RuntimeProfile:
    component_id = {
        "bm25": "bm25-v1",
        "dense": "dense-hashing-demo-v1",
        "hybrid": "hybrid-hashing-demo-v1",
        # Historical CLI behavior was hashing + token overlap, not M5 learned.
        "hybrid_rerank": "hybrid-token-rerank-demo-v1",
    }[retriever]
    policy = "evidence-policy-m2-v1" if mode == "m2" else "evidence-policy-m3-v1"
    verifier = "grounding-v1" if mode == "m2" else "claim-support-v1"
    return RuntimeProfile(
        profile_id=f"legacy-{mode}-{retriever}",
        provider="openai-compatible-v1",
        retriever=component_id,
        policy=policy if mode in {"m2", "m3"} else None,
        verifier=verifier if mode in {"m2", "m3"} else None,
        tool_set=("search-knowledge-v1",) if mode in {"m1", "m2", "m3"} else (),
        mode=mode,
    )


if __name__ == "__main__":
    raise SystemExit(main())
