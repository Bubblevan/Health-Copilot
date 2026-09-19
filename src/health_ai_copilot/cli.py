"""Command-line demo for the reproducible M0 through M3 paths."""

import argparse
import sys

from .agent.model import AgentModelError, AgentOutputMode, OpenAICompatibleAgentModel
from .generation.base import GenerationError
from .generation.openai_compatible import OpenAICompatibleGenerator
from .knowledge.loader import KnowledgeCardLoadError, load_knowledge_cards
from .knowledge.scope import KnowledgeScopeLoadError, load_knowledge_scope
from .pipeline import HealthCopilotPipeline
from .policy.model import OpenAICompatibleEvidencePolicy
from .retrieval.bm25 import BM25Retriever
from .verification.grounding import (
    OpenAICompatibleClaimSupportVerifier,
    OpenAICompatibleGroundingVerifier,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Health-Copilot M0 through M3 pipeline")
    parser.add_argument(
        "--mode",
        choices=("m0", "m1", "m2", "m3"),
        default="m0",
        help="m0: generator; m1: bounded recovery; m2: policy plus grounding; m3: scope plus claim-first",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cards = load_knowledge_cards(args.knowledge_dir)
        retriever = BM25Retriever(cards)
        if args.mode == "m0":
            generator = OpenAICompatibleGenerator()
            pipeline = HealthCopilotPipeline(retriever, generator)
        elif args.mode == "m1":
            agent_model = OpenAICompatibleAgentModel()
            pipeline = HealthCopilotPipeline(retriever, agent_model=agent_model)
        elif args.mode == "m2":
            pipeline = HealthCopilotPipeline(
                retriever,
                agent_model=OpenAICompatibleAgentModel(
                    output_mode=AgentOutputMode.M2_GROUNDED
                ),
                evidence_policy=OpenAICompatibleEvidencePolicy(),
                grounding_verifier=OpenAICompatibleGroundingVerifier(),
            )
        else:
            scope = load_knowledge_scope(args.knowledge_scope, cards)
            pipeline = HealthCopilotPipeline(
                retriever,
                agent_model=OpenAICompatibleAgentModel(
                    output_mode=AgentOutputMode.M3_CLAIM_FIRST
                ),
                evidence_policy=OpenAICompatibleEvidencePolicy(knowledge_scope=scope),
                knowledge_scope=scope,
                claim_support_verifier=OpenAICompatibleClaimSupportVerifier(),
            )
        result = pipeline.answer(args.question)
    except (KnowledgeCardLoadError, KnowledgeScopeLoadError, GenerationError, AgentModelError) as exc:
        print(f"Configuration or knowledge-card error: {exc}", file=sys.stderr)
        return 2

    print(f"route: {result.route.value}")
    print(f"answer: {result.message}")
    if result.safety_reasons:
        print(f"safety_reasons: {', '.join(result.safety_reasons)}")
    if result.citations:
        print("citations:")
        for citation in result.citations:
            print(f"- {citation.source_id}: {citation.title} ({citation.source_url})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
