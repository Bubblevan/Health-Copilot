"""Command-line demo for the reproducible M0 path and bounded M1 path."""

import argparse
import sys

from .agent.model import AgentModelError, OpenAICompatibleAgentModel
from .generation.base import GenerationError
from .generation.openai_compatible import OpenAICompatibleGenerator
from .knowledge.loader import KnowledgeCardLoadError, load_knowledge_cards
from .pipeline import HealthCopilotPipeline
from .retrieval.bm25 import BM25Retriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Health-Copilot M0 or M1 pipeline")
    parser.add_argument(
        "--mode",
        choices=("m0", "m1"),
        default="m0",
        help="m0: deterministic generator path; m1: bounded single-agent recovery path",
    )
    parser.add_argument(
        "--knowledge-dir", default="data/knowledge_cards", help="directory containing JSON cards"
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
        else:
            agent_model = OpenAICompatibleAgentModel()
            pipeline = HealthCopilotPipeline(retriever, agent_model=agent_model)
        result = pipeline.answer(args.question)
    except (KnowledgeCardLoadError, GenerationError, AgentModelError) as exc:
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
