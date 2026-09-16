"""The single read-only knowledge retrieval tool exposed in M1."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..agent.tools import ToolResult, ToolSpec

if TYPE_CHECKING:
    from ..pipeline import Retriever


class SearchKnowledgeTool:
    name = "search_knowledge"

    def __init__(self, retriever: "Retriever", top_k: int = 3) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self._retriever = retriever
        self._top_k = top_k

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description="使用审核知识卡检索一个更适合当前问题的只读查询。",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "非空的改写检索查询",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )

    def validate_arguments(self, arguments: object) -> dict[str, str]:
        if not isinstance(arguments, Mapping):
            raise TypeError("arguments must be an object")
        if set(arguments) != {"query"}:
            raise ValueError("arguments must contain only query")
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        return {"query": query.strip()}

    def execute(self, arguments: object) -> ToolResult:
        query = arguments["query"]  # validated by ToolRegistry
        evidence = self._retriever.search(query, top_k=self._top_k)
        compact = [
            {
                "source_id": item.source_id,
                "title": item.title,
                "excerpt": item.excerpt,
                "score": item.score,
            }
            for item in evidence
        ]
        return ToolResult.success(data=compact, observed_evidence=evidence)
