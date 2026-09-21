"""Official-SDK MCP fixtures used by the M9 product and security tests."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

from ..knowledge.scope import KnowledgeScope
from .client import MCP_PROTOCOL_VERSION


def build_search_knowledge_server(
    retriever,
    *,
    knowledge_scope: KnowledgeScope | None = None,
    server_id: str = "m9-search-knowledge-server",
    ttl_ms: int = 1000,
):
    """Build a read-only MCP server over the existing retriever contract."""

    from mcp.server.caching import CacheHint
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(
        name=server_id,
        version=MCP_PROTOCOL_VERSION,
        cache_hints={"tools/list": CacheHint(ttl_ms=ttl_ms, scope="public")},
    )

    @server.tool(name="search_knowledge", description="Read-only reviewed knowledge-card search", structured_output=True)
    def search_knowledge(query: str) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        evidence = retriever.search(query.strip(), top_k=3)
        return {
            "results": [
                {
                    "source_id": item.source_id,
                    "title": item.title,
                    "excerpt": item.excerpt,
                    "source_url": item.source_url,
                    "score": item.score,
                }
                for item in evidence
            ],
            "scope_id": knowledge_scope.scope_id if knowledge_scope else None,
            "scope_version": knowledge_scope.version if knowledge_scope else None,
        }

    return server


def build_security_fixture_server(
    *,
    allowed_file: Path,
    workspace: Path,
    server_id: str = "m9-security-fixture",
):
    """Test-only tools for permission and real process-containment checks."""

    from mcp.server.mcpserver import MCPServer

    server = MCPServer(name=server_id, version=MCP_PROTOCOL_VERSION)

    @server.tool(name="read_allowed_file", description="TEST ONLY: read the mounted allowed file")
    def read_allowed_file() -> dict[str, str]:
        return {"text": allowed_file.read_text(encoding="utf-8")}

    @server.tool(name="read_forbidden_file", description="TEST ONLY: read a caller-selected file")
    def read_forbidden_file(path: str) -> dict[str, str]:
        return {"text": Path(path).read_text(encoding="utf-8")}

    @server.tool(name="write_workspace_file", description="TEST ONLY: write in the workspace")
    def write_workspace_file(name: str, text: str) -> dict[str, str]:
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return {"path": str(target), "status": "written"}

    @server.tool(name="write_outside_workspace", description="TEST ONLY: write outside the workspace")
    def write_outside_workspace(path: str, text: str) -> dict[str, str]:
        target = Path(path)
        target.write_text(text, encoding="utf-8")
        return {"path": str(target), "status": "written"}

    @server.tool(name="network_probe", description="TEST ONLY: attempt a network connection")
    def network_probe(host: str, port: int) -> dict[str, Any]:
        with socket.create_connection((host, port), timeout=1.0):
            return {"connected": True}

    return server


__all__ = ["build_search_knowledge_server", "build_security_fixture_server"]
