"""Test-only stdio MCP server for M9 process-boundary checks.

This file deliberately exposes unsafe-looking operations only as a security
fixture.  It is never registered in a product runtime profile.
"""

from __future__ import annotations

import argparse
import asyncio
import socket
from pathlib import Path

from mcp.server.mcpserver import MCPServer


def build_server(allowed_file: Path, workspace: Path) -> MCPServer:
    server = MCPServer(name="m9-security-fixture", version="2026-07-28")

    @server.tool(name="read_allowed_file", description="TEST ONLY: read the mounted allowed file")
    def read_allowed_file() -> dict[str, str]:
        return {"text": allowed_file.read_text(encoding="utf-8")}

    @server.tool(name="read_forbidden_file", description="TEST ONLY: read an arbitrary path")
    def read_forbidden_file(path: str) -> dict[str, str]:
        return {"text": Path(path).read_text(encoding="utf-8")}

    @server.tool(name="write_workspace_file", description="TEST ONLY: write in the mounted workspace")
    def write_workspace_file(name: str, text: str) -> dict[str, str]:
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return {"path": str(target), "status": "written"}

    @server.tool(name="write_outside_workspace", description="TEST ONLY: write an arbitrary path")
    def write_outside_workspace(path: str, text: str) -> dict[str, str]:
        target = Path(path)
        target.write_text(text, encoding="utf-8")
        return {"path": str(target), "status": "written"}

    @server.tool(name="network_probe", description="TEST ONLY: attempt a network connection")
    def network_probe(host: str, port: int) -> dict[str, bool]:
        with socket.create_connection((host, port), timeout=1.0):
            return {"connected": True}

    return server


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowed-file", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(build_server(args.allowed_file, args.workspace).run_stdio_async())


if __name__ == "__main__":
    main()
