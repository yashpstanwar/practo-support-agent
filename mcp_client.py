"""Task 14 - a SEPARATE process that talks to the MCP server over HTTP.

Start the server first (python mcp_server.py), then:
    python mcp_client.py APT-1002 APT-1007 APT-9999
"""
from __future__ import annotations

import asyncio
import json
import sys

from fastmcp import Client

import config


async def main(record_ids: list[str]) -> None:
    async with Client(config.MCP_URL) as client:
        tools = await client.list_tools()
        print(f"Connected to MCP server at {config.MCP_URL}")
        for t in tools:
            print(f"  tool: {t.name}")
            print(f"    description: {t.description.strip().splitlines()[0]}")
            print(f"    inputSchema: {json.dumps(t.inputSchema)}")
        for rid in record_ids:
            result = await client.call_tool("check_appointment_status", {"record_id": rid})
            print(f"\n>>> tools/call check_appointment_status(record_id={rid!r})")
            print(f"    isError: {result.is_error}")
            for block in result.content:
                print(f"    content[{block.type}]: {getattr(block, 'text', '')}")
            print(f"    structuredContent: {json.dumps(result.structured_content)}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or ["APT-1002", "APT-1007"]))
