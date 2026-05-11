"""
Search tool setup using the Tavily Model Context Protocol (MCP) server.

Using MCP aligns with the 2025/2026 standard for decoupling tool logic
from agent logic. The remote HTTP endpoint means no Node.js is required.
A direct-SDK fallback ensures the app is robust if the MCP server is
temporarily unreachable.
"""
from __future__ import annotations

import logging
import os

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


async def get_search_tools() -> list[BaseTool]:
    """
    Connect to Tavily's hosted MCP server over HTTP and return LangChain tools.

    Falls back to the direct Tavily Python SDK if the MCP endpoint is
    unavailable, so the app always has a working search capability.
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        api_key = os.environ["TAVILY_API_KEY"]
        client = MultiServerMCPClient(
            {
                "tavily": {
                    "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}",
                    "transport": "streamable_http",
                }
            }
        )
        tools = await client.get_tools()
        if tools:
            logger.info("Tavily MCP tools loaded: %s", [t.name for t in tools])
            return tools
        logger.warning("Tavily MCP returned no tools; falling back to direct SDK.")
    except Exception as exc:
        logger.warning("Tavily MCP unavailable (%s); falling back to direct SDK.", exc)

    # Fallback: direct Tavily Python SDK wrapped as a LangChain tool
    from langchain_community.tools.tavily_search import TavilySearchResults

    logger.info("Using TavilySearchResults (direct SDK fallback).")
    return [TavilySearchResults(max_results=5)]
