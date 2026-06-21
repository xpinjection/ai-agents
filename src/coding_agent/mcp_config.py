"""Load MCP tools from a Claude Code-style ``.mcp.json`` at the project root.

Deep Agents' ``create_deep_agent`` does not auto-discover ``.mcp.json`` (that is a feature of the
separate ``dcode`` CLI harness), so we parse it here and convert each server entry to the
connection schema expected by ``langchain_mcp_adapters.MultiServerMCPClient``.
"""

import json
import logging
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)


def _to_connection(name: str, spec: dict) -> dict | None:
    """Convert one Claude Code ``.mcp.json`` server entry to a MultiServerMCPClient connection."""
    # stdio server: launched via a local command
    if spec.get("command"):
        conn: dict = {"transport": "stdio", "command": spec["command"]}
        if spec.get("args"):
            conn["args"] = spec["args"]
        if spec.get("env"):
            conn["env"] = spec["env"]
        return conn

    # remote server: reachable over http / sse
    url = spec.get("url")
    if url:
        server_type = (spec.get("type") or "http").lower()
        transport = "sse" if server_type == "sse" else "streamable_http"
        conn = {"transport": transport, "url": url}
        if spec.get("headers"):
            conn["headers"] = spec["headers"]
        return conn

    logger.warning("Skipping MCP server '%s': no 'command' or 'url' in .mcp.json entry.", name)
    return None


def _find_mcp_config(project_dir: Path) -> Path | None:
    for candidate in (project_dir / ".mcp.json", project_dir / ".deepagents" / ".mcp.json"):
        if candidate.is_file():
            return candidate
    return None


async def load_mcp_tools(project_dir: Path) -> list:
    """Load MCP tools declared in ``<project_dir>/.mcp.json``.

    Each server is loaded independently so that one failing server (missing command, auth required,
    or langgraph dev's blocking-call guard) does not drop the tools of the healthy ones. Returns an
    empty list if no config is present; never raises — MCP issues must not crash agent startup.
    """
    config_path = _find_mcp_config(project_dir)
    if config_path is None:
        logger.info("No .mcp.json found under %s; continuing without MCP tools.", project_dir)
        return []

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read %s: %s. Continuing without MCP tools.", config_path, e)
        return []

    tools: list = []
    for srv_name, spec in raw.get("mcpServers", {}).items():
        conn = _to_connection(srv_name, spec)
        if conn is None:
            continue
        try:
            client = MultiServerMCPClient({srv_name: conn})
            srv_tools = await client.get_tools()
        except Exception as e:  # noqa: BLE001 - isolate per-server failures, never crash startup
            hint = ""
            if "blocking call" in str(e).lower():
                hint = (
                    " (langgraph dev's blocking-call guard tripped on the stdio MCP client; run "
                    "`langgraph dev --allow-blocking`, or use an http/sse server, to load it)"
                )
            logger.warning("Skipping MCP server '%s': %s%s", srv_name, e, hint)
            continue
        tools.extend(srv_tools)
        logger.info("Loaded %d tool(s) from MCP server '%s'.", len(srv_tools), srv_name)

    return tools
