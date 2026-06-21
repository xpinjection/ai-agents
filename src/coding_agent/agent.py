"""Claude Code-like coding agent built on the deepagents library.

Capabilities:
- Operates on a project directory (from ``CODING_AGENT_PROJECT_DIR``) with filesystem + shell access
  via a ``LocalShellBackend``.
- Loads project memory (``CLAUDE.md``) and skills (``.claude/skills/``).
- Loads MCP tools from ``.mcp.json``.
- Delegates read-only investigation to an ``Explore`` subagent (``gpt-5.4-nano``).
- Plans with ``write_todos`` and gates file writes / edits / shell behind human approval.
"""

import asyncio
import logging

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import FilesystemBackend, LocalShellBackend
from langchain_openai import ChatOpenAI

from coding_agent.config import CodingAgentConfig, load_config
from coding_agent.mcp_config import load_mcp_tools
from coding_agent.prompts import EXPLORE_PROMPT, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _build_explore_subagent(cfg: CodingAgentConfig) -> dict:
    """Read-only investigation subagent, modeled on Claude Code's Explore agent.

    Built as a pre-compiled subagent over a non-executing ``FilesystemBackend`` — so it has no
    ``execute`` shell tool — with all filesystem writes denied. The result is a hard read-only
    sandbox: Explore can only ``ls``/``read_file``/``glob``/``grep``. (Tool-level permissions are
    unsupported on execution-capable backends, so the parent's ``LocalShellBackend`` cannot be
    reused for the subagent.)
    """
    explore_agent = create_deep_agent(
        model=ChatOpenAI(model=cfg.explore_model, use_responses_api=True),
        system_prompt=EXPLORE_PROMPT,
        backend=FilesystemBackend(root_dir=str(cfg.project_dir), virtual_mode=True),
        permissions=[FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")],
    )
    return {
        "name": "Explore",
        "description": (
            "Read-only code investigator for preliminary, broad exploration of the project "
            "directory. Delegate a specific question about the codebase before planning; it "
            "returns concise conclusions with file:line references, not raw file dumps."
        ),
        "runnable": explore_agent,
    }


def _memory_files(cfg: CodingAgentConfig) -> list[str]:
    """Project memory always loaded into the system prompt (paths relative to backend root)."""
    return ["./CLAUDE.md"] if (cfg.project_dir / "CLAUDE.md").is_file() else []


def _skill_dirs(cfg: CodingAgentConfig) -> list[str]:
    """Skill directories loaded on demand (paths relative to backend root)."""
    return ["./.claude/skills/"] if (cfg.project_dir / ".claude" / "skills").is_dir() else []


def _build_agent(cfg: CodingAgentConfig, mcp_tools: list):
    """Assemble the compiled deep agent (synchronous).

    Run via ``asyncio.to_thread`` from the async factory: ``create_deep_agent``, the backends, and
    the memory/skill discovery all perform blocking filesystem calls (``os.stat``/``os.getcwd``)
    that langgraph's blockbuster-guarded event loop rejects when run inline.
    """
    # GPT-5.x with reasoning_effort + tool calling requires OpenAI's Responses API
    # (/v1/responses); the default Chat Completions API rejects that combination.
    model = ChatOpenAI(
        model=cfg.model,
        reasoning_effort=cfg.reasoning_effort,
        use_responses_api=True,
    )
    return create_deep_agent(
        model=model,
        tools=mcp_tools,
        system_prompt=SYSTEM_PROMPT,
        # virtual_mode=True presents the project as a POSIX filesystem rooted at the project dir
        # (paths like /src/...). Required on Windows: with virtual_mode=False the file tools fall
        # back to drive-root semantics (ls "/" lists C:\) and reject Windows absolute paths.
        backend=LocalShellBackend(root_dir=str(cfg.project_dir), virtual_mode=True),
        memory=_memory_files(cfg),
        skills=_skill_dirs(cfg),
        subagents=[_build_explore_subagent(cfg)],
        interrupt_on={
            "write_file": True,
            "edit_file": True,
            "execute": True,
        },
    )


async def create_coding_agent(config: dict | None = None):
    """Graph factory for the coding agent.

    Async because MCP tools load asynchronously. Registered in ``langgraph.json``; ``langgraph dev``
    passes a config dict and manages checkpointing itself (required for the ``interrupt_on``
    human-in-the-loop approval gates).

    Config loading and graph assembly run in a worker thread (``asyncio.to_thread``) because they
    touch the filesystem; langgraph runs this factory on its async event loop, which forbids
    blocking calls.

    Args:
        config: Optional LangGraph server config dict (passed automatically by ``langgraph dev``).
            Ignored — the agent is configured entirely from environment variables.

    Returns:
        A compiled deep-agent graph.
    """
    cfg = await asyncio.to_thread(load_config)
    mcp_tools = await load_mcp_tools(cfg.project_dir)
    agent = await asyncio.to_thread(_build_agent, cfg, mcp_tools)

    logger.info(
        "Coding agent ready: project=%s model=%s effort=%s explore=%s mcp_tools=%d",
        cfg.project_dir,
        cfg.model,
        cfg.reasoning_effort,
        cfg.explore_model,
        len(mcp_tools),
    )
    return agent
