"""Environment-driven configuration for the coding agent."""

import os
from dataclasses import dataclass
from pathlib import Path

# Default project directory: this repository root.
# src/coding_agent/config.py -> src/coding_agent -> src -> <repo root>
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CodingAgentConfig:
    """Resolved settings for the coding agent."""

    project_dir: Path
    model: str
    reasoning_effort: str
    explore_model: str


def load_config() -> CodingAgentConfig:
    """Build the coding-agent configuration from environment variables.

    Environment variables (all optional, sensible defaults applied):
        CODING_AGENT_PROJECT_DIR    Absolute path to the project the agent operates on.
                                    Defaults to this repository root.
        CODING_AGENT_MODEL          Main model id. Defaults to ``gpt-5.5``.
        CODING_AGENT_REASONING_EFFORT  Reasoning effort: low | medium | high. Defaults to ``medium``.
        CODING_AGENT_EXPLORE_MODEL  Model id for the read-only Explore subagent.
                                    Defaults to ``gpt-5.4-nano``.
    """
    # Use `or` (not a getenv default) so an empty value like `CODING_AGENT_PROJECT_DIR=` in .env
    # falls back to the repo root instead of resolving to "".
    raw_dir = os.getenv("CODING_AGENT_PROJECT_DIR") or str(_REPO_ROOT)
    project_dir = Path(raw_dir).expanduser()
    if not project_dir.is_absolute():
        # Anchor relative paths to the repo root. This is deterministic and avoids Path.resolve(),
        # which calls os.getcwd() — a blocking call that langgraph's async event loop rejects.
        project_dir = _REPO_ROOT / project_dir
    if not project_dir.is_dir():
        raise ValueError(
            f"CODING_AGENT_PROJECT_DIR does not exist or is not a directory: {project_dir}"
        )

    return CodingAgentConfig(
        project_dir=project_dir,
        model=os.getenv("CODING_AGENT_MODEL") or "gpt-5.5",
        reasoning_effort=os.getenv("CODING_AGENT_REASONING_EFFORT") or "medium",
        explore_model=os.getenv("CODING_AGENT_EXPLORE_MODEL") or "gpt-5.4-nano",
    )
