"""Mark's gate + per-task-specialist agents. See docs/ARCHITECTURE.md."""

from reachy_mini_brain.agents.base import Agent
from reachy_mini_brain.agents.registry import (
    AGENT_KEYWORDS,
    DEFAULT_AGENT_NAME,
    MONOLITH_AGENT,
    SHARED_TOOLS,
    build_agents,
)

__all__ = [
    "Agent",
    "AGENT_KEYWORDS",
    "DEFAULT_AGENT_NAME",
    "MONOLITH_AGENT",
    "SHARED_TOOLS",
    "build_agents",
]
