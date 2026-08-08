"""Tool base class and dispatch, mirroring the official conversation app's
Tool convention (name / description / parameters_schema / __call__) so the
schema is directly usable as an OpenAI function-calling tool.
"""

from __future__ import annotations

import logging
import queue
from dataclasses import dataclass, field
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class ToolDependencies:
    """Everything a tool needs to do its job, threaded through from main.py."""

    reachy_mini: Any
    motion_queue: "queue.Queue"
    vision: Any
    close_conversation: Any  # AudioListener.close_conversation callable
    doa: Any = None  # DoaOrienter, so tools can pause sound-orienting during a scan
    memory: Any = None  # MemoryStore for remember/forget tools
    profiles: Any = None  # ProfileManager for set_personality tool
    weather: Any = None  # Weather provider for the weather tool
    search: Any = None   # WebSearch provider (Dell-side, always on)
    bridge: Any = None   # BridgeClient to the MacBook (Slack + Calendar)
    scheduler: Any = None  # Scheduler for timers/reminders/alarms
    todo: Any = None       # TodoStore
    wiki: Any = None       # Wiki provider
    finance: Any = None    # Finance provider
    news: Any = None       # News provider
    transit: Any = None    # Transit provider (Google Maps directions)
    listener: Any = None   # AudioListener (for record_memo)
    identity: Any = None   # Identity (face+voice recognition, enrollment)
    desk: Any = None       # DeskController (GenioDesk Tuya local control)
    light: Any = None      # LightController (desk-lights Tuya smart plug)
    # Name of the person currently being talked to, if recognized this turn
    # (set by the brain loop from a voice/face match). Drives per-person memory.
    current_speaker: str | None = None
    available_moves: list[str] = field(default_factory=list)
    # Set by go_to_sleep instead of queueing the motion immediately, so the
    # brain loop can run the sleep motion only AFTER all speech for the turn
    # has finished (see the sleep sequencing note in main.py).
    pending_sleep: bool = False
    # Set by the desk tool to ARM a move that still needs the user's spoken
    # confirmation. The desk tool NEVER moves the desk itself; the deterministic
    # yes/no gate in main.py's brain loop fires this on the next turn (or clears
    # it on "no"/timeout). Shape: {"kind":"preset"|"nudge", ..., "armed_at":ts}.
    # This is what makes every desk move confirm-before-act and impossible in one
    # LLM step (a sensitive-action requirement from the owner).
    pending_desk: "dict | None" = None
    # Face tracking is meant to be ON at all times while awake. The control
    # loop re-asserts it periodically so nothing can silently leave it off;
    # only an explicit "stop following" from the user clears this.
    tracking_desired: bool = True


class Tool:
    """Base class for an LLM-callable tool."""

    name: str = ""
    description: str = ""
    parameters_schema: Dict[str, Any] = {"type": "object", "properties": {}}

    def to_openai_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        raise NotImplementedError


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def to_openai_tools(self) -> list[Dict[str, Any]]:
        return [t.to_openai_tool() for t in self._tools.values()]

    def dispatch(self, deps: ToolDependencies, name: str, arguments: Dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            logger.warning("Unknown tool requested: %s", name)
            return f"Unknown tool '{name}'."
        try:
            return tool.run(deps, **arguments)
        except Exception:
            logger.exception("Tool '%s' failed", name)
            return f"Tool '{name}' failed to execute."
