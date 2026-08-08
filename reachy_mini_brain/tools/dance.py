from typing import Any

from reachy_mini_brain.dances import available_dances
from reachy_mini_brain.tools.core import Tool, ToolDependencies

_DANCES = available_dances()


class DanceTool(Tool):
    """Perform a rhythmic dance move."""

    name = "dance"
    description = (
        "Perform a rhythmic, beat-synced dance with your head and body. Use when "
        "the user asks you to dance, or to celebrate/have fun with music-like "
        "motion. This is more dynamic than play_emotion. Pick the dance whose "
        "name fits the mood; optionally set bpm to match music tempo."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "enum": _DANCES,
                "description": "Which dance move to perform.",
            },
            "bpm": {"type": "number", "description": "Beats per minute (tempo). Default 120."},
            "beats": {"type": "number", "description": "How many beats to dance for. Default 8."},
        },
        "required": ["name"],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        name = str(kwargs.get("name", "")).strip()
        if name not in _DANCES:
            return f"I don't know the dance '{name}'."
        deps.motion_queue.put({
            "type": "dance",
            "name": name,
            "bpm": float(kwargs.get("bpm", 120.0) or 120.0),
            "beats": float(kwargs.get("beats", 8.0) or 8.0),
        })
        return f"Dancing '{name}'."
