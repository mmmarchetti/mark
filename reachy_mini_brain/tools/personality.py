from typing import Any

from reachy_mini_brain.profiles import PROFILES
from reachy_mini_brain.tools.core import Tool, ToolDependencies

_OPTIONS = ", ".join(f"'{k}' ({label})" for k, (label, _) in PROFILES.items())


class SetPersonalityTool(Tool):
    """Switch the robot's personality/tone."""

    name = "set_personality"
    description = (
        "Change your own personality/tone when the user asks you to act "
        f"differently (e.g. be more playful, calmer, more professional). Options: {_OPTIONS}."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "personality": {
                "type": "string",
                "enum": list(PROFILES.keys()),
                "description": "Which personality to switch to.",
            },
        },
        "required": ["personality"],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        name = str(kwargs.get("personality", "")).strip().lower()
        if deps.profiles is None:
            return "Personality switching isn't available right now."
        if deps.profiles.set(name):
            label = PROFILES[name][0]
            return f"Switched to my {label} personality."
        return f"I don't have a '{name}' personality."
