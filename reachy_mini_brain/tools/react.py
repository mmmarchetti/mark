from typing import Any

from reachy_mini_brain.tools.core import Tool, ToolDependencies

_GESTURES = ["perk", "wiggle", "droop", "none"]
_SOUNDS = ["chime", "beep", "success", "error", "pop", "none"]


class ReactTool(Tool):
    """Quick expressive reaction: antenna gesture and/or a sound effect."""

    name = "react"
    description = (
        "Give a quick physical/audio reaction with your antennas and/or a sound "
        "effect - use it to punctuate a moment: perk up when curious/surprised, "
        "wiggle when happy/excited, droop when sad. Sounds: chime, beep, success, "
        "error, pop. This is lighter than play_emotion; use it liberally to feel alive."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "gesture": {"type": "string", "enum": _GESTURES},
            "sound": {"type": "string", "enum": _SOUNDS},
        },
        "required": [],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        gesture = kwargs.get("gesture", "none")
        sound = kwargs.get("sound", "none")
        if gesture and gesture != "none":
            deps.motion_queue.put({"type": "antennas", "pattern": gesture})
        if sound and sound != "none":
            deps.motion_queue.put({"type": "sound", "name": sound})
        return "Reacted." if (gesture != "none" or sound != "none") else "Nothing to do."
