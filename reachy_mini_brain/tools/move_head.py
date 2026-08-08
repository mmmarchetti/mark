from typing import Any

from reachy_mini_brain.tools.core import Tool, ToolDependencies


class MoveHeadTool(Tool):
    """Move the robot's head (and optionally rotate its body) to look somewhere."""

    name = "move_head"
    description = (
        "Move the robot's head to look in a direction, e.g. to look at the user, "
        "look up/down/sideways, or nod/shake for yes/no. Set body_yaw as well to "
        "physically rotate the whole body - needed to look far to the side or "
        "behind, since the head alone cannot turn more than about 65 degrees away "
        "from the body. Angles in degrees."
    )
    # Ranges here mirror the documented safety limits in the SDK's AGENTS.md.
    parameters_schema = {
        "type": "object",
        "properties": {
            "yaw": {
                "type": "number",
                "description": (
                    "Head left(-)/right(+) angle in degrees relative to the body, "
                    "range -65 to 65. When body_yaw is also set the head automatically "
                    "follows the body, so leave this at 0 to simply face the new direction."
                ),
            },
            "pitch": {"type": "number", "description": "Head down(-)/up(+) angle in degrees, range -40 to 40."},
            "roll": {"type": "number", "description": "Head tilt angle in degrees, range -40 to 40."},
            "body_yaw": {
                "type": "number",
                "description": (
                    "Body rotation in degrees, range -160 to 160. Rotates the whole "
                    "robot to face a new direction. Keep within 65 degrees of the head yaw."
                ),
            },
            "duration": {"type": "number", "description": "Seconds for the movement. Default 1.0."},
        },
        "required": [],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        action = {
            "type": "goto_head",
            "yaw": float(kwargs.get("yaw", 0.0) or 0.0),
            "pitch": float(kwargs.get("pitch", 0.0) or 0.0),
            "roll": float(kwargs.get("roll", 0.0) or 0.0),
            "duration": float(kwargs.get("duration", 1.0) or 1.0),
        }
        if kwargs.get("body_yaw") is not None:
            action["body_yaw"] = float(kwargs["body_yaw"])
        deps.motion_queue.put(action)
        return "Movement queued."
