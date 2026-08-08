from typing import Any

from reachy_mini_brain.tools.core import Tool, ToolDependencies

# Moves that shouldn't be picked as a conversational reaction: they are
# lifecycle animations the app drives itself (wake/sleep), not expressions.
_EXCLUDED_MOVES = {"mini-deep-sleep", "wake-mini-up", "sleep1", "waiting", "dying1"}


class PlayEmotionTool(Tool):
    """Play a recorded emotion/dance move from the emotions library."""

    name = "play_emotion"
    description = (
        "Play a recorded expressive movement on the robot's body. Use this often - "
        "react physically to the conversation the way a person would with body "
        "language: when greeting, agreeing, disagreeing, being surprised, amused, "
        "proud, confused, impatient, sympathetic, and so on. Pick the move whose "
        "name best matches the feeling you are expressing right now, and vary your "
        "choices instead of reusing the same few."
    )

    def __init__(self, available_moves: list[str] | None = None) -> None:
        moves = sorted(set(available_moves or []) - _EXCLUDED_MOVES)
        # Constrain the model to real move names with an enum. Without the full
        # list it only ever reused the handful of names mentioned in prose,
        # so the robot's body language barely varied.
        name_schema: dict[str, Any] = {
            "type": "string",
            "description": "Name of the move to play, chosen to match the emotion you are expressing.",
        }
        if moves:
            name_schema["enum"] = moves
        self.parameters_schema = {
            "type": "object",
            "properties": {"name": name_schema},
            "required": ["name"],
        }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        name = str(kwargs.get("name", "")).strip()
        if name not in deps.available_moves:
            return f"No such move '{name}'."
        deps.motion_queue.put({"type": "play_move", "name": name})
        return f"Playing '{name}'."
