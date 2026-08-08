from typing import Any

from reachy_mini_brain.tools.core import Tool, ToolDependencies


class CameraTool(Tool):
    """Take a picture and describe it using the local vision model."""

    name = "camera"
    description = (
        "Take a picture with the camera to see what is in front of the robot. "
        "Use this when the user asks you to look at something, describe the scene, "
        "or comment on their appearance. The camera is live; each call captures the "
        "current moment. If asked to look without specifying at what, just call this "
        "and describe what you see."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "What to look for or ask about in the picture.",
            },
        },
        "required": ["question"],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        question = str(kwargs.get("question", "What do you see?")).strip()
        return deps.vision.describe(deps.reachy_mini, question)
