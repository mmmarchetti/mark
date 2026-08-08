from typing import Any

from reachy_mini_brain.tools.core import Tool, ToolDependencies


class StopListeningTool(Tool):
    """Stop actively listening without going to sleep - lighter than go_to_sleep."""

    name = "stop_listening"
    description = (
        "Stop actively listening/paying attention for now, without physically "
        "going to sleep. Use when the user asks you to stop listening, leave them "
        "alone, or says something like 'that's all for now' without asking you to "
        "sleep. The wake word will be required again before the next conversation."
    )
    parameters_schema = {"type": "object", "properties": {}, "required": []}

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        deps.close_conversation()
        return "Stopped actively listening until the wake word is used again."
