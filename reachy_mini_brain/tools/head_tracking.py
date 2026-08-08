import logging
from typing import Any

from reachy_mini_brain import config
from reachy_mini_brain.tools.core import Tool, ToolDependencies

logger = logging.getLogger(__name__)


class HeadTrackingTool(Tool):
    """Turn face-following on or off."""

    name = "head_tracking"
    description = (
        "Turn face tracking on or off. When on, the robot automatically keeps its "
        "head pointed at the person it sees, following them as they move. Use this "
        "when the user asks you to follow them, look at them, keep watching them, "
        "or conversely to stop following / look away / stay still."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "enabled": {
                "type": "boolean",
                "description": "True to start following the user's face, False to stop.",
            },
        },
        "required": ["enabled"],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        enabled = bool(kwargs.get("enabled", True))
        try:
            # Record the intent too - the control loop re-asserts tracking
            # continuously, so without this an explicit "stop following" would
            # just be switched straight back on again.
            deps.tracking_desired = enabled
            if enabled:
                deps.reachy_mini.start_head_tracking(weight=config.HEAD_TRACKING_WEIGHT)
                return "Now following the user's face."
            deps.reachy_mini.stop_head_tracking()
            return "Stopped following."
        except Exception:
            logger.exception("head_tracking tool failed")
            return "I couldn't change face tracking right now."
