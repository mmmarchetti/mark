from typing import Any

from reachy_mini_brain.tools.core import Tool, ToolDependencies


class GoToSleepTool(Tool):
    """Put the robot to sleep (low-power idle pose, motors disabled)."""

    name = "go_to_sleep"
    description = "Put the robot to sleep when the user says goodbye or asks it to rest/sleep."
    parameters_schema = {"type": "object", "properties": {}, "required": []}

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        # Force the wake word to be required on the next interaction - without
        # this, a stray utterance during the (up to 5min) still-open
        # conversation window could be treated as a real turn while "asleep".
        deps.close_conversation()
        # Do NOT queue the motion here. The control loop would start lowering
        # the head within ~50ms, i.e. while the goodbye line is still being
        # spoken - and with wobbling active that pushed the head into the
        # shell (confirmed live). The brain loop runs the sleep motion after
        # the turn's speech completes instead.
        deps.pending_sleep = True
        return "Saying goodnight, then going to sleep."
