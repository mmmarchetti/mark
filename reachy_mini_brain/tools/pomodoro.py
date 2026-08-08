from typing import Any

from reachy_mini_brain.tools.core import Tool, ToolDependencies


class PomodoroTool(Tool):
    """Start a focus/pomodoro session with spoken work/break check-ins."""

    name = "focus_session"
    description = (
        "Start a Pomodoro-style focus session: a work period then a break, with the "
        "robot checking in by voice at each transition. Use when the user wants to "
        "focus, study, or start a work sprint. Defaults 25 min work / 5 min break."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "work_min": {"type": "number", "description": "Work minutes (default 25)."},
            "break_min": {"type": "number", "description": "Break minutes (default 5)."},
        },
        "required": [],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        if deps.scheduler is None:
            return "I can't start a focus session right now."
        work = float(kwargs.get("work_min", 25) or 25)
        brk = float(kwargs.get("break_min", 5) or 5)
        deps.scheduler.add_oneshot(
            work * 60, f"Nice work - that {int(work)} minute focus block is done. Take a {int(brk)} minute break.",
            label="pomodoro-work")
        deps.scheduler.add_oneshot(
            (work + brk) * 60, "Break's over. Ready for another focus block?",
            label="pomodoro-break")
        return f"Focus session started: {int(work)} minutes of work, then a {int(brk)} minute break. Go."
