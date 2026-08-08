from typing import Any

from reachy_mini_brain.tools.core import Tool, ToolDependencies


class SetReminderTool(Tool):
    """Set a timer, reminder, or alarm."""

    name = "set_reminder"
    description = (
        "Set a timer, reminder, or alarm. Compute delay_seconds from what the user "
        "says (e.g. 'in 10 minutes' = 600, 'in 2 hours' = 7200; for a clock time like "
        "'at 7am' compute seconds from now). Give a short spoken message to say when it "
        "fires (e.g. 'Time to leave', 'Your 10 minute timer is up')."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "delay_seconds": {"type": "number", "description": "Seconds from now until it fires."},
            "message": {"type": "string", "description": "What to say aloud when it fires."},
            "label": {"type": "string", "description": "Short label to find/cancel it later (e.g. 'pasta timer')."},
            "persist": {"type": "boolean", "description": "True for reminders/alarms that should survive a restart; false for quick timers."},
        },
        "required": ["delay_seconds", "message"],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        if deps.scheduler is None:
            return "Reminders aren't available right now."
        delay = float(kwargs.get("delay_seconds", 0) or 0)
        if delay < 1:
            return "That time is too soon or unclear."
        deps.scheduler.add_oneshot(
            delay, str(kwargs.get("message", "Reminder")).strip(),
            persist=bool(kwargs.get("persist", False)),
            label=str(kwargs.get("label", "")).strip(),
        )
        mins = int(delay // 60)
        return f"Okay, set for {mins} minute{'s' if mins != 1 else ''} from now." if mins else "Okay, set."


class RemindersTool(Tool):
    """List or cancel pending reminders/timers."""

    name = "reminders"
    description = "List the user's pending timers/reminders, or cancel one by label."
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "cancel"]},
            "which": {"type": "string", "description": "For cancel: the label (or id) of the reminder to cancel."},
        },
        "required": ["action"],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        if deps.scheduler is None:
            return "Reminders aren't available right now."
        if kwargs.get("action") == "cancel":
            n = deps.scheduler.cancel(str(kwargs.get("which", "")))
            return f"Cancelled {n}." if n else "I didn't find that one."
        pending = deps.scheduler.list_pending()
        if not pending:
            return "You have no pending reminders."
        parts = []
        for p in pending[:8]:
            m = int(p["in_s"] // 60)
            parts.append(f"{p.get('label') or p['message']} in {m} min" if m else f"{p.get('label') or p['message']} soon")
        return "Pending: " + "; ".join(parts) + "."
