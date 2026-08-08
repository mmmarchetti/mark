import datetime as dt
from typing import Any

from reachy_mini_brain.tools.core import Tool, ToolDependencies

# How many events to hand the model to speak. Voice-design guidance (Alexa/
# Google) caps spoken lists at ~3-6 short items before cognitive load hurts;
# the model is told to voice at most 2-3 and offer "and N more".
_MAX_SPOKEN = 5


def _spoken_time(when_iso: str, all_day: bool) -> str:
    """A speech-ready time token in 24h words the TTS reads cleanly. The model
    localizes/naturalizes it further (e.g. '9 da manha' / '9 AM')."""
    if all_day or not when_iso:
        return "all day"
    try:
        d = dt.datetime.fromisoformat(when_iso.replace("Z", "+00:00"))
        h, m = d.hour, d.minute
        return f"{h}:{m:02d}" if m else f"{h}:00"
    except Exception:
        return ""


def _relative_day(when_iso: str, today: dt.date) -> str:
    """today | tomorrow | <weekday> | <M/D> - so the model never does date math."""
    try:
        d = dt.datetime.fromisoformat(when_iso.replace("Z", "+00:00")).date() \
            if "T" in when_iso else dt.date.fromisoformat(when_iso[:10])
    except Exception:
        return ""
    delta = (d - today).days
    if delta <= 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    if delta < 7:
        return d.strftime("%A").lower()   # weekday name, model translates
    return d.strftime("%-m/%-d")


class CalendarAgendaTool(Tool):
    """Read the user's Google Calendar (via the MacBook bridge).

    Returns a COMPACT STRUCTURED summary (count + next event + a few upcoming,
    each with a relative day and a spoken time) - NOT a flat list. The hard,
    error-prone bits for a small model (date math, counting, truncation) are
    computed here; the model only has to phrase it naturally in the user's
    language. See the SPEAKING ABOUT THE CALENDAR block in the system prompt.
    """

    name = "calendar_agenda"
    description = (
        "Check the user's Google Calendar and tell them what's coming up. Use when "
        "they ask about their schedule, today's/tomorrow's events, or their agenda. "
        "Read-only. Speak the result naturally in the user's language - lead with the "
        "count and the next event, mention only a couple, don't read the whole list."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "days": {"type": "integer", "description": "How many days ahead to include (default 1 = today)."},
        },
        "required": [],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        if deps.bridge is None:
            return "Calendar isn't available right now."
        days = int(kwargs.get("days", 1) or 1)
        data, err = deps.bridge.get("/calendar/upcoming", {"days": days})
        if err:
            return err
        events = data.get("events", [])
        window = "today" if days <= 1 else f"the next {days} days"
        if not events:
            return f"NO_EVENTS for {window}. Tell the user their schedule is clear, briefly."
        today = dt.date.today()

        def _fmt(e: dict) -> str:
            title = (e.get("title") or "an event").strip()
            iso = e.get("start", "")
            all_day = bool(e.get("all_day"))
            rel = _relative_day(iso, today)
            when = "all day" if all_day else _spoken_time(iso, all_day)
            # Compact "day @ time: title" the model rephrases; NOT to be read verbatim.
            return f"{rel} {when}: {title}" if rel else f"{when}: {title}"

        total = len(events)
        shown = events[:_MAX_SPOKEN]
        lines = [
            f"CALENDAR DATA (window: {window}). Speak this NATURALLY in the user's "
            f"language - lead with the count and the next item, mention 2-3 at most, "
            f"do NOT read the whole list. Use relative days and spoken times.",
            f"total_events: {total}",
            f"next_event: {_fmt(events[0])}",
            "upcoming:",
        ]
        lines += [f"  - {_fmt(e)}" for e in shown]
        remaining = total - len(shown)
        if remaining > 0:
            lines.append(f"remaining_not_shown: {remaining} (offer to list the rest)")
        return "\n".join(lines)


class CalendarCreateTool(Tool):
    """Create a Google Calendar event (write - needs confirmation)."""

    name = "calendar_create_event"
    description = (
        "Create a new event on the user's Google Calendar. This CHANGES their "
        "calendar, so first tell the user exactly what you'll create (title, date, "
        "time) and ask them to confirm; only call this with confirmed=true after "
        "they clearly say yes."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Event title."},
            "start": {"type": "string", "description": "Start datetime in ISO 8601, e.g. 2026-07-25T15:00:00."},
            "end": {"type": "string", "description": "End datetime in ISO 8601. Optional; defaults to 1h after start."},
            "confirmed": {"type": "boolean", "description": "Must be true; only set after the user confirms."},
        },
        "required": ["title", "start", "confirmed"],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        if not kwargs.get("confirmed"):
            # Guard rail: never write without explicit confirmation.
            return "Not confirmed yet. Tell the user the exact event and ask them to confirm first."
        if deps.bridge is None:
            return "Calendar isn't available right now."
        body = {
            "title": str(kwargs.get("title", "")).strip(),
            "start": str(kwargs.get("start", "")).strip(),
            "end": str(kwargs.get("end", "")).strip() or None,
        }
        data, err = deps.bridge.post("/calendar/event", body)
        if err:
            return err
        return "Done - the event is on the calendar." if data.get("ok") else "I couldn't create that event."
