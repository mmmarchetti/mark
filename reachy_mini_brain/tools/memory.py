from typing import Any

from reachy_mini_brain.tools.core import Tool, ToolDependencies


class RememberTool(Tool):
    """Store a fact about the user for future conversations."""

    name = "remember"
    description = (
        "Save a durable fact about the user so you remember it in future "
        "conversations, even after a restart. Use when the user tells you "
        "something about themselves worth keeping (their name, preferences, "
        "important people/dates, how they like things done). Store one concise "
        "fact per call, written in the third person, e.g. 'The user's name is "
        "Marcos' or 'The user prefers short answers'."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "fact": {"type": "string", "description": "A single concise fact to remember."},
        },
        "required": ["fact"],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        fact = str(kwargs.get("fact", "")).strip()
        if not fact:
            return "Nothing to remember."
        if deps.memory is None:
            return "Memory isn't available right now."
        ok = deps.memory.add(fact, owner=deps.current_speaker)
        return "Got it, I'll remember that." if ok else "I already knew that."


class ForgetTool(Tool):
    """Forget previously stored facts matching a phrase."""

    name = "forget"
    description = (
        "Forget stored facts about the user that match a phrase. Use when the "
        "user asks you to forget something or corrects outdated information."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "about": {"type": "string", "description": "Phrase to match facts to forget."},
        },
        "required": ["about"],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        about = str(kwargs.get("about", "")).strip()
        if not about:
            return "Forget what?"
        if deps.memory is None:
            return "Memory isn't available right now."
        n = deps.memory.forget(about, owner=deps.current_speaker)
        return f"Okay, I've forgotten that." if n else "I didn't have anything like that stored."
