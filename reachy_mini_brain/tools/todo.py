from typing import Any

from reachy_mini_brain.tools.core import Tool, ToolDependencies


class TodoTool(Tool):
    """Manage the user's to-do list."""

    name = "todo"
    description = (
        "Manage the user's personal to-do list. action 'add' with text to add a task, "
        "'list' to read the open tasks, 'done' with text/number to complete one."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "list", "done"]},
            "text": {"type": "string", "description": "Task text (for add) or which item to complete (for done)."},
        },
        "required": ["action"],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        if deps.todo is None:
            return "The to-do list isn't available right now."
        action = kwargs.get("action")
        if action == "add":
            return deps.todo.add(str(kwargs.get("text", "")))
        if action == "done":
            return deps.todo.complete(str(kwargs.get("text", "")))
        return deps.todo.list_open()
