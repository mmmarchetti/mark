from typing import Any

from reachy_mini_brain.tools.core import Tool, ToolDependencies


class WebSearchTool(Tool):
    """Search the internet for current information."""

    name = "web_search"
    description = (
        "Search the internet for current or factual information the user asks about - "
        "news, facts, how-to, prices, people, anything you don't already know. Use it "
        "whenever a question needs up-to-date or external info. Summarize the results "
        "briefly in the user's language."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
        },
        "required": ["query"],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        if deps.search is None:
            return "Web search isn't available right now."
        return deps.search.search(str(kwargs.get("query", "")))
