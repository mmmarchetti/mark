from typing import Any

from reachy_mini_brain.tools.core import Tool, ToolDependencies


class WeatherTool(Tool):
    """Get current weather and today's forecast."""

    name = "weather"
    description = (
        "Get the current weather and today's forecast. Use when the user asks "
        "about the weather, temperature, rain, or how the day is. Defaults to the "
        "user's current location (auto-detected); pass a city name to check "
        "somewhere else. Report the result naturally in the user's language."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "place": {
                "type": "string",
                "description": "Optional city/place name. Omit for the current location.",
            },
        },
        "required": [],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        if deps.weather is None:
            return "Weather isn't available right now."
        place = (kwargs.get("place") or "").strip() or None
        return deps.weather.describe(place)
