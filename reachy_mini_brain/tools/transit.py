from typing import Any

from reachy_mini_brain.tools.core import Tool, ToolDependencies


class TransitTool(Tool):
    """Directions, commute time, and travel estimates via Google Maps."""

    name = "directions"
    description = (
        "Get travel time or directions between two places. Use for 'how long to work?', "
        "'commute time', 'how far is X', 'directions to Y'. You can say 'home' or 'work' "
        "and it uses the saved addresses. Default origin is home if not given. "
        "Modes: driving, walking, bicycling, transit (public transport)."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "destination": {"type": "string", "description": "Where to go (address, place, or 'work'/'home')."},
            "origin": {"type": "string", "description": "Starting point; defaults to home."},
            "mode": {"type": "string", "enum": ["driving", "walking", "bicycling", "transit"]},
        },
        "required": ["destination"],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        if deps.transit is None:
            return "Maps aren't available right now."
        return deps.transit.directions(
            destination=str(kwargs.get("destination", "")).strip(),
            origin=(str(kwargs.get("origin")).strip() if kwargs.get("origin") else None),
            mode=str(kwargs.get("mode", "driving")).strip().lower(),
        )
