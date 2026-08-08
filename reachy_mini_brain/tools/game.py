from typing import Any

from reachy_mini_brain import games
from reachy_mini_brain.tools.core import Tool, ToolDependencies


class PlayGameTool(Tool):
    """Play a quick game with the user."""

    name = "play_game"
    description = (
        "Play a game with the user. Games: 'rock_paper_scissors' (uses the camera to "
        "read the player's hand - tell them to hold their hand up first), 'trivia', "
        "'twenty_questions', 'guess_number'. Use when the user asks to play a game or "
        "picks one. For the LLM-driven games this just starts them; keep playing over "
        "the following turns."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "game": {
                "type": "string",
                "enum": ["rock_paper_scissors", "trivia", "twenty_questions", "guess_number"],
            },
        },
        "required": ["game"],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        game = str(kwargs.get("game", "")).strip().lower()
        if game == "rock_paper_scissors":
            return games.play_rock_paper_scissors(deps.reachy_mini, deps.vision)
        setup = games.game_setup(game)
        if setup:
            return setup
        return "I don't know that game. I can play rock paper scissors, trivia, twenty questions, or guess the number."
