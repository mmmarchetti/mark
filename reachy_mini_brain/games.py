"""Small camera/vision games. Rock-paper-scissors reads the player's hand from
a camera frame via the local vision model; the LLM-driven games (trivia,
twenty-questions) just get kicked off with a setup instruction and then run
conversationally through normal turns.
"""

import logging
import random

logger = logging.getLogger(__name__)

_RPS = ("rock", "paper", "scissors")
_BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}


def play_rock_paper_scissors(reachy_mini, vision) -> str:
    """Capture the player's hand, compare to Mark's random throw, return a
    spoken-ready result line for the LLM to voice."""
    mine = random.choice(_RPS)
    theirs = None
    try:
        frame = reachy_mini.media.get_frame()
        if frame is not None and vision is not None:
            ans = vision.describe_frames(
                [frame],
                "The person is playing rock paper scissors. Look ONLY at their hand. "
                "Reply with exactly one lowercase word: rock, paper, scissors, or none.",
            ).strip().lower()
            for w in _RPS:
                if w in ans:
                    theirs = w
                    break
    except Exception:
        logger.exception("RPS vision read failed")

    if theirs is None:
        return (f"You threw {mine}. Tell the user you couldn't clearly see their hand - "
                f"ask them to hold rock, paper, or scissors up to the camera and try again.")

    if theirs == mine:
        outcome = f"It's a tie - you both threw {mine}."
    elif _BEATS[mine] == theirs:
        outcome = f"You win! You threw {mine}, they threw {theirs}."
    else:
        outcome = f"They win! You threw {mine}, they threw {theirs}."
    return (f"Rock-paper-scissors result (announce it playfully, react with an emotion): "
            f"{outcome}")


# LLM-driven games: return a setup instruction the model then narrates and runs
# over subsequent conversational turns. Kept text-only and short.
_SETUPS = {
    "trivia": (
        "Start a trivia game with the user. Ask ONE fun trivia question now, wait for "
        "their answer on the next turn, tell them if they're right, then offer another. "
        "Keep a running score in your head. Keep each message short."
    ),
    "twenty_questions": (
        "Start a game of twenty questions. Tell the user to think of an object; you will "
        "ask up to twenty yes/no questions to guess it. Ask your FIRST yes/no question now. "
        "Keep it to one short question per turn."
    ),
    "guess_number": (
        "Start a number-guessing game: pick a secret number between 1 and 100, tell the "
        "user to guess, and respond 'higher' or 'lower' each turn until they get it. "
        "Announce the game and ask for their first guess now."
    ),
}


def game_setup(name: str) -> str | None:
    return _SETUPS.get(name)
