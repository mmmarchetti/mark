"""Personality profiles: swappable character/tone for the robot.

Each profile is a short persona snippet appended to the base system prompt.
The active profile persists to disk so it survives restarts, and can be
switched by voice (via the set_personality tool) or config.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_PATH = Path.home() / ".reachy_mini_brain_profile.json"

# name -> (human label, persona instructions)
PROFILES: dict[str, tuple[str, str]] = {
    "friendly": (
        "Friendly companion",
        "Your personality: warm, upbeat and encouraging, like a friendly companion. "
        "You are curious about the user and enjoy small talk, but stay concise.",
    ),
    "assistant": (
        "Focused assistant",
        "Your personality: a calm, efficient assistant. Be helpful and to the point, "
        "minimal chit-chat, professional but not cold.",
    ),
    "playful": (
        "Playful jokester",
        "Your personality: playful and witty, you love light jokes and teasing banter, "
        "and you are expressive with your body. Keep it good-natured and never mean.",
    ),
    "zen": (
        "Calm and zen",
        "Your personality: serene, gentle and reassuring. You speak slowly and calmly, "
        "and you help the user feel relaxed.",
    ),
    "storyteller": (
        "Storyteller",
        "Your personality: a warm, engaging storyteller. When asked for a story or a "
        "joke, tell a vivid, fun, age-appropriate one - here you MAY speak for several "
        "sentences (ignore the one-sentence limit for the story itself). Otherwise stay "
        "friendly and concise.",
    ),
}
# Profiles allowed to speak at length (the usual 1-sentence rule is relaxed).
LONG_FORM_PROFILES = {"storyteller"}
DEFAULT_PROFILE = "friendly"


class ProfileManager:
    def __init__(self) -> None:
        self.active = self._load()

    def _load(self) -> str:
        try:
            if STATE_PATH.exists():
                name = json.loads(STATE_PATH.read_text(encoding="utf-8")).get("active")
                if name in PROFILES:
                    return name
        except Exception:
            logger.exception("Failed to load profile state")
        return DEFAULT_PROFILE

    def _save(self) -> None:
        try:
            STATE_PATH.write_text(json.dumps({"active": self.active}), encoding="utf-8")
        except Exception:
            logger.exception("Failed to save profile state")

    def set(self, name: str) -> bool:
        name = (name or "").strip().lower()
        if name not in PROFILES:
            return False
        self.active = name
        self._save()
        logger.info("Personality set to %r", name)
        return True

    def persona_text(self) -> str:
        return PROFILES[self.active][1]

    @staticmethod
    def names() -> list[str]:
        return list(PROFILES.keys())
