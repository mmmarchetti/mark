"""Manual reply-language selection for Mark.

Mark used to auto-detect the reply language from each utterance via STT, but STT
mis-detects short/accented speech (e.g. "Gracias" -> Spanish, an English weather
question answered in Portuguese), which caused language mixing. Instead the user
picks ONE language via a button on the brain control panel; Mark always replies
in it. This also keeps the LLM prompt prefix byte-identical across turns, so the
local model's prompt cache stays hot (sub-second replies).
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_PATH = Path.home() / ".reachy_mini_brain_language.json"
SUPPORTED = ("en", "pt")
DEFAULT = "en"


class LanguageManager:
    def __init__(self) -> None:
        self.active = self._load()

    def _load(self) -> str:
        try:
            if STATE_PATH.exists():
                val = json.loads(STATE_PATH.read_text(encoding="utf-8")).get("active")
                if val in SUPPORTED:
                    return val
        except Exception:
            logger.exception("Failed to load language state")
        return DEFAULT

    def _save(self) -> None:
        try:
            STATE_PATH.write_text(json.dumps({"active": self.active}), encoding="utf-8")
        except Exception:
            logger.exception("Failed to save language state")

    def set(self, lang: str) -> bool:
        if lang not in SUPPORTED:
            return False
        self.active = lang
        self._save()
        logger.info("Reply language set to %s", lang)
        return True

    def toggle(self) -> str:
        self.set("pt" if self.active == "en" else "en")
        return self.active
