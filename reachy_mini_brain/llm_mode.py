"""Runtime LLM backend selection for Mark (brain control-panel buttons).

Mark can answer from two backends: the fast LAN model on the MacBook (~0.6s warm)
or the OpenAI cloud (5-13s, but always available). Which one is used per turn is
picked here, live, from a button on the brain page - mirroring LanguageManager so
the choice survives a restart:

  * local  - always use the MacBook model first (still falls back to the cloud on a
             hard failure so Mark never goes silent).
  * cloud  - always use the OpenAI cloud; never touch the MacBook.
  * auto   - the application decides BY RESPONSE TIME: prefer local while it is
             healthy and fast; if a local call fails/times out or its measured
             latency creeps above a target, transparently use the cloud and
             re-probe local a little later. This is the recommended default.

The value is read on EVERY turn (not cached at init), so flipping a button takes
effect on the next utterance with no restart.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_PATH = Path.home() / ".reachy_mini_brain_llm_mode.json"
SUPPORTED = ("local", "cloud", "auto")
DEFAULT = "auto"


class LLMModeManager:
    def __init__(self, default: str = DEFAULT) -> None:
        self._default = default if default in SUPPORTED else DEFAULT
        self.active = self._load()

    def _load(self) -> str:
        try:
            if STATE_PATH.exists():
                val = json.loads(STATE_PATH.read_text(encoding="utf-8")).get("active")
                if val in SUPPORTED:
                    return val
        except Exception:
            logger.exception("Failed to load LLM-mode state")
        return self._default

    def _save(self) -> None:
        try:
            STATE_PATH.write_text(json.dumps({"active": self.active}), encoding="utf-8")
        except Exception:
            logger.exception("Failed to save LLM-mode state")

    def set(self, mode: str) -> bool:
        if mode not in SUPPORTED:
            return False
        self.active = mode
        self._save()
        logger.info("LLM backend mode set to %s", mode)
        return True
