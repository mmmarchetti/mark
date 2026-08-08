import logging
import time
from pathlib import Path
from typing import Any

import soundfile as sf

from reachy_mini_brain.tools.core import Tool, ToolDependencies

logger = logging.getLogger(__name__)
MEMO_DIR = Path.home() / "reachy_memos"


class RecordMemoTool(Tool):
    """Save the user's last spoken message as an audio memo."""

    name = "record_memo"
    description = (
        "Save the user's most recent spoken message as an audio voice-memo file. "
        "Use when they say something like 'save that', 'remember this recording', or "
        "'make a voice memo of that'."
    )
    parameters_schema = {
        "type": "object",
        "properties": {"label": {"type": "string", "description": "Optional short name for the memo."}},
        "required": [],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        listener = deps.listener
        if listener is None or getattr(listener, "last_utterance", None) is None:
            return "I don't have a recent recording to save."
        try:
            MEMO_DIR.mkdir(parents=True, exist_ok=True)
            label = str(kwargs.get("label", "")).strip().replace(" ", "_") or "memo"
            path = MEMO_DIR / f"{label}_{time.strftime('%Y%m%d-%H%M%S')}.wav"
            sf.write(str(path), listener.last_utterance, 16000)
            return f"Saved a voice memo as {path.name}."
        except Exception:
            logger.exception("record_memo failed")
            return "I couldn't save that memo."
