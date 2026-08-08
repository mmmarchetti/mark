"""Conversation transcript log - append-only JSONL of every turn, searchable
from the web panel. Cheap, robust, and easy to grep offline.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

PATH = Path.home() / ".reachy_logs" / "transcript.jsonl"

# Self-capping so the log can never grow without bound on the Dell. When the
# file passes MAX_BYTES we trim it to the newest KEEP_LINES turns (a full turn
# is well under 1 KB, so ~10k turns of history). Trims are rare - the check is a
# cheap os.stat after each write, and a rewrite only fires when over the cap.
MAX_BYTES = int(os.getenv("REACHY_TRANSCRIPT_MAX_BYTES", str(8 * 1024 * 1024)))  # 8 MB
KEEP_LINES = int(os.getenv("REACHY_TRANSCRIPT_KEEP_LINES", "10000"))


class Transcript:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        PATH.parent.mkdir(parents=True, exist_ok=True)

    def log_turn(self, *, speaker: str, language: str, user_text: str,
                 reply: str, tools: list[str] | None = None) -> None:
        rec = {
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%d %H:%M:%S"),
            "speaker": speaker or "user",
            "language": language,
            "user_text": user_text,
            "reply": reply,
            "tools": tools or [],
        }
        try:
            with self._lock, PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._trim_if_large()
        except Exception:
            logger.exception("Failed to write transcript")

    def _trim_if_large(self) -> None:
        """Keep the newest KEEP_LINES lines once the file passes MAX_BYTES."""
        try:
            if PATH.stat().st_size <= MAX_BYTES:
                return
            with self._lock:
                lines = PATH.read_text(encoding="utf-8").splitlines()
                if len(lines) <= KEEP_LINES:
                    return
                tail = lines[-KEEP_LINES:]
                tmp = PATH.with_suffix(".jsonl.tmp")
                tmp.write_text("\n".join(tail) + "\n", encoding="utf-8")
                os.replace(tmp, PATH)  # atomic swap
                logger.info("Transcript trimmed to newest %d turns (was %d).",
                            len(tail), len(lines))
        except Exception:
            logger.exception("Transcript trim failed")

    def search(self, query: str = "", limit: int = 50) -> list[dict]:
        q = (query or "").lower().strip()
        out: list[dict] = []
        try:
            if not PATH.exists():
                return []
            # Read tail-ish: load all lines (files stay small); filter newest-first.
            lines = PATH.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines):
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not q or q in rec.get("user_text", "").lower() or q in rec.get("reply", "").lower():
                    out.append(rec)
                    if len(out) >= limit:
                        break
        except Exception:
            logger.exception("Transcript search failed")
        return out
