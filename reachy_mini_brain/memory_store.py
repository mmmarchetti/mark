"""Simple persistent long-term memory: a list of facts on disk.

Facts survive restarts and are injected into the system prompt so the robot
remembers things about the user across sessions (name, preferences, etc.).
Kept intentionally simple - a JSON list of short strings - which is plenty for
a desktop companion and easy to inspect/edit by hand.
"""

import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path.home() / ".reachy_mini_brain_memory.json"


class MemoryStore:
    def __init__(self, path: Path | None = None, max_facts: int = 100) -> None:
        self.path = Path(path) if path else DEFAULT_PATH
        self.max_facts = max_facts
        self._lock = threading.Lock()
        self._facts: list[str] = self._load()

    def _load(self) -> list[dict]:
        """Facts are stored as {"text": str, "owner": str|None}. Older files
        held a plain list of strings; those load as shared facts (owner=None)."""
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    out = []
                    for x in data:
                        if isinstance(x, dict) and "text" in x:
                            out.append({"text": str(x["text"]), "owner": x.get("owner")})
                        else:
                            out.append({"text": str(x), "owner": None})
                    return out
        except Exception:
            logger.exception("Failed to load memory from %s", self.path)
        return []

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps(self._facts, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            logger.exception("Failed to save memory to %s", self.path)

    def add(self, fact: str, owner: str | None = None) -> bool:
        fact = fact.strip()
        if not fact:
            return False
        with self._lock:
            # Avoid near-duplicates for the same owner (case-insensitive).
            if any(fact.lower() == f["text"].lower() and f["owner"] == owner for f in self._facts):
                return False
            self._facts.append({"text": fact, "owner": owner})
            del self._facts[: max(0, len(self._facts) - self.max_facts)]
            self._save()
        logger.info("Memory added (owner=%s): %s", owner, fact)
        return True

    def forget(self, query: str, owner: str | None = None) -> int:
        """Remove matching facts. If owner is given, only that person's facts
        (and shared facts) are eligible, so one person can't wipe another's."""
        query = query.strip().lower()
        if not query:
            return 0
        with self._lock:
            before = len(self._facts)
            self._facts = [
                f for f in self._facts
                if not (query in f["text"].lower() and (owner is None or f["owner"] in (None, owner)))
            ]
            removed = before - len(self._facts)
            if removed:
                self._save()
        if removed:
            logger.info("Memory forgot %d fact(s) matching %r (owner=%s)", removed, query, owner)
        return removed

    def all(self, owner: str | None = None) -> list[str]:
        """Facts visible to `owner`: shared facts plus that person's own. With
        owner=None, returns every fact (used for counts/backup)."""
        with self._lock:
            if owner is None:
                return [f["text"] for f in self._facts]
            return [f["text"] for f in self._facts if f["owner"] in (None, owner)]

    def as_prompt_block(self, speaker: str | None = None) -> str:
        # When nobody's recognized, only surface shared facts - not every
        # enrolled person's private notes.
        with self._lock:
            if speaker is None:
                facts = [f["text"] for f in self._facts if f["owner"] is None]
            else:
                facts = [f["text"] for f in self._facts if f["owner"] in (None, speaker)]
        if not facts:
            return ""
        lines = "\n".join(f"- {f}" for f in facts)
        return (
            "Things you remember about the user from previous conversations "
            "(use them naturally, don't recite them):\n" + lines
        )
