"""Simple persistent local to-do list (JSON). No external account."""

import json
import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

STORE = Path.home() / ".reachy_mini_todo.json"


class TodoStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: list[dict] = self._load()

    def _load(self) -> list[dict]:
        try:
            if STORE.exists():
                return json.loads(STORE.read_text())
        except Exception:
            logger.exception("todo load failed")
        return []

    def _save(self) -> None:
        try:
            STORE.write_text(json.dumps(self._items, ensure_ascii=False))
        except Exception:
            logger.exception("todo save failed")

    def add(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "Add what?"
        with self._lock:
            self._items.append({"text": text, "done": False, "ts": time.time()})
            self._save()
        return f"Added: {text}."

    def list_open(self) -> str:
        with self._lock:
            open_ = [i["text"] for i in self._items if not i["done"]]
        if not open_:
            return "Your to-do list is empty."
        return "To-do: " + "; ".join(f"{i+1}. {t}" for i, t in enumerate(open_)) + "."

    def complete(self, text_or_index: str) -> str:
        q = (text_or_index or "").strip().lower()
        with self._lock:
            open_items = [i for i in self._items if not i["done"]]
            target = None
            if q.isdigit():
                idx = int(q) - 1
                if 0 <= idx < len(open_items):
                    target = open_items[idx]
            else:
                for i in open_items:
                    if q in i["text"].lower():
                        target = i
                        break
            if target is None:
                return "I couldn't find that item."
            target["done"] = True
            self._save()
            return f"Done: {target['text']}."
