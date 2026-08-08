"""One background scheduler thread for time-based spoken events: timers,
reminders, alarms, pomodoro check-ins, daily briefings. When an item is due it
fires a spoken event via the on_fire(text, language) callback (which pushes to
the brain's event_queue). Persists reminders/alarms so they survive a restart.
"""

import json
import logging
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

STORE = Path.home() / ".reachy_mini_reminders.json"


class Scheduler:
    def __init__(self, on_fire, default_language="pt") -> None:
        self.on_fire = on_fire            # callable(text, language)
        self.default_language = default_language
        self._lock = threading.Lock()
        self._items: list[dict] = self._load()      # persisted (reminders/alarms)
        self._ephemeral: list[dict] = []             # timers/pomodoro (not persisted)
        self._recurring: list[dict] = []             # daily briefings etc.
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- persistence ----
    def _load(self) -> list[dict]:
        try:
            if STORE.exists():
                return [x for x in json.loads(STORE.read_text()) if x.get("due", 0) > time.time()]
        except Exception:
            logger.exception("reminder load failed")
        return []

    def _save(self) -> None:
        try:
            STORE.write_text(json.dumps(self._items))
        except Exception:
            logger.exception("reminder save failed")

    # ---- public API (used by tools) ----
    def add_oneshot(self, delay_s: float, message: str, language=None, persist=False, label="") -> str:
        item = {"id": uuid.uuid4().hex[:8], "due": time.time() + max(1, delay_s),
                "message": message, "language": language or self.default_language, "label": label}
        with self._lock:
            (self._items if persist else self._ephemeral).append(item)
            if persist:
                self._save()
        return item["id"]

    def add_daily(self, hour: int, minute: int, builder, language=None, key="") -> None:
        """Recurring daily event. `builder()` returns the message text at fire time."""
        with self._lock:
            self._recurring.append({"hour": hour, "minute": minute, "builder": builder,
                                    "language": language or self.default_language,
                                    "key": key, "last": ""})

    def list_pending(self) -> list[dict]:
        with self._lock:
            now = time.time()
            return [{"id": i["id"], "label": i.get("label", ""), "message": i["message"],
                     "in_s": int(i["due"] - now)}
                    for i in sorted(self._items + self._ephemeral, key=lambda x: x["due"])
                    if i["due"] > now]

    def cancel(self, id_or_label: str) -> int:
        q = (id_or_label or "").lower().strip()
        removed = 0
        with self._lock:
            for lst in (self._items, self._ephemeral):
                before = len(lst)
                lst[:] = [i for i in lst if i["id"] != q and q not in i.get("label", "").lower()]
                removed += before - len(lst)
            self._save()
        return removed

    # ---- thread ----
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="scheduler")
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _fire(self, text: str, language: str) -> None:
        try:
            self.on_fire(text, language)
        except Exception:
            logger.exception("scheduler on_fire failed")

    def _run(self) -> None:
        logger.info("Scheduler started.")
        while not self._stop.is_set():
            time.sleep(1.0)
            now = time.time()
            due: list[dict] = []
            with self._lock:
                for lst, persist in ((self._items, True), (self._ephemeral, False)):
                    ready = [i for i in lst if i["due"] <= now]
                    for i in ready:
                        lst.remove(i)
                        due.append(i)
                    if persist and ready:
                        self._save()
                # recurring daily
                hm = time.strftime("%H:%M")
                today = time.strftime("%Y-%m-%d")
                for r in self._recurring:
                    if hm == f"{r['hour']:02d}:{r['minute']:02d}" and r["last"] != today:
                        r["last"] = today
                        try:
                            due.append({"message": r["builder"](), "language": r["language"]})
                        except Exception:
                            logger.exception("daily builder failed")
            for i in due:
                if i.get("message"):
                    self._fire(i["message"], i["language"])
        logger.info("Scheduler stopped.")
