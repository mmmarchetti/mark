"""Proactive notifications (spoken aloud). When the MacBook bridge is up, polls
for IMMINENT calendar events (starting within a short window) and announces them
- only when Mark is awake, idle, and not mid-conversation. No-ops when the bridge
isn't configured/reachable, so it's safe to always run.

Only genuinely imminent, TIMED events are announced. Previously it announced
every upcoming event (incl. all-day items and next-day meetings) the first time
it saw them, which spammed "Heads up: you have..." every poll and once spoke over
Mark as he was falling asleep.
"""

import datetime as dt
import logging
import threading
import time

from reachy_mini_brain import config

logger = logging.getLogger(__name__)


class Notifier:
    def __init__(self, bridge, listener, sleeping, on_announce, language=None) -> None:
        self.bridge = bridge
        self.listener = listener
        self.sleeping = sleeping
        self.on_announce = on_announce  # callable(text, language)
        self.language = language        # LanguageManager (manual reply language)
        self.enabled = True
        self._seen_events: set[str] = set()
        self._last_slack_ann = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.bridge is None or not getattr(self.bridge, "configured", False):
            logger.info("Notifier idle (bridge not configured).")
            return
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="notifier")
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _quiet_ok(self) -> bool:
        # Don't interrupt sleep, speech, or an active conversation.
        return (self.enabled and not self.sleeping.is_set()
                and not self.listener.speaking
                and time.time() >= self.listener._conversation_open_until)

    def _run(self) -> None:
        time.sleep(config.NOTIFIER_STARTUP_DELAY_S)
        logger.info("Proactive notifier started.")
        while not self._stop.is_set():
            time.sleep(config.NOTIFIER_POLL_S)
            if not self._quiet_ok():
                continue
            self._check_calendar()
        logger.info("Proactive notifier stopped.")

    def _check_calendar(self) -> None:
        data, err = self.bridge.get("/calendar/upcoming", {"days": 1})
        if err or not data:
            return
        now = dt.datetime.now(dt.timezone.utc)
        lead = dt.timedelta(minutes=config.NOTIFIER_LEAD_MINUTES)
        for e in data.get("events", []):
            key = f"{e.get('start','') or e.get('when','')}|{e.get('title','')}"
            if key in self._seen_events:
                continue
            # Only announce TIMED events that start within the lead window (e.g.
            # the next 30 min). Skip all-day items and anything further out - the
            # calendar tool handles "what's on today"; this is a last-minute nudge.
            if e.get("all_day"):
                self._seen_events.add(key)  # never a proactive heads-up
                continue
            start = self._parse_start(e.get("start"))
            if start is None:
                self._seen_events.add(key)
                continue
            if start < now:
                self._seen_events.add(key)  # already started/past
                continue
            if start - now > lead:
                continue  # not imminent yet - re-check next poll, don't mark seen
            self._seen_events.add(key)
            # Re-check the quiet gate right before speaking (tightens the race
            # against Mark going to sleep between the poll and here).
            if self._quiet_ok():
                lang = self.language.active if self.language is not None else config.DEFAULT_LANGUAGE
                title = e.get("title", "an event")
                when = e.get("when", "")
                text = (f"Lembrete: você tem '{title}' em breve, {when}." if lang == "pt"
                        else f"Heads up: you have '{title}' coming up, {when}.")
                self.on_announce(text, lang)
                break  # one announcement per cycle

    @staticmethod
    def _parse_start(iso: str | None):
        """Parse an event 'start' ISO string to an aware UTC datetime, or None."""
        if not iso:
            return None
        try:
            d = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
            if d.tzinfo is None:  # date-only or naive -> treat as local midnight
                d = d.replace(tzinfo=dt.timezone.utc)
            return d.astimezone(dt.timezone.utc)
        except Exception:
            return None
