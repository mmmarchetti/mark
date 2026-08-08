"""Face-presence events: notice when a person arrives or leaves.

Polls the daemon's face tracker (already running for head tracking) and, with
debouncing, emits "someone appeared" / "the person left" events. main turns an
arrival into a spontaneous greeting turn so the robot acknowledges people
rather than only ever reacting to speech.
"""

import logging
import threading
import time

from reachy_mini_brain import config

logger = logging.getLogger(__name__)


class PresenceMonitor:
    def __init__(self, reachy_mini, on_arrival, on_departure) -> None:
        self.reachy_mini = reachy_mini
        self.on_arrival = on_arrival
        self.on_departure = on_departure
        self.enabled = True
        self._present = False
        self._last_seen = 0.0
        self._first_seen = 0.0
        self._last_event = 0.0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="presence")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _face_detected(self) -> bool:
        try:
            face = self.reachy_mini.get_tracked_face(wait=False)
            return bool(getattr(face, "detected", False))
        except Exception:
            return False

    def _run(self) -> None:
        time.sleep(config.PRESENCE_STARTUP_DELAY_S)  # stagger vs other daemon-polling threads
        logger.info("Presence monitor started.")
        while not self._stop.is_set():
            time.sleep(config.PRESENCE_POLL_S)
            if not self.enabled:
                continue
            now = time.time()
            detected = self._face_detected()

            if detected:
                self._last_seen = now
                if not self._present:
                    if self._first_seen == 0.0:
                        self._first_seen = now
                    # Require a face to persist briefly before greeting, so a
                    # one-frame false positive doesn't trigger a greeting.
                    elif now - self._first_seen >= config.PRESENCE_ARRIVAL_CONFIRM_S:
                        self._present = True
                        self._first_seen = 0.0
                        if now - self._last_event >= config.PRESENCE_EVENT_COOLDOWN_S:
                            self._last_event = now
                            self._fire(self.on_arrival, "arrival")
            else:
                self._first_seen = 0.0
                if self._present and (now - self._last_seen) >= config.PRESENCE_DEPARTURE_S:
                    self._present = False
                    if now - self._last_event >= config.PRESENCE_EVENT_COOLDOWN_S:
                        self._last_event = now
                        self._fire(self.on_departure, "departure")
        logger.info("Presence monitor stopped.")

    @staticmethod
    def _fire(cb, kind: str) -> None:
        logger.info("Presence event: %s", kind)
        try:
            cb()
        except Exception:
            logger.exception("Presence %s callback failed", kind)
