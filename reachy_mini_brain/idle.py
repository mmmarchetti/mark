"""Idle life: when Mark is awake but nobody's interacting for a while, do a small
spontaneous behavior (glance around, a subtle emotion) so it feels alive rather
than frozen. Suppressed during conversation, speech, sleep, or when a face is
actively being tracked.
"""

import logging
import random
import threading
import time

from reachy_mini_brain import config

logger = logging.getLogger(__name__)


class IdleLife:
    def __init__(self, reachy_mini, motion_queue, listener, sleeping) -> None:
        self.reachy_mini = reachy_mini
        self.motion_queue = motion_queue
        self.listener = listener
        self.sleeping = sleeping
        self.enabled = True
        self.last_interaction = time.time()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def poke(self) -> None:
        """Call on any interaction to reset the idle timer."""
        self.last_interaction = time.time()

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="idle")
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        time.sleep(config.IDLE_STARTUP_DELAY_S)
        logger.info("Idle-life started.")
        while not self._stop.is_set():
            time.sleep(config.IDLE_POLL_S)
            if not self.enabled or self.sleeping.is_set() or self.listener.speaking:
                continue
            # In a conversation? stay out of the way.
            if time.time() < self.listener._conversation_open_until:
                self.poke()
                continue
            if time.time() - self.last_interaction < config.IDLE_AFTER_S:
                continue
            # If a face is visible, face tracking owns the head - don't fight it.
            try:
                if getattr(self.reachy_mini.get_tracked_face(wait=False), "detected", False):
                    self.poke()
                    continue
            except Exception:
                pass
            # Do a small idle action, then reset the clock (with jitter).
            self.last_interaction = time.time() - random.uniform(0, config.IDLE_AFTER_S * 0.5)
            try:
                self._idle_action()
            except Exception:
                logger.exception("idle action failed")
        logger.info("Idle-life stopped.")

    def _idle_action(self) -> None:
        action = random.choice(["glance", "glance", "antenna", "emotion"])
        if action == "glance":
            yaw = random.uniform(-25, 25)
            self.motion_queue.put({"type": "goto_head", "yaw": yaw, "pitch": random.uniform(-8, 8),
                                   "roll": 0.0, "duration": 1.2})
            # drift back to centre after a beat
            self.motion_queue.put({"type": "goto_head", "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "duration": 1.5})
        elif action == "antenna":
            self.motion_queue.put({"type": "antennas", "pattern": random.choice(["perk", "wiggle"])})
        else:
            self.motion_queue.put({"type": "play_move", "name": random.choice(config.IDLE_EMOTIONS)})
