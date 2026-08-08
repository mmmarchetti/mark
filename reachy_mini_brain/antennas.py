"""Antennas as physical buttons.

The antenna motors use a low-P PID, so they are semi-passive and safe to push
by hand (per the SDK's interaction-patterns skill).

Detection is EDGE-TRIGGERED with an ADAPTIVE baseline, learned the hard way:
a fixed startup baseline plus "fire while deflected" produced an infinite
toggle loop when an antenna settled at a constant offset (0.57 rad) after an
emotion/wake move - it re-fired every cooldown forever. So instead:

* The rest baseline is a slow exponential moving average that tracks drift, but
  is frozen while a press is in progress so a held press isn't absorbed.
* A press fires ONCE on the rising edge (deflection crosses the press
  threshold and holds briefly). It will not fire again until the antenna is
  RELEASED (deflection falls back below a lower release threshold) - so a stuck
  or persistently-deflected antenna can never repeat-trigger.
"""

import logging
import threading
import time

from reachy_mini_brain import config

logger = logging.getLogger(__name__)


class AntennaButtons:
    def __init__(self, reachy_mini, on_press) -> None:
        self.reachy_mini = reachy_mini
        self.on_press = on_press
        self.enabled = True
        self._baseline: list[float] | None = None
        self._armed = True          # can a new press fire?
        self._press_candidate_since = 0.0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="antennas")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _read(self):
        try:
            return self.reachy_mini.get_present_antenna_joint_positions()
        except Exception:
            return None

    def _run(self) -> None:
        time.sleep(1.0)
        samples = []
        for _ in range(10):
            a = self._read()
            if a is not None:
                samples.append(a)
            time.sleep(0.05)
        if not samples:
            logger.warning("Antenna buttons: could not read baseline, disabling.")
            return
        n = len(samples[0])
        self._baseline = [sum(s[i] for s in samples) / len(samples) for i in range(n)]
        logger.info("Antenna buttons ready (baseline=%s).", [round(x, 2) for x in self._baseline])

        alpha = config.ANTENNA_BASELINE_ALPHA
        while not self._stop.is_set():
            time.sleep(config.ANTENNA_POLL_S)
            if not self.enabled or self._baseline is None:
                # While disabled (e.g. during a move/sleep) DISARM, so that when
                # re-enabled we don't fire on whatever deflected pose the move
                # left the antennas in - arming only happens after a clear
                # release back to rest (below).
                self._armed = False
                self._press_candidate_since = 0.0
                continue
            a = self._read()
            if a is None:
                continue

            deflection = max(
                abs(a[i] - self._baseline[i]) for i in range(min(len(a), len(self._baseline)))
            )
            now = time.time()
            pressed = deflection >= config.ANTENNA_PRESS_THRESHOLD_RAD
            released = deflection < config.ANTENNA_RELEASE_THRESHOLD_RAD

            if pressed:
                # Freeze the baseline (don't adapt toward a held press), and
                # only fire on a fresh, armed press held for the debounce time.
                if self._armed:
                    if self._press_candidate_since == 0.0:
                        self._press_candidate_since = now
                    elif now - self._press_candidate_since >= config.ANTENNA_PRESS_MIN_S:
                        self._armed = False  # edge consumed - won't fire again until released
                        self._press_candidate_since = 0.0
                        logger.info("Antenna press (deflection=%.2f rad).", deflection)
                        try:
                            self.on_press()
                        except Exception:
                            logger.exception("Antenna on_press callback failed")
                # If not armed (already fired, still held), do nothing - this is
                # exactly what stops the old infinite re-trigger loop.
            else:
                self._press_candidate_since = 0.0
                # Adapt the baseline toward the current rest position so slow
                # drift (settling after a move) is tracked, not mistaken for a
                # press. Re-arm once the antenna has clearly returned to rest.
                for i in range(len(self._baseline)):
                    self._baseline[i] = (1 - alpha) * self._baseline[i] + alpha * a[i]
                if released:
                    self._armed = True
