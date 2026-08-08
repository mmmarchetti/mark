"""Sound Direction-of-Arrival: orient toward a speaker not yet in view.

Runs in a background thread. DoA is a COMPLEMENT to face tracking, not a
competitor: when a face is already being tracked the daemon is aiming the head
precisely, so we stay out of the way. We only nudge the head toward the sound
when speech is heard but no face is visible - i.e. someone is talking from a
direction the robot isn't looking - so it turns to find them, after which face
tracking takes over.
"""

import logging
import threading
import time

import numpy as np

from reachy_mini_brain import config

logger = logging.getLogger(__name__)


class DoaOrienter:
    def __init__(self, reachy_mini) -> None:
        self.reachy_mini = reachy_mini
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.enabled = True  # paused during sleep / scans

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="doa")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _face_visible(self) -> bool:
        try:
            face = self.reachy_mini.get_tracked_face(wait=False)
            return bool(getattr(face, "detected", False))
        except Exception:
            return False

    def _run(self) -> None:
        last_angle = None
        last_face_seen = 0.0
        last_turn = 0.0
        speech_run_s = 0.0  # how long sound has persisted from ~the same direction
        # Small stagger so the background threads (DoA/presence/antennas) and
        # the head-tracking re-assert don't all hit the daemon websocket in the
        # same instant at startup, which caused a transient disconnect.
        time.sleep(config.DOA_STARTUP_DELAY_S)
        logger.info("DoA orienting thread started.")
        while not self._stop.is_set():
            time.sleep(config.DOA_POLL_S)
            if not self.enabled:
                continue

            # Keep a running "face seen recently" timestamp. Face detection
            # flickers (True/False/False/True even with someone right there),
            # so a single False frame must NOT let DoA grab the head away from
            # tracking - only a SUSTAINED absence counts as "no one visible".
            if self._face_visible():
                last_face_seen = time.time()

            try:
                doa = self.reachy_mini.media.get_DoA()
            except Exception:
                logger.exception("get_DoA failed")
                continue
            if doa is None:
                continue
            angle, speech = doa
            now = time.time()
            if not speech:
                speech_run_s = 0.0
                continue
            # Require sound to persist for a bit before reacting, so a single
            # word / passing noise doesn't make the head twitch.
            speech_run_s += config.DOA_POLL_S

            # Defer to face tracking unless no face has been seen for a while.
            if now - last_face_seen < config.DOA_FACE_GRACE_S:
                last_angle = None
                continue
            # Conservative gates: sustained speech, a real directional change,
            # and a cooldown so it turns occasionally rather than constantly.
            if speech_run_s < config.DOA_MIN_SPEECH_S:
                continue
            if last_angle is not None and abs(angle - last_angle) < config.DOA_MIN_CHANGE_RAD:
                continue
            if now - last_turn < config.DOA_TURN_COOLDOWN_S:
                continue
            last_angle = angle
            last_turn = now
            try:
                # Convert the head-relative bearing to a world point, but only
                # turn PART-WAY toward it (a small glance, not a full snap) and
                # clamp the horizontal angle so the head never swings far.
                capped = max(-config.DOA_MAX_ANGLE_RAD,
                             min(config.DOA_MAX_ANGLE_RAD, angle * config.DOA_TURN_FRACTION))
                p_head = np.array([np.sin(capped), np.cos(capped), 0.0])
                r_world_head = self.reachy_mini.get_current_head_pose()[:3, :3]
                p_world = r_world_head @ p_head
                self.reachy_mini.look_at_world(
                    float(p_world[0]), float(p_world[1]), float(p_world[2]),
                    duration=config.DOA_MOVE_DURATION_S,
                )
                logger.info("DoA: turned toward speech at %.2f rad (no face visible).", angle)
            except Exception:
                logger.exception("DoA look_at_world failed")
        logger.info("DoA orienting thread stopped.")
