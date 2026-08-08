import logging
import threading
import time
from typing import Any

from reachy_mini_brain import config
from reachy_mini_brain.tools.core import Tool, ToolDependencies

logger = logging.getLogger(__name__)

# Body yaw angles (degrees) to scan through. Well inside the documented
# +/-160 deg body limit. The sweep runs left -> centre -> right and then
# returns to centre.
_SCAN_ANGLES_DEG = (-60.0, 0.0, 60.0)
_MOVE_DURATION_S = 1.2
_MOVE_TIMEOUT_S = 8.0
# Just enough pause for the camera frame not to be motion-blurred. Kept small
# on purpose: the scan must feel like one continuous sweep.
_SETTLE_S = 0.25


class LookAroundTool(Tool):
    """Sweep the body around, grabbing a frame per corner, then describe it all."""

    name = "look_around"
    description = (
        "Physically rotate the robot's body to scan the whole room and describe "
        "everything around it. Use this when the user asks you to look around, "
        "check/scan the environment or the room, see what is around you, or find "
        "something nearby - anything that needs more than the single forward view "
        "the camera tool gives. This turns the robot's body, so it sees to the "
        "sides too, and takes several seconds."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "What to look for while scanning, e.g. describe the room, find the person.",
            },
        },
        "required": [],
    }

    @staticmethod
    def _goto(deps: ToolDependencies, body_yaw: float, wait: bool) -> bool:
        done = threading.Event() if wait else None
        action = {
            "type": "goto_head",
            "yaw": 0.0,  # head follows the body (resolved in main._dispatch_motion)
            "pitch": 0.0,
            "roll": 0.0,
            "body_yaw": body_yaw,
            "duration": _MOVE_DURATION_S,
        }
        if done is not None:
            action["done"] = done
        deps.motion_queue.put(action)
        if done is None:
            return True
        if not done.wait(timeout=_MOVE_TIMEOUT_S):
            logger.warning("look_around: timed out waiting for body_yaw=%s", body_yaw)
            return False
        return True

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        question = str(kwargs.get("question") or "Describe what you see.").strip()

        # Face tracking AND sound orienting would both fight this deliberate
        # sweep (each pulls the head back toward the user), so pause them.
        tracking_paused = False
        try:
            deps.reachy_mini.stop_head_tracking()
            tracking_paused = True
        except Exception:
            logger.exception("look_around: could not pause head tracking")
        if deps.doa is not None:
            deps.doa.enabled = False

        try:
            # Phase 1: sweep and CAPTURE ONLY. Frame grabs take milliseconds,
            # so the body keeps moving corner to corner without the long stops
            # that per-angle vision inference used to cause.
            frames = []
            labels = []
            for angle in _SCAN_ANGLES_DEG:
                if not self._goto(deps, angle, wait=True):
                    continue
                time.sleep(_SETTLE_S)
                frame = deps.reachy_mini.media.get_frame()
                if frame is None:
                    logger.warning("look_around: no camera frame at body_yaw=%s", angle)
                    continue
                frames.append(frame)
                labels.append(
                    "to the left" if angle < 0 else ("to the right" if angle > 0 else "straight ahead")
                )

            # Always finish facing forward rather than leaving the body twisted.
            self._goto(deps, 0.0, wait=True)
        finally:
            if tracking_paused:
                try:
                    deps.reachy_mini.start_head_tracking(weight=config.HEAD_TRACKING_WEIGHT)
                except Exception:
                    logger.exception("look_around: could not resume head tracking")
            if deps.doa is not None:
                deps.doa.enabled = True

        if not frames:
            return "I couldn't complete the scan of the room."

        # Phase 2: only now, after the whole sweep, process every captured
        # frame - in ONE request rather than one per angle.
        prompt = (
            f"These {len(frames)} photos were taken by a robot turning on the spot, "
            f"in this order: {', '.join(labels)}. Together they cover the room around it. "
            f"{question} Answer with one short combined description of the surroundings."
        )
        return deps.vision.describe_frames(frames, prompt)
