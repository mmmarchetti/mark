"""Rhythmic dances from reachy_mini_dances_library.

Unlike the pre-recorded emotion clips (played via async_play_move), these are
procedural moves: a function that returns per-frame head/antenna offsets given
a beat position. We run them in a short real-time control loop synced to a BPM,
mirroring the library's own dance_demo.py.
"""

import logging
import time

import numpy as np
from reachy_mini import utils

logger = logging.getLogger(__name__)

# ~10deg anti-shake antenna rest (see main.py ANTENNA_REST): parking antennas at
# 0deg leaves them in an unstable equilibrium that oscillates.
try:
    from reachy_mini.reachy_mini import INIT_ANTENNAS_JOINT_POSITIONS as _ANTENNA_REST
except Exception:
    _ANTENNA_REST = [-0.1745, 0.1745]

try:
    from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES
    _AVAILABLE = dict(AVAILABLE_MOVES)
except Exception:  # pragma: no cover - library optional
    logger.exception("Dances library not available")
    _AVAILABLE = {}

# A few that are too intense / flagged as combos to auto-offer.
_SKIP = {"headbanger_combo"}

CONTROL_DT = 0.01  # 100 Hz, as in the library's demo


def available_dances() -> list[str]:
    return sorted(k for k in _AVAILABLE if k not in _SKIP)


def play_dance(reachy_mini, name: str, bpm: float = 120.0, beats: float = 8.0,
               amplitude: float = 1.0, stop_flag=None) -> bool:
    """Run one procedural dance for `beats` beats at `bpm`. Returns True if played.

    `stop_flag` (optional callable -> bool) lets a caller cut it short.
    """
    entry = _AVAILABLE.get(name)
    if entry is None:
        return False
    move_fn, base_params, _ = entry

    params = base_params.copy()
    for key in list(params):
        if "amplitude" in key or "_amp" in key:
            params[key] *= amplitude

    neutral_pos = np.zeros(3)
    neutral_eul = np.zeros(3)
    beats_per_sec = bpm / 60.0
    t_beats = 0.0
    logger.info("Dancing %r (%.0f bpm, %.0f beats).", name, bpm, beats)
    while t_beats < beats:
        if stop_flag is not None and stop_flag():
            break
        loop_start = time.time()
        offsets = move_fn(t_beats, **params)
        final_pos = neutral_pos + offsets.position_offset
        final_eul = neutral_eul + offsets.orientation_offset
        reachy_mini.set_target(
            utils.create_head_pose(*final_pos, *final_eul, degrees=False),
            antennas=offsets.antennas_offset,
        )
        t_beats += beats_per_sec * CONTROL_DT
        # Maintain the control rate.
        dt = time.time() - loop_start
        if dt < CONTROL_DT:
            time.sleep(CONTROL_DT - dt)

    # Settle back to neutral so we don't leave the head at a dance extreme.
    try:
        reachy_mini.goto_target(head=utils.create_head_pose(), antennas=list(_ANTENNA_REST), duration=0.4)
    except Exception:
        logger.exception("Dance settle-to-neutral failed")
    return True
