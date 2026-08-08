"""Speaker volume control via ALSA amixer on the Reachy Mini Audio card.

The daemon owns audio playback, but the OS mixer level is separate and is what
we've been tuning by hand (`amixer -c <card> sset 'PCM',0/1 <pct>%`). This wraps
that so the web panel can read and set it.
"""

import json
import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_CARD_CACHE: int | None = None

# Volume is NOT restored by anything at startup, and ALSA re-enumeration on a
# USB-C replug resets the card's saved mixer state (observed: PCM,1 mono fell
# back to -12dB while PCM,0 stayed at 0dB). Persist the intended level here so
# main.py can re-assert BOTH PCM controls on every start. Mirrors the other
# ~/.reachy_mini_brain_*.json settings files.
_VOLUME_STORE = Path.home() / ".reachy_mini_brain_volume.json"
_DEFAULT_VOLUME = 100


def _card() -> int | None:
    global _CARD_CACHE
    if _CARD_CACHE is not None:
        return _CARD_CACHE
    try:
        out = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=3).stdout
        for line in out.splitlines():
            if "Reachy Mini Audio" in line:
                m = re.match(r"card (\d+):", line)
                if m:
                    _CARD_CACHE = int(m.group(1))
                    return _CARD_CACHE
    except Exception:
        logger.exception("Could not determine Reachy audio card")
    return None


def _read_control_percent(card: int, control: str) -> int | None:
    try:
        out = subprocess.run(
            ["amixer", "-c", str(card), "sget", control],
            capture_output=True, text=True, timeout=3,
        ).stdout
        m = re.search(r"\[(\d+)%\]", out)
        if m:
            return int(m.group(1))
    except Exception:
        logger.exception("read control %s failed", control)
    return None


def get_volume_percent() -> int | None:
    """Return the EFFECTIVE speaker level.

    The card has two independent PCM playback controls - PCM,0 (stereo) and
    PCM,1 (mono). The mono channel gates the single speaker, so reporting only
    PCM,0 (the old behaviour) masked a PCM,1 stuck at -12dB and showed "100%"
    while the robot was quiet. Return the MIN of the two so the panel reflects
    what you actually hear.
    """
    card = _card()
    if card is None:
        return None
    vals = [
        v for v in (
            _read_control_percent(card, "PCM,0"),
            _read_control_percent(card, "PCM,1"),
        ) if v is not None
    ]
    return min(vals) if vals else None


def set_volume_percent(percent: int, persist: bool = True) -> bool:
    card = _card()
    if card is None:
        return False
    pct = max(0, min(100, int(percent)))
    ok = True
    for control in ("PCM,0", "PCM,1"):
        try:
            subprocess.run(
                ["amixer", "-c", str(card), "sset", control, f"{pct}%"],
                capture_output=True, text=True, timeout=3, check=True,
            )
        except Exception:
            logger.exception("set_volume failed for %s", control)
            ok = False
    if persist:
        _save_persisted_volume(pct)
    return ok


def get_persisted_volume() -> int:
    """The last volume the user set, restored across restarts/replugs. Defaults
    to 100 the first time (no store yet)."""
    try:
        if _VOLUME_STORE.exists():
            v = int(json.loads(_VOLUME_STORE.read_text()).get("volume", _DEFAULT_VOLUME))
            return max(0, min(100, v))
    except Exception:
        logger.exception("get_persisted_volume failed")
    return _DEFAULT_VOLUME


def _save_persisted_volume(pct: int) -> None:
    try:
        _VOLUME_STORE.write_text(json.dumps({"volume": int(pct)}))
    except Exception:
        logger.exception("save_persisted_volume failed")


def reassert_volume() -> bool:
    """Force both PCM controls back to the persisted level. Call at startup so a
    replug that reset ALSA's saved state (PCM,1 -> -12dB) doesn't leave the
    speaker quiet. Does NOT re-persist (it's reading the stored value)."""
    return set_volume_percent(get_persisted_volume(), persist=False)
