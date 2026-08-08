"""GenioDesk (Tuya Wi-Fi) control + the LLM-callable desk tool.

SAFETY MODEL (per the owner's explicit requirements):
  1. Every state-changing desk move is CONFIRM-BEFORE-ACT. The tool NEVER moves
     the desk itself - it only ARMS a pending action (deps.pending_desk) and
     returns a directive for Mark to ask the user to confirm. The move fires ONLY
     from the DETERMINISTIC yes/no gate in main.py's brain loop on the next turn
     (modeled on the go_to_sleep pending_sleep pattern). The LLM can therefore
     never move the desk in a single step, and a stray "yes-ish" word inside a
     long sentence won't trigger it (the gate requires a short, clear yes).
  2. No free / open-ended up-down. Up/down moves ONLY a specific number of
     centimeters the user names. If no cm is given, the tool arms nothing and
     tells Mark to ask "how many centimeters?" - it never guesses or free-runs.
  3. Presets are the SAFE primitive: dp3 level_N self-stops at the saved height,
     so sit / stand / end-of-day need no active stop watchdog.
  4. The cm-nudge (the only non-self-stopping move) is HARD-BOUNDED: clamped to
     [min_raw, max_raw] (the saved preset envelope), a max travel-time cap, an
     immediate stop on any fault, and a coast margin - and it stays DISABLED
     until a raw<->cm calibration exists (~/.desk_calibration.json), because the
     device reports raw units, not cm.

Local-only control (tinytuya v3.4) over the LAN - no cloud call per command.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

from reachy_mini_brain.tools.core import Tool, ToolDependencies

logger = logging.getLogger(__name__)

# Datapoint (DP) map - CONFIRMED by live local reads on this exact desk:
#   dp1  up_down     enum up/down/stop  (command AND live motion state)
#   dp2  work_state  enum up/down/stop
#   dp3  level       preset command level_1/level_2/level_3/level_4 (self-stops)
#   dp5  fault       bitmap (0 = healthy)
#   dp101 height     LIVE height in RAW units (NOT cm; L1=287 L4=314 L2=425)
DP_UPDOWN = "1"
DP_PRESET = "3"
DP_FAULT = "5"
DP_HEIGHT = "101"

# Friendly intent -> preset level. Sitting / "start my day" = 1; standing = 2;
# "end/finish my day" = 4 (chair fits under the desk). Level 3 is unused/free.
INTENT_TO_LEVEL = {
    "sitting": 1, "sit": 1, "start_of_day": 1, "start_my_day": 1,
    "standing": 2, "stand": 2,
    "end_of_day": 4, "end_my_day": 4, "finish_my_day": 4,
    "1": 1, "2": 2, "3": 3, "4": 4,
}

# Spoken description of each preset, per language, for confirmation + ack.
LEVEL_DESC = {
    1: {"en": "sitting position", "pt": "posição sentado"},
    2: {"en": "standing position", "pt": "posição em pé"},
    3: {"en": "the third preset", "pt": "o terceiro preset"},
    4: {"en": "end of day height, so your chair fits under the desk",
        "pt": "altura de fim de expediente, para a cadeira caber embaixo da mesa"},
}


def _int_to_words(n: int, lang: str) -> str:
    """Small integer -> spoken words (for cm confirmations spoken directly,
    bypassing the LLM). Covers the realistic nudge range; falls back to digits.
    """
    n = int(round(n))
    en = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
          6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
          12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
          16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
          20: "twenty", 30: "thirty", 40: "forty", 50: "fifty"}
    pt = {0: "zero", 1: "um", 2: "dois", 3: "três", 4: "quatro", 5: "cinco",
          6: "seis", 7: "sete", 8: "oito", 9: "nove", 10: "dez", 11: "onze",
          12: "doze", 13: "treze", 14: "quatorze", 15: "quinze", 16: "dezesseis",
          17: "dezessete", 18: "dezoito", 19: "dezenove", 20: "vinte",
          30: "trinta", 40: "quarenta", 50: "cinquenta"}
    table = pt if lang == "pt" else en
    if n in table:
        return table[n]
    if 21 <= n < 60:
        tens, ones = (n // 10) * 10, n % 10
        if ones and tens in table and ones in table:
            join = " e " if lang == "pt" else "-"
            return f"{table[tens]}{join}{table[ones]}"
    return str(n)


class DeskController:
    """Local Tuya control of the GenioDesk. Thread-safe for one nudge at a time."""

    def __init__(
        self,
        device_id: str,
        ip: str,
        key_path: str,
        version: float = 3.4,
        calib_path: str = "",
        min_raw: int = 287,
        max_raw: int = 425,
        nudge_time_cap_s: float = 40.0,
        coast_margin_raw: float = 3.0,
        status_timeout_s: float = 6.0,
    ) -> None:
        self.device_id = device_id
        self.ip = ip
        self.key_path = os.path.expanduser(key_path)
        self.version = version
        self.calib_path = os.path.expanduser(calib_path) if calib_path else ""
        self.min_raw = int(min_raw)
        self.max_raw = int(max_raw)
        self.nudge_time_cap_s = float(nudge_time_cap_s)
        self.coast_margin_raw = float(coast_margin_raw)
        self.status_timeout_s = float(status_timeout_s)

        self._dev = None
        self._lock = threading.Lock()          # serialize device I/O
        self._nudge_lock = threading.Lock()     # only one nudge at a time
        self._nudge_stop = threading.Event()
        self.raw_per_cm: Optional[float] = None  # slope, for cm nudges
        self.cm_per_raw: Optional[float] = None  # inverse slope, for display
        self.cm_at_zero: float = 0.0             # offset: cm = cm_per_raw*raw + cm_at_zero
        self._load_key()
        self._load_calibration()

    # --- setup -----------------------------------------------------------
    def _load_key(self) -> None:
        try:
            with open(self.key_path) as f:
                self._key = f.read().strip()
        except Exception:
            logger.exception("Desk local_key unreadable at %s", self.key_path)
            self._key = ""

    def _load_calibration(self) -> None:
        """Load raw<->cm calibration if present. Absent file => cm-nudge disabled
        (only self-stopping presets work). File shape (written by the setup step):
        {"raw_per_cm": float, "cm_per_raw": float, "cm_at_zero": float,
         "min_raw": int, "max_raw": int}. Only raw_per_cm is required for nudges;
        cm_per_raw/cm_at_zero give accurate absolute cm readouts."""
        self.raw_per_cm = None
        self.cm_per_raw = None
        self.cm_at_zero = 0.0
        if not self.calib_path or not os.path.exists(self.calib_path):
            return
        try:
            with open(self.calib_path) as f:
                data = json.load(f)
            rpc = float(data.get("raw_per_cm") or 0)
            if rpc > 0:
                self.raw_per_cm = rpc
                self.cm_per_raw = float(data.get("cm_per_raw") or (1.0 / rpc))
                self.cm_at_zero = float(data.get("cm_at_zero") or 0.0)
            if data.get("min_raw") is not None:
                self.min_raw = int(data["min_raw"])
            if data.get("max_raw") is not None:
                self.max_raw = int(data["max_raw"])
            logger.info("Desk calibration loaded: %.3f raw/cm, bounds [%d, %d].",
                        self.raw_per_cm or 0, self.min_raw, self.max_raw)
        except Exception:
            logger.exception("Desk calibration file unreadable: %s", self.calib_path)

    @property
    def calibrated(self) -> bool:
        return bool(self.raw_per_cm and self.raw_per_cm > 0)

    @property
    def nudging(self) -> bool:
        """True while a bounded cm move is actively running (its watchdog holds
        the nudge lock). Presets self-stop, so they aren't counted here."""
        if self._nudge_lock.acquire(blocking=False):
            self._nudge_lock.release()
            return False
        return True

    @property
    def available(self) -> bool:
        return bool(self._key)

    # --- low-level device I/O -------------------------------------------
    def _device(self):
        if self._dev is None:
            import tinytuya  # lazy: don't hard-depend at import time
            dev = tinytuya.Device(self.device_id, self.ip, self._key,
                                  version=self.version)
            dev.set_socketPersistent(True)
            dev.set_socketTimeout(self.status_timeout_s)
            self._dev = dev
        return self._dev

    def _reset_device(self) -> None:
        try:
            if self._dev is not None:
                self._dev.close()
        except Exception:
            pass
        self._dev = None

    def _status_raw(self) -> Dict[str, Any]:
        with self._lock:
            try:
                st = self._device().status()
            except Exception:
                logger.debug("Desk status failed; resetting socket", exc_info=True)
                self._reset_device()
                try:
                    st = self._device().status()
                except Exception:
                    logger.exception("Desk unreachable")
                    return {}
        return st.get("dps", {}) if isinstance(st, dict) else {}

    def _set(self, dp: str, value: Any) -> bool:
        with self._lock:
            try:
                self._device().set_value(dp, value)
                return True
            except Exception:
                logger.debug("Desk set_value failed; resetting socket", exc_info=True)
                self._reset_device()
                try:
                    self._device().set_value(dp, value)
                    return True
                except Exception:
                    logger.exception("Desk set_value(%s=%s) failed", dp, value)
                    return False

    # --- public reads ----------------------------------------------------
    def status(self) -> Dict[str, Any]:
        dps = self._status_raw()
        return {
            "raw_height": dps.get(DP_HEIGHT),
            "level": dps.get(DP_PRESET),
            "motion": dps.get(DP_UPDOWN),
            "fault": dps.get(DP_FAULT),
            "reachable": bool(dps),
        }

    def read_height_raw(self, retries: int = 3) -> Optional[int]:
        # dp101 reads None intermittently mid-move (device rate-limits status).
        for _ in range(max(1, retries)):
            dps = self._status_raw()
            h = dps.get(DP_HEIGHT)
            if isinstance(h, (int, float)):
                return int(h)
            time.sleep(0.4)
        return None

    def height_cm(self) -> Optional[float]:
        if not self.calibrated:
            return None
        raw = self.read_height_raw()
        if raw is None:
            return None
        if self.cm_per_raw:  # accurate affine map (slope + offset)
            return round(raw * self.cm_per_raw + self.cm_at_zero, 1)
        return round(raw / self.raw_per_cm, 1)

    # --- public writes ---------------------------------------------------
    def set_preset(self, level: int) -> bool:
        """Move to a saved preset. Self-stopping - safe fire-and-forget."""
        if level not in (1, 2, 3, 4):
            return False
        return self._set(DP_PRESET, f"level_{level}")

    def stop(self) -> bool:
        """Immediate stop - a safety action, always allowed without confirm."""
        self._nudge_stop.set()  # abort any nudge watchdog too
        return self._set(DP_UPDOWN, "stop")

    def start_nudge(self, delta_cm: float) -> bool:
        """Begin a bounded cm move on a watchdog thread. Returns False if the
        cm control isn't calibrated or another nudge is already running."""
        if not self.calibrated:
            return False
        if not self._nudge_lock.acquire(blocking=False):
            return False
        self._nudge_stop.clear()
        threading.Thread(target=self._run_nudge, args=(float(delta_cm),),
                         daemon=True, name="desk-nudge").start()
        return True

    def _run_nudge(self, delta_cm: float) -> None:
        try:
            start = self.read_height_raw()
            if start is None:
                logger.warning("Desk nudge aborted: can't read height.")
                return
            target = start + delta_cm * self.raw_per_cm
            # HARD clamp into the saved-preset envelope - never runaway.
            target = max(self.min_raw, min(self.max_raw, target))
            going_up = target > start
            if abs(target - start) <= self.coast_margin_raw:
                logger.info("Desk nudge: already within margin of target.")
                return
            direction = "up" if going_up else "down"
            logger.info("Desk nudge %s: %d -> %.0f raw (%.1f cm).",
                        direction, start, target, delta_cm)
            if not self._set(DP_UPDOWN, direction):
                return
            t0 = time.time()
            while time.time() - t0 < self.nudge_time_cap_s:
                time.sleep(0.4)
                if self._nudge_stop.is_set():
                    break
                dps = self._status_raw()
                fault = dps.get(DP_FAULT)
                if fault:  # any non-zero fault -> stop now
                    logger.warning("Desk fault %s during nudge - stopping.", fault)
                    break
                h = dps.get(DP_HEIGHT)
                if not isinstance(h, (int, float)):
                    continue
                # Stop early by the coast margin so momentum doesn't overshoot.
                if going_up and h >= target - self.coast_margin_raw:
                    break
                if (not going_up) and h <= target + self.coast_margin_raw:
                    break
        except Exception:
            logger.exception("Desk nudge crashed - forcing stop.")
        finally:
            self._set(DP_UPDOWN, "stop")
            self._nudge_lock.release()

    # --- execute an armed (already-confirmed) action ---------------------
    def execute_pending(self, pend: Dict[str, Any], language: str,
                        light: "Optional[LightController]" = None) -> str:
        """Run a confirmed action; return a plain SPOKEN confirmation line
        (this is spoken directly, bypassing the LLM, so no digits/symbols).

        A preset may carry a bundled light change (pend["light"] = "on"/"off"),
        used by the "start my day" (sit + lights on) and "finish my day" (end +
        lights off) routines. The light is flipped alongside the move."""
        lang = "pt" if language == "pt" else "en"
        kind = pend.get("kind")
        if kind == "preset":
            level = int(pend.get("level", 0))
            ok = self.set_preset(level)
            desc = LEVEL_DESC.get(level, {}).get(lang, "")
            if not ok:
                return ("Não consegui falar com a mesa agora." if lang == "pt"
                        else "I couldn't reach the desk just now.")
            # Bundled light change (start/finish my day).
            light_word = ""
            want = pend.get("light")
            if want in ("on", "off") and light is not None and light.available:
                light.set_on(want == "on")
                if lang == "pt":
                    light_word = (" e acendendo as luzes" if want == "on"
                                  else " e apagando as luzes")
                else:
                    light_word = (" and turning the lights on" if want == "on"
                                  else " and turning the lights off")
            return (f"Certo, movendo a mesa para a {desc}{light_word}." if lang == "pt"
                    else f"Okay, moving the desk to the {desc}{light_word}.")
        if kind == "nudge":
            delta = float(pend.get("delta_cm", 0))
            ok = self.start_nudge(delta)
            n = _int_to_words(abs(delta), lang)
            up = delta > 0
            if not ok:
                return ("Não consegui mover a mesa agora." if lang == "pt"
                        else "I couldn't move the desk just now.")
            if lang == "pt":
                verb = "subindo" if up else "descendo"
                return f"Certo, {verb} a mesa {n} centímetros."
            verb = "up" if up else "down"
            return f"Okay, moving the desk {verb} {n} centimeters."
        return ("Não entendi o comando da mesa." if lang == "pt"
                else "I didn't understand that desk command.")


class LightController:
    """Local Tuya control of the smart plug driving the top-of-desk lights.

    A plain on/off plug - NOT sensitive like the desk, so no confirm gate: on/off
    is immediate. Thread-safe for the single-turn calls Mark makes.

    SEND-THEN-VERIFY on FRESH connections (learned live 2026-08-05): this proto-3.5
    plug over a PERSISTENT tinytuya socket returns UNRELIABLE frames - after a
    command it can echo a stale/queued frame or an unsolicited power-reading update
    (dps 18/19/20) instead of the switch state, so a command that never took still
    looked like success (`set_value` returned a dict, no exception) and Mark said
    "turning the lights on" while nothing switched. Fix: use a NEW non-persistent
    connection per op, and after each set VERIFY the switch actually changed with a
    fresh status read, retrying until it matches. Fresh reads don't carry buffered
    frames, so the verify is trustworthy (proved 5/5 first-try vs the persistent
    path desyncing)."""

    def __init__(self, device_id: str, ip: str, key_path: str,
                 version: float = 3.5, switch_dp: str = "1",
                 status_timeout_s: float = 5.0, verify_tries: int = 3,
                 verify_delay_s: float = 0.6) -> None:
        self.device_id = device_id
        self.ip = ip
        self.key_path = os.path.expanduser(key_path)
        self.version = version
        self.switch_dp = str(switch_dp)
        self.status_timeout_s = float(status_timeout_s)
        self.verify_tries = int(verify_tries)
        self.verify_delay_s = float(verify_delay_s)
        self._lock = threading.Lock()
        try:
            with open(self.key_path) as f:
                self._key = f.read().strip()
        except Exception:
            logger.exception("Light local_key unreadable at %s", self.key_path)
            self._key = ""

    @property
    def available(self) -> bool:
        return bool(self._key)

    def _fresh_device(self):
        """A NEW non-persistent tinytuya device - no buffered/stale frames."""
        import tinytuya
        dev = tinytuya.Device(self.device_id, self.ip, self._key,
                              version=self.version)
        dev.set_socketTimeout(self.status_timeout_s)
        return dev

    def _read_switch(self) -> Optional[bool]:
        """One fresh status read of the switch dp (True/False, or None if unknown
        - e.g. the frame carried only a power-reading update)."""
        dev = self._fresh_device()
        try:
            dps = (dev.status() or {}).get("dps", {})
        finally:
            try:
                dev.close()
            except Exception:
                pass
        v = dps.get(self.switch_dp)
        return v if isinstance(v, bool) else None

    def set_on(self, on: bool) -> bool:
        """Set the switch and CONFIRM it actually changed (send-then-verify).
        Returns True only when a fresh read shows the requested state."""
        want = bool(on)
        with self._lock:
            last = None
            for attempt in range(1, self.verify_tries + 1):
                try:
                    dev = self._fresh_device()
                    try:
                        dev.set_value(self.switch_dp, want)
                    finally:
                        try:
                            dev.close()
                        except Exception:
                            pass
                    time.sleep(self.verify_delay_s)
                    last = self._read_switch()
                    if last == want:
                        return True
                    logger.info("Light set %s not confirmed (read=%r), retry %d/%d",
                                want, last, attempt, self.verify_tries)
                except Exception:
                    logger.debug("Light set attempt %d failed", attempt, exc_info=True)
            logger.warning("Light set_value(%s=%s) NOT confirmed after %d tries "
                           "(last read=%r)", self.switch_dp, want,
                           self.verify_tries, last)
            return False

    def is_on(self) -> Optional[bool]:
        with self._lock:
            for _ in range(2):
                try:
                    v = self._read_switch()
                    if v is not None:
                        return v
                except Exception:
                    pass
            return None


class DeskTool(Tool):
    """Control the standing desk AND the desk lights. Desk MOVES are sensitive:
    they ARM for confirmation and never move in one step. Lights are NOT
    sensitive: on/off is immediate. Routines: 'start my day' = sit + lights on;
    'finish my day' = end-of-day height + lights off (both confirmed, since they
    move the desk). Up/down needs an explicit centimeters value or Mark asks."""

    name = "desk"
    description = (
        "Control the user's standing desk and the lights on top of it. Actions: "
        "'start_my_day' (sit down AND turn the lights on) when the user says start "
        "my day / good morning / let's begin; 'finish_my_day' (go to end-of-day "
        "height AND turn the lights off) when they say finish/end my day / I'm done "
        "/ good night at the desk; 'preset' to just move to sitting, standing, or "
        "end_of_day WITHOUT changing lights; 'up'/'down' to move by a number of "
        "centimeters; 'light_on'/'light_off' to just switch the desk lights; "
        "'stop' to stop the desk; 'status' for the current height. "
        "For 'up'/'down' you MUST pass 'centimeters' - the exact amount the user "
        "said; if they did NOT say a number, call action 'ask_cm' instead (never "
        "guess a number). Every DESK MOVE (start_my_day, finish_my_day, preset, "
        "up, down) is confirmed first: after you call it, tell the user exactly "
        "what will happen and ask them to confirm - it happens only after they say "
        "yes on the next turn. Lights (light_on/light_off) happen immediately."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start_my_day", "finish_my_day", "preset", "up", "down",
                         "light_on", "light_off", "stop", "status", "ask_cm"],
                "description": (
                    "start_my_day = sit + lights on; finish_my_day = end-of-day "
                    "height + lights off; preset = move to a saved position only; "
                    "up/down = move by centimeters (requires 'centimeters'); "
                    "light_on/light_off = switch the desk lights immediately; "
                    "ask_cm = user wants up/down but gave no amount, so ask how "
                    "many; stop = stop the desk immediately; status = report height."
                ),
            },
            "position": {
                "type": "string",
                "enum": ["sitting", "standing", "end_of_day"],
                "description": (
                    "For action 'preset' only. sitting; standing; end_of_day = "
                    "finish-my-day height (chair fits under)."
                ),
            },
            "centimeters": {
                "type": "number",
                "description": "For action up/down: how many centimeters to move.",
            },
        },
        "required": ["action"],
    }

    def run(self, deps: ToolDependencies, **kwargs: Any) -> str:
        desk: Optional[DeskController] = getattr(deps, "desk", None)
        light: Optional[LightController] = getattr(deps, "light", None)
        action = str(kwargs.get("action", "")).strip().lower()

        # --- LIGHTS: not sensitive -> immediate, no confirmation ---
        if action in ("light_on", "light_off"):
            if light is None or not light.available:
                return "The desk lights aren't set up right now."
            want_on = action == "light_on"
            ok = light.set_on(want_on)
            if not ok:
                return "I couldn't reach the desk lights just now."
            return "Turned the lights on." if want_on else "Turned the lights off."

        if desk is None or not desk.available:
            return "The desk control isn't set up right now."

        # --- read-only, immediate ---
        if action == "status":
            st = desk.status()
            if not st.get("reachable"):
                return "I can't reach the desk right now."
            cm = desk.height_cm()
            light_txt = ""
            if light is not None and light.available:
                on = light.is_on()
                if on is not None:
                    light_txt = " Lights are on." if on else " Lights are off."
            if cm is not None:
                return (f"The desk is at about {cm} centimeters "
                        f"(preset {st.get('level')}).{light_txt}")
            return (f"The desk is at raw height {st.get('raw_height')} "
                    f"(preset {st.get('level')}); centimeter reading isn't "
                    f"calibrated yet.{light_txt}")

        # --- stop is a SAFETY action: fire immediately, no confirmation ---
        if action == "stop":
            deps.pending_desk = None  # cancel any armed move too
            ok = desk.stop()
            return "Stopped the desk." if ok else "I couldn't reach the desk to stop it."

        # --- up/down with no amount: arm NOTHING, ask how many cm ---
        if action == "ask_cm":
            return ("Tell the user you can move the desk up or down, and ask them "
                    "how many centimeters. Do not move the desk.")

        # --- ROUTINES: start/finish my day = preset move + bundled light ---
        # These MOVE the desk, so they still ARM + confirm; the light is applied
        # together with the move once the user says yes.
        if action == "start_my_day":
            # routine="start_my_day" tells the confirm gate in main.py to speak
            # the next calendar meeting AFTER the desk has moved + lights are on
            # and the spoken confirmation is delivered (owner's requested order).
            deps.pending_desk = {"kind": "preset", "level": 1, "light": "on",
                                 "routine": "start_my_day", "armed_at": time.time()}
            return ("You are about to start the day: move the desk to the sitting "
                    "position and turn the lights on. Tell the user exactly this "
                    "and ask them to confirm. Do NOT do it yet - only after they "
                    "say yes.")
        if action == "finish_my_day":
            deps.pending_desk = {"kind": "preset", "level": 4, "light": "off",
                                 "armed_at": time.time()}
            return ("You are about to finish the day: move the desk to the "
                    "end-of-day height so the chair fits under it, and turn the "
                    "lights off. Tell the user exactly this and ask them to "
                    "confirm. Do NOT do it yet - only after they say yes.")

        # --- plain preset move (no light change): ARM + ask to confirm ---
        if action == "preset":
            pos = str(kwargs.get("position", "")).strip().lower()
            level = INTENT_TO_LEVEL.get(pos)
            if level is None:
                return ("Ask the user whether they want sitting, standing, or "
                        "end-of-day height.")
            deps.pending_desk = {"kind": "preset", "level": level,
                                 "armed_at": time.time()}
            desc_en = LEVEL_DESC.get(level, {}).get("en", "")
            return (f"You are about to move the desk to the {desc_en}. Tell the "
                    "user exactly this and ask them to confirm before it moves. "
                    "Do NOT move it yet - it moves only after they say yes.")

        # --- cm nudge: require an amount, require calibration, then ARM ---
        if action in ("up", "down"):
            cm = kwargs.get("centimeters")
            try:
                cm = abs(float(cm))
            except (TypeError, ValueError):
                cm = 0.0
            if cm <= 0:
                return ("Tell the user you can move the desk up or down, and ask "
                        "them how many centimeters. Do not move the desk.")
            if not desk.calibrated:
                return ("Tell the user that moving by centimeters isn't calibrated "
                        "yet, so for now you can only use the sitting, standing, or "
                        "end-of-day presets. Do not move the desk.")
            delta = cm if action == "up" else -cm
            deps.pending_desk = {"kind": "nudge", "delta_cm": delta,
                                 "armed_at": time.time()}
            return (f"You are about to move the desk {action} by {cm:g} "
                    "centimeters. Tell the user exactly this and ask them to "
                    "confirm before it moves. Do NOT move it yet - it moves only "
                    "after they say yes.")

        return "I didn't understand that desk command."
