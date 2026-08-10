"""Central configuration for the Reachy Mini bilingual conversation brain."""

import os

from dotenv import load_dotenv

load_dotenv()

# --- LLM (cloud) ---
OPENAI_MODEL = os.getenv("REACHY_LLM_MODEL", "gpt-5.4-nano")
LLM_MAX_TOKENS = int(os.getenv("REACHY_LLM_MAX_TOKENS", "80"))
# Longer cap for storyteller mode (see profiles.LONG_FORM_PROFILES).
LLM_MAX_TOKENS_LONG = int(os.getenv("REACHY_LLM_MAX_TOKENS_LONG", "400"))
# Fail fast on a stuck cloud call: without an explicit timeout the OpenAI client
# defaults to 600s + 2 silent retries, which is exactly the "no answer / huge
# hang" symptom. A short timeout lets the router fall back (or speak the error).
CLOUD_TIMEOUT_S = float(os.getenv("REACHY_LLM_CLOUD_TIMEOUT_S", "8"))
CLOUD_MAX_RETRIES = int(os.getenv("REACHY_LLM_CLOUD_MAX_RETRIES", "1"))
# Rough $/1M-token prices for the cloud model, used ONLY to show an estimated
# cost on the dashboard (real billing needs an org-admin key we don't use).
# Set these to the current gpt-5.4-nano rate; 0 = don't show a cost.
CLOUD_PRICE_IN_PER_1M = float(os.getenv("REACHY_OPENAI_PRICE_IN", "0"))
CLOUD_PRICE_OUT_PER_1M = float(os.getenv("REACHY_OPENAI_PRICE_OUT", "0"))

# --- LLM (local, on the second Mac over Tailscale) ---
# An OpenAI-compatible mlx_lm.server. When set, the brain routes LOCAL-FIRST and
# falls back to the cloud on any error/timeout/empty reply. Unset -> cloud only
# (today's behavior). See the hybrid-router plan.
LOCAL_LLM_URL = os.getenv("REACHY_LOCAL_LLM_URL", "")  # e.g. http://100.84.161.85:8081/v1
LOCAL_LLM_MODEL = os.getenv("REACHY_LOCAL_LLM_MODEL", "")  # e.g. mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
# Warm/cached turns are ~0.5s. A COLD load (model paged out after idle, or first
# turn post-startup) was measured at ~11s when the M4's RAM was tight and ~4s
# when it had headroom. 12s was right at that edge and tripped on a slow cold
# load -> "Connection error" -> cloud. 18s rides out even a worst-case cold load
# so it stays local instead of falling back. Warm turns are unaffected (they
# finish in <1s and never approach this ceiling).
LOCAL_LLM_TIMEOUT_S = float(os.getenv("REACHY_LOCAL_LLM_TIMEOUT_S", "18"))
# mlx_lm.server usually honors stream_options.include_usage for token counts; if a
# given build rejects it (breaking local streaming), set this to 0 to drop it.
LOCAL_LLM_STREAM_USAGE = os.getenv("REACHY_LOCAL_LLM_STREAM_USAGE", "1") not in ("0", "false", "False")
# After a local failure, skip local for this long so we don't hammer a Mac that's
# off (or busy serving the code model during an `llm-on` session) every turn.
# 15s (was 30s): the model now recovers/reloads fast, so a single cold-load blip
# shouldn't force half a minute of cloud-only turns - retry local sooner.
LOCAL_LLM_COOLDOWN_S = float(os.getenv("REACHY_LOCAL_LLM_COOLDOWN_S", "15"))
LLM_ROUTING = os.getenv("REACHY_LLM_ROUTING", "local_first")  # local_first | cloud_only
# Runtime backend mode chosen from the brain page (local | cloud | auto), persisted
# in ~/.reachy_mini_brain_llm_mode.json and read live each turn. This env var only
# seeds the FIRST run before a button is pressed; after that the saved file wins.
# "auto" = the app picks by RESPONSE TIME: prefer local until its recent average
# latency exceeds LLM_AUTO_TARGET_S (then use cloud and re-probe local later).
LLM_MODE_DEFAULT = os.getenv("REACHY_LLM_MODE", "auto")  # local | cloud | auto
# In "auto" mode, if the local backend's recent average latency is above this many
# seconds, prefer the cloud until local recovers. Warm local is ~0.6s; the cloud is
# 5-13s, so anything under a few seconds still strongly favors staying local.
LLM_AUTO_TARGET_S = float(os.getenv("REACHY_LLM_AUTO_TARGET_S", "4.0"))
# In "auto" mode, exponential backoff on how often to RE-PROBE a slow/busy local
# backend. The 24 GB M4 fits only ONE ~17 GB model at a time, so each probe while
# you're coding forces an ~18-22s chat<->code model SWAP that evicts your coder.
# Backoff starts at LOCAL_LLM_COOLDOWN_S and DOUBLES per slow/failed local turn up
# to this cap; a fast local turn (<= target) resets it, and pressing any backend
# button resets it too (see llm.py). This turns "one swap every 15s during a
# coding session" into "a swap or two, then settle on cloud." Tip: for a long
# coding session just press the API (cloud) button - zero swaps, coder stays warm.
LLM_AUTO_BACKOFF_MAX_S = float(os.getenv("REACHY_LLM_AUTO_BACKOFF_MAX_S", "300"))

# --- Multi-agent router (gate + per-task specialists; see docs/ARCHITECTURE.md) ---
# ROUTER_ENABLED now defaults ON: the deterministic gate + six specialists are the
# standard layout after the live rollout (cache preserved across domain switches;
# routing 30/30). Set MARK_ROUTER_ENABLED=0 for the instant revert to MONOLITH_AGENT
# (byte-for-byte the pre-refactor single prompt). The other two stay OFF by default.
# ROUTER_ENABLED: turn deterministic keyword+sticky routing on.
ROUTER_ENABLED = os.getenv("MARK_ROUTER_ENABLED", "1") not in ("0", "false", "no")
# ROUTER_CLASSIFY_ENABLED: on a genuine keyword tie, allow ONE tiny CLOUD-only
# classify to break it (a local classify would evict the local prompt cache).
ROUTER_CLASSIFY_ENABLED = os.getenv("MARK_ROUTER_CLASSIFY_ENABLED", "0") not in ("0", "false", "no")
# ROUTER_HANDOFF_ENABLED: if the first completion's tool calls all fall outside
# the routed specialist's advisory set, re-run the turn once for the owning
# specialist (cheap - only the cached-prefix tail recomputes).
ROUTER_HANDOFF_ENABLED = os.getenv("MARK_ROUTER_HANDOFF_ENABLED", "0") not in ("0", "false", "no")

# --- M4 host metrics (the MacBook Pro that runs the local LLM) ---
# The local model runs on a *different* machine than this brain, so the dashboard
# can only show that Mac's CPU/RAM/GPU/VRAM if the brain PULLS them. A tiny
# read-only agent on the M4 (mac_bridge/m4_metrics/mark_metrics_agent.py) exposes
# GET /metrics; a poll thread here feeds the dashboard's "M4 (LLM host)" panel.
# Empty URL -> the panel is simply absent (graceful, like the local LLM itself).
# Default derived from LOCAL_LLM_URL's host when set (see _m4_metrics_url()).
M4_METRICS_URL = os.getenv("REACHY_M4_METRICS_URL", "")  # e.g. http://100.84.161.85:8790/metrics
M4_METRICS_SECRET = os.getenv("REACHY_M4_METRICS_SECRET", "")  # matches the agent's MARK_METRICS_SECRET
M4_METRICS_POLL_S = float(os.getenv("REACHY_M4_METRICS_POLL_S", "5"))

# --- Safe shutdown (brain-page "Shut down Mark" button) ---
# Stopping these two USER systemd units is the safe power-off path. ORDER matters:
# stop the brain FIRST (its unit has KillSignal=SIGINT, so systemctl stop triggers
# the app's graceful _safe_shutdown -> goto_sleep parks the head WHILE torque is
# still on, then disable_motors cuts torque with the head already resting = no
# head-drop), THEN stop the daemon (releases the motor/media backend). Both units
# are Restart=on-failure, so a clean stop does NOT auto-relaunch them. This does
# NOT touch the M4 local LLM (a separate machine used by other services). After
# this completes, the physical OFF button is safe to press. All overridable.
SHUTDOWN_ENABLED = os.getenv("REACHY_SHUTDOWN_ENABLED", "1") not in ("0", "false", "False")
SHUTDOWN_BRAIN_UNIT = os.getenv("REACHY_SHUTDOWN_BRAIN_UNIT", "reachy-brain.service")
SHUTDOWN_DAEMON_UNIT = os.getenv("REACHY_SHUTDOWN_DAEMON_UNIT", "reachy-daemon.service")


def m4_metrics_url() -> str:
    """Resolved M4 metrics URL: explicit env, else derive from LOCAL_LLM_URL host.

    The M4 metrics agent shares the LLM host, so if only LOCAL_LLM_URL is set
    (e.g. http://100.84.161.85:8080/v1) we reuse that host on the metrics port.
    """
    if M4_METRICS_URL:
        return M4_METRICS_URL
    if LOCAL_LLM_URL:
        try:
            from urllib.parse import urlparse
            host = urlparse(LOCAL_LLM_URL).hostname
            if host:
                return f"http://{host}:8790/metrics"
        except Exception:
            pass
    return ""

# --- STT (local, GPU) ---
STT_MODEL = os.getenv("REACHY_STT_MODEL", "large-v3-turbo")
STT_DEVICE = os.getenv("REACHY_STT_DEVICE", "cuda")
STT_COMPUTE_TYPE = os.getenv("REACHY_STT_COMPUTE_TYPE", "int8_float16")

# --- TTS (local, CPU - Piper is fast enough without GPU) ---
# Piper for both languages - one consistent engine/voice character instead of
# switching between Kokoro (en) and Piper (pt), which sounded jarring live.
TTS_VOICES = {
    "en": {"piper_model": "en_US-ryan-high.onnx"},
    "pt": {"piper_model": "pt_BR-faber-medium.onnx"},
}
DEFAULT_LANGUAGE = os.getenv("REACHY_DEFAULT_LANGUAGE", "en")

# --- Idle life (spontaneous behaviors when nobody's interacting) ---
IDLE_STARTUP_DELAY_S = float(os.getenv("REACHY_IDLE_STARTUP_DELAY_S", "8"))
IDLE_POLL_S = float(os.getenv("REACHY_IDLE_POLL_S", "10"))
IDLE_AFTER_S = float(os.getenv("REACHY_IDLE_AFTER_S", "90"))  # quiet time before an idle move
IDLE_EMOTIONS = ["curious1", "thoughtful1", "boredom1", "attentive1", "waiting"]

# --- Custom neural wake-word ("Hey Mark") ---
# A small classifier (scripts/train_wakeword.py) over openWakeWord embeddings.
# OFF by default: with only two Piper voices for synthetic training data it
# can't robustly reject the near-homophones "market"/"Marcos", so the whole-word
# fuzzy STT matcher stays the primary trigger (it handles those correctly).
# Enable to experiment, or after retraining on a richer dataset.
WAKEWORD_NEURAL_ENABLED = os.getenv("REACHY_WAKEWORD_NEURAL", "0") not in ("0", "false", "no")

# --- Identity: recognize enrolled people by face + voice ---
IDENTITY_ENABLED = os.getenv("REACHY_IDENTITY_ENABLED", "1") == "1"
IDENTITY_FACE_THRESHOLD = float(os.getenv("REACHY_IDENTITY_FACE_THRESHOLD", "0.42"))
# Voice raised 0.72 -> 0.78: 0.72 was loose for Resemblyzer and let strangers
# clear it (Mark greeting non-enrolled people by name). Env-tunable if too strict.
IDENTITY_VOICE_THRESHOLD = float(os.getenv("REACHY_IDENTITY_VOICE_THRESHOLD", "0.78"))
# A match must beat the runner-up person by at least this cosine margin, else
# it's treated as "unknown" (prevents confident WRONG-name matches when two
# enrolled people score nearly the same). See identity.Identity._best.
IDENTITY_MATCH_MARGIN = float(os.getenv("REACHY_IDENTITY_MATCH_MARGIN", "0.05"))
# Don't re-greet the same recognized person more often than this.
IDENTITY_GREET_COOLDOWN_S = float(os.getenv("REACHY_IDENTITY_GREET_COOLDOWN_S", "600"))
# How long a face recognized on arrival stays valid as the "who's speaking"
# fallback when the per-turn VOICE match misses. The person who just walked up
# is almost always the one talking; keeps per-person memory attaching without
# grabbing a fresh camera frame on the turn's latency path.
IDENTITY_FACE_SPEAKER_TTL_S = float(os.getenv("REACHY_IDENTITY_FACE_SPEAKER_TTL_S", "180"))

# --- Proactive notifier (spoken calendar/slack heads-ups; needs the bridge) ---
NOTIFIER_STARTUP_DELAY_S = float(os.getenv("REACHY_NOTIFIER_STARTUP_DELAY_S", "12"))
NOTIFIER_POLL_S = float(os.getenv("REACHY_NOTIFIER_POLL_S", "120"))
# Only proactively announce a calendar event when it starts within this many
# minutes (a last-minute nudge). Prevents spamming every upcoming/all-day event.
NOTIFIER_LEAD_MINUTES = int(os.getenv("REACHY_NOTIFIER_LEAD_MINUTES", "30"))

# --- Daily briefing (spoken good-morning: weather + calendar + news) ---
BRIEFING_ENABLED = os.getenv("REACHY_BRIEFING_ENABLED", "0") not in ("0", "false", "no")
BRIEFING_HOUR = int(os.getenv("REACHY_BRIEFING_HOUR", "8"))
BRIEFING_MINUTE = int(os.getenv("REACHY_BRIEFING_MINUTE", "0"))

# --- Auto-recovery watchdog (motor-fault self-heal) ---
WATCHDOG_STARTUP_DELAY_S = float(os.getenv("REACHY_WATCHDOG_STARTUP_DELAY_S", "20"))
WATCHDOG_POLL_S = float(os.getenv("REACHY_WATCHDOG_POLL_S", "5"))
WATCHDOG_CONFIRM_S = float(os.getenv("REACHY_WATCHDOG_CONFIRM_S", "15"))
WATCHDOG_ENABLED = os.getenv("REACHY_WATCHDOG_ENABLED", "1") not in ("0", "false", "no")
# Memory self-guard: the app's own RSS is polled by the watchdog; if it grows
# far past the normal baseline (~2.8 GB with all models loaded) it means a slow
# leak is under way, so we gracefully cycle (markctl recover) LONG before the
# kernel OOM-killer would kill the process (and freeze the whole desktop).
# A prior incident leaked to ~27 GB and was OOM-killed; 5 GB is ~1.8x baseline.
MEMORY_GUARD_ENABLED = os.getenv("REACHY_MEMORY_GUARD_ENABLED", "1") not in ("0", "false", "no")
MEMORY_GUARD_MB = float(os.getenv("REACHY_MEMORY_GUARD_MB", "5000"))
MEMORY_GUARD_CONFIRM_S = float(os.getenv("REACHY_MEMORY_GUARD_CONFIRM_S", "30"))

# --- Maps / transit (Google Maps Directions API) ---
# Empty key disables the transit tool gracefully (it says it's not set up).
MAPS_KEY = os.getenv("REACHY_MAPS_KEY", "")
HOME_ADDRESS = os.getenv("REACHY_HOME_ADDRESS", "")
WORK_ADDRESS = os.getenv("REACHY_WORK_ADDRESS", "")

# --- Web search (Dell-side, always-on) ---
# Keyless DuckDuckGo by default; set a Brave key for higher quality/quota.
BRAVE_KEY = os.getenv("REACHY_BRAVE_KEY", "")
SEARCH_MAX_RESULTS = int(os.getenv("REACHY_SEARCH_MAX_RESULTS", "3"))

# --- Standing desk (GenioDesk, Tuya Wi-Fi module, local LAN control) ---
# Empty device id disables the desk tool gracefully. Local control only
# (tinytuya v3.4) - no cloud call per command. The local_key lives OUTSIDE the
# repo in a 0600 file. SAFETY: every move is confirm-before-act (see tools/desk.py
# + the deterministic gate in main.py); up/down is cm-only and bounded; presets
# self-stop. The cm-nudge stays disabled until a raw<->cm calibration exists,
# because the device reports RAW units (L1=287, L4=314, L2=425), not centimeters.
DESK_ENABLED = os.getenv("REACHY_DESK_ENABLED", "1") not in ("0", "false", "no")
DESK_DEVICE_ID = os.getenv("REACHY_DESK_DEVICE_ID", "")  # set your Tuya device id in .env
DESK_IP = os.getenv("REACHY_DESK_IP", "")  # set your desk's LAN IP in .env
DESK_VERSION = float(os.getenv("REACHY_DESK_VERSION", "3.4"))
DESK_KEY_PATH = os.getenv("REACHY_DESK_KEY_PATH", "~/.desk_local_key")
DESK_CALIB_PATH = os.getenv("REACHY_DESK_CALIB_PATH", "~/.desk_calibration.json")
# Saved-preset raw envelope - the cm-nudge is HARD-clamped to this range so it
# can never drive past the sitting/standing limits. (L1=287 low, L2=425 high.)
DESK_MIN_RAW = int(os.getenv("REACHY_DESK_MIN_RAW", "287"))
DESK_MAX_RAW = int(os.getenv("REACHY_DESK_MAX_RAW", "425"))
DESK_NUDGE_TIME_CAP_S = float(os.getenv("REACHY_DESK_NUDGE_TIME_CAP_S", "40"))
# How long a spoken confirmation stays valid before the armed move auto-expires.
DESK_CONFIRM_TTL_S = float(os.getenv("REACHY_DESK_CONFIRM_TTL_S", "45"))

# --- Desk lights (Tuya smart plug over the top-of-desk lights, local LAN) ---
# A simple on/off smart plug (EKAZA 20A, category cz). Local control only
# (tinytuya). on/off = dp switch_1 (raw dp "1"). Key lives OUTSIDE the repo in a
# 0600 file. Lights are NOT sensitive like the desk, so on/off is immediate (no
# confirm gate). The "start my day" routine also turns the lights ON; "finish my
# day" turns them OFF (bundled with the preset moves, see tools/desk.py).
LIGHT_ENABLED = os.getenv("REACHY_LIGHT_ENABLED", "1") not in ("0", "false", "no")
LIGHT_DEVICE_ID = os.getenv("REACHY_LIGHT_DEVICE_ID", "")  # set your Tuya device id in .env
LIGHT_IP = os.getenv("REACHY_LIGHT_IP", "")  # set your light's LAN IP in .env
LIGHT_VERSION = float(os.getenv("REACHY_LIGHT_VERSION", "3.5"))
LIGHT_KEY_PATH = os.getenv("REACHY_LIGHT_KEY_PATH", "~/.light_local_key")
LIGHT_SWITCH_DP = os.getenv("REACHY_LIGHT_SWITCH_DP", "1")  # switch_1

# --- MacBook bridge (Slack + Calendar over Tailscale) ---
# The Mark Bridge runs on the MacBook; Mark reaches it over Tailscale. Empty URL
# disables the calendar/slack tools gracefully (they'll say it's not set up).
BRIDGE_URL = os.getenv("REACHY_BRIDGE_URL", "")  # e.g. http://marcosma-ltmmjf9:8770
BRIDGE_SECRET = os.getenv("REACHY_BRIDGE_SECRET", "")
BRIDGE_TIMEOUT_S = float(os.getenv("REACHY_BRIDGE_TIMEOUT_S", "8"))

# --- Weather (Open-Meteo, free/no-key; location auto-detected from IP) ---
# Optional manual override; leave blank to auto-detect from the machine's IP.
WEATHER_LAT = os.getenv("REACHY_WEATHER_LAT", "")
WEATHER_LON = os.getenv("REACHY_WEATHER_LON", "")
WEATHER_PLACE = os.getenv("REACHY_WEATHER_PLACE", "")
WEATHER_TIMEZONE = os.getenv("REACHY_WEATHER_TIMEZONE", "")

# --- Vision (local, GPU, on-demand via Ollama) ---
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
# 8b was measured live only getting 48%/52% GPU/CPU split alongside our own
# ~2GB STT+TTS footprint on this 8GB card, forcing slow CPU inference and
# repeated 60s+ timeouts. 4b fits 100% on GPU (3.5GB) with headroom to spare
# and responds in ~3s instead.
VISION_MODEL = os.getenv("REACHY_VISION_MODEL", "qwen3-vl:4b-instruct-q4_K_M")
# keep_alive:0 (fully unload after every call) was causing 30s+ cold-reload
# waits on every single camera use, sometimes exceeding our old 30s timeout
# entirely - confirmed live (ReadTimeoutError). Keeping it warm for a few
# minutes trades a bit of VRAM headroom for much better latency during an
# active conversation; it still unloads on its own once idle.
VISION_KEEP_ALIVE = os.getenv("REACHY_VISION_KEEP_ALIVE", "10m")
VISION_TIMEOUT_S = float(os.getenv("REACHY_VISION_TIMEOUT_S", "60"))
# Smaller image + capped output = much faster camera turns (were 8-14s live).
VISION_MAX_DIMENSION = int(os.getenv("REACHY_VISION_MAX_DIM", "512"))
VISION_MAX_TOKENS = int(os.getenv("REACHY_VISION_MAX_TOKENS", "120"))

# --- Wake word (fuzzy match, no dedicated model) ---
# The robot's name is Mark. Single-word entries are matched as WHOLE WORDS
# (see _fuzzy_contains_wake_word), so "mark" won't fire on "market"/"remark".
# Variants cover common STT mishearings of "Mark".
# Kept to short exact words (whole-word matched) + phrases. A 5+ char variant
# would enable fuzzy matching that wrongly catches "market"/"remark"/"Marcos".
WAKE_WORDS = ("mark", "marc", "hey mark", "hey marc", "ok mark", "oi mark")
ROBOT_NAME = os.getenv("REACHY_ROBOT_NAME", "Mark")
# Raised from 0.6 after a live test showed common short words (e.g. Portuguese
# "acho") scoring 0.60 against "reachy" by chance, while a genuine mishearing
# ("richie" vs "richy") scored 0.73 - 0.68 cleanly separates the two.
WAKE_FUZZY_THRESHOLD = float(os.getenv("REACHY_WAKE_FUZZY_THRESHOLD", "0.68"))
# Tokens shorter than this are never fuzzy-matched - short common words are
# the main source of false positives with ratio-based matching.
WAKE_FUZZY_MIN_TOKEN_LEN = int(os.getenv("REACHY_WAKE_FUZZY_MIN_TOKEN_LEN", "5"))
# How long (seconds) a conversation stays "open" after the last turn before
# the wake word is required again. Raised from 12s after live feedback - that
# was shorter than some tool-calling round-trips themselves, forcing the wake
# word to be repeated almost every turn. Ends early via the stop_listening or
# go_to_sleep tools, whichever the user asks for.
CONVERSATION_TIMEOUT_S = float(os.getenv("REACHY_CONVERSATION_TIMEOUT_S", "300.0"))

# --- Face tracking ---
# Daemon-side visual head tracking blend factor (0-1). 1.0 lets tracking fully
# own the head; lower values let our own motions/emotions still show through
# while still biasing the head toward the person's face.
# 1.0 = tracking fully owns the head, so it follows the face crisply instead of
# being pulled ~20% off-target by wobbling/other motion (was 0.8, followed weakly).
HEAD_TRACKING_WEIGHT = float(os.getenv("REACHY_HEAD_TRACKING_WEIGHT", "1.0"))
# How often (seconds) the control loop re-asserts tracking while awake, so it
# is effectively always on and can't be silently left off.
HEAD_TRACKING_REASSERT_S = float(os.getenv("REACHY_HEAD_TRACKING_REASSERT_S", "5.0"))

# --- Antennas as physical buttons (edge-triggered, adaptive baseline) ---
ANTENNA_POLL_S = float(os.getenv("REACHY_ANTENNA_POLL_S", "0.05"))
# Press when deflection from the (adaptive) rest baseline crosses this...
ANTENNA_PRESS_THRESHOLD_RAD = float(os.getenv("REACHY_ANTENNA_PRESS_THRESHOLD_RAD", "0.45"))
# ...and re-arm only after it falls back below this (hysteresis).
ANTENNA_RELEASE_THRESHOLD_RAD = float(os.getenv("REACHY_ANTENNA_RELEASE_THRESHOLD_RAD", "0.2"))
ANTENNA_PRESS_MIN_S = float(os.getenv("REACHY_ANTENNA_PRESS_MIN_S", "0.12"))
# How fast the rest baseline tracks drift (per poll) while NOT pressed. Small so
# a real press still stands out, but enough to absorb a settled offset in ~2-3s.
ANTENNA_BASELINE_ALPHA = float(os.getenv("REACHY_ANTENNA_BASELINE_ALPHA", "0.05"))

# --- Face presence (greet on arrival, notice departure) ---
PRESENCE_STARTUP_DELAY_S = float(os.getenv("REACHY_PRESENCE_STARTUP_DELAY_S", "3.0"))
PRESENCE_POLL_S = float(os.getenv("REACHY_PRESENCE_POLL_S", "0.5"))
PRESENCE_ARRIVAL_CONFIRM_S = float(os.getenv("REACHY_PRESENCE_ARRIVAL_CONFIRM_S", "1.5"))
PRESENCE_DEPARTURE_S = float(os.getenv("REACHY_PRESENCE_DEPARTURE_S", "6.0"))
PRESENCE_EVENT_COOLDOWN_S = float(os.getenv("REACHY_PRESENCE_EVENT_COOLDOWN_S", "30.0"))
# Whether the robot spontaneously greets people who appear. Off by default for
# a focused conversation; the arrival still opens the conversation window.
PRESENCE_GREET_ENABLED = os.getenv("REACHY_PRESENCE_GREET", "1") not in ("0", "false", "no")

# --- Sound direction of arrival (turn toward an unseen speaker) ---
DOA_STARTUP_DELAY_S = float(os.getenv("REACHY_DOA_STARTUP_DELAY_S", "2.0"))
DOA_POLL_S = float(os.getenv("REACHY_DOA_POLL_S", "0.4"))
# Only turn toward sound if NO face has been seen for this long - stops DoA
# from grabbing the head during brief face-detection dropouts. Longer = more
# conservative (waits longer before ever reacting to sound).
DOA_FACE_GRACE_S = float(os.getenv("REACHY_DOA_FACE_GRACE_S", "6.0"))
# A directional change must exceed this to react (bigger = ignores small shifts).
DOA_MIN_CHANGE_RAD = float(os.getenv("REACHY_DOA_MIN_CHANGE_RAD", "0.6"))  # ~34deg
DOA_MOVE_DURATION_S = float(os.getenv("REACHY_DOA_MOVE_DURATION_S", "0.7"))
# Sound must persist this long before the head reacts (ignores single words/noise).
DOA_MIN_SPEECH_S = float(os.getenv("REACHY_DOA_MIN_SPEECH_S", "1.2"))
# Minimum gap between DoA turns, so it glances occasionally, not constantly.
DOA_TURN_COOLDOWN_S = float(os.getenv("REACHY_DOA_TURN_COOLDOWN_S", "6.0"))
# Turn only part-way toward the sound (a glance), and never past this angle.
DOA_TURN_FRACTION = float(os.getenv("REACHY_DOA_TURN_FRACTION", "0.5"))
DOA_MAX_ANGLE_RAD = float(os.getenv("REACHY_DOA_MAX_ANGLE_RAD", "0.6"))  # ~34deg cap

# --- Sleep breathing ambiance ---
# Sleep-breathing loudness is set by the WAV PEAK AMPLITUDE (0-1) rendered into
# the looped clip - NOT a playbin volume: the SDK routes the breath playbin
# through a custom audio-sink tee bin with no volume element, so playbin.volume
# is a silent no-op. The "Sleep breathing volume" slider maps 0-100% onto
# [0, BREATH_PEAK_MAX] and regenerates the clip live (see _set_sleep_volume).
# 0.12 = the "Sleep volume 15%" the owner set (slider maps 0-100% onto
# [0, BREATH_PEAK_MAX 0.8], so 15% -> 0.15 * 0.8 = 0.12).
BREATH_PEAK = float(os.getenv("REACHY_BREATH_PEAK", "0.12"))
# Slider ceiling (100%). Must be > the default peak or the slider has no upward
# headroom (the old value 0.4 == BREATH_PEAK was exactly why "louder" did
# nothing). 0.8 puts the 0.4 default at ~50% with room to go louder; the small
# sleep speaker distorts above ~0.85, so we stay under 1.0.
BREATH_PEAK_MAX = float(os.getenv("REACHY_BREATH_PEAK_MAX", "0.8"))
# DEPRECATED: former playbin-gain knob. No longer wired to the slider (the
# custom sink ignored it). Kept only so old .env overrides don't crash import.
BREATH_GAIN = float(os.getenv("REACHY_BREATH_GAIN", "0.45"))
# Breath cycles rendered into the clip that gets gapless-looped (~4s each).
# Looping is seamless (playbin about-to-finish), so this only needs to be long
# enough that the same noise pattern isn't obviously recognisable.
BREATH_CYCLES_PER_FILE = int(os.getenv("REACHY_BREATH_CYCLES_PER_FILE", "8"))

# --- Barge-in (interrupting the robot while it speaks) ---
# The hardware AEC cancels the robot's own voice well (measured ~1.1x baseline
# while speaking), so we listen during playback. A higher VAD threshold and a
# minimum sustained duration keep residual echo/noise from false-triggering.
BARGE_IN_VAD_THRESHOLD = float(os.getenv("REACHY_BARGE_IN_VAD_THRESHOLD", "0.85"))
BARGE_IN_MIN_SPEECH_MS = float(os.getenv("REACHY_BARGE_IN_MIN_SPEECH_MS", "450"))

# --- Conversation turn-taking ---
# Keep the mic muted this long AFTER speech playback returns. The audio
# pipeline still has ~120ms buffered at that point and the tail was being
# transcribed as if the user had spoken (phantom turns like "Gracias.").
SPEECH_TAIL_MUTE_S = float(os.getenv("REACHY_SPEECH_TAIL_MUTE_S", "0.4"))
# Utterances older than this when finally dequeued are dropped - the robot was
# busy speaking or running a tool, and replying now would answer something the
# user has already moved past.
MAX_UTTERANCE_AGE_S = float(os.getenv("REACHY_MAX_UTTERANCE_AGE_S", "8.0"))

# --- VAD ---
VAD_SAMPLE_RATE = 16000
VAD_MIN_SILENCE_MS = 500
# Default VAD threshold. Runtime-adjustable via the "Mic sensitivity" slider
# (audio_io.set_mic_from_ui).
# Raised 0.285 -> 0.40: 0.285 ("Sensitivity 90%") let borderline ambient noise
# clips reach STT, which then hallucinated phantom speech. 0.40 maps to ~57% on
# the slider (VAD_THRESHOLD_MAX 0.6 down to MIN 0.25; 0.6 - x*0.35 = 0.40 ->
# ~57%). Env-overridable via REACHY_VAD_THRESHOLD and still live-tunable.
VAD_THRESHOLD = float(os.getenv("REACHY_VAD_THRESHOLD", "0.40"))

# --- Mic sensitivity & noise reduction (hardware capture gain is already maxed
# at 0 dB, so these software levers are how we hear quieter speech) ---
# Software input gain applied to mic audio before VAD/STT. Default tuned so you
# don't have to raise your voice; the noise gate below counters amplified hiss.
# 3.7 = software gain that pairs with Sensitivity 90% (1.0 + 0.9*(4.0-1.0)).
MIC_GAIN = float(os.getenv("REACHY_MIC_GAIN", "3.7"))
MIC_GAIN_MAX = float(os.getenv("REACHY_MIC_GAIN_MAX", "4.0"))  # slider at 100%
# VAD threshold range the sensitivity slider maps across (low=very sensitive).
VAD_THRESHOLD_MIN = float(os.getenv("REACHY_VAD_THRESHOLD_MIN", "0.25"))
VAD_THRESHOLD_MAX = float(os.getenv("REACHY_VAD_THRESHOLD_MAX", "0.6"))
# Noise gate (RMS floor). 0 = off; the slider maps 0..100% to 0..MAX.
# 0.015 = the "Noise reduction 50%" the owner set (0.50 * MAX 0.03).
MIC_NOISE_GATE = float(os.getenv("REACHY_MIC_NOISE_GATE", "0.015"))
MIC_NOISE_GATE_MAX = float(os.getenv("REACHY_MIC_NOISE_GATE_MAX", "0.03"))
