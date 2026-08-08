"""Reachy Mini Brain: bilingual (pt-BR/en-US) always-listening conversational app.

Pipeline: mic -> VAD -> wake-word fuzzy match -> STT (faster-whisper) ->
GPT-5.4 Nano (chat.completions + tools) -> TTS (Kokoro) -> speaker, with
vision on demand via local Qwen3-VL (Ollama) and motion via a queue drained
by this SDK control loop (see reachy_mini/skills/ai-integration.md).

NOTE: per explicit instruction, this app must not be run against the
physical robot without the operator's go-ahead for the first recorded test.
"""

import asyncio
import logging
import queue
import threading
import time

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402  (must follow require_version)

from reachy_mini import ReachyMini, ReachyMiniApp  # noqa: E402
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini.utils import create_head_pose

# Antenna rest pose. NOT [0, 0]: at the exact 0deg vertical position the antenna
# gearbox backlash puts the motor in an unstable equilibrium (inverted pendulum)
# and it shakes/oscillates (Pollen's known "shaky antennas" issue). The SDK's
# INIT_ANTENNAS_JOINT_POSITIONS is offset ~10deg (-0.1745, +0.1745 rad) so gravity
# takes up the play and it sits still. Every antenna gesture must return HERE, not
# to 0, or Mark re-parks the antennas at the shaking point after each perk/wiggle.
try:
    from reachy_mini.reachy_mini import INIT_ANTENNAS_JOINT_POSITIONS as ANTENNA_REST
except Exception:  # older SDK without the offset - fall back to the same ~10deg
    ANTENNA_REST = [-0.1745, 0.1745]

from reachy_mini_brain import config
from reachy_mini_brain.audio_io import AudioListener
from reachy_mini_brain.breathing import write_breath_wav
from reachy_mini_brain.antennas import AntennaButtons
from reachy_mini_brain.bridge_client import BridgeClient
from reachy_mini_brain.dances import play_dance
from reachy_mini_brain.doa import DoaOrienter
from reachy_mini_brain.memory_store import MemoryStore
from reachy_mini_brain.presence import PresenceMonitor
from reachy_mini_brain.finance import Finance
from reachy_mini_brain.identity import Identity
from reachy_mini_brain.idle import IdleLife
from reachy_mini_brain.metrics import Metrics
from reachy_mini_brain.news import News
from reachy_mini_brain.notifier import Notifier
from reachy_mini_brain.language import LanguageManager
from reachy_mini_brain.llm_mode import LLMModeManager
from reachy_mini_brain.profiles import ProfileManager
from reachy_mini_brain.scheduler import Scheduler
from reachy_mini_brain.todo_store import TodoStore
from reachy_mini_brain.transit import Transit
from reachy_mini_brain.wiki import Wiki
from reachy_mini_brain.transcript import Transcript
from reachy_mini_brain.watchdog import RecoveryWatchdog
from reachy_mini_brain.weather import Weather
from reachy_mini_brain.websearch import WebSearch
from reachy_mini_brain.llm import LLMBrain
from reachy_mini_brain.agents import build_agents, MONOLITH_AGENT
from reachy_mini_brain.router import Router
from reachy_mini_brain.stt import STTEngine
from reachy_mini_brain.tts import TTSEngine
from reachy_mini_brain.tools import ToolDependencies, build_default_registry
from reachy_mini_brain.tools.desk import DeskController, LightController
from reachy_mini_brain.vision import VisionTool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from pathlib import Path as _Path  # noqa: E402
SOUNDS_DIR = _Path(__file__).parent / "sounds"

ERROR_FALLBACK = {
    "pt": "Desculpa, tive um problema aqui. Pode repetir?",
    "en": "Sorry, I hit a glitch there. Could you say that again?",
}


class ReachyMiniBrain(ReachyMiniApp):
    # Enables the built-in FastAPI settings server (see _register_web_panel).
    custom_app_url: str | None = "http://0.0.0.0:8042"
    request_media_backend: str | None = None

    # Handles for the gapless sleep-breathing playback (see _start_breathing).
    _breath_playbin = None
    _breath_handler_id = None
    _tracking_last_asserted = 0.0
    # Sleep-breathing loudness, INDEPENDENT of the main speaker volume. The main
    # Volume slider drives the ALSA PCM mixer (whole card); breathing loudness is
    # instead the GStreamer playbin `volume` (a gdouble gain applied only to the
    # breath playback), so the two sliders never interfere. The WAV is generated
    # at a fixed peak (headroom); _breath_gain (0-1) is the live-adjustable knob.
    _breath_peak = None    # WAV generation peak amplitude (set from config)
    _breath_gain = None    # playbin volume 0.0-1.0 (the sleep slider), set at run()
    _breath_dirty = False  # regenerate/restart breathing mid-sleep on change

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        # The daemon only auto-wakes on its OWN startup (wake_up_on_start=True).
        # If our app is relaunched against an already-running daemon (e.g. after
        # our own _safe_shutdown() disabled motors on a previous run), nothing
        # else re-enables torque/repositions the head - confirmed live (robot
        # stayed drooped, motors off, after an app restart with no daemon
        # restart). So always wake up for ourselves at startup, regardless of
        # what state a previous session left the robot in.
        # Hold off the periodic tracking re-assert for a few seconds so it
        # doesn't fire while the websocket/threads are still settling.
        self._tracking_last_asserted = time.time() + config.HEAD_TRACKING_REASSERT_S
        if self._breath_peak is None:
            self._breath_peak = config.BREATH_PEAK
        if self._breath_gain is None:
            self._breath_gain = config.BREATH_GAIN

        logger.info("Waking up...")
        try:
            reachy_mini.enable_motors()
            reachy_mini.wake_up()
        except Exception:
            logger.exception("wake_up on startup failed")

        try:
            # SDK-native audio-reactive head movement: analyzes anything played
            # via media.play_sound()/push_audio_sample() and adds subtle head
            # motion in sync, automatically. No need to hand-roll this.
            reachy_mini.enable_wobbling()
        except Exception:
            logger.exception("enable_wobbling failed")

        try:
            # Daemon-side face tracking so the robot actually follows whoever
            # is talking to it, rather than only moving when a tool says so.
            reachy_mini.start_head_tracking(weight=config.HEAD_TRACKING_WEIGHT)
            logger.info("Face tracking enabled (weight=%.2f)", config.HEAD_TRACKING_WEIGHT)
        except Exception:
            logger.exception("start_head_tracking failed")

        # Turn toward whoever speaks when we can't see them yet (complements
        # face tracking; defers to it whenever a face is actually visible).
        doa = DoaOrienter(reachy_mini)
        doa.start()

        logger.info("Loading models (STT, TTS, tools, emotions library)...")
        stt = STTEngine()
        tts = TTSEngine()
        vision = VisionTool()
        moves = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
        available_moves = moves.list_moves()

        motion_queue: "queue.Queue[dict]" = queue.Queue()
        listener = AudioListener(stt)
        memory = MemoryStore()
        profiles = ProfileManager()
        language = LanguageManager()
        llm_mode = LLMModeManager(default=config.LLM_MODE_DEFAULT)
        weather = Weather()
        search = WebSearch()
        bridge = BridgeClient()
        transcript = Transcript()
        metrics = Metrics()
        todo = TodoStore()
        wiki = Wiki()
        finance = Finance()
        news = News()
        transit = Transit()
        identity = Identity() if config.IDENTITY_ENABLED else None
        desk = None
        if config.DESK_ENABLED and config.DESK_DEVICE_ID:
            try:
                desk = DeskController(
                    device_id=config.DESK_DEVICE_ID,
                    ip=config.DESK_IP,
                    key_path=config.DESK_KEY_PATH,
                    version=config.DESK_VERSION,
                    calib_path=config.DESK_CALIB_PATH,
                    min_raw=config.DESK_MIN_RAW,
                    max_raw=config.DESK_MAX_RAW,
                    nudge_time_cap_s=config.DESK_NUDGE_TIME_CAP_S,
                )
                logger.info("Desk control enabled (calibrated=%s).", desk.calibrated)
            except Exception:
                logger.exception("Desk control failed to init; disabling it.")
                desk = None
        light = None
        if config.LIGHT_ENABLED and config.LIGHT_DEVICE_ID:
            try:
                light = LightController(
                    device_id=config.LIGHT_DEVICE_ID,
                    ip=config.LIGHT_IP,
                    key_path=config.LIGHT_KEY_PATH,
                    version=config.LIGHT_VERSION,
                    switch_dp=config.LIGHT_SWITCH_DP,
                )
                logger.info("Desk-light control enabled (available=%s).", light.available)
            except Exception:
                logger.exception("Desk-light control failed to init; disabling it.")
                light = None
        if config.WATCHDOG_ENABLED:
            RecoveryWatchdog().start()
        # Re-assert the persisted speaker volume on BOTH PCM controls. ALSA's
        # saved mixer state gets reset by a USB-C replug (PCM,1 mono fell back to
        # -12dB while the panel still showed 100%), so nothing kept the speaker
        # loud across restarts until now.
        try:
            from reachy_mini_brain.volume import reassert_volume, get_persisted_volume
            if reassert_volume():
                logger.info("Speaker volume re-asserted to %d%% on both PCM controls.", get_persisted_volume())
            else:
                logger.warning("Could not re-assert speaker volume at startup.")
        except Exception:
            logger.exception("Volume re-assert at startup failed")
        logger.info("Loaded %d remembered fact(s); personality=%s.", len(memory.all()), profiles.active)
        deps = ToolDependencies(
            reachy_mini=reachy_mini,
            motion_queue=motion_queue,
            vision=vision,
            close_conversation=listener.close_conversation,
            doa=doa,
            memory=memory,
            profiles=profiles,
            weather=weather,
            search=search,
            bridge=bridge,
            todo=todo,
            wiki=wiki,
            finance=finance,
            news=news,
            transit=transit,
            listener=listener,
            identity=identity,
            desk=desk,
            light=light,
            available_moves=available_moves,
        )
        tools = build_default_registry(available_moves)
        brain = LLMBrain(tools, memory=memory, profiles=profiles, metrics=metrics,
                         language=language, llm_mode=llm_mode)
        # Multi-agent gate/router. Reuses the brain's CLOUD client+model for the
        # rare ambiguous-turn classify (never the local one - that would evict
        # the local prompt cache). Behind config.ROUTER_ENABLED (default off);
        # when off, every turn uses MONOLITH_AGENT (pre-refactor behaviour).
        specialists = build_agents(tools)
        router = Router(agents=specialists, classify_client=brain.client,
                        classify_model=config.OPENAI_MODEL)
        sleeping = threading.Event()

        # Browser control panel (volume, personality, tracking, memory, status).
        self._register_web_panel(reachy_mini, memory, profiles, deps, listener,
                                 sleeping, metrics, transcript, language, llm_mode,
                                 brain)

        # Set by the audio thread when the user talks over the robot; checked
        # by on_speak/handle_turn to cut the current reply short.
        interrupted = threading.Event()
        listener.on_barge_in = interrupted.set

        # Per-turn capture for the transcript/metrics (reset each turn).
        turn_reply: list[str] = []

        def on_speak(text: str, language: str) -> None:
            if interrupted.is_set():
                # An earlier part of this turn was interrupted - skip the rest.
                return
            turn_reply.append(text)
            # The voice follows the SELECTED reply language (the same one the LLM
            # was told to write in via LANGUAGE_DIRECTIVE), NOT a per-sentence
            # auto-detect. detect_language() tripped on replies that embed a
            # foreign proper noun: an ENGLISH calendar reminder carrying the
            # Portuguese event title "Tabulacao - Agenda de Trabalho" was
            # misclassified as Portuguese and spoken by the PT voice (confirmed
            # live 2026-07-27). The language is pinned via the brain-panel toggle,
            # so the voice trusts that config and stays consistent with the text.
            speak_language = language if language in ("en", "pt") else config.DEFAULT_LANGUAGE
            logger.info("Speaking (%s): %s", speak_language, text)
            t0 = time.time()
            # Do NOT mute while speaking: the hardware AEC cancels our own voice
            # well enough (measured ~1.1x baseline), so the listener stays live
            # to catch a barge-in. It only watches for a sustained interruption
            # in this state, not normal utterances.
            listener.speaking = True
            try:
                finished = tts.speak(reachy_mini, text, speak_language, interrupt=interrupted)
            finally:
                listener.speaking = False
                # Brief tail mute + flush: the pipeline still has ~120ms
                # buffered when speak() returns, previously transcribed as
                # phantom turns ('Gracias.').
                listener.muted = True
                time.sleep(config.SPEECH_TAIL_MUTE_S)
                dropped = listener.flush_pending()
                listener.muted = False
                speak_dt = time.time() - t0
                metrics.record_stage("tts", speak_dt)
                logger.info(
                    "[timing] speak %.2fs%s%s",
                    speak_dt,
                    "" if finished else " (interrupted)",
                    f" (dropped {dropped})" if dropped else "",
                )

        audio_thread = threading.Thread(
            target=listener.run, args=(reachy_mini, stop_event), daemon=True, name="audio-listener"
        )
        audio_thread.start()

        # Live system metrics (GPU/CPU/RAM) for the dashboard, sampled in the bg.
        metrics.start_system_sampler(interval_s=2.0, stop_event=stop_event)
        # Report local-server health to the dashboard. No-op without a local LLM.
        # NOTE: no active prompt-cache "prewarm" thread - mlx_lm.server is
        # single-threaded and wedges (uninterruptible) if a prewarm request
        # overlaps a real turn, which is worse than the one-time ~8s cold start.
        # Instead, the fixed reply language (LanguageManager) keeps the prompt
        # prefix byte-identical across turns, so the cache stays warm naturally
        # once the first real turn primes it. A single, guarded prewarm runs
        # shortly after startup only (see _local_llm_prewarm_once).
        if config.LOCAL_LLM_URL:
            threading.Thread(target=self._local_llm_keepwarm, args=(metrics, stop_event),
                             daemon=True, name="local-llm-keepwarm").start()
            threading.Thread(target=self._local_llm_prewarm_once,
                             args=(brain, listener, sleeping, stop_event),
                             daemon=True, name="local-llm-prewarm").start()
        # Pull the M4 (LLM host) CPU/RAM/GPU into the dashboard's "LLM host"
        # panel. Separate machine, so we poll its read-only metrics agent.
        if config.m4_metrics_url():
            threading.Thread(target=self._m4_metrics_poll, args=(metrics, stop_event),
                             daemon=True, name="m4-metrics-poll").start()

        # Presence-driven spontaneous turns (greet on arrival, etc.) are queued
        # here as (event_text, language) and drained by the brain loop like a
        # user turn, but injected as a system instruction rather than speech.
        event_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()

        # Scheduler fires reminders/timers/alarms as spoken events. A fired
        # reminder is phrased as a system instruction so Mark says it naturally.
        def _fire_scheduled(message: str, language: str) -> None:
            if not sleeping.is_set():
                listener.note_activity()
            event_queue.put((
                f"A reminder/timer you set just went off. Tell the user this, "
                f"naturally and briefly: {message}", language,
            ))

        scheduler = Scheduler(_fire_scheduled, default_language=config.DEFAULT_LANGUAGE)
        scheduler.start()
        deps.scheduler = scheduler

        # Idle life: small spontaneous moves when nobody's interacting.
        idle = IdleLife(reachy_mini, motion_queue, listener, sleeping)
        idle.start()

        # Proactive spoken heads-ups (calendar/slack) when the Mac bridge is up.
        def _announce(text: str, language: str) -> None:
            listener.note_activity()
            event_queue.put((f"Proactively tell the user, briefly and naturally: {text}", language))

        notifier = Notifier(bridge, listener, sleeping, _announce, language=language)
        notifier.start()

        # Claude Code -> Mark alerts. A hook on the MacBook POSTs here (over
        # Tailscale) when Claude Code needs the user's attention; Mark speaks it
        # aloud - but ONLY when awake. We reuse the base class's already-running
        # settings FastAPI (port 8042); adding a route post-startup works (the
        # web panel does the same). The route only ENQUEUES via _announce, so it
        # never touches the robot directly (no control-lock contention), and the
        # brain loop drops queued events while asleep - so the awake-gate below
        # is belt-and-suspenders (explicit, and reports back to the caller).
        if self.settings_app is not None:
            from fastapi import Header, Request

            @self.settings_app.post("/notify")
            async def _notify(request: Request,
                              x_mark_secret: str | None = Header(default=None)):
                if config.BRIDGE_SECRET and x_mark_secret != config.BRIDGE_SECRET:
                    from fastapi import HTTPException
                    raise HTTPException(status_code=401, detail="bad secret")
                try:
                    body = await request.json()
                except Exception:
                    body = {}
                event = str(body.get("event") or "Notification")
                project = str(body.get("project") or "").strip()
                # Awake-only: never wake or speak for an alert while asleep.
                if sleeping.is_set():
                    logger.info("Notify dropped (asleep): event=%s project=%s", event, project)
                    return {"delivered": False, "reason": "asleep"}
                # Speak the alert in Mark's currently-selected reply language.
                lang = language.active if language is not None else config.DEFAULT_LANGUAGE
                if lang == "pt":
                    where = f" no projeto {project}" if project else ""
                    text = (f"O Claude Code terminou{where}." if event == "Stop"
                            else f"O Claude Code precisa de você{where}.")
                else:
                    where = f" in the {project} project" if project else ""
                    text = (f"Claude Code finished{where}." if event == "Stop"
                            else f"Claude Code needs you{where}.")
                logger.info("Notify -> announce (%s): %s", lang, text)
                # Physical heads-up: a quick antenna perk so Mark visibly signals
                # before speaking (drained by the control loop; safe while awake).
                try:
                    motion_queue.put_nowait({"type": "antennas", "pattern": "perk"})
                except Exception:
                    pass
                _announce(text, lang)
                return {"delivered": True}

        # Optional daily good-morning briefing (weather + calendar + news).
        if config.BRIEFING_ENABLED:
            def _briefing() -> str:
                parts = ["Good morning."]
                try:
                    parts.append(weather.describe())
                except Exception:
                    pass
                try:
                    d, err = bridge.get("/calendar/upcoming", {"days": 1})
                    if d and d.get("events"):
                        parts.append("On your calendar: " + "; ".join(
                            f"{e['when']} {e['title']}" for e in d["events"][:5]))
                except Exception:
                    pass
                try:
                    parts.append(news.headlines("top", config.DEFAULT_LANGUAGE))
                except Exception:
                    pass
                return " ".join(parts)
            scheduler.add_daily(config.BRIEFING_HOUR, config.BRIEFING_MINUTE,
                                lambda: f"Give the user this morning briefing, naturally: {_briefing()}",
                                key="briefing")

        _greeted_at: dict[str, float] = {}
        # Last face recognized on arrival (name + when). Reused as the speaker
        # fallback when the per-turn VOICE match misses, so per-person memory
        # still attaches - WITHOUT grabbing a fresh frame on the turn's latency
        # path (arrival already ran match_face). The person talking is almost
        # always the one who just walked up.
        _last_face: dict[str, float | str | None] = {"name": None, "at": 0.0}

        def _on_arrival() -> None:
            if not config.PRESENCE_GREET_ENABLED:
                return
            # Only greet when idle; never interrupt an ongoing exchange.
            if listener.speaking or time.time() < listener._conversation_open_until:
                return
            # Recognize the arriving face, if enrolled, to greet by name.
            name = None
            if identity is not None:
                try:
                    frame = reachy_mini.media.get_frame()
                    if frame is not None:
                        name = identity.match_face(frame)
                except Exception:
                    logger.exception("face id on arrival failed")
            now = time.time()
            if name:
                # Cache for the per-turn speaker fallback (used before the greet
                # cooldown returns, so it's remembered even on a repeat arrival).
                _last_face["name"] = name
                _last_face["at"] = now
                if now - _greeted_at.get(name, 0.0) < config.IDENTITY_GREET_COOLDOWN_S:
                    return  # already greeted this person recently
                _greeted_at[name] = now
                logger.info("Recognized arriving face: %s", name)
            listener.note_activity()  # open the conversation so a reply can follow
            who = f"You recognize this person as {name}. " if name else ""
            by_name = f", addressing them by name ({name})" if name else ""
            event_queue.put((
                f"A person just appeared in front of you. {who}Say a short, warm, natural "
                f"greeting in one sentence{by_name}. Do NOT use the camera or any other tool - "
                "just say hi; you can play a quick friendly emotion if you like.",
                config.DEFAULT_LANGUAGE,
            ))

        def _on_departure() -> None:
            logger.info("Person left the view.")

        presence = PresenceMonitor(reachy_mini, _on_arrival, _on_departure)
        presence.start()

        def _on_antenna_press() -> None:
            # A physical press is a no-voice way to get attention or dismiss:
            # if a conversation is open, close it; otherwise open it and prompt.
            if time.time() < listener._conversation_open_until:
                listener.close_conversation()
                logger.info("Antenna press -> stopped listening.")
            else:
                listener.note_activity()
                logger.info("Antenna press -> listening.")
                event_queue.put((
                    "The user pressed one of your antennas to get your attention. "
                    "Acknowledge briefly and ask how you can help.",
                    config.DEFAULT_LANGUAGE,
                ))

        antenna_buttons = AntennaButtons(reachy_mini, _on_antenna_press)
        antenna_buttons.start()

        history: list[dict] = []

        def brain_loop() -> None:
            while not stop_event.is_set():
                system_event = False
                try:
                    text, language, captured_at = listener.utterance_queue.get(timeout=0.5)
                except queue.Empty:
                    # No speech - check for a spontaneous event (e.g. someone
                    # appeared) to act on instead. Skip events while asleep.
                    try:
                        text, language = event_queue.get_nowait()
                    except queue.Empty:
                        # Idle: honor a pending sleep requested with no active turn
                        # (e.g. the brain-page Sleep button), which otherwise only
                        # gets consumed at the end of a spoken turn.
                        if deps.pending_sleep and not sleeping.is_set():
                            deps.pending_sleep = False
                            motion_queue.put({"type": "sleep"})
                        continue
                    if sleeping.is_set():
                        continue
                    system_event = True
                    captured_at = time.time()

                # Skip speech that waited too long to be picked up (the robot
                # was speaking or running a tool). Answering it now would be
                # answering a question the user has moved on from. (Events are
                # always fresh.)
                age = time.time() - captured_at
                if not system_event and age > config.MAX_UTTERANCE_AGE_S:
                    logger.info("[timing] dropping stale utterance (%.1fs old): %r", age, text)
                    continue

                # Fresh turn: clear any interruption flag from a prior barge-in
                # so this turn is allowed to speak.
                interrupted.clear()
                turn_t0 = time.time()
                if sleeping.is_set():
                    sleeping.clear()
                    # Soft restart: start each wake with a FRESH conversation so
                    # stale pre-sleep context doesn't leak in (and, on the local
                    # model, so tool-call state can't drift across a long session).
                    # In-process only - the daemon/audio/wake-word stay untouched.
                    history.clear()
                    self._stop_breathing()
                    try:
                        reachy_mini.wake_up()
                        # All were turned off for the sleep period.
                        reachy_mini.enable_wobbling()
                        reachy_mini.start_head_tracking(weight=config.HEAD_TRACKING_WEIGHT)
                        doa.enabled = True
                        presence.enabled = True
                        antenna_buttons.enabled = True
                    except Exception:
                        logger.exception("wake_up failed")
                if system_event:
                    logger.info("Spontaneous event turn: %s", text)
                else:
                    logger.info("Heard (%s): %s  [%.2fs since end of speech]", language, text, age)
                # Identify who's speaking (cheap CPU voice match on the just-
                # captured utterance) so memory and persona can be per-person.
                deps.current_speaker = None
                if identity is not None and not system_event:
                    try:
                        audio = getattr(listener, "last_utterance", None)
                        if audio is not None:
                            deps.current_speaker = identity.match_voice(audio)
                        if deps.current_speaker:
                            logger.info("Recognized speaker by voice: %s", deps.current_speaker)
                        # Fallback: if voice didn't match, use the face recognized
                        # on arrival (if recent). No fresh frame/inference here, so
                        # it adds no latency - it just lets per-person memory attach
                        # when the voice model is unsure but we saw who walked up.
                        elif (_last_face["name"]
                              and time.time() - _last_face["at"]
                                  <= config.IDENTITY_FACE_SPEAKER_TTL_S):
                            deps.current_speaker = _last_face["name"]
                            logger.info("Speaker by recent face: %s", deps.current_speaker)
                    except Exception:
                        logger.exception("voice id failed")
                turn_reply.clear()
                turn_tools: list[str] = []

                # --- DETERMINISTIC desk-move gate (confirm-before-act) ---
                # The desk tool only ARMS deps.pending_desk; a real move fires
                # ONLY here, and ONLY on a clear spoken "yes" while the arm is
                # still fresh. This makes every desk move confirm-before-act and
                # impossible in a single LLM step (the owner's sensitive-action
                # requirement). It also honors a bare "stop" to halt an active
                # cm nudge immediately, and cancels on any "no"/hesitation.
                if deps.desk is not None and not system_event:
                    if deps.desk.nudging and self._is_stop_command(text):
                        deps.desk.stop()
                        deps.pending_desk = None
                        line = "Parei a mesa." if language == "pt" else "Stopped the desk."
                        on_speak(line, language)
                        turn_reply.append(line)
                        transcript.log_turn(speaker="user", language=language,
                                            user_text=text, reply=line, tools=["desk_stop"])
                        continue
                    pend = deps.pending_desk
                    if pend:
                        if time.time() - float(pend.get("armed_at", 0)) > config.DESK_CONFIRM_TTL_S:
                            deps.pending_desk = None  # arm expired -> normal turn
                        else:
                            signal = self._confirmation_signal(text)
                            if signal == "yes":
                                logger.info("Desk move CONFIRMED by user: %s", pend)
                                routine = pend.get("routine")
                                deps.pending_desk = None
                                line = deps.desk.execute_pending(pend, language,
                                                                 light=deps.light)
                                on_speak(line, language)
                                turn_reply.append(line)
                                tools_done = ["desk_execute"]
                                # "start my day": AFTER the desk has moved + lights
                                # on + the confirmation is spoken, add the next
                                # meeting from the agenda (owner's requested order).
                                if routine == "start_my_day":
                                    agenda = self._next_meeting_line(bridge, language)
                                    if agenda:
                                        on_speak(agenda, language)
                                        turn_reply.append(agenda)
                                        tools_done.append("calendar_next")
                                transcript.log_turn(speaker="user", language=language,
                                                    user_text=text, reply=" ".join(turn_reply),
                                                    tools=tools_done)
                                continue
                            if signal == "no":
                                logger.info("Desk move CANCELLED by user.")
                                deps.pending_desk = None
                                line = ("Ok, deixo a mesa como está." if language == "pt"
                                        else "Okay, leaving the desk as it is.")
                                on_speak(line, language)
                                turn_reply.append(line)
                                transcript.log_turn(speaker="user", language=language,
                                                    user_text=text, reply=line,
                                                    tools=["desk_cancel"])
                                continue
                            # Unclear: keep the arm (still within TTL) and let the
                            # LLM answer naturally; a later clear yes still works.
                            logger.info("Desk confirmation unclear (%r); keeping it armed.", text)

                # --- GATE / ROUTER (tier 1) ---
                # Pick the specialist for this turn. Off by default -> always the
                # MONOLITH_AGENT, i.e. the pre-refactor single-prompt behaviour.
                # Routing never gates tool dispatch; it only chooses the prompt
                # focus, so a misroute can still reach the right tool.
                if config.ROUTER_ENABLED:
                    decision = router.route(text, system_event, deps, history)
                    turn_agent = decision.agent
                    deps.current_specialist = turn_agent.name
                    logger.info("[route] specialist=%s src=%s scores=%s",
                                turn_agent.name, decision.source, decision.scores)
                else:
                    turn_agent = MONOLITH_AGENT

                try:
                    nonlocal_history = brain.handle_turn(
                        deps, history, text, language, on_speak,
                        interrupted=interrupted, is_system_event=system_event,
                        tools_used=turn_tools, agent=turn_agent,
                    )
                    history[:] = self._truncate_history(nonlocal_history, max_len=20)
                    dt = time.time() - turn_t0
                    logger.info(
                        "[timing] TURN TOTAL %.2fs (%.2fs from end of user speech)",
                        dt, time.time() - captured_at,
                    )
                    metrics.record_turn(dt, turn_tools)
                    if not system_event and getattr(listener, "last_stt_latency", None):
                        metrics.record_stage("stt", listener.last_stt_latency)
                    # Deterministic backstops: the local model sometimes SAYS it
                    # will do a state action ("I'll go to sleep now / good night";
                    # "I'll stop listening") but forgets to call the tool. If the
                    # reply clearly signals the action and the tool didn't fire,
                    # do it anyway so Mark's body always follows its words.
                    reply_text = " ".join(turn_reply)
                    if ("go_to_sleep" not in turn_tools and not deps.pending_sleep
                            and self._reply_means_sleep(reply_text)):
                        logger.info("Sleep intent spoken without tool call - forcing sleep.")
                        deps.close_conversation()
                        deps.pending_sleep = True
                    elif ("stop_listening" not in turn_tools and "go_to_sleep" not in turn_tools
                            and not deps.pending_sleep
                            and self._reply_means_stop_listening(reply_text)):
                        logger.info("Stop-listening intent spoken without tool call - forcing it.")
                        deps.close_conversation()
                    # Backstop: Mark narrated a web search ("I'm searching the
                    # web now") but never called web_search, so nothing was ever
                    # searched (confirmed live 2026-07-27). Run it deterministically
                    # from the user's own words and enqueue the results as a system
                    # event so the next loop summarizes+speaks them in the pinned
                    # language - instead of leaving a dangling "please wait".
                    if (not system_event and "web_search" not in turn_tools
                            and not deps.pending_sleep
                            and deps.search is not None
                            and self._reply_means_web_search(reply_text)):
                        query = self._search_query_from_text(text)
                        logger.info("Web-search intent spoken without tool call - "
                                    "running it directly for %r.", query)
                        try:
                            results = deps.search.search(query)
                            event_queue.put((
                                "You just searched the web and got these results. "
                                "Tell the user the answer briefly and naturally, in "
                                f"one or two sentences: {results}",
                                language,
                            ))
                        except Exception:
                            logger.exception("Backstop web search failed for %r", query)
                    transcript.log_turn(
                        speaker="system" if system_event else "user",
                        language=language, user_text=text,
                        reply=" ".join(turn_reply), tools=turn_tools,
                    )
                except Exception:
                    # Previously failed turns went completely silent - looked
                    # like an unexplained hang live. Always say SOMETHING.
                    logger.exception("LLM turn failed")
                    try:
                        on_speak(ERROR_FALLBACK.get(language, ERROR_FALLBACK["pt"]), language)
                    except Exception:
                        logger.exception("Error fallback speech also failed")
                    history.clear()  # drop possibly-corrupted history rather than repeat the crash

                # Speech for this turn is finished (on_speak blocks), so it is
                # now safe to actually lower the head into the sleep pose.
                if deps.pending_sleep:
                    deps.pending_sleep = False
                    motion_queue.put({"type": "sleep"})

        brain_thread = threading.Thread(target=brain_loop, daemon=True, name="brain")
        brain_thread.start()

        logger.info("Mark is running. Say '%s' to start talking.", config.ROBOT_NAME)

        # Control loop: drain motion_queue and dispatch to the SDK.
        try:
            while not stop_event.is_set():
                try:
                    action = motion_queue.get(timeout=0.05)
                except queue.Empty:
                    if sleeping.is_set():
                        # Volume changed from the web panel while asleep -
                        # regenerate the breath clip at the new level and
                        # restart playback (kept on THIS thread; all GStreamer
                        # calls must stay off the FastAPI thread).
                        if self._breath_dirty:
                            self._breath_dirty = False
                            self._stop_breathing()
                            self._start_breathing(reachy_mini)
                        # Watchdog: if the sleep breathing ever stops on its
                        # own (playbin dropped out of PLAYING), restart it
                        # rather than leaving the robot silently asleep.
                        self._ensure_breathing(reachy_mini)
                    else:
                        # Keep face tracking on at all times while awake, so a
                        # movement, an emotion or a stray tool call can never
                        # leave the robot not following the user.
                        self._ensure_tracking(reachy_mini, deps)
                    continue

                try:
                    if action.get("type") == "sleep":
                        # Order matters. Wobbling adds audio-reactive offsets
                        # and tracking keeps re-aiming the head; either one
                        # layered on top of the already-low sleep pose is what
                        # drove the head into the body shell. Turn both OFF
                        # first, then lower the head - both are restored on wake.
                        try:
                            reachy_mini.disable_wobbling()
                            reachy_mini.stop_head_tracking()
                            doa.enabled = False  # no orienting while asleep
                            presence.enabled = False  # no greetings while asleep
                            antenna_buttons.enabled = False
                        except Exception:
                            logger.exception("disabling wobbling/tracking before sleep failed")
                        try:
                            reachy_mini.goto_sleep()
                        except Exception:
                            logger.exception("goto_sleep failed")
                        sleeping.set()
                        # Require "Hey Mark" to wake again: close any still-open
                        # conversation window (up to CONVERSATION_TIMEOUT_S in the
                        # future) so an ambient-noise / phantom transcription can't
                        # slip past the wake-word gate and wake him in his sleep.
                        listener.close_conversation()
                        # Non-blocking: GStreamer loops the audio on its own
                        # thread, so the control loop stays free to keep
                        # draining the motion queue while asleep.
                        self._start_breathing(reachy_mini)
                    elif action.get("type") == "wake":
                        # Manual wake (e.g. the brain-page button) - mirror the
                        # brain loop's wake sequence, on THIS control thread so
                        # all SDK motion calls stay off the FastAPI thread.
                        if sleeping.is_set():
                            sleeping.clear()
                            history.clear()  # fresh conversation on wake
                            self._stop_breathing()
                            try:
                                reachy_mini.wake_up()
                                reachy_mini.enable_wobbling()
                                reachy_mini.start_head_tracking(weight=config.HEAD_TRACKING_WEIGHT)
                                doa.enabled = True
                                presence.enabled = True
                                antenna_buttons.enabled = True
                            except Exception:
                                logger.exception("manual wake_up failed")
                    else:
                        # An emotion/dance drives the antennas under motor
                        # control, which would read as a false "press", so
                        # pause button detection for the duration of the move.
                        moves_antennas = action.get("type") in ("play_move", "dance")
                        if moves_antennas:
                            antenna_buttons.enabled = False
                        try:
                            self._dispatch_motion(reachy_mini, moves, action)
                        finally:
                            if moves_antennas and not sleeping.is_set():
                                antenna_buttons.enabled = True
                except Exception:
                    logger.exception("Motion action failed: %s", action)
                finally:
                    # Tools that need to wait for a movement to actually finish
                    # (e.g. look_around, which captures a frame per angle)
                    # attach a "done" event and block on it.
                    done = action.get("done")
                    if done is not None:
                        done.set()
        finally:
            self._safe_shutdown(reachy_mini)

    @staticmethod
    def _local_llm_keepwarm(metrics, stop_event) -> None:
        """Report local-server reachability + model to the dashboard every ~30s.

        Only a GET /models probe - deliberately NO generation ping: mlx_lm keeps
        the model resident for the process lifetime (the only cold start is the
        server launch itself), and a dummy generation would EVICT the server's
        prompt cache, forcing Mark's big (~4k-token) system+tools prefix to be
        reprocessed from scratch every turn (~6s instead of the cached ~0.5s).
        """
        import json as _json
        import urllib.request

        base = config.LOCAL_LLM_URL.rstrip("/")
        want = config.LOCAL_LLM_MODEL
        prev = None  # last reachability, to log only on TRANSITIONS (not every 30s)
        while not stop_event.is_set():
            reachable, model, why = False, None, ""
            try:
                with urllib.request.urlopen(base + "/models", timeout=4) as r:
                    data = _json.loads(r.read().decode())
                ids = [it.get("id") for it in (data.get("data") or [])]
                reachable = True
                # A server may advertise several cached models; prefer the one we
                # actually route to, so the dashboard shows the real chat model.
                model = want if want in ids else (ids[0] if ids else want)
            except Exception as ex:
                reachable = False
                why = f"{type(ex).__name__}: {ex}"
            # Log only when the state flips, so the journal shows a clean timeline
            # of exactly when the local LLM dropped/recovered - not 2 lines/min.
            if prev is not None and reachable != prev:
                if reachable:
                    logger.info("Local LLM health: RECOVERED (reachable again, model=%s).", model)
                else:
                    logger.warning("Local LLM health: LOST (unreachable) - %s", why)
            prev = reachable
            metrics.record_local_health(reachable, model)
            stop_event.wait(30)

    @staticmethod
    def _m4_metrics_poll(metrics, stop_event) -> None:
        """Pull the M4 (local-LLM host) CPU/RAM/GPU every ~5s for the dashboard.

        The local model runs on a different machine than this brain, so its
        resource use (the GPU/VRAM the model actually occupies, which is what
        drives local-LLM latency) is only visible if we fetch it. A read-only
        agent on the M4 (mac_bridge/m4_metrics) exposes GET /metrics; we just
        forward its JSON into metrics.record_m4. Cheap, no side effects.
        """
        import json as _json
        import urllib.request

        url = config.m4_metrics_url()
        secret = config.M4_METRICS_SECRET
        period = max(2.0, config.M4_METRICS_POLL_S)
        while not stop_event.is_set():
            try:
                req = urllib.request.Request(url)
                if secret:
                    req.add_header("X-Mark-Secret", secret)
                with urllib.request.urlopen(req, timeout=4) as r:
                    data = _json.loads(r.read().decode())
                metrics.record_m4(True, data)
            except Exception:
                metrics.record_m4(False, None)
            stop_event.wait(period)

    @staticmethod
    def _reply_means_sleep(reply: str) -> bool:
        """True if Mark's spoken reply clearly announces going to sleep (EN/PT),
        used as a backstop when the model narrates sleep without calling the tool.
        Kept specific to avoid false positives (e.g. "did you sleep well?").
        """
        r = (reply or "").lower()
        phrases = (
            "going to sleep", "go to sleep now", "i'll sleep", "i will sleep",
            "going to rest now", "good night", "goodnight",
            "vou dormir", "indo dormir", "vou descansar", "boa noite",
        )
        return any(p in r for p in phrases)

    @staticmethod
    def _reply_means_web_search(reply: str) -> bool:
        """True if Mark's reply ACTIVELY narrates searching the web (EN/PT) -
        backstop for when the model says "I'm searching the web..." but forgets
        to call the web_search tool (confirmed live 2026-07-27: the turn spoke
        "I'm searching the web now / please wait" and then ended with NO tool
        call, so nothing was ever searched).

        Deliberately FIRST-PERSON, ACTIVE phrasing only ("searching", "I'll
        search", "let me look up") - never the bare capability phrase "search the
        web", which also appears in Mark's "I can help with... search the web..."
        menu and must NOT trigger a phantom search.
        """
        r = (reply or "").lower()
        phrases = (
            "searching the web", "searching online", "searching the internet",
            "let me search", "let me look that up", "let me look it up",
            "i'll search", "i will search", "i'll look that up", "i'll look it up",
            "vou pesquisar", "estou pesquisando", "pesquisando na", "vou procurar",
            "deixa eu pesquisar", "deixa eu procurar", "vou dar uma pesquisada",
        )
        return any(p in r for p in phrases)

    # Leading filler tokens of a "search the web for ..." command, stripped off
    # so the DDG query is the SUBJECT, not the imperative. Token-based (not a
    # regex) so it is WORD-BOUNDARY safe - a substring approach matched "in"
    # inside "internet" and corrupted the query to "ternet..." (caught in test).
    _SEARCH_LEAD_TOKENS = frozenset((
        "please", "can", "could", "you", "search", "look", "up", "for",
        "google", "find", "on", "in", "at", "the", "web", "internet", "online",
        "about", "me", "a",
        # pt-BR
        "pesquise", "pesquisa", "pesquisar", "pesquisando", "procure", "procura",
        "procurar", "busque", "busca", "buscar", "na", "no", "sobre", "por",
        "rede", "me", "sobre",
    ))

    @classmethod
    def _search_query_from_text(cls, text: str) -> str:
        """Strip a leading 'search the web for ...' command off the user's
        utterance so the DDG query is the SUBJECT, not the imperative. Drops
        leading filler tokens (verb/preposition/web-word/article) until the first
        content word; falls back to the whole text (DDG tolerates it) if that
        would leave nothing.
        """
        q = (text or "").strip()
        tokens = q.split()
        i = 0
        while i < len(tokens) and tokens[i].strip(".,:;?!\"'").lower() in cls._SEARCH_LEAD_TOKENS:
            i += 1
        rest = " ".join(tokens[i:]).strip(" .?!,:;\"'")
        return rest or q

    @staticmethod
    def _reply_means_stop_listening(reply: str) -> bool:
        """True if Mark's reply says it will stop listening / stand by (EN/PT) -
        backstop for when the model narrates it without calling stop_listening.
        """
        r = (reply or "").lower()
        phrases = (
            "stop listening", "i'll stop listening", "i will stop listening",
            "leave you alone", "let you be", "standing by", "call me when",
            "parar de ouvir", "parar de escutar", "vou parar", "te deixo em paz",
            "me chama quando", "estou aqui se precisar",
        )
        return any(p in r for p in phrases)

    # Desk-move confirmation: single-word yes/no signals matched WORD-BOUNDARY
    # safe (a set membership over tokens), plus multi-word phrases matched as
    # substrings. Kept conservative: the desk moves ONLY on a clear yes.
    _DESK_YES_TOKENS = frozenset((
        "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm",
        "confirmed", "go", "do", "please",
        # pt-BR
        "sim", "isso", "confirmo", "confirmar", "pode", "manda", "vai",
        "claro", "beleza", "positivo", "aham",
    ))
    _DESK_YES_PHRASES = (
        "go ahead", "do it", "go for it", "please do", "yes please",
        "pode ir", "pode sim", "isso mesmo", "manda ver", "pode mandar",
        "tá bom", "ta bom", "pode fazer",
    )
    _DESK_NO_TOKENS = frozenset((
        "no", "nope", "nah", "cancel", "stop", "wait", "don't", "dont",
        # pt-BR
        "não", "nao", "cancela", "cancelar", "para", "pare", "espera",
        "espere", "deixa", "deixe",
    ))
    _DESK_NO_PHRASES = (
        "do not", "never mind", "nevermind", "hold on", "not now",
        "melhor não", "melhor nao", "deixa pra lá", "deixa pra la",
        "peraí", "perai", "pera aí", "agora não", "agora nao",
    )

    @classmethod
    def _confirmation_signal(cls, text: str) -> str:
        """Classify a reply to a desk confirmation as 'yes', 'no', or '' (unclear).
        NO is checked FIRST as the fail-safe: any hesitation/stop word cancels."""
        t = (text or "").lower()
        tokens = {w.strip(".,:;?!\"'()") for w in t.split()}
        if any(p in t for p in cls._DESK_NO_PHRASES) or (tokens & cls._DESK_NO_TOKENS):
            return "no"
        if any(p in t for p in cls._DESK_YES_PHRASES) or (tokens & cls._DESK_YES_TOKENS):
            return "yes"
        return ""

    @staticmethod
    def _is_stop_command(text: str) -> bool:
        """Bare emphatic stop, to halt a moving desk immediately (EN/PT)."""
        t = (text or "").lower()
        tokens = {w.strip(".,:;?!\"'()") for w in t.split()}
        return bool(tokens & {"stop", "para", "pare", "parar", "halt"})

    @staticmethod
    def _next_meeting_line(bridge, language: str) -> str | None:
        """Spoken one-liner for the next calendar meeting, for "start my day".

        Reads the next 24h from the MacBook bridge and returns a short, already
        speech-safe sentence (the bridge pre-formats `when` for TTS). Returns
        None on any failure so the routine never blocks or crashes if the Mac /
        calendar is unavailable - the desk move + lights already happened. Never
        surfaces the bridge's error text here (that path stays for direct
        calendar questions); a briefing just quietly skips the agenda line.
        """
        pt = language == "pt"
        try:
            data, err = bridge.get("/calendar/upcoming", {"days": 1})
        except Exception:
            logger.debug("start_my_day calendar fetch raised", exc_info=True)
            return None
        if err or not data:
            logger.info("start_my_day: no calendar (%s)", err or "no data")
            return None
        events = data.get("events") or []
        if not events:
            return ("Sua agenda está livre hoje." if pt
                    else "Your calendar is clear for today.")
        ev = events[0]
        when, title = ev.get("when", ""), ev.get("title", "")
        if pt:
            return f"Seu próximo compromisso: {title}, {when}."
        return f"Your next meeting: {title}, {when}."

    @staticmethod
    def _local_llm_prewarm_once(brain, listener, sleeping, stop_event) -> None:
        """Prime the local prompt cache ONCE, shortly after startup, so the first
        real turn is fast instead of paying the ~8s cold prefix cost.

        Deliberately a single shot, not a loop: mlx_lm.server is single-threaded,
        so a prewarm that overlaps a real turn wedges it. After this, the fixed
        reply language keeps the prefix stable and real turns keep the cache warm.
        Only fires if Mark is idle at the ~20s mark; if busy, we skip it (the real
        turn already in flight warms the cache anyway).
        """
        if stop_event.wait(20):
            return
        idle = (not listener.speaking
                and time.time() >= getattr(listener, "_conversation_open_until", 0)
                and not sleeping.is_set())
        if not idle:
            return
        try:
            if brain.prewarm_local():
                logger.info("Local prompt cache primed at startup.")
        except Exception:
            logger.debug("Local prewarm error", exc_info=True)

    @staticmethod
    def _truncate_history(messages: list[dict], max_len: int) -> list[dict]:
        """Keep only the last `max_len` messages, but never cut in the middle
        of a tool_calls/tool-response pair - confirmed live that a naive
        messages[-max_len:] slice can leave an orphaned 'tool' role message at
        the start with no preceding 'tool_calls' message, which the OpenAI API
        rejects outright (400 Invalid parameter) and crashes every subsequent
        turn until the process restarts.
        """
        if len(messages) <= max_len:
            return messages
        window = messages[-max_len:]
        for i, msg in enumerate(window):
            if msg.get("role") == "user":
                return window[i:]
        return []  # no safe cut point found in the window - drop it all

    def _start_breathing(self, reachy_mini: ReachyMini) -> None:
        """Start a truly continuous breathing ambiance for the sleep state.

        Looping is done with GStreamer's own gapless mechanism: playbin emits
        `about-to-finish` shortly before the clip ends, and re-setting its
        `uri` in that callback makes it roll straight into the next pass with
        no gap, forever, on GStreamer's thread.

        This replaced two approaches that failed on hardware: repeatedly
        calling push_audio_sample() (only the first push was ever audible) and
        re-triggering a short clip with play_sound() on a timer (audible once,
        then silence - each call rebuilds a playbin and audio-sink bin while
        the shared TTS pipeline still holds the device). Verified live: the
        signal fired ~4x over 30s with an 8s clip, exactly as expected.
        """
        output_rate = reachy_mini.media.get_output_audio_samplerate()
        rate = output_rate if output_rate > 0 else 24000
        path = "/tmp/reachy_breath_cycle.wav"
        peak = self._breath_peak if self._breath_peak is not None else config.BREATH_PEAK
        duration = write_breath_wav(path, rate, cycles=config.BREATH_CYCLES_PER_FILE, peak=peak)

        reachy_mini.media.play_sound(path)

        playbin = getattr(reachy_mini.media.audio, "_playbin", None)
        if playbin is None:
            logger.warning("Breathing: playbin unavailable, cannot loop seamlessly.")
            return

        uri = f"file://{path}"

        # NOTE: loudness comes from the WAV `peak` above (regenerated on change),
        # NOT from playbin volume - the SDK's custom audio-sink tee bin has no
        # volume element, so setting it here would be a silent no-op. Left unset.

        def _loop_again(pb) -> None:
            pb.set_property("uri", uri)

        self._breath_playbin = playbin
        self._breath_handler_id = playbin.connect("about-to-finish", _loop_again)
        logger.info("Sleeping - continuous breathing started (%.0fs clip, peak %.2f, gapless loop).",
                    duration, peak)

    def _ensure_tracking(self, reachy_mini: ReachyMini, deps) -> None:
        """Re-assert face tracking periodically while awake.

        Enabling it once at startup isn't enough in practice - anything that
        turns it off (sleep, a scan, an explicit tool call) would otherwise
        leave the robot permanently not following the user. Re-sending the
        command is idempotent, so this just keeps the desired state true.
        """
        if not deps.tracking_desired:
            return
        now = time.time()
        if now - self._tracking_last_asserted < config.HEAD_TRACKING_REASSERT_S:
            return
        self._tracking_last_asserted = now
        try:
            reachy_mini.start_head_tracking(weight=config.HEAD_TRACKING_WEIGHT)
        except ConnectionError as e:
            # Transient websocket blip (e.g. daemon momentarily busy); the next
            # periodic re-assert will recover it - no need for a full traceback.
            logger.warning("Re-asserting head tracking skipped: %s", e)
        except Exception:
            logger.exception("Re-asserting head tracking failed")

    def _ensure_breathing(self, reachy_mini: ReachyMini) -> None:
        """Restart the breathing if its playbin is no longer actually playing.

        Cheap safety net for the "it only breathed once" failure mode: the
        gapless loop is verified working, but if the playbin ever drops out of
        PLAYING for any reason the robot would go silently quiet with no way
        to recover until woken.
        """
        playbin = self._breath_playbin
        if playbin is None:
            self._start_breathing(reachy_mini)
            return
        try:
            # 0 timeout: just read the last known state, never block the loop.
            _, state, _ = playbin.get_state(0)
            if state not in (Gst.State.PLAYING, Gst.State.PAUSED):
                logger.warning("Breathing playback stopped (state=%s); restarting.", state)
                self._stop_breathing()
                self._start_breathing(reachy_mini)
        except Exception:
            logger.exception("Breathing watchdog check failed")

    def _stop_breathing(self) -> None:
        """Stop the breathing ambiance without touching the shared pipeline.

        Deliberately does NOT call media.stop_playing(): that sets the SHARED
        capture+playback pipeline to NULL and kills the microphone with it
        (confirmed live). Tearing down just this playbin is the same thing
        play_sound() itself does to its predecessor, so it is safe.
        """
        playbin, handler_id = self._breath_playbin, self._breath_handler_id
        self._breath_playbin, self._breath_handler_id = None, None
        if playbin is None:
            return
        try:
            if handler_id is not None:
                playbin.disconnect(handler_id)  # stop re-queueing first
            playbin.set_state(Gst.State.NULL)
            logger.info("Breathing stopped (woken up).")
        except Exception:
            logger.exception("Failed to stop breathing playback")

    @staticmethod
    def _dispatch_motion(reachy_mini: ReachyMini, moves: RecordedMoves, action: dict) -> None:
        kind = action.get("type")
        if kind == "goto_head":
            head_yaw_deg = action["yaw"]
            kwargs = {}
            if "body_yaw" in action:
                # Clamp to the documented +/-160 deg body limit.
                body_yaw_deg = max(-160.0, min(160.0, action["body_yaw"]))
                # The head pose matrix is in WORLD frame (see the SDK's
                # AGENTS.md): sending body_yaw alone just pivots the body
                # *under* a head that keeps facing the same way - confirmed
                # live, the body turned but the head didn't. So treat the
                # tool's `yaw` as relative to the body and add the body
                # rotation into the world-frame head yaw, which makes the
                # head follow the body.
                relative_head_yaw = max(-65.0, min(65.0, head_yaw_deg))  # documented head-body delta limit
                head_yaw_deg = relative_head_yaw + body_yaw_deg
                kwargs["body_yaw"] = float(np.deg2rad(body_yaw_deg))
            pose = create_head_pose(
                yaw=head_yaw_deg, pitch=action["pitch"], roll=action["roll"], degrees=True
            )
            reachy_mini.goto_target(head=pose, duration=max(action["duration"], 0.1), **kwargs)
        elif kind == "play_move":
            move = moves.get(action["name"])
            asyncio.run(reachy_mini.async_play_move(move, initial_goto_duration=1.0))
        elif kind == "dance":
            play_dance(
                reachy_mini, action["name"],
                bpm=action.get("bpm", 120.0), beats=action.get("beats", 8.0),
            )
        elif kind == "antennas":
            ReachyMiniBrain._express_antennas(reachy_mini, action.get("pattern", "wiggle"))
        elif kind == "sound":
            path = SOUNDS_DIR / f"{action.get('name','pop')}.wav"
            if path.exists():
                reachy_mini.media.play_sound(str(path))
        else:
            logger.warning("Unknown motion action type: %s", kind)

    @staticmethod
    def _express_antennas(reachy_mini: ReachyMini, pattern: str) -> None:
        """Quick expressive antenna gestures. Antennas are [right, left] radians.
        Each gesture RETURNS TO ANTENNA_REST (the ~10deg anti-shake offset), NOT
        to [0, 0] - parking them at the 0deg vertical leaves them in the unstable
        equilibrium that makes the antenna oscillate (the "shaky antenna" bug).
        """
        rest = list(ANTENNA_REST)
        try:
            if pattern == "perk":
                reachy_mini.goto_target(antennas=[0.6, 0.6], duration=0.25)
                reachy_mini.goto_target(antennas=rest, duration=0.4)
            elif pattern == "droop":
                reachy_mini.goto_target(antennas=[-0.7, -0.7], duration=0.5)
                reachy_mini.goto_target(antennas=rest, duration=0.6)
            else:  # wiggle
                for a in ([0.5, -0.5], [-0.5, 0.5], [0.4, -0.4], rest):
                    reachy_mini.goto_target(antennas=a, duration=0.18)
        except Exception:
            logger.exception("antenna expression failed")

    @staticmethod
    def _sleep_pct_to_peak(pct: int) -> float:
        """Map the 0-100% sleep slider to a WAV peak amplitude (0..BREATH_PEAK_MAX).

        The peak is what actually sets breathing loudness (see _set_sleep_volume),
        so the slider maps linearly onto [0, BREATH_PEAK_MAX]. 100% = the max
        headroom peak; 0% = silent.
        """
        pct = max(0, min(100, int(pct)))
        return round(pct / 100.0 * config.BREATH_PEAK_MAX, 4)

    def _breath_peak_to_percent(self) -> int:
        # Inverse of _sleep_pct_to_peak: reflect the WAV peak (0..MAX) as 0-100%
        # so the panel slider shows the real current loudness on refresh.
        peak = self._breath_peak if self._breath_peak is not None else config.BREATH_PEAK
        mx = config.BREATH_PEAK_MAX or 1.0
        return int(round(max(0.0, min(1.0, peak / mx)) * 100))

    def _register_web_panel(self, reachy_mini, memory, profiles, deps, listener, sleeping,
                            metrics=None, transcript=None, language=None,
                            llm_mode=None, brain=None) -> None:
        """Register control-panel API routes on the base class's settings_app.

        The base ReachyMiniApp already stood up a FastAPI server (because
        custom_app_url is set) and serves static/index.html; we just add the
        JSON endpoints the page talks to.
        """
        app = self.settings_app
        if app is None:
            logger.warning("Settings app not available; web panel disabled.")
            return

        from pydantic import BaseModel

        from reachy_mini_brain.profiles import PROFILES
        from reachy_mini_brain.volume import get_volume_percent, set_volume_percent

        @app.get("/api/status")
        def _status():
            return {
                "personality": profiles.active,
                "personalities": {k: v[0] for k, v in PROFILES.items()},
                "tracking": deps.tracking_desired,
                "volume": get_volume_percent(),
                "sleep_volume": self._breath_peak_to_percent(),
                "mic_sensitivity": listener.mic_sensitivity_pct(),
                "noise_reduction": listener.mic_noise_pct(),
                "asleep": sleeping.is_set(),
                "deaf": getattr(listener, "deaf", False),
                "facts": memory.all(),
                "default_language": config.DEFAULT_LANGUAGE,
                "language": language.active if language is not None else config.DEFAULT_LANGUAGE,
                "llm_mode": llm_mode.active if llm_mode is not None else "auto",
                "llm_local_configured": bool(config.LOCAL_LLM_URL and config.LOCAL_LLM_MODEL),
                # Cheap booleans (no network I/O - just "is a key loaded") so the
                # page can show/hide the desk + lights cards on the fast poll. The
                # LIVE state (height/level/lights on) is a separate on-demand read
                # (/api/desk_state) to avoid hitting the Tuya devices every 4s.
                "desk_configured": bool(getattr(deps, "desk", None) and deps.desk.available),
                "light_configured": bool(getattr(deps, "light", None) and deps.light.available),
                "identity_configured": deps.identity is not None,
            }

        class LanguageBody(BaseModel):
            language: str

        @app.post("/api/language")
        def _set_language(body: LanguageBody):
            # Manual reply-language switch (EN/PT) from the brain control panel.
            ok = language.set(body.language) if language is not None else False
            return {"ok": ok, "language": language.active if language is not None else None}

        class LLMModeBody(BaseModel):
            mode: str  # "local" | "cloud" | "auto"

        @app.post("/api/llm_mode")
        def _set_llm_mode(body: LLMModeBody):
            # Which LLM backend answers: force the MacBook model ("local"), force the
            # OpenAI cloud ("cloud"), or let the app pick by response time ("auto").
            # Read live each turn by the router - takes effect on the next utterance,
            # no restart. Persisted so it survives a restart.
            ok = llm_mode.set(body.mode) if llm_mode is not None else False
            # An explicit backend choice should take effect NOW: clear any auto
            # re-probe backoff so the next turn re-evaluates local immediately
            # (otherwise a long backoff from a prior coding session could delay it).
            if ok and brain is not None:
                brain.reset_auto_backoff()
            return {"ok": ok, "llm_mode": llm_mode.active if llm_mode is not None else None}

        class VolumeBody(BaseModel):
            percent: int

        @app.post("/api/volume")
        def _set_volume(body: VolumeBody):
            pct = max(0, min(100, int(body.percent)))
            set_volume_percent(pct)
            return {"volume": pct}

        @app.post("/api/sleep_volume")
        def _set_sleep_volume(body: VolumeBody):
            # Sleep-breathing loudness, INDEPENDENT of the main speaker volume.
            #
            # We drive the WAV PEAK AMPLITUDE (_breath_peak), NOT the playbin
            # "volume" property: the SDK routes the breath playbin through a
            # custom audio-sink tee bin with no volume element, so setting
            # playbin.volume was a silent no-op (the reported "slider does
            # nothing" bug). Changing the rendered peak provably changes the
            # loudness. We can't safely rebuild GStreamer from this FastAPI
            # thread, so we flag _breath_dirty and let the control loop (which
            # owns all GStreamer calls) regenerate + restart the clip on its
            # next tick - near-instant while asleep.
            pct = max(0, min(100, int(body.percent)))
            self._breath_peak = self._sleep_pct_to_peak(pct)
            self._breath_dirty = True  # control loop regenerates the clip
            return {"sleep_volume": pct}

        @app.get("/api/metrics")
        def _metrics():
            return metrics.snapshot() if metrics is not None else {}

        @app.get("/dashboard")
        def _dashboard():
            # Live metrics dashboard (GPU/VRAM/CPU/RAM + local vs cloud LLM +
            # per-stage latency + cost). Served from static/dashboard.html.
            from pathlib import Path

            from fastapi.responses import FileResponse, HTMLResponse
            page = Path(__file__).parent / "static" / "dashboard.html"
            if page.exists():
                return FileResponse(str(page))
            return HTMLResponse("<h1>dashboard.html missing</h1>", status_code=404)

        @app.get("/api/transcript")
        def _transcript(q: str = "", limit: int = 50):
            return {"entries": transcript.search(q, limit) if transcript is not None else []}

        class MicBody(BaseModel):
            sensitivity: int | None = None
            noise: int | None = None

        @app.post("/api/mic")
        def _set_mic(body: MicBody):
            # Live mic tuning - hear quieter speech / reject more noise, no restart.
            listener.set_mic_from_ui(sensitivity_pct=body.sensitivity, noise_pct=body.noise)
            return {"mic_sensitivity": listener.mic_sensitivity_pct(),
                    "noise_reduction": listener.mic_noise_pct()}

        class PersonaBody(BaseModel):
            personality: str

        @app.post("/api/personality")
        def _set_personality(body: PersonaBody):
            ok = profiles.set(body.personality)
            return {"ok": ok, "personality": profiles.active}

        class TrackingBody(BaseModel):
            enabled: bool

        @app.post("/api/tracking")
        def _set_tracking(body: TrackingBody):
            deps.tracking_desired = bool(body.enabled)
            try:
                if body.enabled:
                    reachy_mini.start_head_tracking(weight=config.HEAD_TRACKING_WEIGHT)
                else:
                    reachy_mini.stop_head_tracking()
            except Exception:
                logger.exception("web panel tracking toggle failed")
            return {"tracking": deps.tracking_desired}

        class SleepBody(BaseModel):
            asleep: bool

        @app.post("/api/sleep")
        def _set_sleep(body: SleepBody):
            # Manual sleep/wake from the brain page. Never call SDK motion here
            # (FastAPI thread) - route through the control loop: sleep via the
            # same pending_sleep path the tool uses; wake via a "wake" motion
            # action handled on the control thread.
            if body.asleep:
                if not sleeping.is_set():
                    deps.close_conversation()
                    deps.pending_sleep = True
            else:
                if sleeping.is_set():
                    try:
                        deps.motion_queue.put_nowait({"type": "wake"})
                    except Exception:
                        logger.exception("web panel wake enqueue failed")
            return {"asleep": body.asleep}

        class DeafBody(BaseModel):
            deaf: bool

        @app.post("/api/deaf")
        def _set_deaf(body: DeafBody):
            # Meeting mode: stop processing the mic WITHOUT going to sleep, so
            # Mark won't interrupt a meeting but keeps his awake pose/eyes. Just
            # flips a flag the audio loop checks; safe from this FastAPI thread
            # (no SDK/GStreamer calls). Also closes any open conversation window
            # so nothing queued mid-toggle gets answered when you turn it back on.
            listener.deaf = bool(body.deaf)
            if listener.deaf:
                try:
                    deps.close_conversation()
                    listener.flush_pending()
                except Exception:
                    logger.exception("deaf-mode flush failed")
            logger.info("Deaf mode %s (mic %s).",
                        "ON" if listener.deaf else "OFF",
                        "ignored" if listener.deaf else "live")
            return {"deaf": listener.deaf}

        class ForgetBody(BaseModel):
            about: str

        @app.post("/api/forget")
        def _forget(body: ForgetBody):
            n = memory.forget(body.about)
            return {"removed": n, "facts": memory.all()}

        # --- People: face/voice recognition management from the panel ---------
        # Recognition itself is already built (Identity: InsightFace faces +
        # Resemblyzer voices, stored in ~/.reachy_mini_identities.json). The
        # ONLY reliable way to enroll a CLEAN name is to type it here - the voice
        # enroll path takes the name from STT, which garbles it ("Marcos"->
        # "Marte"). So the panel enroll: person sits in front of the camera,
        # types their exact name, taps Capture -> we grab a fresh camera frame
        # for the face embedding (+ voice if an utterance happens to be buffered).
        @app.get("/api/people")
        def _people():
            if deps.identity is None:
                return {"enabled": False, "people": []}
            try:
                return {"enabled": True, "people": deps.identity.people()}
            except Exception:
                logger.exception("people list failed")
                return {"enabled": True, "people": []}

        class EnrollBody(BaseModel):
            name: str

        @app.post("/api/enroll")
        def _enroll(body: EnrollBody):
            idn = deps.identity
            if idn is None:
                return {"ok": False, "error": "recognition not available"}
            name = (body.name or "").strip()
            if not name:
                return {"ok": False, "error": "type a name first"}
            face_emb = voice_emb = None
            try:
                frame = reachy_mini.media.get_frame()
                if frame is not None:
                    face_emb = idn.face_embedding(frame)
            except Exception:
                logger.exception("panel enroll: face capture failed")
            # Bonus voice sample only if a fresh utterance is buffered (the panel
            # user usually isn't speaking, so face is the primary signal).
            try:
                audio = getattr(listener, "last_utterance", None)
                if audio is not None:
                    voice_emb = idn.voice_embedding(audio)
            except Exception:
                logger.exception("panel enroll: voice capture failed")
            if face_emb is None and voice_emb is None:
                return {"ok": False, "error": "no_face",
                        "message": ("I couldn't see a face. Sit in front of the "
                                    "camera, look at Mark, and tap Capture again.")}
            msg = idn.enroll(name, face_emb=face_emb, voice_emb=voice_emb)
            logger.info("Panel enroll %r (face=%s voice=%s)",
                        name, face_emb is not None, voice_emb is not None)
            return {"ok": True, "message": msg,
                    "captured": {"face": face_emb is not None,
                                 "voice": voice_emb is not None},
                    "people": idn.people()}

        class RenameBody(BaseModel):
            old: str
            new: str

        @app.post("/api/identity_rename")
        def _identity_rename(body: RenameBody):
            idn = deps.identity
            if idn is None:
                return {"ok": False, "error": "recognition not available"}
            msg = idn.rename(body.old, body.new)
            return {"ok": True, "message": msg, "people": idn.people()}

        class IdentityForgetBody(BaseModel):
            name: str

        @app.post("/api/identity_forget")
        def _identity_forget(body: IdentityForgetBody):
            idn = deps.identity
            if idn is None:
                return {"ok": False, "error": "recognition not available"}
            msg = idn.forget(body.name)
            return {"ok": True, "message": msg, "people": idn.people()}

        class ShutdownBody(BaseModel):
            confirm: bool = False

        @app.post("/api/shutdown")
        def _shutdown(body: ShutdownBody):
            # Safe power-off of MARK ONLY (this Dell's brain + daemon) so the
            # physical OFF button can then be pressed without a head-drop. Does
            # NOT touch the M4 local LLM (separate machine, shared by other
            # services). Requires confirm:true (the page double-confirms).
            #
            # We DON'T stop the units from here directly: stopping the brain
            # kills THIS process mid-request, so the daemon would never be
            # stopped. Instead we hand the sequence to a DETACHED transient
            # systemd unit (verified: systemd-run --user works from the brain's
            # env) that outlives the brain and stops both in the right order.
            if not config.SHUTDOWN_ENABLED:
                return {"ok": False, "error": "shutdown disabled by config"}
            if not body.confirm:
                return {"ok": False, "error": "confirm required"}
            try:
                self._begin_safe_shutdown()
            except Exception as ex:
                logger.exception("safe shutdown launch failed")
                return {"ok": False, "error": str(ex)}
            return {"ok": True, "message": "Shutting down Mark safely "
                    "(parking head, torque off, stopping services). "
                    "Wait ~15s, then it's safe to press the physical OFF button."}

        # --- Desk + desk-lights control from the brain page ------------------
        # SAFETY (owner's standing rule, mirrored from the voice path):
        #   * LIGHTS are NOT sensitive -> on/off is IMMEDIATE, no confirm.
        #   * Every DESK MOVE IS sensitive -> confirm-before-act. The page double-
        #     taps (arm -> confirm) each move button; on the confirmed tap we run
        #     the SAME already-confirmed execution path the voice "yes" uses
        #     (desk.set_preset / execute_pending), never a one-tap move. STOP is a
        #     safety action -> immediate.
        # These controllers do their own locking and are plain LAN socket I/O (no
        # SDK/GStreamer), so calling them from this FastAPI thread is safe.

        @app.get("/api/desk_state")
        def _desk_state():
            # On-demand LIVE read (NOT on the 4s status poll) so we don't hammer
            # the Tuya desk/plug. Each field degrades to None if unreachable.
            out = {"desk_reachable": False, "level": None, "height_cm": None,
                   "moving": None, "light_on": None}
            desk = getattr(deps, "desk", None)
            light = getattr(deps, "light", None)
            if desk is not None and desk.available:
                try:
                    st = desk.status()
                    out["desk_reachable"] = bool(st.get("reachable"))
                    out["level"] = st.get("level")
                    out["moving"] = (st.get("motion") not in (None, "stop"))
                    out["height_cm"] = desk.height_cm()
                except Exception:
                    logger.debug("desk_state read failed", exc_info=True)
            if light is not None and light.available:
                try:
                    out["light_on"] = light.is_on()
                except Exception:
                    logger.debug("light_state read failed", exc_info=True)
            return out

        class LightBody(BaseModel):
            on: bool

        @app.post("/api/light")
        def _set_light(body: LightBody):
            # Immediate, no confirm (same as the voice light_on/light_off action).
            light = getattr(deps, "light", None)
            if light is None or not light.available:
                return {"ok": False, "error": "lights not configured"}
            ok = light.set_on(bool(body.on))
            return {"ok": ok, "light_on": bool(body.on) if ok else light.is_on()}

        class DeskMoveBody(BaseModel):
            # A preset move; the page has already double-tap-confirmed it.
            position: str  # "sitting" | "standing" | "end_of_day"

        @app.post("/api/desk_preset")
        def _desk_preset(body: DeskMoveBody):
            # A CONFIRMED preset move (the page armed+confirmed it, mirroring the
            # spoken "yes"). Presets self-stop at the saved height, so this is the
            # safe move primitive. Speaks a spoken confirmation like the voice path.
            desk = getattr(deps, "desk", None)
            if desk is None or not desk.available:
                return {"ok": False, "error": "desk not configured"}
            from reachy_mini_brain.tools.desk import INTENT_TO_LEVEL
            level = INTENT_TO_LEVEL.get(str(body.position).strip().lower())
            if level not in (1, 2, 4):
                return {"ok": False, "error": "position must be sitting, standing, or end_of_day"}
            lang = language.active if language is not None else config.DEFAULT_LANGUAGE
            pend = {"kind": "preset", "level": level}
            try:
                line = desk.execute_pending(pend, lang, light=getattr(deps, "light", None))
            except Exception as ex:
                logger.exception("desk preset from panel failed")
                return {"ok": False, "error": str(ex)}
            logger.info("Desk preset from panel: level %d (%s).", level, body.position)
            return {"ok": True, "level": level, "message": line}

        @app.post("/api/desk_stop")
        def _desk_stop():
            # Safety action: stop immediately, no confirm (mirrors spoken "stop").
            desk = getattr(deps, "desk", None)
            if desk is None or not desk.available:
                return {"ok": False, "error": "desk not configured"}
            ok = desk.stop()
            deps.pending_desk = None
            return {"ok": ok}

        logger.info("Web control panel available at %s", self.custom_app_url)

    def _begin_safe_shutdown(self) -> None:
        """Launch a DETACHED unit that stops the brain then the daemon, in order.

        Order + why (Pollen-sanctioned safe power-off):
          1. `systemctl --user stop reachy-brain`  -> the brain unit's
             KillSignal=SIGINT makes systemd send SIGINT, which runs the app's
             graceful cleanup (KeyboardInterrupt -> app.stop() -> _safe_shutdown:
             goto_sleep parks the head with torque ON, then disable_motors cuts
             torque with the head already resting = no head-drop). TimeoutStopSec
             is 20s, so we allow up to ~25s here.
          2. `systemctl --user stop reachy-daemon` -> releases the motor + media
             backend cleanly once the head is safely parked.
        Both units are Restart=on-failure, so a deliberate stop does not relaunch
        them. The sequence runs in its OWN transient unit so that stopping the
        brain (which is serving the HTTP request that triggered this) cannot kill
        the stopper before it reaches the daemon.
        """
        import shlex
        import subprocess

        brain = config.SHUTDOWN_BRAIN_UNIT
        daemon = config.SHUTDOWN_DAEMON_UNIT
        # Give the brain up to 25s to finish its graceful stop before the daemon
        # goes (its own TimeoutStopSec=20 is the hard cap; -w blocks until done).
        script = (
            f"systemctl --user stop -- {shlex.quote(brain)}; "
            f"systemctl --user stop -- {shlex.quote(daemon)}"
        )
        logger.warning("SAFE SHUTDOWN requested: stopping %s then %s (detached).",
                       brain, daemon)
        subprocess.run(
            ["systemd-run", "--user", "--collect",
             "--unit", "mark-safe-shutdown",
             "/bin/bash", "-lc", script],
            check=True, timeout=10,
        )

    def _safe_shutdown(self, reachy_mini: ReachyMini) -> None:
        """Return to a safe rest pose and disable torque before exiting."""
        logger.info("Shutting down: returning to safe pose...")
        self._stop_breathing()  # don't leave the loop running past exit
        try:
            reachy_mini.goto_sleep()
        except Exception:
            logger.exception("goto_sleep failed during shutdown")
        try:
            reachy_mini.disable_motors()
        except Exception:
            logger.exception("disable_motors failed during shutdown")


if __name__ == "__main__":
    app = ReachyMiniBrain()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
