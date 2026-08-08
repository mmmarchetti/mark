"""OpenAI chat.completions LLM brain: bilingual, tool-calling, short replies.

Routes LOCAL-FIRST when a local mlx_lm.server is configured (config.LOCAL_LLM_URL):
a warm local model on the second Mac answers in ~0.6s vs 5-13s for the cloud, and
staying on the LAN also fixes the "no answer" hang. Any local error/timeout/empty
reply transparently falls back to the OpenAI cloud, so Mark always answers.
"""

import json
import logging
import random
import time
from typing import Any

from openai import OpenAI

from reachy_mini_brain import config
from reachy_mini_brain.tools.core import ToolDependencies, ToolRegistry

logger = logging.getLogger(__name__)

_SENTENCE_ENDINGS = ".!?…"


def _sentence_end_index(text: str) -> int | None:
    """Index just past the first sentence-ending punctuation followed by a
    space or end-of-buffer. Avoids splitting on decimals like '3.5' by
    requiring the punctuation not to sit between two digits.
    """
    for i, ch in enumerate(text):
        if ch in _SENTENCE_ENDINGS:
            nxt = text[i + 1] if i + 1 < len(text) else " "
            prev = text[i - 1] if i > 0 else " "
            if ch == "." and prev.isdigit() and nxt.isdigit():
                continue  # decimal number, not a sentence break
            if nxt == " " or i + 1 == len(text):
                # Only split if we already have a reasonable amount of text,
                # so we don't speak one-word fragments.
                if len(text[: i + 1].strip()) >= 8:
                    return i + 1
    return None


# Restated at the top of the system prompt on every turn - see handle_turn.
LANGUAGE_DIRECTIVE = {
    "en": (
        "THIS TURN: the user spoke ENGLISH. Write your entire reply in English "
        "(en-US). Do not use any Portuguese, even if earlier replies were in "
        "Portuguese."
    ),
    "pt": (
        "THIS TURN: the user spoke BRAZILIAN PORTUGUESE. Write your entire reply "
        "in Brazilian Portuguese (pt-BR). Do not use any English, even if earlier "
        "replies were in English."
    ),
}

# Camera/vision take a few seconds; kept for the "must acknowledge" list below.
SLOW_TOOLS = {"camera", "look_around"}

# Instant, side-effect-only tools that are just queued to the motion loop or flip
# a flag - a filler before these would be annoying chatter ("let me see" before a
# nod). Any tool NOT in this set does real work (calendar/weather/search/memory/
# reminders/etc.) and MAY take a moment - Mark speaks a meaningful filler first so
# it never goes silent while the tool + follow-up LLM call run. Per the user's
# request: filler on any tool-using turn except these instant ones.
INSTANT_TOOLS = {
    "play_emotion", "move_head", "dance", "react", "head_tracking",
    "go_to_sleep", "stop_listening", "set_personality", "desk",
}

# Deterministic fallback so the robot never goes silent while a tool runs,
# even if the model doesn't self-generate an acknowledgment (observed to
# happen in practice with gpt-5.4-nano on clear-cut tool calls).
ACKNOWLEDGMENTS = {
    "pt": [
        "Deixa eu ver...",
        "Um segundo, estou nisso...",
        "Peraí que eu verifico...",
        "Olhando agora...",
        "Já te falo...",
    ],
    "en": [
        "Let me see...",
        "One second, on it...",
        "Let me check...",
        "Looking now...",
        "Just a moment...",
    ],
}

SYSTEM_PROMPT = """You are Mark, a small expressive desktop robot. Your name is \
Mark and you respond to it. \
You are bilingual: you understand and speak both Brazilian Portuguese (pt-BR) \
and US English (en-US) fluently. ALWAYS reply ENTIRELY in the same language the \
user just spoke to you in - never mix both languages in the same reply. If you \
cannot tell, default to Brazilian Portuguese.

Keep replies SHORT - usually ONE sentence, at most two. You are speaking out \
loud through a robot speaker, not writing text.

ALWAYS answer in PLAIN SPOKEN TEXT ONLY. Never use emojis, and never use \
symbols or abbreviations - write every unit and symbol as the full spoken word \
in the reply's language. For example write "graus Celsius" / "degrees Celsius" \
instead of "°C", "por cento" / "percent" instead of "%", "reais" instead of \
"R$", "e" / "and" instead of "&". Write numbers and everything else exactly as \
they should be read out loud. Do not use markdown, quotes, parentheses of \
symbols, or any character that is not part of normal speech.

Talk like a normal person in a relaxed chat. Do NOT end every reply with a \
question or an offer to help ("what would you like to do next?", "quer que eu \
ajude com mais alguma coisa?") - only ask a follow-up question when you \
genuinely need information. Most of the time, just answer and stop.

You have a body: a head with 6 degrees of freedom, two antennas, and a camera. \
Use your tools to physically react and look around, not just talk.

GOING TO SLEEP: when the user tells you to sleep, rest, stop, or says goodbye/good \
night (in any wording, e.g. "sleep", "sleepy", "go to sleep", "rest now", "tchau", \
"boa noite", "dorme"), you MUST call the go_to_sleep tool. Do NOT just say you are \
going to sleep - actually call the tool, or your body will not sleep.

Only acknowledge before acting when the action actually takes a moment - using \
the camera (vision analysis takes a few seconds). For instant actions (moving \
your head, playing an emotion/dance, going to sleep) just act, no filler phrase \
needed. When you do need to acknowledge (camera), vary the phrasing, don't \
repeat the same line every time.

RECOGNIZING PEOPLE vs REMEMBERING FACTS - two DIFFERENT tools, don't confuse them:
- When someone wants YOU TO KNOW WHO THEY ARE - "remember me", "I'm Marcos", \
"learn my face", "this is my wife Francielle", "you should recognize me" - call \
the IDENTITY tool with action enroll and their name. That captures their face and \
voice so you greet them by name next time. This is NOT the remember tool. First \
make sure they're looking at the camera and have just spoken.
- Use the remember tool only for PREFERENCES and details (favorite color, likes \
tennis, prefers short answers) - NOT for who someone is. Use forget when asked. \
To stop recognizing someone, call identity with action forget.
- Don't announce that you're saving or enrolling unless it's natural to.

You can also help with real-world things: search the web, look things up on \
Wikipedia, get news headlines, stock/crypto prices and currency conversions, set \
timers/reminders/alarms, manage a to-do list, check the user's Google Calendar, \
and translate between languages (just translate inline when asked). Use the \
matching tool when asked.

CONFIRMATION RULE for actions that change something (creating a calendar event): \
NEVER do it in one step. First tell the user exactly what you will do - the event \
title, date and time - and ask them to confirm. Only after they clearly say yes, \
call the tool with confirmed set to true. If they say no or change it, don't do it.

CONTROLLING THE STANDING DESK AND ITS LIGHTS (the desk tool): the user has a \
motorized standing desk you can move, plus lights on top of the desk you can \
switch. Moving the desk is a SENSITIVE, physical action, so it is ALWAYS \
confirm-before-move - you never move it in one step. Switching the lights is NOT \
sensitive and happens immediately.
- Daily routines: when the user says "start my day" / "good morning" / "let's \
begin", call the desk tool with action start_my_day (this sits the desk down AND \
turns the lights on together). When they say "finish my day" / "end my day" / \
"I'm done" / "good night", call action finish_my_day (this moves the desk to the \
end-of-day height so the chair fits under it AND turns the lights off together).
- Just the desk position (no light change): "sit down" = position sitting; "stand \
up" = position standing; "lower it for the chair" = position end_of_day. Call \
action preset with the right position.
- Just the lights: "turn the lights on" / "lights off" = action light_on or \
light_off. These take effect right away - no confirmation.
- Moving by a specific amount: ONLY move up or down by a NUMBER OF CENTIMETERS the \
user actually says (e.g. "up five centimeters"). Pass action up or down with \
centimeters set to that number. NEVER move the desk a vague or open-ended amount. \
If the user asks to raise/lower it but does NOT say how many centimeters, call the \
desk tool with action ask_cm - do not guess a number.
- Every DESK-MOVING action (start_my_day, finish_my_day, preset, up, down) does \
NOT move yet: the tool hands you back exactly what's about to happen. Tell the user \
that in one short sentence and ask them to confirm. The desk moves ONLY after they \
clearly say yes on their next reply; if they say no, it stays put. Never claim the \
desk has moved (or, for a routine, that the lights changed) before they confirm.
- "stop" always stops the desk immediately (action stop) - no confirmation needed. \
Use action status to tell them the current height and whether the lights are on.

SPEAKING ABOUT THE CALENDAR: when the calendar tool gives you data, you are \
SPEAKING it out loud, not reading a list. The data is already sorted and labeled \
with relative days and times - just phrase it like a person would.
- Lead with the count and the NEXT event. Mention at most two or three events; \
if more remain, say "and a few more, want those?" - NEVER read the whole list.
- Use the relative days and times as given (today, tomorrow, the weekday name) \
and say times the natural spoken way; for all-day items say it lasts all day, \
never invent a clock time.
- Keep it to one or two sentences, no lists, no dates like "27 Jul" or "09:00", \
no symbols. Reply in the user's language (English if they spoke English, \
Brazilian Portuguese if they spoke Portuguese) - never mix.
Good (en): "You've got three things today - first up your AI Builder interview \
at nine, then a Carta de Servicos review at nine forty-five." \
Good (pt): "Voce tem tres compromissos hoje. O proximo e a entrevista AI Builder \
as nove, e depois a revisao da Carta de Servicos as nove e quarenta e cinco." \
Bad: "Calendar: Mon 27 Jul 09:00 Virtual Interview; 09:45 PoC..."
"""


# ---------------------------------------------------------------------------
# Prompt decomposition for the multi-agent router (see agents/ + router.py).
#
# SYSTEM_PROMPT above is left BYTE-FOR-BYTE untouched. We SLICE it (never retype
# it) into a stable cross-cutting preamble plus the domain-specific blocks, so a
# specialist agent can be given the SHARED_PREAMBLE plus only its own focus text.
# The monolith path (handle_turn with agent=None) keeps using SYSTEM_PROMPT via
# _base_system_prompt and is therefore identical to the pre-refactor behaviour.
#
# The assertions guarantee the slices reconstruct the original exactly, so no
# rule can be silently dropped between the preamble and a suffix (decomposition
# drift is the top correctness risk of the refactor).
# ---------------------------------------------------------------------------
_IDX_CAL_CREATE = SYSTEM_PROMPT.index("CONFIRMATION RULE for actions that change something")
_IDX_DESK = SYSTEM_PROMPT.index("CONTROLLING THE STANDING DESK AND ITS LIGHTS")
_IDX_CAL_SPEAK = SYSTEM_PROMPT.index("SPEAKING ABOUT THE CALENDAR")
assert _IDX_CAL_CREATE < _IDX_DESK < _IDX_CAL_SPEAK, "SYSTEM_PROMPT section order changed"

# Stable across every specialist -> stays in the local model's cached prefix.
SHARED_PREAMBLE = SYSTEM_PROMPT[:_IDX_CAL_CREATE]
# Domain blocks, sliced verbatim (each keeps its own leading/trailing blank
# lines, so concatenating them reproduces SYSTEM_PROMPT exactly).
CALENDAR_CREATE_BLOCK = SYSTEM_PROMPT[_IDX_CAL_CREATE:_IDX_DESK]
DESK_BLOCK = SYSTEM_PROMPT[_IDX_DESK:_IDX_CAL_SPEAK]
CALENDAR_SPEAK_BLOCK = SYSTEM_PROMPT[_IDX_CAL_SPEAK:]
# The monolith's domain tail == the original prompt minus the shared preamble.
MONOLITH_SUFFIX = CALENDAR_CREATE_BLOCK + DESK_BLOCK + CALENDAR_SPEAK_BLOCK
assert SHARED_PREAMBLE + MONOLITH_SUFFIX == SYSTEM_PROMPT, "prompt slices do not reconstruct SYSTEM_PROMPT"


class LLMBrain:
    def __init__(self, tools: ToolRegistry, memory=None, profiles=None, metrics=None,
                 language=None, llm_mode=None) -> None:
        # Cloud client: explicit short timeout + capped retries. Without these the
        # SDK default is 600s + 2 silent retries - the exact "no answer / long
        # hang" symptom seen in the logs.
        self.client = OpenAI(timeout=config.CLOUD_TIMEOUT_S,
                             max_retries=config.CLOUD_MAX_RETRIES)
        # Local client (OpenAI-compatible mlx_lm.server on the second Mac), built
        # whenever configured - the runtime LLM MODE (local/cloud/auto, chosen from
        # the brain page) decides per-turn whether to actually use it, so the client
        # must exist even if the seed mode is cloud. api_key is required by the SDK
        # but ignored by the server.
        self.local_client = None
        if config.LOCAL_LLM_URL and config.LOCAL_LLM_MODEL:
            self.local_client = OpenAI(base_url=config.LOCAL_LLM_URL, api_key="local",
                                       timeout=config.LOCAL_LLM_TIMEOUT_S, max_retries=0)
            logger.info("Local LLM client ready: %s @ %s",
                        config.LOCAL_LLM_MODEL, config.LOCAL_LLM_URL)
        # Runtime backend selector (brain-page buttons). None -> behave as "auto".
        self.llm_mode = llm_mode
        # Recent local latency (EWMA), used by "auto" mode to route BY RESPONSE TIME.
        self._local_recent_latency_s = None
        self._auto_reprobe_at = 0.0   # in auto: next time to re-test a "slow" local
        # Exponential backoff for that re-probe. On a 24 GB M4 only ONE ~17 GB model
        # fits, so each auto re-probe during a coding session forces an ~18-22s
        # chat<->code SWAP. Start at the cooldown, double per slow/failed local
        # turn (capped), reset on a fast local turn or a button press.
        self._auto_reprobe_backoff_s = config.LOCAL_LLM_COOLDOWN_S
        self._local_skip_until = 0.0  # cooldown after a local failure
        # mlx_lm.server is single-threaded and wedges under concurrent requests,
        # so serialize every local call (real turns AND startup prewarm) here.
        import threading as _threading
        self._local_lock = _threading.Lock()
        self.tools = tools
        self.memory = memory
        self.profiles = profiles
        self.metrics = metrics
        self.language = language  # LanguageManager: the manually-selected reply language

    def _classify_local_failure(self) -> str:
        """Cheap post-failure probe: is the local server DOWN or UP-but-busy?

        A 2s GET /models right after a failed turn distinguishes the two causes
        that both surface as "Connection error": if the probe connects, the
        server is alive and the turn likely timed out under load / wedged
        (single-threaded mlx_lm); if it's refused/unreachable, the server
        process is actually down (crashed / OOM-killed / not started). Returns a
        short tag for the log. Never raises - diagnostics must not break a turn.
        """
        import socket
        import urllib.error
        import urllib.request
        base = config.LOCAL_LLM_URL.rstrip("/")
        t = time.time()
        try:
            with urllib.request.urlopen(base + "/models", timeout=2) as r:
                r.read(1)
            return f"probe:server-UP-in-{time.time() - t:.2f}s (turn timed out/busy?)"
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", e)
            if isinstance(reason, ConnectionRefusedError):
                return "probe:server-DOWN (connection refused - crashed/not running)"
            if isinstance(reason, socket.timeout):
                return "probe:server-UNRESPONSIVE (probe also timed out - wedged/overloaded)"
            return f"probe:unreachable ({type(reason).__name__}: {reason})"
        except Exception as e:  # noqa: BLE001 - diagnostics only
            return f"probe:error ({type(e).__name__})"

    def _auto_prefers_local(self) -> bool:
        """In 'auto' mode, decide whether to try local BY RESPONSE TIME.

        Prefer local while its recent average latency is under the target. If a
        past turn was slow (avg > target), stop preferring local but RE-PROBE it
        periodically (every cooldown window) - one probe turn can re-learn that
        the model warmed up, so we don't get stuck on the cloud forever. With no
        measurement yet (fresh start), optimistically try local: it's the whole
        point of having it, and a real failure just falls back + records latency.
        """
        lat = self._local_recent_latency_s
        if lat is None or lat <= config.LLM_AUTO_TARGET_S:
            return True
        # Local has been slow; re-probe on an EXPONENTIALLY GROWING interval. On the
        # 24 GB M4 a re-probe during a coding session forces an ~18-22s chat<->code
        # model swap, so we must NOT retry every cooldown - that would evict the
        # coder repeatedly. Each slow/failed local turn doubles the wait (capped),
        # so a coding session settles onto the cloud after a swap or two instead of
        # thrashing. A fast local turn resets the backoff (see _note_local_latency).
        now = time.time()
        if now >= self._auto_reprobe_at:
            self._auto_reprobe_at = now + self._auto_reprobe_backoff_s
            self._auto_reprobe_backoff_s = min(
                self._auto_reprobe_backoff_s * 2, config.LLM_AUTO_BACKOFF_MAX_S)
            return True
        return False

    def reset_auto_backoff(self) -> None:
        """Clear the auto re-probe backoff so the next turn re-evaluates local
        immediately. Called when the user presses a backend button - an explicit
        choice should take effect now, not after a long backoff window.
        """
        self._auto_reprobe_at = 0.0
        self._auto_reprobe_backoff_s = config.LOCAL_LLM_COOLDOWN_S

    def _note_local_latency(self, latency_s: float) -> None:
        """Fold a local turn's latency into an EWMA for 'auto' routing.

        A fast turn (at/under target) also RESETS the re-probe backoff: local is
        healthy again, so we should keep preferring it without an artificial wait.
        """
        prev = self._local_recent_latency_s
        self._local_recent_latency_s = (
            latency_s if prev is None else 0.5 * prev + 0.5 * latency_s)
        if latency_s <= config.LLM_AUTO_TARGET_S:
            self._auto_reprobe_at = 0.0
            self._auto_reprobe_backoff_s = config.LOCAL_LLM_COOLDOWN_S

    def _stream_completion(self, messages, tools, on_sentence, reply_language, interrupted):
        """Route a streamed completion LOCAL-FIRST, falling back to the cloud.

        Returns (full_text, tool_calls_list). Speaking sentence-by-sentence as
        they arrive cuts the perceived pause before the robot starts talking -
        the first sentence plays while the rest is still being generated.

        Local is tried first when configured and not in a post-failure cooldown;
        any error/timeout/empty reply falls back to the cloud so Mark always
        answers. To keep audio coherent we only speak from the backend that
        succeeds - a failed local attempt speaks nothing before the cloud retry.
        """
        # Runtime backend mode from the brain page (default "auto" if unset).
        mode = self.llm_mode.active if self.llm_mode is not None else "auto"
        if mode == "cloud" or self.local_client is None:
            # Forced cloud (button) or no local configured -> straight to cloud.
            use_local = False
        elif mode == "local":
            # Forced local (button): use it whenever it's not in a hard-failure
            # cooldown. Still falls back to cloud on an actual error so Mark never
            # goes silent, but never pre-empts local for being merely "slow".
            use_local = time.time() >= self._local_skip_until
        else:  # "auto" -> decide BY RESPONSE TIME
            use_local = (time.time() >= self._local_skip_until
                         and self._auto_prefers_local())
        # Serialize with any in-flight local call (e.g. startup prewarm). If the
        # lock isn't free almost immediately, don't wait - go cloud so the user
        # never blocks on the single-threaded local server.
        if use_local and not self._local_lock.acquire(timeout=0.2):
            logger.info("Local busy (prewarm/other); using cloud this turn.")
            use_local = False
        if use_local:
            # Buffer local speech instead of speaking live: the local model is
            # sub-second, so deferring speech costs almost nothing, and it means a
            # local failure (or empty reply) speaks NOTHING before the cloud
            # fallback - no double/partial speech. On success we replay in order.
            spoken: list[tuple[str, str]] = []
            collect = lambda s, lang: spoken.append((s, lang))
            _t_local = time.time()
            try:
                text, calls = self._stream_one(
                    self.local_client, config.LOCAL_LLM_MODEL, "local",
                    messages, tools, collect, reply_language, interrupted)
                interrupted_now = interrupted is not None and interrupted.is_set()
                # An empty reply with no tool call = nothing to say; treat as a
                # local miss and let the cloud handle it. An interrupted turn is a
                # legitimate early stop, not a failure.
                if (text.strip() or calls) or interrupted_now:
                    # Learn this turn's local latency so "auto" mode can route by
                    # response time on the next turn.
                    self._note_local_latency(time.time() - _t_local)
                    for s, lang in spoken:
                        on_sentence(s, lang)
                    return text, calls
                logger.info("Local LLM returned empty after %.1fs; falling back to cloud.",
                            time.time() - _t_local)
            except Exception as ex:
                # Log the exception TYPE + how long we waited, then classify the
                # failure with a cheap /models liveness probe: server DOWN (conn
                # refused/unreachable) vs UP-but-slow/busy (probe succeeds, so the
                # real call just timed out under load or wedged). This is the
                # difference between "the Mac's LLM crashed" and "it was busy" -
                # invisible in the old one-line message. Runs on the Dell; the
                # only M4 touch is one tiny GET that happens ONLY on a failure.
                elapsed = time.time() - _t_local
                diag = self._classify_local_failure()
                logger.warning(
                    "Local LLM failed after %.1fs: %s: %s [%s]; falling back to cloud.",
                    elapsed, type(ex).__name__, ex, diag)
                # A failure/timeout is a strong "local is slow" signal for auto mode.
                self._note_local_latency(elapsed)
            finally:
                self._local_lock.release()
            # Back off local for a while so we don't retry a down/busy Mac each turn.
            self._local_skip_until = time.time() + config.LOCAL_LLM_COOLDOWN_S
            if self.metrics is not None:
                self.metrics.record_llm_fallback()

        return self._stream_one(
            self.client, config.OPENAI_MODEL, "cloud",
            messages, tools, on_sentence, reply_language, interrupted)

    def _stream_one(self, client, model, backend, messages, tools, on_sentence,
                    reply_language, interrupted):
        """Stream one completion from a single backend (client+model).

        `backend` is "local" or "cloud" - used only for metrics labeling.
        Records latency + token usage per backend when self.metrics is set.
        """
        # Storyteller mode may speak at length; everything else stays short.
        from reachy_mini_brain.profiles import LONG_FORM_PROFILES
        max_tokens = config.LLM_MAX_TOKENS
        if self.profiles is not None and self.profiles.active in LONG_FORM_PROFILES:
            max_tokens = config.LLM_MAX_TOKENS_LONG
        kwargs = dict(model=model, messages=messages,
                      max_completion_tokens=max_tokens, stream=True)
        if tools is not None:
            kwargs["tools"] = tools
        # Ask for a usage summary in the final chunk (for the cost/token dashboard).
        want_usage = (backend == "cloud") or config.LOCAL_LLM_STREAM_USAGE
        if want_usage:
            kwargs["stream_options"] = {"include_usage": True}

        _t0 = time.time()
        stream = client.chat.completions.create(**kwargs)

        full = []
        buffer = ""
        usage = None
        # Accumulate streamed tool-call fragments by index.
        tool_frags: dict[int, dict] = {}
        for chunk in stream:
            # The final include_usage chunk has empty choices but carries usage.
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                full.append(delta.content)
                buffer += delta.content
                # Emit complete sentences as they form.
                buffer = self._drain_sentences(buffer, on_sentence, reply_language, interrupted)
            for tc in (delta.tool_calls or []):
                slot = tool_frags.setdefault(tc.index, {"id": None, "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments
            if interrupted is not None and interrupted.is_set():
                break

        # Speak whatever partial sentence is left.
        tail = buffer.strip()
        if tail and not (interrupted is not None and interrupted.is_set()):
            on_sentence(tail, reply_language)

        if self.metrics is not None:
            self.metrics.record_llm_call(
                backend=backend, latency_s=time.time() - _t0,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None))

        tool_calls = [
            {"id": f["id"], "name": f["name"], "args": f["args"]}
            for _, f in sorted(tool_frags.items())
            if f["name"]
        ]
        return "".join(full), tool_calls

    @staticmethod
    def _drain_sentences(buffer, on_sentence, reply_language, interrupted):
        """Speak any complete sentences in `buffer`, return the remainder."""
        while True:
            if interrupted is not None and interrupted.is_set():
                return ""
            idx = _sentence_end_index(buffer)
            if idx is None:
                return buffer
            sentence = buffer[:idx].strip()
            buffer = buffer[idx:].lstrip()
            if sentence:
                on_sentence(sentence, reply_language)

    def _base_system_prompt(self, reply_language: str, speaker: str | None = None) -> str:
        parts = [SYSTEM_PROMPT]
        # Persona first, so it colours everything below it.
        if self.profiles is not None:
            parts.append(self.profiles.persona_text())
        if speaker:
            parts.append(
                f"You have recognized the person you're talking to as {speaker} "
                f"(by their face and/or voice). Greet or address them by name "
                f"naturally when it fits; don't announce that you recognized them."
            )
        if self.memory is not None:
            block = self.memory.as_prompt_block(speaker=speaker)
            if block:
                parts.append(block)
        parts.append(LANGUAGE_DIRECTIVE[reply_language])
        return "\n\n".join(parts)

    def _shared_preamble(self) -> str:
        """The stable, cross-cutting system text shared by EVERY specialist.

        Constant across turns and across specialists, so together with the
        always-full tools array it forms a prompt prefix the local mlx model can
        keep cached even when the router switches domains. All per-turn/variable
        content lives in _agent_context (placed after history) instead. See
        docs/ARCHITECTURE.md section 6.
        """
        return SHARED_PREAMBLE

    def _agent_context(self, agent, reply_language: str, speaker: str | None = None) -> str:
        """Per-turn variable tail for a SPECIALIST turn: persona, recognized
        speaker, memory, an ADVISORY tool-preference line, the specialist's focus
        text, then the language directive. Placed AFTER history so switching
        specialists recomputes only this short segment, never the cached prefix.

        The specialist's tool subset is advisory only (named here) - the tools
        array sent to the model is always the full set, and dispatch is never
        gated, so a misroute can still reach the right tool.
        """
        parts = []
        # Persona first, so it colours the rest of the tail (mirrors the monolith).
        if self.profiles is not None:
            parts.append(self.profiles.persona_text())
        if speaker:
            # Same wording as _base_system_prompt (kept in sync deliberately).
            parts.append(
                f"You have recognized the person you're talking to as {speaker} "
                f"(by their face and/or voice). Greet or address them by name "
                f"naturally when it fits; don't announce that you recognized them."
            )
        if self.memory is not None:
            block = self.memory.as_prompt_block(speaker=speaker)
            if block:
                parts.append(block)
        if agent.tool_names:
            prefer = ", ".join(agent.tool_names)
            parts.append(
                f"You are in {agent.label} mode. Prefer these tools when they "
                f"fit: {prefer}. You may still use any other tool if the user "
                f"clearly needs it."
            )
        focus = (agent.focus_suffix or "").strip()
        if focus:
            parts.append(focus)
        parts.append(LANGUAGE_DIRECTIVE[reply_language])
        return "\n\n".join(parts)

    def prewarm_local(self) -> bool:
        """Prime the local model's prompt cache with Mark's real system+tools
        prefix so the FIRST real turn (and turns after a pause) are sub-second
        instead of paying the ~8s cold prefix-processing cost. No-op if no local
        client. Returns True on success. Must run serialized with real turns
        (mlx_lm.server is single-threaded) - the caller guards that.
        """
        if self.local_client is None:
            return False
        lang = self.language.active if self.language is not None else config.DEFAULT_LANGUAGE
        if lang not in LANGUAGE_DIRECTIVE:
            lang = config.DEFAULT_LANGUAGE
        # Warm whatever prefix REAL turns will present as messages[0]. With the
        # router on, specialist turns lead with the stable SHARED_PREAMBLE (the
        # variable content moves after history), so warming that one prefix
        # primes the cache for ALL six specialists at once. With the router off,
        # warm the monolith prompt exactly as before.
        if config.ROUTER_ENABLED:
            system_content = self._shared_preamble()
        else:
            system_content = self._base_system_prompt(lang, None)
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "hi"},
        ]
        # Don't wait if a real turn holds the lock - just skip (that turn warms
        # the cache anyway). Never run concurrently with a real local call.
        if not self._local_lock.acquire(blocking=False):
            return False
        try:
            self.local_client.chat.completions.create(
                model=config.LOCAL_LLM_MODEL, messages=messages,
                tools=self.tools.to_openai_tools(), max_completion_tokens=1, stream=False)
            return True
        except Exception as ex:
            logger.debug("Local prewarm skipped: %s", ex)
            return False
        finally:
            self._local_lock.release()

    def handle_turn(
        self,
        deps: ToolDependencies,
        history: list[dict[str, Any]],
        user_text: str,
        user_language: str,
        on_speak,
        interrupted=None,
        is_system_event=False,
        tools_used=None,
        agent=None,
        handoff_resolver=None,
    ) -> list[dict[str, Any]]:
        """Run one full conversational turn (including any tool round-trips).

        `on_speak(text, language)` is called for every piece of text that
        should be spoken aloud, in order. If `interrupted` (a threading.Event)
        is set mid-turn by a barge-in, remaining LLM/tool work is skipped.

        When `is_system_event` is True, `user_text` is an instruction to the
        robot (e.g. "a person appeared, greet them"), not something the user
        said - it's injected as a system message so the model acts on it
        without treating it as user speech to answer.

        `agent` selects the prompt layout. None or a monolith agent uses the
        original single-prompt layout (byte-for-byte the pre-refactor prompt);
        a specialist agent uses the split layout (stable SHARED_PREAMBLE prefix
        + a per-turn context segment placed after history) so domain switches
        don't evict the local model's cached prefix. The tools array is ALWAYS
        the full set and dispatch is NEVER gated by the agent - a misroute can
        still reach the right tool.

        `handoff_resolver(tool_names) -> agent|None` is the OPTIONAL bounded
        1-hop misroute recovery (gated by config.ROUTER_HANDOFF_ENABLED, default
        OFF). If the first completion is a PURE tool-call turn (no spoken
        content yet) and every proposed tool is owned by one OTHER specialist,
        the turn is re-focused to that owner and the first completion is re-run
        ONCE. Nothing was spoken before the re-run, so there is no double-speech;
        the SHARED_PREAMBLE prefix stays cached so only the short context tail
        recomputes. Any failure/ambiguity leaves the original completion intact.
        """
        def _interrupted() -> bool:
            return interrupted is not None and interrupted.is_set()

        def _persisted_history(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
            # Drop the leading system prompt (msgs[0]) always, plus the per-turn
            # context segment on the specialist path (it's variable and must not
            # persist - keeping it would pollute the cached prefix next turn).
            out = msgs[1:]
            if context_msg is not None:
                out = [m for m in out if m is not context_msg]
            return out

        # Reply language is chosen MANUALLY (LanguageManager), not auto-detected
        # from STT: STT mis-detects short/accented utterances (e.g. "Gracias" ->
        # Spanish; an English question answered in Portuguese), which caused
        # language mixing. A fixed language also keeps the prompt prefix stable
        # so the local model's prompt cache stays hot. Falls back to the
        # STT-detected/default language only if no LanguageManager is wired.
        if self.language is not None:
            reply_language = self.language.active
        else:
            reply_language = user_language if user_language in LANGUAGE_DIRECTIVE else config.DEFAULT_LANGUAGE
        if reply_language not in LANGUAGE_DIRECTIVE:
            reply_language = config.DEFAULT_LANGUAGE

        speaker = getattr(deps, "current_speaker", None)
        use_specialist = agent is not None and not getattr(agent, "is_monolith", False)
        # The per-turn context segment (specialist path only) must NOT persist
        # into history - it's variable and would pollute the cached prefix next
        # turn. Tracked by identity and filtered out of the returned history.
        context_msg = None
        if use_specialist:
            # SPLIT layout: stable SHARED_PREAMBLE prefix (+ always-full tools)
            # stays cached; the variable per-turn context goes AFTER history as
            # its own system message, right before the user/event message, so a
            # domain switch recomputes only this short tail.
            context_seg = self._agent_context(agent, reply_language, speaker)
            messages = [{"role": "system", "content": self._shared_preamble()}] + history
            context_msg = {"role": "system", "content": context_seg}
            messages.append(context_msg)
        else:
            # MONOLITH layout: identical to the pre-refactor code path.
            system_content = self._base_system_prompt(reply_language, speaker)
            messages = [{"role": "system", "content": system_content}] + history
        if is_system_event:
            # Spontaneous action: inject the directive as a system message.
            messages.append({"role": "system", "content": user_text})
        else:
            messages.append({"role": "user", "content": user_text})

        # From here on, speak in the resolved language.
        user_language = reply_language

        _t0 = time.time()
        content, tool_calls = self._stream_completion(
            messages, self.tools.to_openai_tools(), on_speak, user_language, interrupted
        )
        logger.info("[timing] LLM streamed %.2fs", time.time() - _t0)

        # Optional bounded 1-hop misroute handoff (flag-gated, default OFF).
        # ONLY when this was a PURE tool-call turn (no content spoken yet) do we
        # consider re-focusing: nothing has been said, so re-running the first
        # completion for the owning specialist can't cause double-speech. The
        # resolver returns the sole OTHER specialist that owns every proposed
        # tool, or None (unowned / spans specialists / already correct). We hop
        # AT MOST once; the SHARED_PREAMBLE prefix stays cached so only the short
        # context tail recomputes.
        if (config.ROUTER_HANDOFF_ENABLED and handoff_resolver is not None
                and use_specialist and tool_calls and not content
                and not _interrupted()):
            try:
                owner = handoff_resolver([tc["name"] for tc in tool_calls])
            except Exception as ex:  # noqa: BLE001 - handoff must never break a turn
                logger.warning("Handoff resolver failed (%s); keeping route.", type(ex).__name__)
                owner = None
            if owner is not None and owner.name != agent.name:
                logger.info("[route] handoff %s -> %s (tools=%s)",
                            agent.name, owner.name, [tc["name"] for tc in tool_calls])
                agent = owner
                if deps is not None:
                    setattr(deps, "current_specialist", owner.name)
                # Rebuild the split layout for the new owner: same cached
                # SHARED_PREAMBLE prefix, a fresh per-turn context tail. Track
                # the new context_msg by identity so it's stripped from history.
                context_seg = self._agent_context(agent, reply_language, speaker)
                messages = [{"role": "system", "content": self._shared_preamble()}] + history
                context_msg = {"role": "system", "content": context_seg}
                messages.append(context_msg)
                if is_system_event:
                    messages.append({"role": "system", "content": user_text})
                else:
                    messages.append({"role": "user", "content": user_text})
                _th = time.time()
                content, tool_calls = self._stream_completion(
                    messages, self.tools.to_openai_tools(), on_speak, user_language, interrupted
                )
                logger.info("[timing] LLM re-streamed after handoff %.2fs", time.time() - _th)

        # Rebuild the assistant message for the history/tool round-trip.
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if content:
            assistant_msg["content"] = content
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["args"] or "{}"}}
                for tc in tool_calls
            ]
        messages.append(assistant_msg)

        # Speak a meaningful filler before running any tool that does real work
        # (anything not instant), so Mark isn't silent while the tool + the
        # follow-up LLM call run. Only when the model didn't already say
        # something itself, and only once per turn.
        needs_filler = any(tc["name"] not in INSTANT_TOOLS for tc in tool_calls)
        if not content and needs_filler:
            lang = user_language if user_language in ACKNOWLEDGMENTS else config.DEFAULT_LANGUAGE
            on_speak(random.choice(ACKNOWLEDGMENTS[lang]), user_language)

        if tool_calls:
            for call in tool_calls:
                try:
                    args = json.loads(call["args"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                logger.info("Tool call: %s(%s)", call["name"], args)
                if tools_used is not None:
                    tools_used.append(call["name"])
                _tt = time.time()
                result = self.tools.dispatch(deps, call["name"], args)
                logger.info("[timing] tool %s %.2fs", call["name"], time.time() - _tt)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": result,
                    }
                )

            # If the user cut in during the tool call, skip the (costly)
            # follow-up reply entirely - they're already asking something else.
            if _interrupted():
                logger.info("Turn interrupted before follow-up reply; skipping it.")
                return _persisted_history(messages)

            _t1 = time.time()
            # Follow-up reply after tools: stream it too (no more tools here).
            final_content, _ = self._stream_completion(
                messages, None, on_speak, user_language, interrupted
            )
            logger.info("[timing] LLM follow-up streamed %.2fs", time.time() - _t1)
            if final_content:
                messages.append({"role": "assistant", "content": final_content})

        # Strip the leading system prompt (and, on the specialist path, the
        # per-turn context segment) back off before returning updated history.
        return _persisted_history(messages)
