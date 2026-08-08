"""Deterministic-first gate/router for Mark's multi-agent brain.

Given a turn, pick the specialist Agent that should handle it. The overwhelming
majority of turns are decided with ZERO extra LLM calls (system-event map,
keyword score, or sticky continuation); a tiny CLOUD-ONLY classify is used only
when the keyword signal is genuinely ambiguous AND enabled by a flag.

Why cloud-only classify: a LOCAL classify would evict mlx_lm.server's single
prompt-cache slot and cold-start the next real local turn (~0.6s -> ~8s). See
docs/ARCHITECTURE.md sections 6-7.

Nothing here gates tool DISPATCH - routing only chooses the prompt focus + the
advisory tool line, so a misroute can never strand a tool.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass

from reachy_mini_brain import config
from reachy_mini_brain.agents import (
    AGENT_KEYWORDS,
    DEFAULT_AGENT_NAME,
    build_agents,
)
from reachy_mini_brain.agents.base import Agent

logger = logging.getLogger(__name__)

# System events (greeting, reminder-fired, proactive announce, antenna press,
# web-search follow-up) are all "just say this naturally, no tool" - Chat owns
# them. There are no motion events on the event queue.
SYSTEM_EVENT_AGENT = "chat"

# Continuation markers: a short reply led by one of these is treated as "same
# topic as last turn" and stays in the current specialist (bilingual).
_CONTINUATION_PREFIXES = (
    "and ", "e ", "what about", "e o ", "e a ", "e sobre", "how about",
    "also ", "tambem", "then ", "entao", "e depois", "e amanha", "and tomorrow",
    "e ai", "ok ", "okay ", "sim", "yes", "no", "nao",
)
# Chit-chat markers reset routing to Chat regardless of stickiness.
_CHITCHAT_MARKERS = (
    "hi", "hello", "hey", "oi", "ola", "haha", "kkk", "lol", "thanks",
    "thank you", "obrigado", "obrigada", "valeu", "tchau", "bye",
    "good morning", "bom dia", "boa noite", "how are you", "tudo bem",
)


def _normalize(text: str) -> str:
    """Lowercase and strip accents so bilingual keyword matching is robust to
    diacritics (e.g. 'reuniao'/'reunião', 'acao'/'ação')."""
    text = (text or "").lower().strip()
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


@dataclass(frozen=True)
class RouteDecision:
    """The chosen specialist plus why (for the [route] log line)."""
    agent: Agent
    source: str          # "event" | "keyword" | "sticky" | "classify" | "default"
    scores: dict         # per-agent keyword hit counts (for debugging)


class Router:
    def __init__(self, agents: dict[str, Agent] | None = None,
                 classify_client=None, classify_model: str | None = None) -> None:
        # Build the specialists once; the router is stateless beyond this table.
        self.agents = agents if agents is not None else build_agents()
        self.default = self.agents[DEFAULT_AGENT_NAME]
        # Cloud client + model for the optional ambiguous-turn classify. When
        # None, classify is unavailable and ambiguity falls back to sticky/Chat.
        self._classify_client = classify_client
        self._classify_model = classify_model

    # -- keyword scoring ----------------------------------------------------
    def _keyword_score(self, norm_text: str) -> dict:
        scores: dict[str, int] = {}
        for name, words in AGENT_KEYWORDS.items():
            hits = sum(1 for w in words if w in norm_text)
            if hits:
                scores[name] = hits
        return scores

    # -- sticky continuation ------------------------------------------------
    def _is_continuation(self, norm_text: str) -> bool:
        """A short reply or one led by a continuation marker = same topic."""
        if not norm_text:
            return False
        word_count = len(norm_text.split())
        if word_count <= 3:
            return True
        return any(norm_text.startswith(p) for p in _CONTINUATION_PREFIXES)

    def _sticky(self, norm_text: str, deps, scores: dict) -> RouteDecision:
        # A greeting / chit-chat marker always resets to Chat so sticky can't
        # trap the conversation in a task specialist.
        if any(norm_text == m or norm_text.startswith(m + " ")
               for m in _CHITCHAT_MARKERS):
            return RouteDecision(self.default, "default", scores)
        current = getattr(deps, "current_specialist", None)
        if current and current in self.agents and self._is_continuation(norm_text):
            return RouteDecision(self.agents[current], "sticky", scores)
        return RouteDecision(self.default, "default", scores)

    # -- cloud-only classify (rare) -----------------------------------------
    def _classify(self, user_text: str, candidates: list[str], scores: dict):
        """Tiny CLOUD-ONLY classify to break a keyword tie. Returns a
        RouteDecision or None on any failure/unavailability. Never raises."""
        if not (config.ROUTER_CLASSIFY_ENABLED and self._classify_client
                and self._classify_model):
            return None
        options = candidates or list(self.agents)
        allowed = ", ".join(options)
        sys_prompt = (
            "You label a user utterance with the single best handler for a "
            "voice robot. Reply with EXACTLY ONE word from this list and nothing "
            f"else: {allowed}."
        )
        try:
            resp = self._classify_client.chat.completions.create(
                model=self._classify_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_text},
                ],
                temperature=0,
                max_completion_tokens=3,
                stream=False,
            )
            raw = (resp.choices[0].message.content or "").strip().lower()
        except Exception as ex:  # noqa: BLE001 - classify must never break a turn
            logger.warning("Router classify failed (%s); falling back.", type(ex).__name__)
            return None
        # Accept the answer only if it names a known agent (match by prefix so a
        # stray period/word doesn't spoil an otherwise-clear label).
        for name in options:
            if raw == name or raw.startswith(name):
                return RouteDecision(self.agents[name], "classify", scores)
        logger.info("Router classify returned unrecognized label %r; falling back.", raw)
        return None

    # -- main entry ---------------------------------------------------------
    def route(self, user_text: str, is_system_event: bool, deps,
              history=None) -> RouteDecision:
        # 1. System event -> fixed specialist, 0 LLM. Never classify.
        if is_system_event:
            return RouteDecision(self.agents[SYSTEM_EVENT_AGENT], "event", {})

        norm = _normalize(user_text)
        scores = self._keyword_score(norm)

        # 2/3. Keyword winner. Clear single winner -> route, 0 LLM.
        if scores:
            top = max(scores.values())
            winners = [n for n, s in scores.items() if s == top]
            if len(winners) == 1:
                return RouteDecision(self.agents[winners[0]], "keyword", scores)
            # 5. Genuine tie above zero -> ambiguous. Try a cloud classify
            #    (flag-gated); fall back to sticky/Chat on unavailability.
            decision = self._classify(user_text, winners, scores)
            if decision is not None:
                return decision
            return self._sticky(norm, deps, scores)

        # 4. No keyword hits -> sticky continuation or Chat default. 0 LLM.
        return self._sticky(norm, deps, scores)
