"""Build Mark's specialist agents + the monolith fallback, and the bilingual
keyword map the router scores against.

Design rules honoured here (see docs/ARCHITECTURE.md):
- focus_suffix for the calendar-create, desk and calendar-speaking domains is
  SLICED VERBATIM from the monolith SYSTEM_PROMPT (llm.CALENDAR_CREATE_BLOCK /
  DESK_BLOCK / CALENDAR_SPEAK_BLOCK), never retyped, so no rule text drifts.
- Every cross-cutting rule already lives in SHARED_PREAMBLE; a focus_suffix only
  ADDS domain specifics, it never removes a core rule.
- tool_names is advisory only. SHARED_TOOLS (the sleep/stop exits) are appended
  to every specialist so those exits stay named in the advisory line, but tier-0
  gates + backstops handle them regardless.
- MONOLITH_AGENT reproduces the pre-refactor prompt exactly (is_monolith=True →
  handle_turn uses the original single-prompt layout).
"""

from __future__ import annotations

from reachy_mini_brain import llm
from reachy_mini_brain.agents.base import Agent

# Cross-cutting exits every specialist should know it can still call. Tier-0
# gates/backstops fire these regardless of routing; naming them just keeps them
# visible to the model in any mode.
SHARED_TOOLS: tuple[str, ...] = ("go_to_sleep", "stop_listening")


def _with_shared(names: tuple[str, ...]) -> tuple[str, ...]:
    """Append SHARED_TOOLS without duplicating, preserving order."""
    out = list(names)
    for n in SHARED_TOOLS:
        if n not in out:
            out.append(n)
    return tuple(out)


# --- Focus text for domains that have NO dedicated monolith block ----------
# These are concise ADDITIONS (the shared preamble already covers identity,
# language, short-replies, TTS-safe text, "you have a body + tools", the
# identity-vs-remember distinction, sleep/stop and the generic confirm rule).
# They intentionally do not restate any of those.
_CHAT_FOCUS = (
    "You are in a casual conversation. Just chat naturally and briefly. You can "
    "save durable preferences with the remember tool (not for who someone is - "
    "that's the identity tool), record a voice memo, switch your personality, or "
    "play a quick game. When asked to translate, just translate inline - do not "
    "search the web for it."
)
_BODY_FOCUS = (
    "Favour physical expression. Use your head, antennas, emotions, dances and "
    "reactions to respond, and turn face-tracking on or off when asked. Just act "
    "on instant movements - no filler phrase needed before them."
)
_VISION_FOCUS = (
    "Use your camera to see. Take a picture and describe what's in front of you, "
    "or scan the room, when asked what you see. For recognising people, remember "
    "that 'remember me / I'm X / learn my face' is the identity tool (enroll), "
    "which captures their face and voice - first make sure they're looking at the "
    "camera and have just spoken."
)
_KNOWLEDGE_FOCUS = (
    "Answer real-world questions using the matching tool: web search for current "
    "or factual info, Wikipedia for encyclopedic facts, news headlines, stock or "
    "crypto prices and currency conversions, weather, and travel time or "
    "directions between places. Keep the spoken answer to one or two sentences."
)


def build_agents(registry=None) -> dict[str, Agent]:
    """Return the specialist agents keyed by name.

    `registry` is accepted for symmetry/future validation (e.g. asserting every
    advisory tool name is registered) but is not required to build the agents.
    """
    chat = Agent(
        name="chat", label="CHAT", focus_suffix=_CHAT_FOCUS,
        tool_names=_with_shared((
            "remember", "forget", "record_memo", "play_game", "set_personality",
        )),
    )
    body = Agent(
        name="body", label="BODY", focus_suffix=_BODY_FOCUS,
        tool_names=_with_shared((
            "play_emotion", "move_head", "dance", "react", "head_tracking",
            "look_around",
        )),
    )
    vision = Agent(
        name="vision", label="VISION", focus_suffix=_VISION_FOCUS,
        tool_names=_with_shared((
            "camera", "look_around", "identity",
        )),
    )
    knowledge = Agent(
        name="knowledge", label="KNOWLEDGE", focus_suffix=_KNOWLEDGE_FOCUS,
        tool_names=_with_shared((
            "web_search", "wikipedia", "news", "finance", "weather", "directions",
        )),
    )
    # Productivity keeps the monolith's calendar-create confirmation + calendar-
    # speaking blocks verbatim (they govern its tools). Concatenated with a blank
    # line between, matching the monolith's own spacing.
    productivity = Agent(
        name="productivity", label="PRODUCTIVITY",
        focus_suffix=(llm.CALENDAR_CREATE_BLOCK.strip() + "\n\n"
                      + llm.CALENDAR_SPEAK_BLOCK.strip()),
        tool_names=_with_shared((
            "set_reminder", "reminders", "todo", "focus_session",
            "calendar_agenda", "calendar_create_event",
        )),
    )
    # Desk/home keeps the monolith's desk block verbatim.
    desk = Agent(
        name="desk", label="DESK", focus_suffix=llm.DESK_BLOCK.strip(),
        tool_names=_with_shared(("desk",)),
    )
    return {
        a.name: a for a in (chat, body, vision, knowledge, productivity, desk)
    }


# The instant-revert fallback: its rendered prompt is byte-for-byte the
# pre-refactor SYSTEM_PROMPT (is_monolith → original layout in handle_turn), and
# it names all tools. Selected whenever routing is disabled.
MONOLITH_AGENT = Agent(
    name="monolith", label="MONOLITH", focus_suffix=llm.MONOLITH_SUFFIX,
    tool_names=(), is_monolith=True,
)

DEFAULT_AGENT_NAME = "chat"


# --- Bilingual keyword map the router scores against (pt-BR + en-US) --------
# Substring match on a normalized (lowercased, accent-stripped) utterance. Order
# does not matter; the router counts hits per agent. Translation markers live
# under CHAT deliberately (translation must NOT route to knowledge/web search).
AGENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "knowledge": (
        "weather", "tempo", "clima", "temperatura", "forecast", "previsao",
        "news", "noticia", "noticias", "headline", "manchete",
        "stock", "acao", "acoes", "bolsa", "price", "preco", "cotacao",
        "bitcoin", "crypto", "dolar", "euro", "moeda",
        "search", "pesquisa", "pesquisar", "procura na internet", "busca",
        "wikipedia", "wiki", "who is", "quem e", "what is", "o que e",
        "directions", "how do i get", "como chego", "como faco pra chegar",
        "transit", "onibus", "metro", "trajeto", "rota", "commute",
    ),
    "productivity": (
        "remind", "reminder", "lembrete", "lembra de", "me lembra",
        "timer", "alarm", "alarme", "cronometro",
        "todo", "to-do", "task", "tarefa", "lista de tarefas",
        "pomodoro", "focus session", "sessao de foco", "foco",
        "calendar", "calendario", "agenda", "schedule", "reuniao", "meeting",
        "compromisso", "evento",
    ),
    "body": (
        "dance", "danca", "danca", "dancar",
        "move your head", "mexe a cabeca", "vira", "vire", "olha pra", "olhe",
        "nod", "acena", "balanca", "wiggle", "antenna", "antena",
        "emotion", "emocao", "reaction", "reage", "reaja", "look around",
        "olha em volta", "escaneia",
    ),
    "vision": (
        "camera", "take a picture", "tira uma foto", "foto",
        "see", "veja", "ve isso", "what do you see", "o que voce ve",
        "who am i", "quem sou eu", "recognize", "reconhece", "reconhecer",
        "my name is", "meu nome e", "learn my face", "aprende meu rosto",
        "remember me", "lembra de mim", "enroll",
    ),
    "desk": (
        "desk", "mesa", "stand up", "levanta a mesa", "sit down", "abaixa a mesa",
        "raise", "sobe", "subir", "lower", "desce", "descer",
        "lights", "luz", "luzes", "brightness", "start my day", "comeca meu dia",
        "finish my day", "termina meu dia", "bom dia", "boa noite",
    ),
    "chat": (
        "translate", "traduz", "traducao", "como se diz", "how do you say",
        "play a game", "jogar", "vamos jogar", "game",
        "personality", "personalidade", "memo", "recado",
        "remember that", "lembra que", "meu favorito", "i prefer", "eu prefiro",
    ),
}
