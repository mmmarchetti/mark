"""The Agent value object for Mark's gate + per-task-specialist architecture.

An Agent is just data: a name, a short display label used in the advisory
tool-preference line, the domain-specific prompt text (focus_suffix), and the
advisory subset of tool names the router associates with this domain. It carries
NO behaviour - llm.py renders it and the router selects it. See
docs/ARCHITECTURE.md for the full design.

Crucially, tool_names is ADVISORY only: the tools array sent to the model is
always the full registry, and dispatch is never gated by an agent, so a misroute
can still reach the right tool. tool_names drives only the advisory prompt line
and optional misroute/handoff detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Agent:
    #: Stable internal id, e.g. "chat", "body", "knowledge". Used for logging,
    #: sticky routing (deps.current_specialist) and the keyword map.
    name: str
    #: Short human label injected into the advisory line ("You are in <label>
    #: mode."), e.g. "CHAT", "BODY", "KNOWLEDGE".
    label: str
    #: Domain-specific system-prompt text appended (as part of the per-turn
    #: context segment) after the shared preamble + history. Empty for a pure
    #: fall-through agent whose behaviour is fully covered by the preamble.
    focus_suffix: str = ""
    #: Advisory subset of registered tool names for this domain (see class note).
    tool_names: tuple[str, ...] = field(default_factory=tuple)
    #: True only for MONOLITH_AGENT: selects the original single-prompt layout in
    #: handle_turn (byte-for-byte the pre-refactor prompt) instead of the split
    #: SHARED_PREAMBLE + context-segment layout used by specialists.
    is_monolith: bool = False
