# Mark — Architecture

This document describes **how Mark is built today** (the single-LLM "monolith") and the **target
multi-agent architecture** (a gate/router agent + per-task specialist agents) that the brain is
being refactored toward. It also captures the constraints that shape the design — above all,
**real-time voice latency** and the **local model's prompt cache**.

---

## 1. System topology

Mark spans up to four processes/machines:

```
┌───────────────────────────────┐        ┌──────────────────────────────┐
│  Reachy Mini + Daemon          │        │  Local LLM host (Mac)         │
│  ────────────────────          │        │  ───────────────────         │
│  motors, camera, mic array,    │        │  mlx_lm.server                │
│  speaker; shared GStreamer      │        │  Qwen3-30B (OpenAI-compatible)│
│  media pipeline; uvicorn :8000  │        │  reached over Tailscale       │
│  WebSocket SDK                  │        └───────────────▲──────────────┘
└───────────────▲─────────────────┘                        │ local-first
                │ ws://localhost:8000/ws/sdk                │ (fallback ↓)
┌───────────────┴──────────────────────────────────────────┴──────────────┐
│  BRAIN  (this app — reachy_mini_brain)                                    │
│  voice pipeline · LLM loop · 28 tools · control panel :8042               │
└───────────────▲──────────────────────────────────────────┬──────────────┘
                │ Google Calendar / personal integrations   │ OpenAI cloud
        ┌───────┴────────┐                                  │ (fallback)
        │  Mac bridge     │                          ┌───────▼────────┐
        │  (optional)     │                          │  api.openai.com │
        └────────────────┘                          └────────────────┘
```

- **Daemon** owns hardware and the shared media pipeline; the brain never touches ALSA/GStreamer directly for capture.
- **Brain** is the subject of this document.
- **Local LLM host** gives sub-second answers; **OpenAI cloud** is a transparent fallback.
- **Mac bridge** brokers personal integrations (e.g. Google Calendar) so credentials don't live on the robot.

---

## 2. The brain's runtime loop (today)

The brain runs a **daemon thread, `brain_loop()`** (`main.py`), plus a **control loop** that drains a
motion queue to the robot. Per turn:

```
utterance_queue (real speech)  ──┐
                                  ├──►  brain_loop()  ──►  TIER-0 deterministic gates
event_queue    (spontaneous)   ──┘                         (0 LLM calls)
                                                                │ falls through
                                                    LLMBrain.handle_turn()
                                                     • 1 big system prompt
                                                     • all 28 tools, flat
                                                     • stream reply sentence-by-sentence
                                                     • dispatch tool(s) → follow-up reply
                                                                │
                                                    TIER-0 post-turn backstops
                                                     (sleep / stop / web-search safety nets)
```

### Tier-0 deterministic gates (before the LLM — already present)
These run with **zero LLM calls** and are the foundation the multi-agent gate extends:

- **Wake-from-sleep** — on any fresh utterance while asleep: clear history, `wake_up()`, re-enable wobbling / head-tracking / direction-of-arrival / presence / antenna buttons.
- **Stale-utterance drop** — discard speech older than `MAX_UTTERANCE_AGE_S` (the user has moved on).
- **Speaker identification** — `identity.match_voice(audio)` (with a recent-face fallback) sets `deps.current_speaker` so memory + persona can be per-person.
- **Desk confirm-before-move** — a spoken "yes" within a TTL executes an armed desk move deterministically; "no" cancels; "stop" halts an active nudge. A sensitive physical action can never happen in a single LLM step.

### The LLM turn (`LLMBrain.handle_turn`, `llm.py`)
1. Resolve reply language (manually selected, **not** auto-detected from STT — short/accented clips misdetect).
2. Build the system prompt: one ~2,500-word `SYSTEM_PROMPT` + persona + a recognized-speaker line + a memory block + a per-turn language directive.
3. `messages = [system] + history + user/event message`.
4. **Stream** the completion sentence-by-sentence (`on_speak` speaks each sentence as it forms) — this is what makes Mark start talking almost immediately.
5. If the model emitted tool calls: speak a **filler** (unless the tool is "instant"), `dispatch` each tool, append results, then a **follow-up** streamed reply.
6. Return updated history (system prompt stripped).

### Local-first routing (`_stream_completion`)
- Mode is `local` / `cloud` / `auto` (chosen in the panel). `auto` routes **by measured latency** (EWMA) with an exponential re-probe backoff so a slow/evicted local model doesn't thrash.
- The local call is **serialized** (`_local_lock`) — `mlx_lm.server` is single-threaded.
- Local speech is **buffered**: a local miss/timeout speaks **nothing** before the cloud retry, so there's never double or partial speech.
- `prewarm_local()` primes the local model's **prompt cache** with Mark's real system+tools prefix so the first real turn is sub-second instead of ~8 s cold.

### Tier-0 post-turn backstops
If the reply *says* Mark will sleep / stop listening / search the web but **no tool fired**, the loop
does it anyway (deterministic intent detection on the reply text). Sleep is deferred until after
speech via `pending_sleep` → the motion queue.

---

## 3. Tools

- **`Tool`** (`tools/core.py`) — `name`, `description`, `parameters_schema`, `to_openai_tool()` (emits the OpenAI function schema), `run(deps, **kwargs) -> str`.
- **`ToolRegistry`** — `register`, `to_openai_tools()` (all schemas), `dispatch(deps, name, args)`.
- **`ToolDependencies`** — a dataclass threading **all shared services** into every tool: `reachy_mini`, `motion_queue`, `vision`, `memory`, `profiles`, `weather`, `search`, `bridge`, `todo`, `wiki`, `finance`, `news`, `transit`, `listener`, `identity`, `desk`, `light`, `available_moves`, plus mutable per-turn/session state (`current_speaker`, `pending_sleep`, `pending_desk`, `tracking_desired`).
- **`build_default_registry()`** registers a **flat set of 28 tools** across all domains.

This monolithic "one prompt + all tools + one loop" is exactly what the refactor restructures.

---

## 4. Why refactor — and the pattern chosen

**Problem:** one ~2,500-word prompt tries to govern every domain at once (chat, body, vision,
calendar, desk, identity, sleep, …) and all 28 tools are offered on every turn. As capabilities
grow, a single prompt gets harder to keep correct and focused.

**Pattern (chosen): routing.** A classifier/**gate** directs each turn to a **specialized handler**
with a focused prompt and an advisory tool subset. This is Anthropic's "routing" workflow and the
"routines + handoffs" idea from OpenAI's Swarm cookbook.

**Explicitly rejected: a heavy framework** (LangGraph / OpenAI Agents SDK / Swarm). Every routing
**hop adds an LLM call** (→ latency + compounding errors), frameworks add abstraction that obscures
the prompt/tool interface, and Swarm is documented as "not for production." Mark is a **real-time,
local-first voice robot** where latency is the governing constraint, so the router is **hand-rolled**
and **determinism-first**.

---

## 5. Target architecture — gate + per-task specialists

```
TIER 0  Deterministic gate  (EXISTS — extended, not replaced)
        wake / stale-drop / speaker-ID / desk confirm-before-move / stop / sleep+web backstops
        → 0 LLM calls
            │ (falls through)
TIER 1  GATE / ROUTER   (router.py)
        deterministic keyword + sticky classify → pick a specialist   (0 LLM in the common case;
        a tiny CLOUD-only classify ONLY when genuinely ambiguous)
            │
TIER 2  SPECIALIST AGENTS   (focused prompt suffix + advisory tool subset)
   ┌────────┬──────────┬───────────┬─────────────┬───────────────┬────────────┐
  Chat      Body/      Vision/     Knowledge     Productivity     Desk/Home
 (default)  Motion     Identity    web/wiki/     reminders/todo/  desk +
                                   news/finance/ pomodoro/        lights
                                   weather/      calendar
                                   directions
Shared:  ToolDependencies (unchanged) · the llm.py streaming loop · local-first + cloud fallback
```

### Specialist → tool mapping

| Specialist | Tools (advisory subset) |
|---|---|
| **Chat/Companion** *(default)* | `remember`, `forget`, `record_memo`, `play_game`, `set_personality`, inline translation |
| **Body/Motion** | `play_emotion`, `move_head`, `dance`, `react`, `head_tracking`, `look_around` |
| **Vision/Identity** | `camera`, `look_around`, `identity` (enroll/forget) |
| **Knowledge/Info** | `web_search`, `wikipedia`, `news`, `finance`, `weather`, `directions` |
| **Productivity** | `set_reminder`, `reminders`, `todo`, `focus_session`, `calendar_agenda`, `calendar_create` |
| **Desk/Home** | `desk` + lights |
| *(all specialists)* | `go_to_sleep`, `stop_listening` (shared exits) |

---

## 6. The #1 constraint — the local prompt cache (and how the design survives it)

`llm.py` deliberately keeps a **stable prompt prefix** ("keeps the prompt prefix stable so the local
model's prompt cache stays hot"), and `prewarm_local()` primes **one** prefix.

> **Key finding:** `mlx_lm.server` keeps a **single prompt-cache slot**. If each specialist had a
> *different* system prompt **and** a *different* tool subset, every domain switch would change the
> prompt prefix → **cache miss** → local turns regress from **~0.6 s to ~8 s cold**. This is the
> dominant risk of the refactor.

**The design that neutralizes it:**

1. **Split the prompt.** A **stable `SHARED_PREAMBLE`** holds every cross-cutting rule (identity, bilingual/language directive, short-reply, plain-spoken-text/TTS-safe, "you have a body + tools," identity-vs-remember, generic sleep/stop, a generic *confirm-before-physical-action* meta-rule). Domain specifics move into a small per-specialist **`focus_suffix`** (motion vocabulary, calendar-create confirmation + speaking style, desk confirm specifics, knowledge etiquette, companion persona). **Suffixes only add; they never remove a core rule** — so decomposed behavior stays equivalent to the monolith.
2. **Always pass the full 28-tool set** to the model. The per-specialist subset is **advisory only** — named in the variable segment ("You are in KNOWLEDGE mode; prefer Weather / WebSearch / …"). The tools array is **never trimmed**, because trimming it would change the cached prefix. (A 30B model handles 28 tools fine; routing already narrowed the intent.)
3. **Put all variable content last**, as a short context segment right before the user message:
   ```
   [system: SHARED_PREAMBLE + full 28-tool schema]  +  history  +  [context segment]  +  [user msg]
   ```
   A domain switch recomputes only the short tail; the primed prefix stays cached. **One `prewarm_local` now warms the prefix for all six specialists at once** — a strict improvement over the monolith.
4. **Sticky deterministic routing** keeps most consecutive turns in the *same* specialist, so switches — and thus tail recomputes — are rare.

---

## 7. Router algorithm (determinism-first)

`Router.route(user_text, is_system_event, deps, history) -> Agent`:

1. **`system_event` → fixed specialist, 0 LLM** (greeting/presence → Chat; motion event → Body). System events never trigger a classify.
2. **Keyword score** over a bilingual `AGENT_KEYWORDS` map (e.g. *weather/tempo/clima* → Knowledge; *lembr/remind/agenda/reunião* → Productivity; *dance/dança/vire/olhe* → Body; *camera/veja/"quem sou eu"/"meu nome é"* → Vision; *mesa/desk/luz/lights* → Desk; *traduz/translate/jogar/"lembra que"* → Chat — **translation stays in Chat, not Knowledge**).
3. **Clear winner → route there, 0 LLM.**
4. **No hits → sticky:** a short/pronoun-led continuation ("e amanhã?", "and tomorrow?") stays in `deps.current_specialist`; otherwise default to **Chat**. A reset-to-Chat rule (after N no-tool turns / on chit-chat markers) prevents sticky from trapping the conversation.
5. **Genuinely ambiguous (tie above threshold) → tiny classify**, gated by `MARK_ROUTER_CLASSIFY_ENABLED`, run on the **cloud model only** (a *local* classify would evict the hot conversation prefix), temperature 0, `max_tokens ≈ 3`, enum → specialist; any failure → Chat.

### LLM calls per turn
| Turn type | Router | Handle_turn | Total | vs today |
|---|---|---|---|---|
| Plain chat | 0 | 1 | **1** | same |
| Clear tool intent ("what's the weather") | 0 | 2 (tool + follow-up) | **2** | same |
| System event (greeting) | 0 | 1 | **1** | same |
| Genuinely ambiguous | 1 (cloud, tiny) | 1–2 | **+1 small** | only when ambiguous & flag on |

The common cases are **byte-for-byte the same call count** as today; extra cost is bounded to rare
ambiguous turns and **never touches the local cache**.

---

## 8. Module layout (target)

Additive files; surgical edits to `llm.py` / `main.py`; **`ToolDependencies` unchanged**.

| Path | Role |
|---|---|
| `agents/base.py` | `@dataclass(frozen=True) Agent { name, focus_suffix, tool_names, is_system_event_default }` |
| `agents/registry.py` | `build_agents()` → the 6 specialists + `MONOLITH_AGENT`; `AGENT_KEYWORDS`; `SHARED_TOOLS` |
| `agents/__init__.py` | re-exports `Agent`, `build_agents`, `MONOLITH_AGENT` |
| `router.py` | `Router.route()` + `_keyword_score` / `_sticky` / `_classify` (cloud-only) |
| `llm.py` *(edit)* | split `SYSTEM_PROMPT`; `_shared_preamble()` + `_agent_context()`; `handle_turn(..., agent=MONOLITH_AGENT)`; `prewarm_local` primes the shared prefix |
| `tools/core.py` *(edit)* | additive `ToolRegistry.subset(names)` + `is_known(name)` (for misroute detection only; `to_openai_tools()` still returns the full set) |
| `main.py` *(edit)* | in `brain_loop`, after tier-0 gates: `router.route(...)`, set `deps.current_specialist`, pass `agent=`; all behind `MARK_ROUTER_ENABLED` |

**`MONOLITH_AGENT`** — its `focus_suffix` is the original per-domain prompt body verbatim and its
`tool_names` are all 28. With routing disabled, the rendered prompt is **byte-for-byte the
pre-refactor prompt**, making the feature flag a true instant revert.

---

## 9. Safety, backward-compatibility, and rollback

- **Tier-0 gates + post-turn backstops are unchanged** and still run with zero LLM calls.
- **Dispatch is never gated by `agent.tool_names`** — any tool the model calls is dispatched, so a **misroute can never strand a tool** (the wrong specialist can still call the right tool). `tool_names` is used only for the advisory line and optional misroute detection.
- **Optional bounded 1-hop handoff** (`MARK_ROUTER_HANDOFF_ENABLED`, default OFF): considered only on a **pure tool-call turn** (the first completion produced tool calls and **no spoken content yet**) whose tools are all owned by **one other** specialist (`Router.owner_of`). When so, the turn is re-focused to that owner and the first completion is re-run **once** — cheap, since the cached `SHARED_PREAMBLE` prefix stays hot and only the short context tail recomputes. Because nothing was spoken before the re-run, there is **no double-speech**; tools spanning specialists, shared-only exits (sleep/stop), the monolith path, and turns already owned by the routed specialist all skip the hop. Any resolver failure leaves the original completion intact.
- **`ToolDependencies`** gains only one additive field, `current_specialist`.
- **Feature flags** (all default OFF during rollout): `MARK_ROUTER_ENABLED`, `MARK_ROUTER_CLASSIFY_ENABLED`, `MARK_ROUTER_HANDOFF_ENABLED`.
- **Rollback:** `git checkout v0-monolith` (or restore per-file `.bak`) + restart; or, with no redeploy, set `MARK_ROUTER_ENABLED=false` to revert behavior instantly.

---

## 10. Verification

1. **Prompt equivalence** — with routing OFF / `MONOLITH_AGENT`, the logged `messages` array must match the pre-refactor prompt byte-for-byte for the same inputs.
2. **Latency (the decisive test)** — baseline local `[timing]` for a scripted bilingual set (chat, weather, dance, reminder, desk, ambiguous). Then, routing ON, **drive alternating domains every turn** (weather → dance → reminder → desk → weather); every turn must stay local-fast (~0.6 s). Any ~8 s turn means the primed prefix is being invalidated → push the variable segment later in the chat template. *(This resolves the one thing only the live Qwen3 template can answer: where tool schemas render relative to variable system content.)*
3. **Routing accuracy** — one bilingual utterance per specialist; assert the router picks the right one and the right tool fires. Translation stays in **Chat** (no WebSearch).
4. **No-strand / reachability** — say "go to sleep" / "stop listening" from Knowledge or Productivity mode → tier-0 or a backstop still fires; a deliberately misrouted tool still dispatches.
5. **Classify-eviction check** — an ambiguous turn (cloud classify) immediately followed by a normal local turn: the local turn must **not** be cold (proves the classifier didn't evict the local cache).

---

## 11. Open questions / risks

1. **Qwen3 chat-template tool placement** (before/after variable system content) determines whether the late-segment trick fully protects the cached tools region — confirmed only by the latency stress test, not by reading code. **Top risk.**
2. **The classifier must be cloud-only** — a local classify evicts the single local cache slot.
3. **Decomposition drift** — a monolith rule could be lost between preamble and a suffix; mitigated by keeping all cross-cutting rules in `SHARED_PREAMBLE` (suffixes only add) and the byte-equivalence check.
4. **`system_event` → specialist map** depends on the actual `event_queue` event types; enumerate them to finalize the map.
5. **Sticky routing** could trap a conversation in a non-chat specialist without the reset-to-Chat rule.
