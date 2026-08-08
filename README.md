<div align="center">

# 🤖 Mark

**A bilingual, expressive, always-on voice companion for the [Reachy Mini](https://www.pollen-robotics.com/) desktop robot.**

Mark listens for his name, understands Brazilian Portuguese **and** English, answers out loud in a natural voice, moves his head and antennas to react, recognizes the people he knows, and helps with real-world things — the weather, your calendar, reminders, the news, even your standing desk and its lights.

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![platform](https://img.shields.io/badge/platform-Reachy%20Mini-8A2BE2)
![llm](https://img.shields.io/badge/LLM-local--first%20(mlx)%20%2B%20cloud%20fallback-green)
![languages](https://img.shields.io/badge/speaks-pt--BR%20%2B%20en--US-orange)

</div>

---

## What Mark is

Mark is the "brain" application that runs on top of a Reachy Mini robot. The robot supplies the
body — a **6-degree-of-freedom head**, **two expressive antennas**, a **camera**, a **microphone
array**, and a **speaker**. Mark turns that body into a responsive, conversational companion:

- **Say "Hey Mark"** and talk to him in Portuguese or English — he replies in the same language, out loud, in one or two natural sentences.
- **He has a body and uses it** — he looks at you, tracks your face, nods, plays little emotions, dances to a beat, and reacts with his antennas and sound effects.
- **He knows who you are** — enrolled people are greeted by name (face **and** voice recognition, all on-device).
- **He gets things done** — web search, Wikipedia, news, weather, stock/crypto prices, your Google Calendar, reminders/timers, a to-do list, Pomodoro focus sessions, travel times, and control of a Tuya standing desk + lights.
- **He runs locally-first** — a warm local LLM answers in well under a second; the cloud is only a fallback, so Mark stays fast and keeps working even if the internet hiccups.

---

## How Mark hears, thinks, and speaks — the voice pipeline

```
        ┌──────────────────────────── Reachy Mini hardware ───────────────────────────┐
        │  mic array          camera            6-DoF head + antennas       speaker    │
        └────┬───────────────────┬───────────────────▲───────────────────────▲────────┘
             │ audio             │ frames             │ motion                │ audio
     ┌───────▼────────┐   ┌──────▼───────┐    ┌───────┴────────┐     ┌────────┴────────┐
     │  Silero VAD    │   │  Presence /  │    │  Motion queue  │     │   Piper TTS     │
     │ (is this       │   │  DoA / face  │    │  (nods, dances,│     │ (pt-BR / en-US) │
     │  speech?)      │   │  tracking    │    │  emotions)     │     └────────▲────────┘
     └───────┬────────┘   └──────────────┘    └───────▲────────┘              │
             │                                        │                       │ sentences
     ┌───────▼────────┐                               │                       │ (streamed)
     │  Wake word     │   "Hey Mark" (fuzzy match,    │             ┌─────────┴─────────┐
     │  gate          │    optional neural model)     │             │   LLM brain       │
     └───────┬────────┘                               │             │  local-first mlx  │
             │ wake / open conversation               │             │  → cloud fallback │
     ┌───────▼────────┐                               │             │  function-calling │
     │ faster-whisper │   large-v3-turbo on CUDA      │             └─────────┬─────────┘
     │  STT           │   → text + language           │                       │ tool calls
     └───────┬────────┘                               │             ┌─────────▼─────────┐
             │ transcript                             └─────────────┤   28 tools        │
             └────────────────────────────────────────────────────▶│ (weather, desk,   │
                                                                     │  calendar, …)     │
                                                                     └───────────────────┘
```

1. **Voice activity detection (Silero VAD)** decides when you are actually speaking, so the robot isn't transcribing silence.
2. **Wake-word gate** — Mark only engages when he hears his name ("Hey Mark", "Oi Mark", …), matched fuzzily and, optionally, with a trained neural wake-word model. After he wakes, a short **conversation window** stays open so you can keep talking without repeating his name.
3. **Speech-to-text (faster-whisper `large-v3-turbo`, on the GPU)** transcribes what you said and detects the language, with hardening against the classic Whisper "phantom" hallucinations on near-silence.
4. **The LLM brain** decides what to say and which **tools** to call, streaming its reply **sentence by sentence** so Mark starts talking almost immediately. It runs **local-first** (a warm `mlx` model on a nearby Mac over Tailscale, ~0.6 s/turn) and transparently **falls back to the OpenAI cloud** on any local error, timeout, or empty reply.
5. **Text-to-speech (Piper)** speaks the reply in the matching voice (pt-BR or en-US). Speech is **barge-in aware** — start talking over Mark and he stops to listen.
6. **The body** moves in parallel: presence detection greets arrivals, direction-of-arrival turns his head toward whoever spoke, face tracking keeps him looking at you, and a motion queue plays emotions/dances/reactions.

---

## What Mark can do — full capability list

Mark exposes his abilities to the LLM as **28 function-calling tools**, grouped here by domain
(this grouping is also the basis of the [multi-agent refactor](docs/ARCHITECTURE.md)):

### 💬 Chat & Companion
| Ability | What it does |
|---|---|
| Bilingual conversation | Understands & replies in pt-BR or en-US; never mixes languages in one reply |
| `remember` / `forget` | Save or drop durable facts about you (preferences, details) across restarts |
| `record_memo` | Save your last spoken message as an audio voice-memo |
| `play_game` | Play rock-paper-scissors (reads your hand with the camera) |
| `set_personality` | Switch persona/profile (e.g. a chattier or a storyteller mode) |
| Inline translation | Translate between languages on the fly, just by asking |

### 🕺 Body & Motion
| Ability | What it does |
|---|---|
| `play_emotion` | Play a recorded expressive movement to react physically |
| `move_head` | Look in a direction, nod, or shake his head |
| `dance` | Perform a rhythmic, beat-synced dance |
| `react` | Quick antenna + sound-effect punctuation |
| `head_tracking` | Turn automatic face-tracking on/off |
| `look_around` | Rotate the body to scan and describe the whole room |

### 👁️ Vision & Identity
| Ability | What it does |
|---|---|
| `camera` | Take a picture and describe what's in front of him (local vision model) |
| `identity` | Enroll / forget people — recognizes them by **face and voice**, greets by name |

### 🌐 Knowledge & Info
| Ability | What it does |
|---|---|
| `web_search` | Search the internet for current/factual info |
| `wikipedia` | Look up encyclopedic facts |
| `news` | Read the latest headlines |
| `finance` | Stock/crypto prices and currency conversions |
| `weather` | Current conditions and today's forecast |
| `directions` | Travel time / directions between places (e.g. your commute) |

### ✅ Productivity
| Ability | What it does |
|---|---|
| `set_reminder` / `reminders` | Set timers, reminders, and alarms; list or cancel them |
| `todo` | Manage a personal to-do list |
| `focus_session` | Run a Pomodoro-style work/break session with voice check-ins |
| `calendar_agenda` | Read what's coming up on your Google Calendar |
| `calendar_create` | Create a calendar event (always confirm-before-create) |

### 🖥️ Desk & Home
| Ability | What it does |
|---|---|
| `desk` (+ lights) | Raise/lower a Tuya standing desk, sit/stand presets, "start/finish my day" routines, and switch the desk lights. Every **desk move is confirm-before-move** (a sensitive physical action). |

### ⚙️ System behaviors (mostly automatic)
`go_to_sleep` · `stop_listening` · presence-based greetings · direction-of-arrival head turns ·
idle "life" movements · barge-in · a **recovery watchdog** · a meeting notifier and optional
morning briefing.

---

## Architecture at a glance

Mark is split into a **daemon** and a **brain** — plus optional helpers:

- **Daemon** — owns the hardware and the shared media (GStreamer) pipeline; exposes the robot over a local WebSocket SDK.
- **Brain** (this app) — the voice pipeline + LLM + tools + a browser **control panel**. Connects to the daemon over `ws://localhost:8000/ws/sdk`.
- **Local LLM host** — a nearby Mac running an `mlx` server (Qwen3-30B), reached over Tailscale, for sub-second answers.
- **Mac bridge** (optional) — brokers Google Calendar and other personal integrations.

The brain today runs as a **single LLM loop** with all 28 tools and one large system prompt. It is
being refactored into a **gate/router agent + six per-task specialist agents** for clearer
separation of concerns — see **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for the full current
and target designs, the latency/prompt-cache constraints, and the routing algorithm.

---

## Running Mark

Mark runs as a Reachy Mini app (entry point `reachy_mini_brain.main:ReachyMiniBrain`). Day-to-day
it's driven with the `markctl` helper:

```bash
markctl status      # daemon + brain health
markctl start        # start Mark
markctl restart      # cycle the brain app
markctl recover      # full daemon + app cycle (use if the mic goes deaf after a plain restart)
markctl logs         # tail the live log
```

The **control panel** is served at **`http://localhost:8042`** — volume, personality, face-tracking,
mic sensitivity, memory, LLM backend (local/cloud/auto), and live status/metrics.

> **Note:** an app-only `restart` can occasionally leave the microphone deaf (a GStreamer capture
> race). If Mark can speak but not hear, run `markctl recover` for a full daemon + app cycle.

---

## Configuration

**Everything is configured through `REACHY_*` environment variables** (loaded from a local `.env`);
no secrets live in the source. Highlights:

| Area | Variables (defaults in code) |
|---|---|
| Cloud LLM | `REACHY_LLM_MODEL`, `REACHY_LLM_CLOUD_TIMEOUT_S`, `OPENAI_API_KEY` |
| Local LLM | `REACHY_LOCAL_LLM_URL`, `REACHY_LOCAL_LLM_MODEL`, `REACHY_LLM_MODE` (`local`/`cloud`/`auto`) |
| STT | `REACHY_STT_MODEL` (`large-v3-turbo`), `REACHY_STT_DEVICE` (`cuda`), `REACHY_STT_COMPUTE_TYPE` |
| Wake word | `REACHY_ROBOT_NAME`, `REACHY_WAKE_FUZZY_THRESHOLD`, `REACHY_WAKEWORD_NEURAL` |
| Identity | `REACHY_IDENTITY_*` (face/voice thresholds, match margin, greet cooldown) |
| Vision | `REACHY_VISION_MODEL` (Ollama), `OLLAMA_ENDPOINT` |
| Desk / lights | `REACHY_DESK_*`, `REACHY_LIGHT_*` (Tuya device IDs, LAN IPs, local keys) |
| Integrations | `REACHY_BRIDGE_URL`, `REACHY_MAPS_KEY`, `REACHY_BRAVE_KEY`, `REACHY_WEATHER_*` |

---

## Privacy & data

Mark is built to keep personal data **on the device**:

- **Biometrics never leave the machine.** Face (InsightFace) and voice (Resemblyzer) embeddings are computed and matched locally, stored in `~/.reachy_mini_identities.json` — **outside this repository**.
- **Memory, preferences, logs, and transcripts** live in `$HOME` (`~/.reachy_mini_brain_*.json`, `~/.reachy_logs/`) — also outside the repo.
- **The repository contains code only** — no `.env`, no models/voices, no biometrics (enforced by `.gitignore`).
- The **local-first LLM** means most conversations are answered without any cloud round-trip; the OpenAI cloud is used only as a fallback.

---

## Tech stack

Python 3.10+ · [Reachy Mini SDK](https://www.pollen-robotics.com/) · OpenAI (cloud LLM + local
`mlx` server, OpenAI-compatible) · [faster-whisper](https://github.com/SYSTRAN/faster-whisper) ·
[Piper TTS](https://github.com/rhasspy/piper) · [Silero VAD](https://github.com/snakers4/silero-vad) ·
[openWakeWord](https://github.com/dscripka/openWakeWord) · [InsightFace](https://github.com/deepinsight/insightface) ·
[Resemblyzer](https://github.com/resemble-ai/Resemblyzer) · Ollama (local vision) ·
[tinytuya](https://github.com/jasonacox/tinytuya) (desk/lights) · FastAPI/uvicorn (control panel).

---

<div align="center">
<sub>Mark is a personal project — a small robot that's genuinely useful and a little bit alive.</sub>
</div>
