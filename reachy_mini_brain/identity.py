"""Face + voice identity: recognize enrolled people so Mark can greet by name
and keep per-person memory. Models load lazily on CPU (the GPU is busy with
STT/vision). Embeddings are stored locally; nothing leaves the machine.

- Face: InsightFace (buffalo_l via onnxruntime, CPU).
- Voice: Resemblyzer VoiceEncoder (CPU).
Matching is cosine similarity above a threshold.
"""

import json
import logging
import threading
from pathlib import Path

import numpy as np

from reachy_mini_brain import config

logger = logging.getLogger(__name__)

STORE = Path.home() / ".reachy_mini_identities.json"

# Keep several reference embeddings per person per modality. One embedding was
# too fragile: a single noisy reference both MISSED known people and let others
# match the wrong name. Matching now takes the best cosine over the whole list
# (best-of-N), which is far more stable. Cap the list so it can't grow forever.
_MAX_SAMPLES = 5


def _cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def _as_sample_list(value) -> list:
    """Normalize a stored modality value to a list-of-embeddings.

    Backward-compatible: the old format stored ONE flat embedding
    (list[float]); the new format stores a list of embeddings
    (list[list[float]]). Detect a flat vector by its first element being a
    number and wrap it as a single-sample list.
    """
    if not isinstance(value, list) or not value:
        return []
    if isinstance(value[0], (int, float)):
        return [value]          # old single-embedding shape -> wrap
    return value                 # already list-of-embeddings


class Identity:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._face_app = None
        self._voice_enc = None
        self._people: dict[str, dict] = self._load()  # name -> {"face": [...], "voice": [...]}

    # ---- persistence ----
    def _load(self) -> dict:
        try:
            if STORE.exists():
                raw = json.loads(STORE.read_text())
                # Upgrade any old single-embedding entries to the list format
                # in-memory so all downstream code sees list-of-embeddings.
                for rec in raw.values():
                    for kind in ("face", "voice"):
                        if kind in rec:
                            rec[kind] = _as_sample_list(rec[kind])
                return raw
        except Exception:
            logger.exception("identity load failed")
        return {}

    def _save(self) -> None:
        try:
            STORE.write_text(json.dumps(self._people))
        except Exception:
            logger.exception("identity save failed")

    def names(self) -> list[str]:
        return list(self._people.keys())

    def people(self) -> list[dict]:
        """Enrolled people with which biometrics we hold for each - for the UI
        list (no raw embeddings leave this module)."""
        with self._lock:
            return [
                {"name": name, "face": ("face" in p), "voice": ("voice" in p)}
                for name, p in self._people.items()
            ]

    def rename(self, old: str, new: str) -> str:
        """Re-key a person WITHOUT losing their face/voice embeddings (fixes an
        STT-garbled enrollment name like 'Marte' -> 'Marcos'). Case-insensitive
        match on the old name; merges into an existing target if one already
        exists (target's own embeddings win)."""
        old, new = old.strip(), new.strip()
        if not old or not new:
            return "I need both the current name and the new name."
        with self._lock:
            src = next((k for k in self._people if k.lower() == old.lower()), None)
            if src is None:
                return f"I don't have {old} enrolled."
            if src == new:
                return f"{new} is already the name."
            rec = self._people.pop(src)
            dst = next((k for k in self._people if k.lower() == new.lower()), None)
            if dst is not None:
                # Merge into an existing target: CONCATENATE the sample lists so
                # neither person's references are lost (more clean samples =
                # better matching), capped at _MAX_SAMPLES.
                for k in ("face", "voice"):
                    if k in rec:
                        merged = _as_sample_list(self._people[dst].get(k, [])) + _as_sample_list(rec[k])
                        self._people[dst][k] = merged[-_MAX_SAMPLES:]
            else:
                self._people[new] = rec
            self._save()
        return f"Okay, {src} is now {new}."

    # ---- lazy models ----
    def _face(self):
        if self._face_app is None:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(480, 480))
            self._face_app = app
        return self._face_app

    def _voice(self):
        if self._voice_enc is None:
            from resemblyzer import VoiceEncoder
            self._voice_enc = VoiceEncoder("cpu")
        return self._voice_enc

    # ---- embeddings ----
    def face_embedding(self, frame_rgb) -> list | None:
        try:
            faces = self._face().get(np.asarray(frame_rgb)[:, :, ::-1])  # insightface wants BGR
            if not faces:
                return None
            f = max(faces, key=lambda x: x.det_score)
            return f.normed_embedding.tolist()
        except Exception:
            logger.exception("face embedding failed")
            return None

    def voice_embedding(self, audio_16k) -> list | None:
        try:
            from resemblyzer import preprocess_wav
            wav = preprocess_wav(np.asarray(audio_16k, dtype=np.float32), source_sr=16000)
            return self._voice().embed_utterance(wav).tolist()
        except Exception:
            logger.exception("voice embedding failed")
            return None

    # ---- enroll / match ----
    def _append_sample(self, rec: dict, kind: str, emb) -> None:
        """Append one embedding to a person's sample list, capping the length
        (drop the oldest on overflow)."""
        samples = _as_sample_list(rec.get(kind, []))
        samples.append(emb)
        if len(samples) > _MAX_SAMPLES:
            samples = samples[-_MAX_SAMPLES:]
        rec[kind] = samples

    def enroll(self, name: str, face_emb=None, voice_emb=None) -> str:
        name = name.strip()
        if not name:
            return "I need a name to remember you by."
        with self._lock:
            p = self._people.setdefault(name, {})
            if face_emb is not None:
                self._append_sample(p, "face", face_emb)
            if voice_emb is not None:
                self._append_sample(p, "voice", voice_emb)
            self._save()
        got = [k for k in ("face", "voice") if k in self._people[name]]
        return f"Got it, I'll remember {name} by their {' and '.join(got)}."

    def forget(self, name: str) -> str:
        name = name.strip()
        with self._lock:
            # Case-insensitive match on the stored name.
            key = next((k for k in self._people if k.lower() == name.lower()), None)
            if key is None:
                return f"I don't have {name} enrolled."
            del self._people[key]
            self._save()
        return f"Okay, I've forgotten {key}."

    def match_face(self, frame_rgb) -> str | None:
        emb = self.face_embedding(frame_rgb)
        return self._best(emb, "face", config.IDENTITY_FACE_THRESHOLD)

    def match_voice(self, audio_16k) -> str | None:
        emb = self.voice_embedding(audio_16k)
        return self._best(emb, "voice", config.IDENTITY_VOICE_THRESHOLD)

    def _best(self, emb, kind: str, threshold: float) -> str | None:
        if emb is None:
            return None
        # Score each person by their BEST matching sample (best-of-N), then
        # require a margin over the runner-up: an ambiguous match (two people
        # nearly tied) becomes "unknown" instead of a confident wrong name.
        scored: list[tuple[float, str]] = []
        with self._lock:
            for name, p in self._people.items():
                samples = _as_sample_list(p.get(kind, []))
                if not samples:
                    continue
                s = max(_cosine(emb, e) for e in samples)
                scored.append((s, name))
        if not scored:
            return None
        scored.sort(reverse=True)
        best_score, best_name = scored[0]
        if best_score < threshold:
            return None
        if len(scored) > 1 and (best_score - scored[1][0]) < config.IDENTITY_MATCH_MARGIN:
            logger.debug(
                "%s match ambiguous: %s=%.3f vs %s=%.3f (margin<%.3f), returning unknown",
                kind, best_name, best_score, scored[1][1], scored[1][0],
                config.IDENTITY_MATCH_MARGIN,
            )
            return None
        return best_name
