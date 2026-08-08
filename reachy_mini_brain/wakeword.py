"""Custom "Hey Mark" neural wake-word detector. Loads the small classifier
trained by scripts/train_wakeword.py on openWakeWord's audio embeddings and
scores a captured utterance. Used as the PRIMARY wake trigger, with the fuzzy
STT match kept as a fallback (see audio_io.py).

Scoring: embed the (silence-padded) utterance once through openWakeWord's
melspec+embedding front-end, then slide a 16-frame window over the embedding
sequence and take the max probability - so the wake word is detected wherever
it falls in the utterance.
"""

import logging
from pathlib import Path

import numpy as np

from reachy_mini_brain import config

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "models" / "hey_mark.joblib"
_FRAMES = 16  # embedding frames per classifier input (a ~2s window)


class WakeWord:
    def __init__(self) -> None:
        self.available = False
        self._clf = None
        self._threshold = 0.6
        self._af = None
        if not config.WAKEWORD_NEURAL_ENABLED:
            logger.info("Neural wake-word disabled (fuzzy matcher is primary). "
                        "Set REACHY_WAKEWORD_NEURAL=1 to enable.")
            return
        if not MODEL_PATH.exists():
            logger.info("No custom wake-word model at %s; using fuzzy match only.", MODEL_PATH)
            return
        try:
            import joblib
            from openwakeword.utils import AudioFeatures
            data = joblib.load(MODEL_PATH)
            self._clf = data["model"]
            self._threshold = float(data.get("threshold", 0.6))
            self._af = AudioFeatures(ncpu=1)
            self.available = True
            logger.info("Custom wake-word 'Hey Mark' loaded (threshold=%.2f).", self._threshold)
        except Exception:
            logger.exception("Failed to load custom wake-word model; falling back to fuzzy.")

    def score(self, utterance_16k: np.ndarray) -> float:
        """Max wake probability over the utterance (0.0 if unavailable)."""
        if not self.available:
            return 0.0
        try:
            audio = np.asarray(utterance_16k, dtype=np.float32)
            # Pad so at least one full 16-frame window exists (~2s of audio).
            if audio.shape[0] < 2 * 16000:
                audio = np.concatenate([np.zeros(2 * 16000 - audio.shape[0], np.float32), audio])
            pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            emb = np.asarray(self._af._get_embeddings(pcm))  # (F,96)
            if emb.shape[0] < _FRAMES:
                return 0.0
            windows = np.stack([
                emb[i:i + _FRAMES].reshape(-1)
                for i in range(0, emb.shape[0] - _FRAMES + 1)
            ])
            return float(self._clf.predict_proba(windows)[:, 1].max())
        except Exception:
            logger.exception("wake-word scoring failed")
            return 0.0

    def detect(self, utterance_16k: np.ndarray) -> bool:
        return self.score(utterance_16k) >= self._threshold
