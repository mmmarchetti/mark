"""Bilingual speech-to-text using faster-whisper (local, GPU)."""

import logging

import numpy as np
from faster_whisper import WhisperModel

from reachy_mini_brain import config

logger = logging.getLogger(__name__)

# large-v3-turbo invents these canonical caption/subtitle phrases on near-silence
# (confirmed live: 89 phantom turns in transcript.jsonl, e.g. "Gracias.",
# "Продолжение следует..." = RU "to be continued"). The transcribe() params below
# suppress most of them; this denylist is belt-and-suspenders for the exact phrases
# we've seen, applied ONLY to short clips (phantoms are always short) so a real
# sentence containing one of these words is never dropped. "thank you" alone is NOT
# here - it's legitimate; only the known-caption variants are.
_HALLUCINATION_PHRASES = {
    "gracias", "gracias.", "продолжение следует", "продолжение следует...",
    "thanks for watching", "thanks for watching!", "thank you for watching",
    "thank you for watching!", "subscribe", "obrigado", "obrigado.",
}
_HALLUCINATION_MAX_CLIP_S = 2.0


class STTEngine:
    """Wraps faster-whisper for pt-BR / en-US transcription with language auto-detection."""

    def __init__(self) -> None:
        logger.info("Loading STT model %s on %s...", config.STT_MODEL, config.STT_DEVICE)
        self.model = WhisperModel(
            config.STT_MODEL,
            device=config.STT_DEVICE,
            compute_type=config.STT_COMPUTE_TYPE,
        )
        logger.info("STT model loaded.")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> tuple[str, str]:
        """Transcribe a mono float32 audio buffer.

        Returns:
            (text, language) where language is "en" or "pt" (or whatever
            two-letter code Whisper detected, for other languages).
        """
        if sample_rate != 16000:
            raise ValueError(f"faster-whisper expects 16kHz audio, got {sample_rate}")

        segments, info = self.model.transcribe(
            audio,
            language=None,
            vad_filter=False,  # we already do our own VAD upstream
            beam_size=5,
            # Non-speech suppression: without these, large-v3-turbo hallucinates
            # caption phrases on near-silence (see _HALLUCINATION_PHRASES above).
            temperature=0.0,                  # deterministic; less prone to invention
            condition_on_previous_text=False, # don't let a hallucination seed the next clip
            no_speech_threshold=0.6,          # drop segments Whisper itself flags as non-speech
            log_prob_threshold=-1.0,          # drop low-confidence (garbage) segments
            compression_ratio_threshold=2.4,  # drop repetitive/degenerate output
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        language = info.language or config.DEFAULT_LANGUAGE

        # Belt-and-suspenders: drop the exact known caption hallucinations, but only
        # for short clips so a real sentence is never discarded.
        duration_s = len(audio) / float(sample_rate)
        if (
            duration_s < _HALLUCINATION_MAX_CLIP_S
            and text.strip().lower() in _HALLUCINATION_PHRASES
        ):
            logger.debug("Dropped hallucination phrase %r (%.2fs clip)", text, duration_s)
            return "", language

        return text, language
