"""Bilingual text-to-speech: Piper for both languages (single consistent voice
character across pt-BR/en-US, both male voices from the same engine).

Originally used Kokoro for English + Piper for Portuguese, but that meant
switching between two very different-sounding engines depending on detected
language - jarring in a live conversation. Piper has good voices for both
languages, so it's used exclusively now.
"""

import logging
import re
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from piper import PiperVoice
from piper.config import SynthesisConfig

from reachy_mini_brain import config

# Piper's default noise_scale/noise_w_scale inject random prosody/rhythm
# variation per call - confirmed live: identical text produced audibly
# different-length output (33792 vs 32768 samples) across repeated calls,
# which read as "the voice keeps changing" rather than natural variety.
# Zeroing both makes synthesis fully deterministic (confirmed: identical
# output length across repeated calls) for a consistent voice character.
_DETERMINISTIC_SYN_CONFIG = SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0)

logger = logging.getLogger(__name__)

VOICES_DIR = Path(__file__).parent / "voices"

# Emoji desync TTS phonemizers/alignment in general (observed live with Kokoro:
# "words count mismatch" warning + garbled phonemes for the whole sentence,
# not just the emoji itself), so strip them before synthesis regardless of
# engine.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002700-\U000027BF"
    "]+",
    flags=re.UNICODE,
)


def _strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub("", text).strip()


# Symbols Piper mispronounces or drops - expand to spoken words per language,
# BEFORE synthesis, so it never matters what the LLM emitted. Order matters:
# multi-char units (°C) before the bare degree sign.
_SYMBOL_WORDS = {
    "pt": [
        ("°C", " graus Celsius"), ("ºC", " graus Celsius"),
        ("°F", " graus Fahrenheit"), ("ºF", " graus Fahrenheit"),
        ("°", " graus"), ("º", " graus"),
        ("%", " por cento"),
        ("km/h", " quilômetros por hora"), ("km", " quilômetros"),
        ("&", " e "), ("@", " arroba "), ("#", " número "),
        ("R$", " reais"), ("$", " dólares"), ("+", " mais "), ("=", " igual "),
    ],
    "en": [
        ("°C", " degrees Celsius"), ("ºC", " degrees Celsius"),
        ("°F", " degrees Fahrenheit"), ("ºF", " degrees Fahrenheit"),
        ("°", " degrees"), ("º", " degrees"),
        ("%", " percent"),
        ("km/h", " kilometers per hour"), ("km", " kilometers"),
        ("&", " and "), ("@", " at "), ("#", " number "),
        ("$", " dollars"), ("+", " plus "), ("=", " equals "),
    ],
}


def _expand_symbols(text: str, language: str) -> str:
    for sym, word in _SYMBOL_WORDS.get(language, _SYMBOL_WORDS["pt"]):
        text = text.replace(sym, word)
    # Collapse any doubled spaces the replacements introduced.
    return re.sub(r"\s{2,}", " ", text).strip()


class TTSEngine:
    """One Piper voice per language, same engine for a consistent character."""

    def __init__(self) -> None:
        self._voices: dict[str, PiperVoice] = {}
        for lang, cfg in config.TTS_VOICES.items():
            model_path = VOICES_DIR / cfg["piper_model"]
            logger.info("Loading Piper voice for '%s' (%s)...", lang, model_path.name)
            self._voices[lang] = PiperVoice.load(str(model_path))
        logger.info("TTS voices loaded.")

    def _resolve_language(self, language: str) -> str:
        return language if language in self._voices else config.DEFAULT_LANGUAGE

    def synthesize(self, text: str, language: str) -> tuple[np.ndarray, int]:
        """Synthesize `text`, returning (mono float32 audio, its native sample rate)."""
        text = _strip_emoji(text)
        if not text:
            return np.zeros(0, dtype=np.float32), 22050

        lang = self._resolve_language(language)
        text = _expand_symbols(text, lang)
        voice = self._voices[lang]

        chunks = list(voice.synthesize(text, syn_config=_DETERMINISTIC_SYN_CONFIG))
        if not chunks:
            return np.zeros(0, dtype=np.float32), 22050
        rate = chunks[0].sample_rate
        audio_int16 = np.concatenate(
            [np.frombuffer(c.audio_int16_bytes, dtype=np.int16) for c in chunks]
        )
        audio = audio_int16.astype(np.float32) / 32768.0
        return audio, rate

    def speak(self, reachy_mini, text: str, language: str, interrupt=None) -> bool:
        """Synthesize and play `text` through the robot's speaker.

        Blocks until playback finishes. If `interrupt` (a threading.Event) is
        provided and gets set mid-playback, the audio is flushed immediately
        and this returns False (interrupted); otherwise returns True.
        """
        if not text.strip():
            return True

        _t0 = time.time()
        audio, native_rate = self.synthesize(text, language)
        if audio.size == 0:
            return True
        logger.info(
            "[timing] TTS synth %.2fs for %.1fs of speech",
            time.time() - _t0, len(audio) / max(native_rate, 1),
        )

        output_rate = reachy_mini.media.get_output_audio_samplerate()
        if output_rate > 0 and output_rate != native_rate:
            audio_t = torch.from_numpy(audio).unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, native_rate, output_rate)
            audio = audio_t.squeeze(0).numpy()
            rate = output_rate
        else:
            rate = native_rate

        # NOTE: start_recording()/start_playing()/stop_recording()/stop_playing()
        # all act on ONE shared GStreamer pipeline (capture + playback combined,
        # see reachy_mini/media/audio_gstreamer.py). Calling stop_playing() here
        # would set that shared pipeline to NULL and kill the concurrently
        # running microphone capture too - confirmed live on hardware. So we
        # only ever call start_playing() (idempotent, resets pts) and never
        # stop_playing(); the pipeline's lifecycle is owned by
        # start_recording()/stop_recording() in audio_io.py instead.
        reachy_mini.media.start_playing()
        reachy_mini.media.push_audio_sample(audio)

        # Block for the clip duration, but in small steps so a barge-in can
        # cut playback short. clear_player() flushes the queued audio at the
        # GStreamer level (it does NOT tear down the shared pipeline, unlike
        # stop_playing()), so the mic keeps working.
        total = len(audio) / rate
        elapsed = 0.0
        step = 0.05
        while elapsed < total:
            if interrupt is not None and interrupt.is_set():
                try:
                    reachy_mini.media.audio.clear_player()
                except Exception:
                    logger.exception("clear_player during barge-in failed")
                logger.info("[timing] speech interrupted after %.2fs of %.2fs", elapsed, total)
                return False
            time.sleep(step)
            elapsed += step
        return True
