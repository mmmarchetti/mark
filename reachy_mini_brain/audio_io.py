"""Continuous mic listening: streaming VAD segmentation + wake-word fuzzy match.

No dedicated wake-word model: STT stays warm and runs on every detected
utterance; we fuzzy-match the transcript against "reachy" before treating it
as a real turn, unless a conversation is already open (per plan.md decision).
"""

import logging
import queue
import time
from difflib import SequenceMatcher

import numpy as np
import torch
import torchaudio

from reachy_mini_brain import config

logger = logging.getLogger(__name__)

CHUNK_SAMPLES = 512  # silero-vad's expected frame size at 16kHz


def _fuzzy_contains_wake_word(text: str) -> bool:
    text_lower = text.lower()
    tokens = [t.strip(".,!?;:—–\"'") for t in text_lower.split()]
    tokens = [t for t in tokens if t]
    token_set = set(tokens)

    for word in config.WAKE_WORDS:
        if " " in word:
            # Multi-word phrase (e.g. "hey mark"): substring is specific enough.
            if word in text_lower:
                return True
            continue
        # Single word: WHOLE-WORD match, never substring - otherwise a short
        # common name like "mark" would fire on "market", "Denmark", "remark".
        if word in token_set:
            return True
        # Fuzzy only for longer wake words / tokens, to catch STT mishearings
        # without the short-word false positives.
        if len(word) >= config.WAKE_FUZZY_MIN_TOKEN_LEN:
            for token in tokens:
                if len(token) < config.WAKE_FUZZY_MIN_TOKEN_LEN:
                    continue
                if SequenceMatcher(None, token, word).ratio() >= config.WAKE_FUZZY_THRESHOLD:
                    return True
    return False


class AudioListener:
    """Continuously listens to the mic and emits (text, language) turns."""

    def __init__(self, stt_engine) -> None:
        from silero_vad import load_silero_vad

        self.stt = stt_engine
        self.vad_model = load_silero_vad()
        # Custom neural "Hey Mark" wake-word (primary trigger); no-ops if the
        # trained model file is absent, leaving the fuzzy STT match as fallback.
        from reachy_mini_brain.wakeword import WakeWord
        self.wakeword = WakeWord()
        self.utterance_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._conversation_open_until = 0.0
        self.muted = False  # set True to fully ignore mic (e.g. during a scan)
        # Deaf mode: a PERSISTENT "ignore the mic" toggle for meetings, separate
        # from `muted` (which main.py flips on/off around every spoken reply).
        # While deaf, Mark keeps his eyes/pose/awake state but processes NO audio,
        # so he won't butt into a meeting - without going to sleep. Cleared only
        # by the user (the brain-panel "Deaf" button / stop_listening path).
        self.deaf = False
        # Barge-in: when the robot is speaking we DON'T mute (the hardware AEC
        # cancels its own voice well - measured ~1.1x baseline). Instead we
        # listen for a genuine interruption and fire on_barge_in so the caller
        # can stop playback. A run of sustained speech is required to avoid
        # tripping on residual echo.
        self.speaking = False
        self.on_barge_in = None  # optional callable, set by main
        self.last_utterance = None  # np.ndarray of the last captured utterance (16kHz)
        self.last_stt_latency = None  # seconds; last transcribe() wall time (dashboard)

        # --- Runtime-adjustable mic sensitivity (hardware capture gain is
        # already maxed, so these are the levers for hearing quieter speech).
        # All settable live from the web panel; see set_mic_from_ui().
        self.mic_gain = config.MIC_GAIN            # software amplify on the input
        self.vad_threshold = config.VAD_THRESHOLD  # lower = detects quieter speech
        self.noise_gate = config.MIC_NOISE_GATE    # RMS floor: below it = treat as silence

    def set_mic_from_ui(self, sensitivity_pct: int | None = None,
                        noise_pct: int | None = None) -> None:
        """Map two intuitive 0-100 sliders to the internal mic parameters.

        sensitivity: higher = hears quieter speech (more gain, lower VAD thresh).
        noise (reduction): higher = more aggressive silence gate.
        """
        if sensitivity_pct is not None:
            s = max(0, min(100, sensitivity_pct)) / 100.0
            self.mic_gain = 1.0 + s * (config.MIC_GAIN_MAX - 1.0)
            # More sensitive => LOWER VAD threshold.
            self.vad_threshold = config.VAD_THRESHOLD_MAX - s * (
                config.VAD_THRESHOLD_MAX - config.VAD_THRESHOLD_MIN
            )
        if noise_pct is not None:
            n = max(0, min(100, noise_pct)) / 100.0
            self.noise_gate = n * config.MIC_NOISE_GATE_MAX

    def mic_sensitivity_pct(self) -> int:
        # Invert the mapping from vad_threshold for display.
        span = config.VAD_THRESHOLD_MAX - config.VAD_THRESHOLD_MIN
        s = (config.VAD_THRESHOLD_MAX - self.vad_threshold) / span if span else 0.5
        return int(round(max(0.0, min(1.0, s)) * 100))

    def mic_noise_pct(self) -> int:
        m = config.MIC_NOISE_GATE_MAX
        return int(round((self.noise_gate / m if m else 0.0) * 100))

    def flush_pending(self) -> int:
        """Discard queued utterances (used after the robot finishes speaking).

        Anything captured while the robot was talking is almost always its own
        voice leaking back in or noise, and acting on it made replies arrive
        for input from several seconds earlier.
        """
        dropped = 0
        while True:
            try:
                self.utterance_queue.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        return dropped

    def close_conversation(self) -> None:
        """Force the wake word to be required again immediately (stop_listening
        or go_to_sleep tools), rather than waiting out CONVERSATION_TIMEOUT_S.
        """
        self._conversation_open_until = 0.0

    def note_activity(self) -> None:
        """Open the conversation window from outside (e.g. a person appeared),
        so the user can reply to a spontaneous greeting without a wake word.
        """
        self._conversation_open_until = time.time() + config.CONVERSATION_TIMEOUT_S

    @staticmethod
    def _resample_to_16k(audio: np.ndarray, source_rate: int) -> np.ndarray:
        if source_rate == config.VAD_SAMPLE_RATE:
            return audio
        audio_t = torch.from_numpy(audio).float().unsqueeze(0)
        resampled = torchaudio.functional.resample(audio_t, source_rate, config.VAD_SAMPLE_RATE)
        return resampled.squeeze(0).numpy()

    def _vad_prob(self, chunk: np.ndarray) -> float:
        """Speech probability for one VAD frame.

        torch.no_grad() is ESSENTIAL, not an optimization: silero-vad's JIT
        model is STATEFUL (it retains LSTM _state/_context between calls). With
        autograd tracking on, each call's graph chains onto that retained state
        and is never freed - the process grows ~23 KB per call (~2.5 GB/hour at
        this loop's ~31 calls/s), which previously leaked to ~27 GB and got the
        app OOM-killed. This is inference only; grad is never needed here.
        """
        with torch.no_grad():
            return float(self.vad_model(torch.from_numpy(chunk), config.VAD_SAMPLE_RATE).item())

    def run(self, reachy_mini, stop_event) -> None:
        reachy_mini.media.start_recording()
        input_rate = reachy_mini.media.get_input_audio_samplerate()
        logger.info("Listening (input samplerate=%d)...", input_rate)

        pending = np.zeros(0, dtype=np.float32)
        speech_buffer: list[np.ndarray] = []
        in_speech = False
        silence_run_ms = 0.0
        barge_run_ms = 0.0

        samples_seen = 0
        max_amp_since_log = 0.0
        last_heartbeat = time.time()
        none_count = 0

        try:
            while not stop_event.is_set():
                sample = reachy_mini.media.get_audio_sample()
                if sample is None:
                    none_count += 1
                    if time.time() - last_heartbeat > 5.0:
                        logger.info(
                            "Heartbeat: %d samples processed, %d empty polls, max amplitude seen: %.4f",
                            samples_seen, none_count, max_amp_since_log,
                        )
                        last_heartbeat = time.time()
                        max_amp_since_log = 0.0
                    time.sleep(0.01)
                    continue

                samples_seen += sample.size
                max_amp_since_log = max(max_amp_since_log, float(np.abs(sample).max()) if sample.size else 0.0)
                if time.time() - last_heartbeat > 5.0:
                    logger.info(
                        "Heartbeat: %d samples processed, %d empty polls, max amplitude seen: %.4f",
                        samples_seen, none_count, max_amp_since_log,
                    )
                    last_heartbeat = time.time()
                    max_amp_since_log = 0.0

                if self.muted or self.deaf:
                    # Hard mute: fully ignore mic. `muted` is the brief per-reply
                    # tail-mute / body-scan gate; `deaf` is the persistent
                    # meeting toggle. Either one drops all audio and resets the
                    # in-flight VAD state so nothing is transcribed.
                    pending = np.zeros(0, dtype=np.float32)
                    speech_buffer = []
                    in_speech = False
                    silence_run_ms = 0.0
                    barge_run_ms = 0.0
                    continue

                if sample.ndim > 1:
                    sample = sample.mean(axis=1)
                sample = self._resample_to_16k(sample.astype(np.float32), input_rate)
                # Software input gain (hardware capture is already maxed) so
                # quieter speech reaches VAD/STT at a usable level. Clip to
                # avoid overflow/distortion.
                if self.mic_gain != 1.0:
                    sample = np.clip(sample * self.mic_gain, -1.0, 1.0)
                pending = np.concatenate([pending, sample])

                while len(pending) >= CHUNK_SAMPLES:
                    chunk = pending[:CHUNK_SAMPLES]
                    pending = pending[CHUNK_SAMPLES:]

                    chunk_ms = 1000.0 * CHUNK_SAMPLES / config.VAD_SAMPLE_RATE
                    # Noise gate: chunks quieter than the gate are treated as
                    # silence outright (suppresses low-level ambient noise, and
                    # noise the input gain would otherwise amplify into VAD).
                    if self.noise_gate > 0.0:
                        rms = float(np.sqrt(np.mean(chunk * chunk)))
                        if rms < self.noise_gate:
                            prob = 0.0
                        else:
                            prob = self._vad_prob(chunk)
                    else:
                        prob = self._vad_prob(chunk)

                    # While the robot is speaking, watch for a sustained
                    # interruption. A higher threshold + a minimum duration
                    # keeps residual echo / brief noise from cutting it off.
                    if self.speaking:
                        strong = prob >= config.BARGE_IN_VAD_THRESHOLD
                        barge_run_ms = barge_run_ms + chunk_ms if strong else 0.0
                        if barge_run_ms >= config.BARGE_IN_MIN_SPEECH_MS:
                            logger.info("Barge-in detected (prob=%.2f) - interrupting speech.", prob)
                            barge_run_ms = 0.0
                            if self.on_barge_in is not None:
                                try:
                                    self.on_barge_in()
                                except Exception:
                                    logger.exception("on_barge_in callback failed")
                        # Don't also accumulate this as a normal utterance
                        # while speaking; the post-speech capture handles the
                        # actual interrupting words.
                        continue

                    if prob >= self.vad_threshold:
                        if not in_speech:
                            logger.info("Speech detected (VAD prob=%.2f)", prob)
                        in_speech = True
                        silence_run_ms = 0.0
                        speech_buffer.append(chunk)
                    elif in_speech:
                        silence_run_ms += chunk_ms
                        speech_buffer.append(chunk)
                        if silence_run_ms >= config.VAD_MIN_SILENCE_MS:
                            utterance = np.concatenate(speech_buffer)
                            speech_buffer = []
                            in_speech = False
                            silence_run_ms = 0.0
                            self._handle_utterance(utterance)
        finally:
            reachy_mini.media.stop_recording()

    def _handle_utterance(self, utterance: np.ndarray) -> None:
        captured_at = time.time()  # end of speech, i.e. the user stopped talking
        duration_s = len(utterance) / config.VAD_SAMPLE_RATE
        # Keep the last real utterance's audio (16kHz) so a "save this" memo tool
        # can persist it without contending for the shared mic stream.
        self.last_utterance = utterance
        if len(utterance) < config.VAD_SAMPLE_RATE * 0.3:
            logger.info("Utterance too short (%.2fs), skipping STT", duration_s)
            return  # too short to be real speech

        t0 = time.time()
        text, language = self.stt.transcribe(utterance, config.VAD_SAMPLE_RATE)
        self.last_stt_latency = time.time() - t0  # read by the metrics dashboard
        logger.info(
            "[timing] STT %.2fs for %.2fs audio -> language=%s text=%r",
            self.last_stt_latency, duration_s, language, text,
        )
        if not text:
            return

        now = time.time()
        conversation_open = now < self._conversation_open_until
        if not conversation_open:
            # Primary trigger: the custom neural wake-word on the audio. Fall
            # back to the fuzzy STT-text match if the model isn't available or
            # is unsure (so we never regress below the old behavior).
            woke = False
            if self.wakeword.available:
                ww = self.wakeword.score(utterance)
                woke = ww >= self.wakeword._threshold
                logger.info("Wake-word score=%.2f -> %s", ww, "WAKE" if woke else "no")
            if not woke and not _fuzzy_contains_wake_word(text):
                logger.info("Heard but ignored (no wake word, no open conversation): %r", text)
                return

        self._conversation_open_until = now + config.CONVERSATION_TIMEOUT_S
        # Carry the capture time so the brain loop can drop utterances that
        # went stale while the robot was busy speaking or running a tool -
        # replying to input from 5+ seconds ago is what made the conversation
        # feel out of sync.
        self.utterance_queue.put((text, language, captured_at))
