"""Synthesized breathing ambiance played on a loop while the robot sleeps.

Design notes (learned the hard way on real hardware):

* The first version used brown noise at volume=0.06. Measured peak amplitude
  came out at 0.0145 (~-37 dB) and brown noise puts nearly all of its energy
  below ~100 Hz - which the Reachy Mini's small speaker cannot reproduce at
  all. Net result: completely inaudible.
* So: band-pass the noise into the 300-1800 Hz range (where breath noise
  actually lives AND where a small speaker is efficient), and normalise the
  peak AFTER the envelope is applied so the final level is predictable.
"""

import numpy as np
import soundfile as sf
from scipy import signal

from reachy_mini_brain import config

# Breath noise energy band. Low enough to sound like breath rather than hiss,
# high enough that a small speaker can actually reproduce it.
_BAND_LOW_HZ = 300.0
_BAND_HIGH_HZ = 1800.0


def generate_breath_cycle(
    sample_rate: int, duration: float = 4.0, peak: float = 0.18
) -> np.ndarray:
    """One inhale/exhale cycle of soft breathing noise.

    Envelope: ~40% rising (inhale), ~45% falling (exhale), ~15% silent pause,
    i.e. calm slow breathing (~15 breaths/min at duration=4.0).

    `peak` is the final peak amplitude after enveloping and normalisation.
    """
    n = int(sample_rate * duration)
    t = np.linspace(0.0, 1.0, n, endpoint=False)

    white = np.random.default_rng().normal(0.0, 1.0, n)

    # Band-pass into the breath/speaker-friendly range.
    nyquist = sample_rate / 2.0
    low = max(_BAND_LOW_HZ / nyquist, 1e-4)
    high = min(_BAND_HIGH_HZ / nyquist, 0.99)
    sos = signal.butter(2, [low, high], btype="bandpass", output="sos")
    noise = signal.sosfilt(sos, white)

    inhale_end = 0.40
    exhale_end = 0.85
    envelope = np.zeros(n)

    inhale_mask = t < inhale_end
    envelope[inhale_mask] = t[inhale_mask] / inhale_end

    exhale_mask = (t >= inhale_end) & (t < exhale_end)
    exhale_progress = (t[exhale_mask] - inhale_end) / (exhale_end - inhale_end)
    envelope[exhale_mask] = 1.0 - exhale_progress

    # Soften the attack/decay so it swells rather than clicking in.
    envelope = envelope**1.5

    audio = noise * envelope
    max_abs = float(np.abs(audio).max())
    if max_abs > 0:
        # Normalise AFTER enveloping - the noise peak and the envelope peak do
        # not coincide, which is exactly how the first version silently ended
        # up ~12x quieter than intended.
        audio = audio / max_abs * peak
    return audio.astype(np.float32)


def write_breath_wav(
    path: str,
    sample_rate: int,
    duration: float = 4.0,
    peak: float | None = None,
    cycles: int = 1,
) -> float:
    """Write `cycles` back-to-back breath cycles to `path`. Returns total seconds.

    Writing many cycles into ONE file matters: re-triggering playback every
    4 s proved unreliable inside the app (the sleep breathing was heard once
    and then stopped), because each `play_sound` builds a fresh playbin and
    audio-sink bin while the shared TTS pipeline still holds the device. One
    long file means we only re-trigger every couple of minutes instead.

    Each cycle regenerates its own noise, so the loop doesn't sound obviously
    repetitive.
    """
    if peak is None:
        peak = config.BREATH_PEAK
    audio = np.concatenate(
        [generate_breath_cycle(sample_rate, duration=duration, peak=peak) for _ in range(max(1, cycles))]
    )
    sf.write(path, audio, sample_rate)
    return len(audio) / sample_rate
