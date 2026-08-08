"""Train a small custom "Hey Mark" wake-word classifier on openWakeWord's
audio embeddings, using Piper-synthesized speech (no giant external datasets).

Key realism points learned the hard way:
- Real utterances are "Hey Mark <command>" in ONE VAD segment, so positives
  must include wake+command phrases, not just the wake word in silence.
- Confusables ("market", "Marcos", "mercado") must appear as negatives both
  alone AND inside sentences, or sliding-max scoring fires on them.
- The runtime scores an utterance by sliding a 16-frame window over its
  embeddings and taking the MAX, so the decision threshold is selected the
  SAME way on held-out full utterances (max-pooling inflates single-window
  scores; the threshold must account for that).

Output: models/hey_mark.joblib  ->  {"model", "threshold", "win"}
Run:  python -m scripts.train_wakeword
"""

import sys
import numpy as np
import torch
import torchaudio
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reachy_mini_brain.tts import TTSEngine
from openwakeword.utils import AudioFeatures

SR = 16000
WIN = 2 * SR            # 2.0s window -> 16 embedding frames
FRAMES = 16
OUT = Path(__file__).resolve().parents[1] / "reachy_mini_brain" / "models" / "hey_mark.joblib"

# Bare wake phrases (spoken in isolation).
POS_WAKE = [
    "Hey Mark.", "Hey Mark!", "Hey, Mark.", "Mark.", "Mark!", "Mark?",
    "Ok Mark.", "Okay Mark.", "Hey Marc.", "Oi Mark.", "Ei Mark.", "Ô Mark.",
]
# Wake word followed by a command (the common real case, one VAD segment).
POS_CMD = [
    "Hey Mark what's the weather", "Mark what time is it", "Hey Mark tell me a joke",
    "Mark set a timer for five minutes", "Hey Mark turn on the light",
    "Mark are you there", "Hey Mark play some music", "Mark how are you",
    "Oi Mark que horas são", "Ei Mark conta uma piada", "Mark bom dia",
    "Oi Mark tudo bem", "Hey Mark what's on my calendar", "Mark look at me",
]
# Confusables + common speech that must NOT trigger, alone and in sentences.
NEG = [
    "market", "the market", "the stock market fell", "I went to the market",
    "Marcos", "I saw Marcos yesterday", "tell Marcos hello", "remark", "remarkable",
    "dark", "it's getting dark", "park", "the car park", "let's go to the park",
    "start", "let's start now", "smart", "a smart idea", "bark", "the dog will bark",
    "March", "marker", "a red marker", "margin", "Marco Polo", "embark", "denmark",
    "what time is it", "turn on the light", "how are you", "tell me a joke",
    "good morning", "thank you very much", "play some music", "what's the weather",
    "mercado", "vou ao mercado", "marca", "marco", "carro", "no parque",
    "vamos começar", "bom dia", "obrigado", "que horas são", "conta uma piada",
    "liga a luz", "tudo bem", "the quick brown fox jumps", "one two three four five",
    "let me think about it", "can you help me with this",
]


def resample_16k(a, sr):
    if sr == SR:
        return a.astype(np.float32)
    t = torch.from_numpy(a.astype(np.float32)).unsqueeze(0)
    return torchaudio.functional.resample(t, sr, SR).squeeze(0).numpy()


def to_int16(x):
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def window_from(clip, rng, mode):
    """A 2s window containing (part of) the clip. mode 'rand' places a short
    clip anywhere; 'head' keeps the first 2s (wake word + start of command)."""
    w = np.zeros(WIN, dtype=np.float32)
    if mode == "head":
        seg = clip[:WIN]
        w[:len(seg)] = seg
    else:
        c = clip[-WIN:]
        end = int(rng.integers(len(c), WIN + 1))
        w[end - len(c):end] = c
    return w


def augment(w, rng):
    y = w * rng.uniform(0.6, 1.15)
    snr = rng.choice([0.0, 0.0, 0.003, 0.008, 0.015])
    if snr:
        y = y + rng.normal(0, snr, size=y.shape).astype(np.float32)
    return np.clip(y, -1, 1)


def feats_seq(af, win_int16):
    emb = np.asarray(af._get_embeddings(win_int16))
    if emb.shape[0] < FRAMES:
        emb = np.vstack([np.repeat(emb[:1], FRAMES - emb.shape[0], 0), emb])
    return emb


def one_window_feat(af, w):
    return feats_seq(af, to_int16(w))[-FRAMES:].reshape(-1)


def sliding_max_proba(af, clf, audio):
    """Score a full utterance the way the runtime does."""
    if audio.shape[0] < WIN:
        audio = np.concatenate([np.zeros(WIN - audio.shape[0], np.float32), audio])
    emb = feats_seq(af, to_int16(audio))
    if emb.shape[0] < FRAMES:
        return 0.0
    W = np.stack([emb[i:i + FRAMES].reshape(-1) for i in range(emb.shape[0] - FRAMES + 1)])
    return float(clf.predict_proba(W)[:, 1].max())


def main():
    rng = np.random.default_rng(0)
    print("Loading Piper + openWakeWord front-end...")
    tts = TTSEngine()
    af = AudioFeatures(ncpu=2)

    X, y = [], []
    clips_pos, clips_neg = [], []  # kept for threshold selection

    def add(phrases, label, modes, n_aug):
        for lang in ("en", "pt"):
            for ph in phrases:
                a, sr = tts.synthesize(ph, lang)
                if a.size == 0:
                    continue
                clip = resample_16k(a, sr)
                (clips_pos if label else clips_neg).append(clip)
                for _ in range(n_aug):
                    mode = modes[rng.integers(len(modes))]
                    X.append(one_window_feat(af, augment(window_from(clip, rng, mode), rng)))
                    y.append(label)

    print("Positives (wake) ...");    add(POS_WAKE, 1, ["rand"], 6)
    print("Positives (wake+cmd) ..."); add(POS_CMD, 1, ["head"], 6)
    print("Negatives ...");           add(NEG, 0, ["rand", "head"], 3)
    # Noise / silence negatives.
    for _ in range(150):
        k = rng.integers(0, 3)
        w = np.zeros(WIN, np.float32) if k == 0 else rng.normal(0, rng.uniform(0.005, 0.05), WIN).astype(np.float32)
        X.append(one_window_feat(af, w)); y.append(0)
        clips_neg.append(w)

    X = np.array(X); y = np.array(y)
    print(f"train windows: {int((y==1).sum())} pos / {int((y==0).sum())} neg")
    clf = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(X, y)

    # Threshold selection via sliding-max on the FULL held-out clips.
    ps = np.array([sliding_max_proba(af, clf, c) for c in clips_pos])
    ns = np.array([sliding_max_proba(af, clf, c) for c in clips_neg])
    prec, rec, thr = precision_recall_curve(
        np.r_[np.ones(len(ps)), np.zeros(len(ns))], np.r_[ps, ns])
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best = thr[max(0, np.argmax(f1) - 1)]
    # Bias slightly toward precision (avoid false wakes) but keep recall usable.
    threshold = float(min(0.85, max(0.5, best)))
    print(f"pos sliding-max: min={ps.min():.2f} mean={ps.mean():.2f}")
    print(f"neg sliding-max: max={ns.max():.2f} mean={ns.mean():.2f}")
    print(f"chosen threshold={threshold:.2f}  "
          f"(recall@thr={ (ps>=threshold).mean():.2f}, "
          f"false-fire={ (ns>=threshold).mean():.2f})")

    def score(t, lang="en"):
        a, sr = tts.synthesize(t, lang)
        return sliding_max_proba(af, clf, resample_16k(a, sr))
    print("Sanity (want HIGH wake / LOW rest):")
    for t, lg in [("Hey Mark.", "en"), ("Mark what is the weather", "en"),
                  ("Ei Mark", "pt"), ("Oi Mark tudo bem", "pt"),
                  ("market", "en"), ("go to the market", "en"), ("Marcos", "en"),
                  ("what is the weather", "en"), ("bom dia", "pt"), ("vou ao mercado", "pt")]:
        print(f"  {t:26s} {score(t, lg):.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "threshold": threshold, "win": WIN}, OUT)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
