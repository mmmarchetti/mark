"""Lightweight pt-BR vs en-US detection for picking the TTS voice.

We must pick the voice from the language of the TEXT WE ARE ABOUT TO SPEAK,
not from the language STT detected for the user's utterance. Confirmed live:
STT heard "Hey Richie!" as English, the LLM replied in Portuguese anyway, and
the Portuguese text got read by the English voice model - which is exactly the
"the voice keeps changing between replies" symptom.

Only two languages matter here, so a word/character heuristic is enough and
costs nothing at runtime (no extra model, no latency).
"""

import re

# Characters that essentially only appear in Portuguese (of our two languages).
_PT_CHARS = set("ãõçáéíóúâêôàÃÕÇÁÉÍÓÚÂÊÔÀ")

# High-frequency function words, chosen to be unambiguous between the two.
_PT_WORDS = {
    "e", "o", "a", "os", "as", "um", "uma", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "para", "pra", "por", "com", "sem",
    "que", "não", "sim", "eu", "você", "voce", "ele", "ela", "nós", "nos",
    "meu", "minha", "seu", "sua", "isso", "isto", "aqui", "ali", "agora",
    "muito", "mais", "menos", "bem", "bom", "boa", "vou", "vamos", "está",
    "esta", "estou", "tem", "ter", "fazer", "ver", "olhar", "quer", "quero",
    "obrigado", "obrigada", "oi", "olá", "ola", "tchau", "noite", "dia",
}
_EN_WORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at",
    "for", "with", "without", "is", "are", "was", "were", "be", "been",
    "i", "you", "he", "she", "it", "we", "they", "my", "your", "his", "her",
    "this", "that", "these", "those", "here", "there", "now", "very", "more",
    "less", "well", "good", "hi", "hello", "bye", "night", "day", "let",
    "me", "can", "will", "would", "should", "do", "does", "did", "have",
    "has", "had", "look", "see", "want", "going", "gonna", "yeah", "okay",
}

_WORD_RE = re.compile(r"[a-zà-ÿ]+", re.IGNORECASE)


def detect_language(text: str, default: str = "pt") -> str:
    """Return "pt" or "en" for `text`, falling back to `default` when unclear."""
    if not text or not text.strip():
        return default

    # Portuguese-only diacritics are a very strong signal on their own.
    if any(ch in _PT_CHARS for ch in text):
        return "pt"

    words = [w.lower() for w in _WORD_RE.findall(text)]
    if not words:
        return default

    pt_hits = sum(1 for w in words if w in _PT_WORDS)
    en_hits = sum(1 for w in words if w in _EN_WORDS)

    if pt_hits > en_hits:
        return "pt"
    if en_hits > pt_hits:
        return "en"
    return default
