"""Spanish/Rioplatense acoustic analysis primitives.

Provides vowel classification, vowel purity scoring, and consonant
feature analysis for Spanish pronunciation assessment. Designed to
work with either MFA phoneme boundaries or Whisper word timestamps.
"""

import json
from pathlib import Path

import numpy as np

_SPANISH_NORMS_PATH = Path(__file__).parent / "data" / "spanish_vowel_norms.json"
_SPANISH_NORMS = None  # Lazy loaded


def get_spanish_vowel_norms() -> dict:
    """Lazy-load Spanish vowel norms."""
    global _SPANISH_NORMS
    if _SPANISH_NORMS is None:
        with open(_SPANISH_NORMS_PATH) as f:
            _SPANISH_NORMS = json.load(f)["vowels"]
    return _SPANISH_NORMS


def classify_vowel_spanish(f1: float, f2: float) -> str | None:
    """Classify a formant frame into the nearest Spanish vowel.

    Uses Euclidean distance in F1xF2 space against Spanish 5-vowel norms.
    Returns vowel key ("A", "E", "I", "O", "U") or None if too far.
    """
    norms = get_spanish_vowel_norms()
    F1_SCALE = 500.0
    F2_SCALE = 1000.0

    best_vowel = None
    best_dist = float("inf")

    for vowel, ref in norms.items():
        d = ((f1 - ref["f1_mean"]) / F1_SCALE) ** 2 + \
            ((f2 - ref["f2_mean"]) / F2_SCALE) ** 2
        if d < best_dist:
            best_dist = d
            best_vowel = vowel

    if best_dist > 2.25:
        return None
    return best_vowel
