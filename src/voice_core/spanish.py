"""Spanish/Rioplatense acoustic analysis primitives.

Provides vowel classification, vowel purity scoring, and consonant
feature analysis for Spanish pronunciation assessment. Designed to
work with either MFA phoneme boundaries or Whisper word timestamps.
"""

import json
from pathlib import Path

import numpy as np

# Spanish MFA IPA labels → our 5-vowel categories
# MFA Spanish outputs IPA phone labels; we map vowels only
SPANISH_IPA_TO_VOWEL = {
    "a": "A", "a̠": "A",
    "e": "E", "e̞": "E",
    "i": "I",
    "o": "O", "o̞": "O",
    "u": "U",
}

# Consonant labels that are pronunciation targets for Rioplatense
# These are used to identify segments for consonant analysis
SPANISH_CONSONANT_TARGETS = {
    "ʃ": "sheismo",    # Rioplatense LL/Y (correct production)
    "ʒ": "sheismo",    # Voiced variant of sheísmo
    "j": "yeismo",     # Non-Rioplatense glide (what English speakers produce)
    "ʎ": "yeismo",     # Palatal lateral (another non-Rioplatense variant)
    "ɾ": "tap_r",      # Spanish tap-r (correct)
    "r": "trill_r",    # Spanish trill-r (correct)
    "s": "sibilant",   # For s-aspiration detection
    "h": "aspiration", # Aspirated /s/ in Rioplatense
}

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


def score_vowel_purity(f1_frames: np.ndarray, f2_frames: np.ndarray,
                       expected_vowel: str,
                       min_frames: int = 4) -> dict | None:
    """Score vowel purity (monophthong stability) for a vowel segment.

    Measures how stable F1/F2 stay throughout the vowel. Pure Spanish
    monophthongs have minimal F1/F2 drift; English-speaker diphthongization
    shows systematic F2 movement (e.g., /e/ -> [eI] has rising F2).

    Args:
        f1_frames: Per-frame F1 values (Hz) across the vowel segment.
        f2_frames: Per-frame F2 values (Hz) across the vowel segment.
        expected_vowel: Spanish vowel key ("A", "E", "I", "O", "U").
        min_frames: Minimum frames needed for reliable measurement.

    Returns:
        Dict with {purity: 0-1, diphthongized: bool, f1_drift_hz, f2_drift_hz}
        or None if too few frames.
    """
    f1_frames = np.asarray(f1_frames, dtype=float)
    f2_frames = np.asarray(f2_frames, dtype=float)

    if len(f1_frames) < min_frames or len(f2_frames) < min_frames:
        return None

    norms = get_spanish_vowel_norms()
    if expected_vowel not in norms:
        return None

    ref = norms[expected_vowel]

    # Measure drift: linear regression slope across the segment
    t = np.arange(len(f1_frames), dtype=float)
    f1_slope = np.polyfit(t, f1_frames, 1)[0] if len(f1_frames) > 1 else 0.0
    f2_slope = np.polyfit(t, f2_frames, 1)[0] if len(f2_frames) > 1 else 0.0

    # Total drift over the segment (Hz)
    n = len(f1_frames)
    f1_drift = abs(f1_slope * n)
    f2_drift = abs(f2_slope * n)

    # Normalize drift by the vowel's expected range (std)
    f1_norm_drift = f1_drift / ref["f1_std"] if ref["f1_std"] > 0 else 0.0
    f2_norm_drift = f2_drift / ref["f2_std"] if ref["f2_std"] > 0 else 0.0

    # Combined drift score (F2 drift is more perceptually salient for diphthongs)
    combined_drift = 0.3 * f1_norm_drift + 0.7 * f2_norm_drift

    # Purity score: 1.0 = perfectly stable, 0.0 = heavily diphthongized
    purity = max(0.0, min(1.0, 1.0 - combined_drift / 3.0))

    return {
        "purity": round(purity, 3),
        "diphthongized": bool(combined_drift > 1.0),
        "f1_drift_hz": round(f1_drift, 1),
        "f2_drift_hz": round(f2_drift, 1),
    }


def analyze_spanish_words(
    y: np.ndarray,
    sr: int,
    word_timestamps: list[dict],
    target_sounds: list[dict],
) -> dict:
    """Analyze Spanish pronunciation using Whisper word boundaries.

    Fast mode (~100-200ms) that uses word-level timestamps from Whisper
    to locate target sounds and run spectral analysis. No MFA required.

    Args:
        y: Full audio array (float32, mono).
        sr: Sample rate.
        word_timestamps: List of {"word": str, "start": float, "end": float}
            from Whisper's word_timestamps output.
        target_sounds: List of {"word": str, "feature": str} where feature
            is "sheismo", "tap_r", or "vowel_purity".

    Returns:
        Dict with:
            consonant_features: list of per-target classification results
            vowel_scores: list of per-vowel purity results
            summary: {sheismo_score, tap_r_score, vowel_purity_avg}
    """
    from voice_core.spanish_consonants import classify_sheismo, classify_tap_r

    # Build word→timestamp lookup
    ts_lookup: dict[str, dict] = {}
    for wt in word_timestamps:
        ts_lookup[wt["word"].lower().strip()] = wt

    consonant_features = []
    vowel_scores = []

    for target in target_sounds:
        word = target["word"].lower().strip()
        feature = target["feature"]
        ts = ts_lookup.get(word)

        if ts is None:
            consonant_features.append({
                "word": word, "feature": feature,
                "classification": "not_found", "confidence": 0.0,
            })
            continue

        # Extract audio segment
        start_sample = int(ts["start"] * sr)
        end_sample = int(ts["end"] * sr)
        segment = y[start_sample:end_sample]

        if len(segment) < int(0.03 * sr):
            consonant_features.append({
                "word": word, "feature": feature,
                "classification": "too_short", "confidence": 0.0,
            })
            continue

        if feature == "sheismo":
            result = classify_sheismo(segment, sr)
            result["word"] = word
            result["feature"] = feature
            consonant_features.append(result)

        elif feature == "tap_r":
            result = classify_tap_r(segment, sr)
            result["word"] = word
            result["feature"] = feature
            consonant_features.append(result)

        elif feature == "vowel_purity":
            vowel = target.get("vowel", "").upper()
            try:
                import parselmouth
                from parselmouth import praat
                snd = parselmouth.Sound(segment, sampling_frequency=sr)
                formant = praat.call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50.0)
                n_frames = praat.call(formant, "Get number of frames")
                f1_frames = []
                f2_frames = []
                for fi in range(1, n_frames + 1):
                    t = praat.call(formant, "Get time from frame number", fi)
                    f1 = praat.call(formant, "Get value at time", 1, t, "Hertz", "Linear")
                    f2 = praat.call(formant, "Get value at time", 2, t, "Hertz", "Linear")
                    if f1 > 0 and f2 > 0:
                        f1_frames.append(f1)
                        f2_frames.append(f2)

                if f1_frames and vowel:
                    purity_result = score_vowel_purity(
                        np.array(f1_frames), np.array(f2_frames), vowel)
                    if purity_result:
                        purity_result["word"] = word
                        purity_result["vowel"] = vowel
                        vowel_scores.append(purity_result)
            except Exception:
                pass

    # Summary
    sheismo_scores = [c["confidence"] for c in consonant_features
                      if c.get("feature") == "sheismo" and c.get("classification") == "sheismo"]
    tap_scores = [c["confidence"] for c in consonant_features
                  if c.get("feature") == "tap_r" and c.get("classification") == "tap"]
    purity_vals = [v["purity"] for v in vowel_scores]

    summary = {
        "sheismo_score": round(float(np.mean(sheismo_scores)), 3) if sheismo_scores else None,
        "tap_r_score": round(float(np.mean(tap_scores)), 3) if tap_scores else None,
        "vowel_purity_avg": round(float(np.mean(purity_vals)), 3) if purity_vals else None,
    }

    return {
        "consonant_features": consonant_features,
        "vowel_scores": vowel_scores,
        "summary": summary,
    }
