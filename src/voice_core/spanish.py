"""Spanish/Rioplatense acoustic analysis primitives.

Provides vowel classification, vowel purity scoring, and consonant
feature analysis for Spanish pronunciation assessment. Designed to
work with either MFA phoneme boundaries or Whisper word timestamps.
"""

import json
import subprocess
import tempfile
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
    from voice_core.analyze import _classify_vowel
    return _classify_vowel(f1, f2, norms=get_spanish_vowel_norms())


def score_vowel_purity(f1_frames: np.ndarray, f2_frames: np.ndarray,
                       expected_vowel: str,
                       min_frames: int = 4) -> dict | None:
    """Score vowel purity (stability + placement) for a vowel segment.

    Two components:
    1. **Stability** — how stable F1/F2 stay (detects diphthongization,
       e.g., /e/ → [eɪ] shows rising F2).
    2. **Placement** — how close the median F1/F2 are to the Spanish norm
       (detects wrong vowel quality, e.g., English /æ/ "apple" instead of
       Spanish /a/ "casa" — steady but in the wrong formant space).

    Args:
        f1_frames: Per-frame F1 values (Hz) across the vowel segment.
        f2_frames: Per-frame F2 values (Hz) across the vowel segment.
        expected_vowel: Spanish vowel key ("A", "E", "I", "O", "U").
        min_frames: Minimum frames needed for reliable measurement.

    Returns:
        Dict with {purity: 0-1, diphthongized: bool, misplaced: bool,
                    f1_drift_hz, f2_drift_hz, f1_offset_hz, f2_offset_hz,
                    stability: 0-1, placement: 0-1}
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

    # 1. Filter formant-tracking outliers using the segment's own median.
    #    Burg-algorithm trackers frequently jump between formant tracks,
    #    producing values like F1=80 or F2=3351 for an /O/ vowel. We reject
    #    frames far from the segment median rather than from citation norms,
    #    so contextual variants (post-lateral /a/ with high F2) aren't lost.
    f1_med = np.median(f1_frames)
    f2_med = np.median(f2_frames)
    f1_mad = max(np.median(np.abs(f1_frames - f1_med)), 30.0)  # floor 30 Hz
    f2_mad = max(np.median(np.abs(f2_frames - f2_med)), 50.0)  # floor 50 Hz
    mask = ((np.abs(f1_frames - f1_med) < 3.5 * f1_mad) &
            (np.abs(f2_frames - f2_med) < 3.5 * f2_mad))
    f1_clean = f1_frames[mask]
    f2_clean = f2_frames[mask]

    if len(f1_clean) < min_frames:
        return None

    # 2. Trim edges (~15% each side) to reduce coarticulation transitions
    n_full = len(f1_clean)
    trim = max(1, n_full // 6)
    f1_mid = f1_clean[trim:n_full - trim]
    f2_mid = f2_clean[trim:n_full - trim]
    if len(f1_mid) < 3:
        f1_mid = f1_clean
        f2_mid = f2_clean

    # 3. Stability: measure drift via linear regression slope
    t = np.arange(len(f1_mid), dtype=float)
    f1_slope = np.polyfit(t, f1_mid, 1)[0] if len(f1_mid) > 1 else 0.0
    f2_slope = np.polyfit(t, f2_mid, 1)[0] if len(f2_mid) > 1 else 0.0

    n = len(f1_mid)
    f1_drift = abs(f1_slope * n)
    f2_drift = abs(f2_slope * n)

    # Normalize drift by the vowel's expected range (std)
    f1_norm_drift = f1_drift / ref["f1_std"] if ref["f1_std"] > 0 else 0.0
    f2_norm_drift = f2_drift / ref["f2_std"] if ref["f2_std"] > 0 else 0.0

    # Combined drift score (F2 drift is more perceptually salient for diphthongs)
    combined_drift = 0.3 * f1_norm_drift + 0.7 * f2_norm_drift

    # Stability score: 1.0 = perfectly stable, 0.0 = heavily diphthongized
    stability = max(0.0, min(1.0, 1.0 - combined_drift / 4.0))

    # 4. Placement: how close is the median F1/F2 to the Spanish norm?
    #    Uses Mahalanobis-like distance normalized by each formant's std.
    #    English /æ/ "apple" vs Spanish /a/: F1 offset ~90Hz, F2 offset ~450Hz
    #    — the F2 difference alone is ~3 standard deviations.
    f1_mean = float(np.mean(f1_mid))
    f2_mean = float(np.mean(f2_mid))
    f1_offset = abs(f1_mean - ref["f1_mean"])
    f2_offset = abs(f2_mean - ref["f2_mean"])

    f1_norm_offset = f1_offset / ref["f1_std"] if ref["f1_std"] > 0 else 0.0
    f2_norm_offset = f2_offset / ref["f2_std"] if ref["f2_std"] > 0 else 0.0

    # Euclidean distance in normalized formant space
    norm_distance = (f1_norm_offset**2 + f2_norm_offset**2) ** 0.5

    # Placement score: 1.0 = on target, 0.0 = far off
    # Divisor 4.0: within ~1.5 std on each axis scores 0.85+,
    # 3+ std away (like /æ/ for /a/) scores ~0.3
    placement = max(0.0, min(1.0, 1.0 - norm_distance / 4.0))

    # 5. Combined purity = blend of stability and placement
    #    Both matter: a steady wrong vowel and an unstable right vowel
    #    are both problems. Weight equally.
    purity = 0.5 * stability + 0.5 * placement

    return {
        "purity": round(purity, 3),
        "diphthongized": bool(combined_drift > 0.8),
        "misplaced": bool(norm_distance > 2.0),
        "f1_drift_hz": round(f1_drift, 1),
        "f2_drift_hz": round(f2_drift, 1),
        "f1_offset_hz": round(f1_offset, 1),
        "f2_offset_hz": round(f2_offset, 1),
        "stability": round(stability, 3),
        "placement": round(placement, 3),
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
    from voice_core.spanish_stress import strip_accents

    # Build word→timestamp lookup (accent-stripped keys for robust matching)
    ts_lookup: dict[str, dict] = {}
    for wt in word_timestamps:
        key = strip_accents(wt["word"].lower().strip())
        ts_lookup[key] = wt

    consonant_features = []
    vowel_scores = []

    for target in target_sounds:
        word = target["word"].lower().strip()
        feature = target["feature"]
        ts = ts_lookup.get(strip_accents(word))

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

    # Stress analysis
    from voice_core.spanish_stress import detect_stress

    stress_results = detect_stress(y, sr, word_timestamps)

    # Summary
    sheismo_scores = [c["confidence"] for c in consonant_features
                      if c.get("feature") == "sheismo" and c.get("classification") == "sheismo"]
    tap_scores = [c["confidence"] for c in consonant_features
                  if c.get("feature") == "tap_r" and c.get("classification") == "tap"]
    purity_vals = [v["purity"] for v in vowel_scores]

    stress_correct = [s for s in stress_results if s["correct"]]

    summary = {
        "sheismo_score": round(float(np.mean(sheismo_scores)), 3) if sheismo_scores else None,
        "tap_r_score": round(float(np.mean(tap_scores)), 3) if tap_scores else None,
        "vowel_purity_avg": round(float(np.mean(purity_vals)), 3) if purity_vals else None,
        "stress_accuracy": round(len(stress_correct) / len(stress_results), 3) if stress_results else None,
    }

    return {
        "consonant_features": consonant_features,
        "vowel_scores": vowel_scores,
        "stress_analysis": stress_results,
        "summary": summary,
    }


def _mfa_available() -> bool:
    """Check if MFA is installed and accessible."""
    from voice_core.phoneme_align import MFA_BINARY
    try:
        result = subprocess.run(
            [MFA_BINARY, "version"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def analyze_spanish(
    wav_path: str,
    transcript: str,
    target_sounds: list[dict] | None = None,
    dialect: str = "rioplatense",
) -> dict:
    """Full Spanish pronunciation analysis using MFA alignment.

    Runs MFA forced alignment to get phoneme boundaries, then extracts
    formants and classifies consonants at each target phoneme.

    Falls back gracefully if MFA is not installed — returns structure
    with mfa_available=False and empty results.

    Args:
        wav_path: Path to WAV file.
        transcript: Spanish transcript of the audio.
        target_sounds: Optional list of {"word": str, "feature": str}.
            If None, analyzes all detected vowels and Rioplatense consonants.
        dialect: "rioplatense" (default) — affects which consonants to check.

    Returns:
        Dict with {mfa_available, phoneme_alignment, consonant_features,
                    vowel_scores, summary}
    """
    from voice_core.spanish_consonants import classify_sheismo, classify_tap_r

    empty_result = {
        "mfa_available": False,
        "phoneme_alignment": None,
        "consonant_features": [],
        "vowel_scores": [],
        "summary": {"sheismo_score": None, "tap_r_score": None, "vowel_purity_avg": None},
    }

    if not _mfa_available():
        return empty_result

    try:
        from voice_core.phoneme_align import align
        import soundfile as sf

        wav_path = str(Path(wav_path).resolve())

        # Run MFA alignment
        with tempfile.TemporaryDirectory() as tmpdir:
            tg_path = str(Path(tmpdir) / "aligned.TextGrid")
            align(wav_path, transcript, tg_path, language="es")

            # Load audio for segment extraction
            y, sr = sf.read(wav_path, dtype="float32")

            # Extract vowel formants using Spanish IPA mapping
            from praatio import textgrid
            tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=False)
            phones = tg.getTier("phones")

            vowel_scores = []
            consonant_features = []

            import parselmouth
            from parselmouth import praat

            for start, end, label in phones.entries:
                dur = end - start
                if dur < 0.03:
                    continue

                # Check vowels
                if label in SPANISH_IPA_TO_VOWEL:
                    vowel_key = SPANISH_IPA_TO_VOWEL[label]
                    start_sample = int(start * sr)
                    end_sample = int(end * sr)
                    segment = y[start_sample:end_sample]

                    snd = parselmouth.Sound(segment, sampling_frequency=sr)
                    formant = praat.call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50.0)
                    n_frames = praat.call(formant, "Get number of frames")
                    f1_frames, f2_frames = [], []
                    for fi in range(1, n_frames + 1):
                        t = praat.call(formant, "Get time from frame number", fi)
                        f1 = praat.call(formant, "Get value at time", 1, t, "Hertz", "Linear")
                        f2 = praat.call(formant, "Get value at time", 2, t, "Hertz", "Linear")
                        if f1 > 0 and f2 > 0:
                            f1_frames.append(f1)
                            f2_frames.append(f2)

                    if f1_frames:
                        purity = score_vowel_purity(
                            np.array(f1_frames), np.array(f2_frames), vowel_key)
                        if purity:
                            purity["phoneme"] = label
                            purity["vowel"] = vowel_key
                            purity["start"] = start
                            purity["end"] = end
                            vowel_scores.append(purity)

                # Check consonants
                if label in SPANISH_CONSONANT_TARGETS:
                    target_type = SPANISH_CONSONANT_TARGETS[label]
                    start_sample = int(start * sr)
                    end_sample = int(end * sr)
                    segment = y[start_sample:end_sample]

                    if target_type in ("sheismo", "yeismo"):
                        result = classify_sheismo(segment, sr)
                        result["phoneme"] = label
                        result["feature"] = "sheismo"
                        result["start"] = start
                        result["end"] = end
                        consonant_features.append(result)
                    elif target_type == "tap_r":
                        result = classify_tap_r(segment, sr)
                        result["phoneme"] = label
                        result["feature"] = "tap_r"
                        result["start"] = start
                        result["end"] = end
                        consonant_features.append(result)

        # Summary
        sheismo_scores = [c["confidence"] for c in consonant_features
                          if c.get("classification") == "sheismo"]
        tap_scores = [c["confidence"] for c in consonant_features
                      if c.get("classification") == "tap"]
        purity_vals = [v["purity"] for v in vowel_scores]

        return {
            "mfa_available": True,
            "phoneme_alignment": tg_path,
            "consonant_features": consonant_features,
            "vowel_scores": vowel_scores,
            "summary": {
                "sheismo_score": round(float(np.mean(sheismo_scores)), 3) if sheismo_scores else None,
                "tap_r_score": round(float(np.mean(tap_scores)), 3) if tap_scores else None,
                "vowel_purity_avg": round(float(np.mean(purity_vals)), 3) if purity_vals else None,
            },
        }

    except Exception as e:
        empty_result["error"] = str(e)
        return empty_result
