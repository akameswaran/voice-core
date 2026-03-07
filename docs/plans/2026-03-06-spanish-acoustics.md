# Spanish/Rioplatense Acoustic Primitives — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Spanish vowel norms, vowel purity scoring, Rioplatense consonant analysis (sheísmo, tap-r), and a top-level `analyze_spanish()` / `analyze_spanish_words()` API to voice-core so downstream coaches can score pronunciation.

**Architecture:** New data files for Spanish vowel norms and IPA mappings. New `spanish_consonants.py` module for fricative/tap classification. Extend existing `phoneme_align.py` with Spanish MFA support. New `spanish.py` as the top-level API. All backward-compatible — existing English analysis untouched.

**Tech Stack:** praat-parselmouth (formants), librosa/scipy/numpy (spectral analysis), MFA (phoneme alignment — optional, graceful fallback), praatio (TextGrid parsing)

---

### Task 1: Spanish Vowel Norms Data File

**Files:**
- Create: `src/voice_core/data/spanish_vowel_norms.json`
- Create: `tests/test_spanish_vowel_norms.py`

**Context:** The existing English norms (`data/vowel_norms.json`) have 13 ARPABET vowels with F1/F2/F3 mean+std. Spanish has 5 pure monophthongs. We use a simple key scheme: `A`, `E`, `I`, `O`, `U` (uppercase single letters, distinct from ARPABET).

Reference values from Quilis (1981) and Martínez Celdrán (2003), mixed-gender averages. These are well-established in Spanish phonetics literature. Stdev values are estimated from typical cross-speaker variation.

**Step 1: Write the failing test**

```python
# tests/test_spanish_vowel_norms.py
"""Tests for Spanish vowel norms data file."""

import json
from pathlib import Path

NORMS_PATH = Path(__file__).resolve().parent.parent / "src" / "voice_core" / "data" / "spanish_vowel_norms.json"

EXPECTED_VOWELS = {"A", "E", "I", "O", "U"}


def test_file_exists_and_loads():
    assert NORMS_PATH.exists(), f"Missing {NORMS_PATH}"
    data = json.loads(NORMS_PATH.read_text())
    assert "vowels" in data


def test_has_all_five_vowels():
    data = json.loads(NORMS_PATH.read_text())
    assert set(data["vowels"].keys()) == EXPECTED_VOWELS


def test_each_vowel_has_required_fields():
    data = json.loads(NORMS_PATH.read_text())
    required = {"f1_mean", "f1_std", "f2_mean", "f2_std", "f3_mean", "f3_std", "label"}
    for vowel, vals in data["vowels"].items():
        missing = required - set(vals.keys())
        assert not missing, f"{vowel} missing: {missing}"


def test_formant_values_are_plausible():
    """F1 should be 250-900 Hz, F2 800-2500 Hz, F3 2000-3500 Hz."""
    data = json.loads(NORMS_PATH.read_text())
    for vowel, v in data["vowels"].items():
        assert 250 < v["f1_mean"] < 900, f"{vowel} F1={v['f1_mean']}"
        assert 800 < v["f2_mean"] < 2500, f"{vowel} F2={v['f2_mean']}"
        assert 2000 < v["f3_mean"] < 3500, f"{vowel} F3={v['f3_mean']}"
        assert v["f1_std"] > 0
        assert v["f2_std"] > 0
        assert v["f3_std"] > 0
```

**Step 2: Run test to verify it fails**

Run: `cd /home/ak/Projects/VoiceCoaches/voice-core && uv run pytest tests/test_spanish_vowel_norms.py -v`
Expected: FAIL — file does not exist

**Step 3: Create the norms file**

```json
{
  "_description": "Spanish 5-vowel system formant norms. Mixed-gender averages from Quilis (1981) and Martínez Celdrán (2003). Used for vowel classification and purity scoring in Spanish pronunciation analysis.",
  "_source": "Quilis, A. (1981) Fonética acústica de la lengua española; Martínez Celdrán, E. (2003) JIPA 33(2)",

  "vowels": {
    "A": {"f1_mean": 750.0, "f1_std": 120.0, "f2_mean": 1250.0, "f2_std": 150.0, "f3_mean": 2550.0, "f3_std": 300.0, "label": "/a/ (casa)"},
    "E": {"f1_mean": 450.0, "f1_std":  80.0, "f2_mean": 1900.0, "f2_std": 200.0, "f3_mean": 2600.0, "f3_std": 280.0, "label": "/e/ (mesa)"},
    "I": {"f1_mean": 300.0, "f1_std":  70.0, "f2_mean": 2300.0, "f2_std": 250.0, "f3_mean": 2900.0, "f3_std": 300.0, "label": "/i/ (sí)"},
    "O": {"f1_mean": 500.0, "f1_std":  90.0, "f2_mean":  900.0, "f2_std": 150.0, "f3_mean": 2550.0, "f3_std": 300.0, "label": "/o/ (todo)"},
    "U": {"f1_mean": 350.0, "f1_std":  80.0, "f2_mean":  800.0, "f2_std": 150.0, "f3_mean": 2400.0, "f3_std": 300.0, "label": "/u/ (tú)"}
  }
}
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/ak/Projects/VoiceCoaches/voice-core && uv run pytest tests/test_spanish_vowel_norms.py -v`
Expected: 4 PASS

**Step 5: Commit**

```bash
git add src/voice_core/data/spanish_vowel_norms.json tests/test_spanish_vowel_norms.py
git commit -m "feat: add Spanish 5-vowel formant norms data file"
```

---

### Task 2: Spanish Vowel Classification

**Files:**
- Create: `src/voice_core/spanish.py`
- Create: `tests/test_spanish.py`

**Context:** `analyze.py:65-92` has `_classify_vowel(f1, f2)` that classifies against English norms. We need a Spanish equivalent that classifies against the 5 Spanish vowels. This goes in a new `spanish.py` module — the public API entry point for all Spanish analysis.

**Step 1: Write the failing tests**

```python
# tests/test_spanish.py
"""Tests for Spanish acoustic analysis module."""

import numpy as np
import pytest


def test_classify_vowel_a():
    from voice_core.spanish import classify_vowel_spanish
    # /a/ centroid: F1≈750, F2≈1250
    assert classify_vowel_spanish(740.0, 1260.0) == "A"


def test_classify_vowel_e():
    from voice_core.spanish import classify_vowel_spanish
    # /e/ centroid: F1≈450, F2≈1900
    assert classify_vowel_spanish(460.0, 1880.0) == "E"


def test_classify_vowel_i():
    from voice_core.spanish import classify_vowel_spanish
    # /i/ centroid: F1≈300, F2≈2300
    assert classify_vowel_spanish(310.0, 2280.0) == "I"


def test_classify_vowel_o():
    from voice_core.spanish import classify_vowel_spanish
    # /o/ centroid: F1≈500, F2≈900
    assert classify_vowel_spanish(510.0, 910.0) == "O"


def test_classify_vowel_u():
    from voice_core.spanish import classify_vowel_spanish
    # /u/ centroid: F1≈350, F2≈800
    assert classify_vowel_spanish(340.0, 790.0) == "U"


def test_classify_rejects_consonant():
    from voice_core.spanish import classify_vowel_spanish
    # Very far from any vowel
    assert classify_vowel_spanish(100.0, 5000.0) is None


def test_get_spanish_vowel_norms_loads():
    from voice_core.spanish import get_spanish_vowel_norms
    norms = get_spanish_vowel_norms()
    assert set(norms.keys()) == {"A", "E", "I", "O", "U"}
    assert norms["A"]["f1_mean"] > 0
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/ak/Projects/VoiceCoaches/voice-core && uv run pytest tests/test_spanish.py -v`
Expected: FAIL — `voice_core.spanish` does not exist

**Step 3: Implement**

```python
# src/voice_core/spanish.py
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

    Uses Euclidean distance in F1×F2 space against Spanish 5-vowel norms.
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
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/ak/Projects/VoiceCoaches/voice-core && uv run pytest tests/test_spanish.py -v`
Expected: 7 PASS

**Step 5: Commit**

```bash
git add src/voice_core/spanish.py tests/test_spanish.py
git commit -m "feat: Spanish vowel classification against 5-vowel norms"
```

---

### Task 3: Vowel Purity Scoring

**Files:**
- Modify: `src/voice_core/spanish.py`
- Modify: `tests/test_spanish.py`

**Context:** English speakers diphthongize Spanish vowels: /e/ → [eɪ] and /o/ → [oʊ]. A pure monophthong has stable F1/F2 throughout; a diphthong shows F2 drift. We measure trajectory stability by computing the standard deviation of F1/F2 across the vowel segment relative to the vowel's expected range.

This function takes per-frame F1/F2 values from a vowel segment (extracted by Parselmouth) and scores how stable they are.

**Step 1: Write the failing tests**

Add to `tests/test_spanish.py`:

```python
def test_vowel_purity_stable_vowel():
    """A stable vowel (no drift) should score near 1.0."""
    from voice_core.spanish import score_vowel_purity
    # Simulate stable /a/: F1≈750, F2≈1250, minimal variation
    rng = np.random.default_rng(42)
    f1_frames = 750.0 + rng.normal(0, 10, size=20)
    f2_frames = 1250.0 + rng.normal(0, 15, size=20)
    result = score_vowel_purity(f1_frames, f2_frames, expected_vowel="A")
    assert result["purity"] > 0.85
    assert result["diphthongized"] is False


def test_vowel_purity_diphthongized():
    """A drifting vowel (e→eɪ) should score low and flag diphthongization."""
    from voice_core.spanish import score_vowel_purity
    # Simulate /e/ diphthongizing to [eɪ]: F2 drifts from 1900→2200 over time
    n = 20
    f1_frames = np.linspace(450, 400, n)  # F1 drops slightly
    f2_frames = np.linspace(1900, 2200, n)  # F2 rises — diphthong!
    result = score_vowel_purity(f1_frames, f2_frames, expected_vowel="E")
    assert result["purity"] < 0.6
    assert result["diphthongized"] is True


def test_vowel_purity_too_few_frames():
    """Should return None if fewer than 4 frames."""
    from voice_core.spanish import score_vowel_purity
    result = score_vowel_purity(np.array([750.0, 745.0]), np.array([1250.0, 1260.0]), "A")
    assert result is None
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/ak/Projects/VoiceCoaches/voice-core && uv run pytest tests/test_spanish.py::test_vowel_purity_stable_vowel tests/test_spanish.py::test_vowel_purity_diphthongized tests/test_spanish.py::test_vowel_purity_too_few_frames -v`
Expected: FAIL — `score_vowel_purity` not found

**Step 3: Implement**

Add to `src/voice_core/spanish.py`:

```python
def score_vowel_purity(f1_frames: np.ndarray, f2_frames: np.ndarray,
                       expected_vowel: str,
                       min_frames: int = 4) -> dict | None:
    """Score vowel purity (monophthong stability) for a vowel segment.

    Measures how stable F1/F2 stay throughout the vowel. Pure Spanish
    monophthongs have minimal F1/F2 drift; English-speaker diphthongization
    shows systematic F2 movement (e.g., /e/ → [eɪ] has rising F2).

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
    # A drift of 1 std is significant; 2+ std is strong diphthongization
    f1_norm_drift = f1_drift / ref["f1_std"] if ref["f1_std"] > 0 else 0.0
    f2_norm_drift = f2_drift / ref["f2_std"] if ref["f2_std"] > 0 else 0.0

    # Combined drift score (F2 drift is more perceptually salient for diphthongs)
    combined_drift = 0.3 * f1_norm_drift + 0.7 * f2_norm_drift

    # Purity score: 1.0 = perfectly stable, 0.0 = heavily diphthongized
    # Map combined_drift through sigmoid-like curve: drift of 1.5 std → purity ~0.5
    purity = max(0.0, min(1.0, 1.0 - combined_drift / 3.0))

    return {
        "purity": round(purity, 3),
        "diphthongized": combined_drift > 1.0,
        "f1_drift_hz": round(f1_drift, 1),
        "f2_drift_hz": round(f2_drift, 1),
    }
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/ak/Projects/VoiceCoaches/voice-core && uv run pytest tests/test_spanish.py -v`
Expected: 10 PASS (7 previous + 3 new)

**Step 5: Commit**

```bash
git add src/voice_core/spanish.py tests/test_spanish.py
git commit -m "feat: vowel purity scoring with diphthongization detection"
```

---

### Task 4: Sheísmo Classifier

**Files:**
- Create: `src/voice_core/spanish_consonants.py`
- Create: `tests/test_spanish_consonants.py`

**Context:** Rioplatense LL/Y is pronounced /ʃ/ (like English "sh"). English speakers typically produce /j/ (English "y" glide). These are spectrally very different:
- /ʃ/ (fricative): energy concentrated at 2.5-4 kHz, high spectral centroid, noise-like
- /j/ (glide): low-frequency formant structure, low spectral centroid, periodic

The existing `_estimate_sibilant_centroid()` in `analyze.py:1505-1544` detects high-freq energy frames. We build a more targeted classifier that takes an audio segment known to contain LL/Y and classifies the production.

**Step 1: Write the failing tests**

```python
# tests/test_spanish_consonants.py
"""Tests for Rioplatense consonant analysis."""

import numpy as np
import pytest


def _make_noise(duration_s: float, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    """Generate band-limited noise to simulate a fricative."""
    from scipy.signal import butter, sosfilt
    n_samples = int(duration_s * sr)
    noise = np.random.default_rng(42).normal(0, 0.3, n_samples)
    sos = butter(4, [low_hz, high_hz], btype="band", fs=sr, output="sos")
    return sosfilt(sos, noise).astype(np.float32)


def _make_tone(duration_s: float, sr: int, freq_hz: float) -> np.ndarray:
    """Generate a simple tone to simulate a glide/vowel-like sound."""
    t = np.arange(int(duration_s * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_classify_sheismo_fricative():
    """High-freq noise (2.5-5 kHz) should be classified as sheismo."""
    from voice_core.spanish_consonants import classify_sheismo
    sr = 16000
    # Simulate /ʃ/: broadband noise centered around 3.5 kHz
    y = _make_noise(0.15, sr, 2500, 5000)
    result = classify_sheismo(y, sr)
    assert result["classification"] == "sheismo"
    assert result["confidence"] > 0.6


def test_classify_sheismo_glide():
    """Low-freq periodic sound should be classified as yeismo (glide)."""
    from voice_core.spanish_consonants import classify_sheismo
    sr = 16000
    # Simulate /j/: low-frequency tone (300 Hz fundamental)
    y = _make_tone(0.15, sr, 300.0)
    result = classify_sheismo(y, sr)
    assert result["classification"] == "yeismo"


def test_classify_sheismo_short_segment():
    """Very short segment should return unknown."""
    from voice_core.spanish_consonants import classify_sheismo
    sr = 16000
    y = np.zeros(int(0.01 * sr), dtype=np.float32)  # 10ms — too short
    result = classify_sheismo(y, sr)
    assert result["classification"] == "unknown"
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/ak/Projects/VoiceCoaches/voice-core && uv run pytest tests/test_spanish_consonants.py -v`
Expected: FAIL — `voice_core.spanish_consonants` does not exist

**Step 3: Implement**

```python
# src/voice_core/spanish_consonants.py
"""Rioplatense Spanish consonant analysis.

Classifies productions of consonants that distinguish Rioplatense from
other Spanish dialects: sheísmo (LL/Y → /ʃ/), tap-r vs approximant-r,
and s-aspiration.
"""

import numpy as np
import librosa


def classify_sheismo(y: np.ndarray, sr: int,
                     min_duration: float = 0.03) -> dict:
    """Classify an LL/Y segment as sheísmo (/ʃ/) or yeísmo (/j/).

    Sheísmo: LL/Y → /ʃ/ (Rioplatense fricative, energy at 2.5-4 kHz)
    Yeísmo: LL/Y → /j/ (glide, low-frequency formant structure)

    Args:
        y: Audio segment containing the LL/Y production.
        sr: Sample rate.
        min_duration: Minimum segment duration in seconds.

    Returns:
        Dict with {classification: "sheismo"|"yeismo"|"unknown",
                    confidence: 0-1, spectral_centroid_hz: float,
                    high_freq_ratio: float}
    """
    if len(y) / sr < min_duration:
        return {"classification": "unknown", "confidence": 0.0,
                "spectral_centroid_hz": 0.0, "high_freq_ratio": 0.0}

    n_fft = min(2048, len(y))
    hop_length = n_fft // 4

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Energy in bands
    high_mask = freqs >= 2500
    low_mask = (freqs >= 100) & (freqs < 2500)

    high_energy = np.sum(S[high_mask, :] ** 2)
    low_energy = np.sum(S[low_mask, :] ** 2)
    total_energy = high_energy + low_energy + 1e-10

    high_freq_ratio = float(high_energy / total_energy)

    # Spectral centroid of the full segment
    centroid_frames = librosa.feature.spectral_centroid(
        S=S, freq=freqs)[0]
    centroid_hz = float(np.mean(centroid_frames)) if len(centroid_frames) > 0 else 0.0

    # Classification logic:
    # /ʃ/ (sheismo): high_freq_ratio > 0.4, centroid > 2500 Hz
    # /j/ (yeismo): low-freq dominant, centroid < 1500 Hz
    if high_freq_ratio > 0.4 and centroid_hz > 2500:
        classification = "sheismo"
        confidence = min(1.0, high_freq_ratio * 1.5)
    elif high_freq_ratio < 0.2 or centroid_hz < 1500:
        classification = "yeismo"
        confidence = min(1.0, (1.0 - high_freq_ratio) * 1.2)
    else:
        classification = "intermediate"
        confidence = 0.5

    return {
        "classification": classification,
        "confidence": round(confidence, 3),
        "spectral_centroid_hz": round(centroid_hz, 1),
        "high_freq_ratio": round(high_freq_ratio, 3),
    }
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/ak/Projects/VoiceCoaches/voice-core && uv run pytest tests/test_spanish_consonants.py -v`
Expected: 3 PASS

**Step 5: Commit**

```bash
git add src/voice_core/spanish_consonants.py tests/test_spanish_consonants.py
git commit -m "feat: sheísmo classifier for Rioplatense LL/Y production"
```

---

### Task 5: Tap-R Detector

**Files:**
- Modify: `src/voice_core/spanish_consonants.py`
- Modify: `tests/test_spanish_consonants.py`

**Context:** Spanish /ɾ/ (tap) has a brief (~20-30ms) amplitude dip (tongue briefly contacts alveolar ridge). English /ɹ/ (approximant) has sustained low F3 (~1600-2000 Hz) and no closure. We detect the tap by looking for a brief amplitude minimum and checking F3.

**Step 1: Write the failing tests**

Add to `tests/test_spanish_consonants.py`:

```python
def _make_tap_r(sr: int) -> np.ndarray:
    """Simulate a tap-r: vowel + brief silence + vowel."""
    vowel1 = _make_tone(0.04, sr, 500.0) * 0.3
    closure = np.zeros(int(0.025 * sr), dtype=np.float32)  # 25ms closure
    vowel2 = _make_tone(0.04, sr, 500.0) * 0.3
    return np.concatenate([vowel1, closure, vowel2])


def _make_english_r(sr: int) -> np.ndarray:
    """Simulate English approximant /ɹ/: continuous low-frequency sound."""
    return _make_tone(0.10, sr, 350.0)


def test_detect_tap_r():
    """Tap-r (brief closure) should be detected."""
    from voice_core.spanish_consonants import classify_tap_r
    sr = 16000
    y = _make_tap_r(sr)
    result = classify_tap_r(y, sr)
    assert result["classification"] == "tap"
    assert result["has_closure"] is True


def test_detect_english_r():
    """English approximant (no closure) should be detected."""
    from voice_core.spanish_consonants import classify_tap_r
    sr = 16000
    y = _make_english_r(sr)
    result = classify_tap_r(y, sr)
    assert result["classification"] == "approximant"
    assert result["has_closure"] is False


def test_tap_r_short_segment():
    """Too-short segment returns unknown."""
    from voice_core.spanish_consonants import classify_tap_r
    sr = 16000
    y = np.zeros(int(0.01 * sr), dtype=np.float32)
    result = classify_tap_r(y, sr)
    assert result["classification"] == "unknown"
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/ak/Projects/VoiceCoaches/voice-core && uv run pytest tests/test_spanish_consonants.py::test_detect_tap_r tests/test_spanish_consonants.py::test_detect_english_r tests/test_spanish_consonants.py::test_tap_r_short_segment -v`
Expected: FAIL — `classify_tap_r` not found

**Step 3: Implement**

Add to `src/voice_core/spanish_consonants.py`:

```python
def classify_tap_r(y: np.ndarray, sr: int,
                   min_duration: float = 0.03,
                   closure_max_ms: float = 40.0,
                   closure_threshold_db: float = -20.0) -> dict:
    """Classify an r-position segment as tap /ɾ/ or approximant /ɹ/.

    Spanish tap-r has a brief (~20-30ms) amplitude dip from tongue closure.
    English approximant-r has continuous sound with no closure.

    Args:
        y: Audio segment containing the r production.
        sr: Sample rate.
        min_duration: Minimum segment duration in seconds.
        closure_max_ms: Maximum closure duration to count as tap (ms).
        closure_threshold_db: dB threshold below peak for closure detection.

    Returns:
        Dict with {classification: "tap"|"approximant"|"unknown",
                    has_closure: bool, closure_duration_ms: float,
                    confidence: float}
    """
    if len(y) / sr < min_duration:
        return {"classification": "unknown", "has_closure": False,
                "closure_duration_ms": 0.0, "confidence": 0.0}

    # Compute amplitude envelope (RMS in short windows)
    frame_length = int(0.005 * sr)  # 5ms frames
    hop = frame_length // 2
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]

    if len(rms) < 3:
        return {"classification": "unknown", "has_closure": False,
                "closure_duration_ms": 0.0, "confidence": 0.0}

    # Convert to dB
    rms_db = 20 * np.log10(rms + 1e-10)
    peak_db = np.max(rms_db)

    # Find frames below closure threshold (relative to peak)
    closure_frames = rms_db < (peak_db + closure_threshold_db)

    # Find contiguous closure regions
    has_closure = False
    closure_ms = 0.0

    if np.any(closure_frames):
        # Find longest contiguous run of closure frames
        diffs = np.diff(closure_frames.astype(int))
        starts = np.where(diffs == 1)[0] + 1
        ends = np.where(diffs == -1)[0] + 1

        # Handle edge cases
        if closure_frames[0]:
            starts = np.concatenate([[0], starts])
        if closure_frames[-1]:
            ends = np.concatenate([ends, [len(closure_frames)]])

        if len(starts) > 0 and len(ends) > 0:
            lengths = ends[:len(starts)] - starts[:len(ends)]
            if len(lengths) > 0:
                longest = np.max(lengths)
                closure_ms = float(longest * hop / sr * 1000)
                has_closure = 5.0 < closure_ms < closure_max_ms

    if has_closure:
        classification = "tap"
        confidence = min(1.0, 0.5 + closure_ms / 40.0)
    else:
        classification = "approximant"
        # Confidence is higher if RMS is very uniform (no dip at all)
        rms_cv = float(np.std(rms) / (np.mean(rms) + 1e-10))
        confidence = min(1.0, max(0.5, 1.0 - rms_cv))

    return {
        "classification": classification,
        "has_closure": has_closure,
        "closure_duration_ms": round(closure_ms, 1),
        "confidence": round(confidence, 3),
    }
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/ak/Projects/VoiceCoaches/voice-core && uv run pytest tests/test_spanish_consonants.py -v`
Expected: 6 PASS (3 sheísmo + 3 tap-r)

**Step 5: Commit**

```bash
git add src/voice_core/spanish_consonants.py tests/test_spanish_consonants.py
git commit -m "feat: tap-r vs approximant-r classifier for Spanish"
```

---

### Task 6: Spanish MFA Alignment Support

**Files:**
- Modify: `src/voice_core/phoneme_align.py:27-91`
- Modify: `tests/test_spanish.py`

**Context:** The existing `align()` hardcodes `english_mfa` for both acoustic model and dictionary. We need to support Spanish MFA models. MFA provides pre-trained Spanish models: `spanish_mfa` (acoustic) and `spanish_mfa` (dictionary). We also need an IPA-to-Spanish vowel mapping since MFA outputs IPA labels.

Note: MFA may not be installed in all environments. Tests for alignment should be skipped if MFA is unavailable. The vowel mapping tests don't need MFA.

**Step 1: Write the failing tests**

Add to `tests/test_spanish.py`:

```python
def test_spanish_ipa_to_vowel_mapping():
    """Spanish IPA vowel labels should map to our 5-vowel keys."""
    from voice_core.spanish import SPANISH_IPA_TO_VOWEL
    assert SPANISH_IPA_TO_VOWEL["a"] == "A"
    assert SPANISH_IPA_TO_VOWEL["e"] == "E"
    assert SPANISH_IPA_TO_VOWEL["i"] == "I"
    assert SPANISH_IPA_TO_VOWEL["o"] == "O"
    assert SPANISH_IPA_TO_VOWEL["u"] == "U"
    # Stressed variants
    assert SPANISH_IPA_TO_VOWEL.get("a̠") == "A"


def test_spanish_ipa_consonant_targets():
    """Consonant targets should include sheísmo and tap-r labels."""
    from voice_core.spanish import SPANISH_CONSONANT_TARGETS
    assert "ʃ" in SPANISH_CONSONANT_TARGETS
    assert "ʒ" in SPANISH_CONSONANT_TARGETS
    assert "ɾ" in SPANISH_CONSONANT_TARGETS
    assert "j" in SPANISH_CONSONANT_TARGETS
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/ak/Projects/VoiceCoaches/voice-core && uv run pytest tests/test_spanish.py::test_spanish_ipa_to_vowel_mapping tests/test_spanish.py::test_spanish_ipa_consonant_targets -v`
Expected: FAIL — names not found

**Step 3: Implement**

Add to `src/voice_core/spanish.py`:

```python
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
```

Also modify `phoneme_align.py` to support Spanish. Add after line 40:

```python
# Language → (acoustic_model, dictionary_model) for MFA
MFA_MODELS = {
    "en": ("english_mfa", "english_mfa"),
    "es": ("spanish_mfa", "spanish_mfa"),
}
```

Modify the `align()` function signature and subprocess call (lines 53-91):

```python
def align(wav_path: str, transcript: str,
          output_textgrid: Optional[str] = None,
          language: str = "en") -> str:
    """Run MFA forced alignment on a single audio file.

    Args:
        wav_path: Path to WAV file.
        transcript: Orthographic transcript of the audio.
        output_textgrid: Output TextGrid path. If None, writes next to WAV.
        language: Language code ("en" or "es").

    Returns:
        Path to the output TextGrid file.
    """
    wav_path = str(Path(wav_path).resolve())
    if output_textgrid is None:
        output_textgrid = wav_path.replace(".wav", ".TextGrid")

    acoustic_model, dictionary_model = MFA_MODELS.get(language, MFA_MODELS["en"])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                      dir=Path(wav_path).parent,
                                      delete=False) as f:
        f.write(transcript)
        txt_path = f.name

    try:
        result = subprocess.run(
            ["conda", "run", "-n", MFA_CONDA_ENV,
             "mfa", "align_one",
             wav_path, txt_path,
             acoustic_model, dictionary_model,
             output_textgrid],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"MFA alignment failed: {result.stderr[:500]}")
    finally:
        os.unlink(txt_path)

    return output_textgrid
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/ak/Projects/VoiceCoaches/voice-core && uv run pytest tests/test_spanish.py tests/test_imports.py -v`
Expected: All PASS. The `test_imports.py` smoke tests confirm phoneme_align still imports.

**Step 5: Commit**

```bash
git add src/voice_core/spanish.py src/voice_core/phoneme_align.py tests/test_spanish.py
git commit -m "feat: Spanish IPA mappings + language param for MFA alignment"
```

---

### Task 7: Top-Level API — `analyze_spanish_words()`

**Files:**
- Modify: `src/voice_core/spanish.py`
- Modify: `tests/test_spanish.py`

**Context:** This is the Whisper-boundary fast mode — the primary API spanish-voice-coach will call. Takes word timestamps from Whisper, expected words with annotated target sounds, and runs spectral analysis on relevant segments. No MFA required.

Each word in `target_sounds` specifies what to check: `{"word": "calle", "feature": "sheismo"}` means extract the audio for "calle" and run the sheísmo classifier.

**Step 1: Write the failing tests**

Add to `tests/test_spanish.py`:

```python
def test_analyze_spanish_words_returns_structure():
    """Should return vowel purity and consonant scores."""
    from voice_core.spanish import analyze_spanish_words
    sr = 16000
    # Create 1 second of noise (we're testing structure, not real audio)
    y = np.random.default_rng(42).normal(0, 0.1, sr).astype(np.float32)

    word_timestamps = [
        {"word": "calle", "start": 0.0, "end": 0.3},
        {"word": "pero", "start": 0.4, "end": 0.7},
    ]
    target_sounds = [
        {"word": "calle", "feature": "sheismo"},
        {"word": "pero", "feature": "tap_r"},
    ]

    result = analyze_spanish_words(y, sr, word_timestamps, target_sounds)
    assert "consonant_features" in result
    assert "summary" in result
    assert len(result["consonant_features"]) == 2


def test_analyze_spanish_words_empty_targets():
    """No targets should return empty results."""
    from voice_core.spanish import analyze_spanish_words
    sr = 16000
    y = np.zeros(sr, dtype=np.float32)
    result = analyze_spanish_words(y, sr, [], [])
    assert result["consonant_features"] == []
    assert result["vowel_scores"] == []
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/ak/Projects/VoiceCoaches/voice-core && uv run pytest tests/test_spanish.py::test_analyze_spanish_words_returns_structure tests/test_spanish.py::test_analyze_spanish_words_empty_targets -v`
Expected: FAIL — `analyze_spanish_words` not found

**Step 3: Implement**

Add to `src/voice_core/spanish.py`:

```python
from voice_core.spanish_consonants import classify_sheismo, classify_tap_r


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
            # For vowel purity we'd need formant tracking on the segment.
            # This is a simplified version using Parselmouth.
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
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/ak/Projects/VoiceCoaches/voice-core && uv run pytest tests/test_spanish.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/voice_core/spanish.py tests/test_spanish.py
git commit -m "feat: analyze_spanish_words() API for Whisper-boundary fast mode"
```

---

### Task 8: Top-Level API — `analyze_spanish()` (MFA mode)

**Files:**
- Modify: `src/voice_core/spanish.py`
- Modify: `tests/test_spanish.py`

**Context:** The MFA-based full analysis. Calls `phoneme_align.align()` with `language="es"`, extracts formants at vowel midpoints using Spanish IPA mappings, runs vowel purity on each vowel segment, and classifies consonant targets from the TextGrid. Falls back gracefully if MFA is not installed.

**Step 1: Write the failing tests**

Add to `tests/test_spanish.py`:

```python
def test_analyze_spanish_graceful_fallback():
    """Should return error info (not crash) if MFA is unavailable."""
    from voice_core.spanish import analyze_spanish
    import tempfile, soundfile as sf
    sr = 16000
    y = np.random.default_rng(42).normal(0, 0.1, sr).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, y, sr)
        result = analyze_spanish(f.name, "hola")
    # If MFA is not installed, should still return structure with mfa_available=False
    assert "mfa_available" in result
    if not result["mfa_available"]:
        assert result["consonant_features"] == []
        assert result["vowel_scores"] == []


def test_analyze_spanish_returns_expected_keys():
    """Should return all expected top-level keys."""
    from voice_core.spanish import analyze_spanish
    import tempfile, soundfile as sf
    sr = 16000
    y = np.random.default_rng(42).normal(0, 0.1, sr).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, y, sr)
        result = analyze_spanish(f.name, "hola")
    for key in ["mfa_available", "consonant_features", "vowel_scores", "summary"]:
        assert key in result, f"Missing key: {key}"
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/ak/Projects/VoiceCoaches/voice-core && uv run pytest tests/test_spanish.py::test_analyze_spanish_graceful_fallback tests/test_spanish.py::test_analyze_spanish_returns_expected_keys -v`
Expected: FAIL — `analyze_spanish` not found

**Step 3: Implement**

Add to `src/voice_core/spanish.py`:

```python
import subprocess
import tempfile
from pathlib import Path


def _mfa_available() -> bool:
    """Check if MFA is installed and accessible."""
    try:
        result = subprocess.run(
            ["conda", "run", "-n", "mfa", "mfa", "version"],
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
        from voice_core.phoneme_align import align, extract_vowel_formants
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
```

**Step 4: Run all tests**

Run: `cd /home/ak/Projects/VoiceCoaches/voice-core && uv run pytest tests/test_spanish.py tests/test_spanish_vowel_norms.py tests/test_spanish_consonants.py tests/test_imports.py -v`
Expected: All PASS. The `analyze_spanish` tests should return `mfa_available=False` gracefully if MFA isn't installed.

**Step 5: Commit**

```bash
git add src/voice_core/spanish.py tests/test_spanish.py
git commit -m "feat: analyze_spanish() full MFA pipeline with graceful fallback"
```

---

### Task 9: Run Full Test Suite & Verify No Regressions

**Files:** None modified — verification only.

**Step 1: Run all existing tests**

Run: `cd /home/ak/Projects/VoiceCoaches/voice-core && uv run pytest tests/ -v`
Expected: All PASS — no regressions in existing English analysis.

**Step 2: Run import smoke test**

Run: `cd /home/ak/Projects/VoiceCoaches/voice-core && uv run python -c "from voice_core.spanish import analyze_spanish, analyze_spanish_words, classify_vowel_spanish, score_vowel_purity, get_spanish_vowel_norms; from voice_core.spanish_consonants import classify_sheismo, classify_tap_r; print('All imports OK')"`
Expected: "All imports OK"

**Step 3: Commit**

No changes to commit if all passes. If `test_imports.py` needs updating to include new modules:

```bash
# Add to test_imports.py if needed:
# def test_import_spanish(): from voice_core import spanish
# def test_import_spanish_consonants(): from voice_core import spanish_consonants
git add tests/test_imports.py
git commit -m "test: add import smoke tests for Spanish modules"
```
