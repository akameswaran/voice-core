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
    y = _make_noise(0.15, sr, 2500, 5000)
    result = classify_sheismo(y, sr)
    assert result["classification"] == "sheismo"
    assert result["confidence"] > 0.6


def test_classify_sheismo_glide():
    """Low-freq periodic sound should be classified as yeismo (glide)."""
    from voice_core.spanish_consonants import classify_sheismo
    sr = 16000
    y = _make_tone(0.15, sr, 300.0)
    result = classify_sheismo(y, sr)
    assert result["classification"] == "yeismo"


def test_classify_sheismo_short_segment():
    """Very short segment should return unknown."""
    from voice_core.spanish_consonants import classify_sheismo
    sr = 16000
    y = np.zeros(int(0.01 * sr), dtype=np.float32)
    result = classify_sheismo(y, sr)
    assert result["classification"] == "unknown"


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
