"""Tests for per-vowel z-score aggregation in the resonance pipeline.

Tests the _compute_per_vowel_zscores function introduced in Task 1:
per-vowel F1 z-scoring against Hillenbrand (1995) female speaker norms.

Also tests BW4 extraction introduced in Task 2.
"""

import math
import numpy as np
import parselmouth
import pytest

from voice_core.analyze import _compute_per_vowel_zscores, analyze_formants


def _make_test_sound(duration: float = 1.0, sr: int = 16000) -> parselmouth.Sound:
    """Create a synthetic voiced signal suitable for formant analysis.

    Uses a sum of harmonics to create a vowel-like signal with clear
    formant structure that Praat can analyse reliably.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Fundamental + several harmonics to give Praat something to track
    f0 = 150.0
    signal = np.zeros_like(t)
    for harmonic in range(1, 10):
        signal += (1.0 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t)
    signal = (signal / np.max(np.abs(signal))).astype(np.float32)
    return parselmouth.Sound(signal, sampling_frequency=sr)


class TestPerVowelZscores:
    """Behavioral tests for _compute_per_vowel_zscores."""

    # ------------------------------------------------------------------
    # Test 1: Empty input returns empty dict
    # ------------------------------------------------------------------
    def test_empty_input_returns_empty_dict(self):
        """Empty lists should produce an empty result dict with no crash."""
        result = _compute_per_vowel_zscores([], [], [], [])
        assert result == {}

    # ------------------------------------------------------------------
    # Test 2: Single vowel IH classifies correctly
    # ------------------------------------------------------------------
    def test_single_vowel_ih_produces_ih_key(self):
        """F1/F2 values that map to IH should produce an 'ih' key in output.

        IH centroid: F1≈467, F2≈1903. Using F1=600, F2=1900 reliably
        classifies as IH.
        """
        n = 5
        f1 = [600.0] * n
        f2 = [1900.0] * n
        f3 = [2500.0] * n
        f4 = [0.0] * n
        result = _compute_per_vowel_zscores(f1, f2, f3, f4)

        assert "ih" in result, f"Expected 'ih' key, got keys: {list(result.keys())}"

    # ------------------------------------------------------------------
    # Test 3: F1 z-score calculation is correct
    # ------------------------------------------------------------------
    def test_f1_zscore_calculation_is_correct(self):
        """Z-score should equal (mean_f1 - hillenbrand_mean) / hillenbrand_std.

        For IH: Hillenbrand mean=483 Hz, std=56 Hz.
        With F1=600: z = (600 - 483) / 56 = 2.089 (rounded to 3 dp).
        """
        f1 = [600.0] * 5
        f2 = [1900.0] * 5
        f3 = [2500.0] * 5
        f4 = [0.0] * 5
        result = _compute_per_vowel_zscores(f1, f2, f3, f4)

        ih = result["ih"]
        expected_z = round((600.0 - 483.0) / 56.0, 3)
        assert ih["f1_zscore"] == pytest.approx(expected_z, abs=0.001), (
            f"Expected f1_zscore={expected_z}, got {ih['f1_zscore']}"
        )

    # ------------------------------------------------------------------
    # Test 4: F1 above female norm gives positive z-score
    # ------------------------------------------------------------------
    def test_f1_above_female_norm_gives_positive_zscore(self):
        """F1=600 Hz on IH is above the Hillenbrand female mean (483 Hz),
        so the z-score should be positive (≈2.089).
        """
        f1 = [600.0] * 3
        f2 = [1900.0] * 3
        f3 = [2500.0] * 3
        f4 = [0.0] * 3
        result = _compute_per_vowel_zscores(f1, f2, f3, f4)

        assert result["ih"]["f1_zscore"] > 0, (
            "F1 above female norm should yield positive z-score"
        )

    # ------------------------------------------------------------------
    # Test 5: F1 at female norm gives near-zero z-score
    # ------------------------------------------------------------------
    def test_f1_at_female_norm_gives_near_zero_zscore(self):
        """F1 exactly at the Hillenbrand female mean (483 Hz) for IH should
        produce a z-score of 0.0.
        """
        f1 = [483.0] * 4
        f2 = [1900.0] * 4
        f3 = [2500.0] * 4
        f4 = [0.0] * 4
        result = _compute_per_vowel_zscores(f1, f2, f3, f4)

        assert result["ih"]["f1_zscore"] == pytest.approx(0.0, abs=0.001), (
            "F1 at female norm should give z-score ≈ 0.0"
        )

    # ------------------------------------------------------------------
    # Test 6: Multiple vowels produce separate entries
    # ------------------------------------------------------------------
    def test_multiple_vowels_produce_separate_entries(self):
        """A mix of IH frames (F1=600, F2=1900) and EH frames
        (F1=594, F2=1631) should produce both 'ih' and 'eh' keys.
        """
        f1 = [600.0] * 3 + [594.0] * 3
        f2 = [1900.0] * 3 + [1631.0] * 3
        f3 = [2500.0] * 6
        f4 = [0.0] * 6
        result = _compute_per_vowel_zscores(f1, f2, f3, f4)

        assert "ih" in result, f"Expected 'ih' key, got: {list(result.keys())}"
        assert "eh" in result, f"Expected 'eh' key, got: {list(result.keys())}"

    # ------------------------------------------------------------------
    # Test 7: n_frames count is accurate
    # ------------------------------------------------------------------
    def test_n_frames_count_is_accurate(self):
        """5 IH frames should produce n_frames == 5 in the 'ih' entry."""
        n = 5
        f1 = [600.0] * n
        f2 = [1900.0] * n
        f3 = [2500.0] * n
        f4 = [0.0] * n
        result = _compute_per_vowel_zscores(f1, f2, f3, f4)

        assert result["ih"]["n_frames"] == n, (
            f"Expected n_frames={n}, got {result['ih']['n_frames']}"
        )

    # ------------------------------------------------------------------
    # Test 8: Unrecognized vowels are ignored (no crash, no output key)
    # ------------------------------------------------------------------
    def test_unrecognized_vowels_are_ignored(self):
        """Frames that classify to a vowel not in the 7 Hillenbrand targets
        (e.g. AY: F1≈800, F2≈1446) should produce no output entry and
        not raise an exception.
        """
        # AY is in the vowel_norms centroids but not in _ARPABET_TO_HILLENBRAND
        f1 = [800.0] * 4
        f2 = [1446.0] * 4
        f3 = [2400.0] * 4
        f4 = [0.0] * 4
        result = _compute_per_vowel_zscores(f1, f2, f3, f4)

        # Should not crash and should not produce any output for unmapped vowels
        assert result == {}, (
            f"Unrecognized vowel frames should produce empty dict, got: {result}"
        )

    # ------------------------------------------------------------------
    # Bonus Test 9: f1_mean_hz reflects the actual mean of input values
    # ------------------------------------------------------------------
    def test_f1_mean_hz_reflects_input_mean(self):
        """The f1_mean_hz field should equal the arithmetic mean of the
        F1 values provided for that vowel bucket.

        Use F1 values that all reliably classify as IH (stay near the centroid
        at F1≈467, F2≈1903): 480, 500, 520 all classify as IH at F2=1900.
        """
        f1_vals = [480.0, 500.0, 520.0]
        f2 = [1900.0] * 3
        f3 = [2500.0] * 3
        f4 = [0.0] * 3
        result = _compute_per_vowel_zscores(f1_vals, f2, f3, f4)

        expected_mean = round(sum(f1_vals) / len(f1_vals), 1)  # 500.0
        assert "ih" in result, (
            f"Expected 'ih' key in result, got: {list(result.keys())}"
        )
        assert result["ih"]["f1_mean_hz"] == pytest.approx(expected_mean, abs=0.1), (
            f"Expected f1_mean_hz={expected_mean}, got {result['ih']['f1_mean_hz']}"
        )

    # ------------------------------------------------------------------
    # Bonus Test 10: f4 values are aggregated when provided
    # ------------------------------------------------------------------
    def test_f4_mean_hz_aggregated_when_provided(self):
        """When valid F4 values (> 0) are supplied, f4_mean_hz should reflect
        their mean for that vowel.
        """
        n = 3
        f1 = [600.0] * n
        f2 = [1900.0] * n
        f3 = [2500.0] * n
        f4 = [3100.0, 3200.0, 3300.0]
        result = _compute_per_vowel_zscores(f1, f2, f3, f4)

        expected_f4_mean = round(sum(f4) / len(f4), 1)  # 3200.0
        assert result["ih"]["f4_mean_hz"] == pytest.approx(expected_f4_mean, abs=0.1), (
            f"Expected f4_mean_hz={expected_f4_mean}, got {result['ih']['f4_mean_hz']}"
        )

    # ------------------------------------------------------------------
    # Bonus Test 11: Frames with zero F1 are skipped
    # ------------------------------------------------------------------
    def test_zero_f1_frames_are_skipped(self):
        """Frames with F1 <= 0 should be silently skipped (treated as
        unvoiced/unavailable), not included in any bucket.
        """
        # 2 valid IH frames + 3 zero-F1 frames
        f1 = [600.0, 600.0, 0.0, 0.0, 0.0]
        f2 = [1900.0, 1900.0, 1900.0, 1900.0, 1900.0]
        f3 = [2500.0] * 5
        f4 = [0.0] * 5
        result = _compute_per_vowel_zscores(f1, f2, f3, f4)

        assert result["ih"]["n_frames"] == 2, (
            f"Zero-F1 frames should be skipped; expected n_frames=2, "
            f"got {result['ih']['n_frames']}"
        )


class TestBW4Extraction:
    """Integration tests for BW4 extraction in analyze_formants (Task 2)."""

    # ------------------------------------------------------------------
    # Test 1: bw4_mean_hz key is present in analyze_formants output
    # ------------------------------------------------------------------
    def test_bw4_present_in_analyze_formants_output(self):
        """analyze_formants should include a 'bw4_mean_hz' key in its result."""
        snd = _make_test_sound()
        result = analyze_formants(snd)
        assert "bw4_mean_hz" in result, (
            f"Expected 'bw4_mean_hz' key in result, got keys: {list(result.keys())}"
        )

    # ------------------------------------------------------------------
    # Test 2: bw4_mean_hz is positive on a voiced signal
    # ------------------------------------------------------------------
    def test_bw4_is_positive(self):
        """BW4 should be > 0 when Praat successfully tracks F4 on a voiced signal."""
        snd = _make_test_sound()
        result = analyze_formants(snd)
        assert result["bw4_mean_hz"] > 0, (
            f"Expected bw4_mean_hz > 0, got {result['bw4_mean_hz']}"
        )

    # ------------------------------------------------------------------
    # Test 3: bw4_mean_hz is within a plausible acoustic range
    # ------------------------------------------------------------------
    def test_bw4_is_reasonable_range(self):
        """BW4 should be within 0–5000 Hz (Praat bandwidths for voiced speech
        are typically 50–500 Hz for well-resolved formants).
        """
        snd = _make_test_sound()
        result = analyze_formants(snd)
        bw4 = result["bw4_mean_hz"]
        assert 0 < bw4 < 5000, (
            f"Expected 0 < bw4_mean_hz < 5000 Hz, got {bw4}"
        )
