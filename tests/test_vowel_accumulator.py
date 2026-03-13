"""Tests for VowelAccumulator: rolling per-vowel bucket with 10s sliding window."""

import pytest
from voice_core.vowel_accumulator import (
    VowelAccumulator,
    MONOPHTHONG_VOWELS,
    DIPHTHONG_VOWELS,
)


class TestVowelAccumulatorBasic:
    """Basic add() and expiry tests."""

    def test_add_single_frame(self):
        """Test adding a single frame."""
        acc = VowelAccumulator()
        features = {"f1": 700, "f2": 1500, "f4": 3500}
        acc.add("AE", 0.0, features)

        counts = acc.get_vowel_counts()
        assert counts["AE"] == 1

    def test_add_multiple_frames_same_vowel(self):
        """Test adding multiple frames with the same vowel."""
        acc = VowelAccumulator()
        for i in range(5):
            features = {"f1": 700, "f2": 1500, "f4": 3500}
            acc.add("AE", float(i) * 0.02, features)

        counts = acc.get_vowel_counts()
        assert counts["AE"] == 5

    def test_add_frames_different_vowels(self):
        """Test adding frames with different vowels."""
        acc = VowelAccumulator()
        acc.add("AE", 0.0, {"f1": 700, "f2": 1500, "f4": 3500})
        acc.add("IH", 0.02, {"f1": 300, "f2": 2000, "f4": 3600})
        acc.add("UH", 0.04, {"f1": 350, "f2": 800, "f4": 3400})

        counts = acc.get_vowel_counts()
        assert counts["AE"] == 1
        assert counts["IH"] == 1
        assert counts["UH"] == 1

    def test_window_expiry_basic(self):
        """Test that frames older than 10s are expired."""
        acc = VowelAccumulator(window_size_s=10.0)

        # Add frames at times 0.5, 1.5, 2.5, ..., 9.5 (all should stay)
        for i in range(10):
            features = {"f1": 700, "f2": 1500, "f4": 3500}
            acc.add("AE", 0.5 + float(i), features)

        counts = acc.get_vowel_counts()
        assert counts["AE"] == 10

        # Add frame at time 10.5 (cutoff = 10.5 - 10 = 0.5, frames < 0.5 expire)
        # We have 0.5-9.5, all >= 0.5, so keep all + add 10.5 = 11 frames
        acc.add("AE", 10.5, {"f1": 700, "f2": 1500, "f4": 3500})
        counts = acc.get_vowel_counts()
        assert counts["AE"] == 11

        # Add frame at time 20.5 (cutoff = 20.5 - 10 = 10.5, frames < 10.5 expire)
        # We have 0.5-10.5, frames 0.5-9.5 expire, keep 10.5-20.5
        acc.add("AE", 20.5, {"f1": 700, "f2": 1500, "f4": 3500})
        counts = acc.get_vowel_counts()
        assert counts["AE"] == 2

    def test_window_expiry_boundary(self):
        """Test boundary condition: frame exactly at cutoff_ts is kept."""
        acc = VowelAccumulator(window_size_s=10.0)

        # Add frame at time 0
        acc.add("AE", 0.0, {"f1": 700, "f2": 1500, "f4": 3500})

        # Add frame at time 10.0 (cutoff_ts = 10.0 - 10.0 = 0, so 0 is kept)
        acc.add("AE", 10.0, {"f1": 700, "f2": 1500, "f4": 3500})

        counts = acc.get_vowel_counts()
        assert counts["AE"] == 2

        # Add frame at time 10.01 (cutoff_ts = 10.01 - 10.0 = 0.01, so 0 is expired)
        acc.add("AE", 10.01, {"f1": 700, "f2": 1500, "f4": 3500})

        counts = acc.get_vowel_counts()
        assert counts["AE"] == 2


class TestMonophthongFiltering:
    """Tests for monophthong filtering in get_f4_scoring_stats()."""

    def test_f4_scoring_stats_monophthongs_only(self):
        """Test that get_f4_scoring_stats() returns only monophthongs."""
        acc = VowelAccumulator()

        # Add monophthong
        acc.add("AE", 0.0, {"f1": 700, "f2": 1500, "f4": 3500})
        acc.add("AE", 0.02, {"f1": 720, "f2": 1520, "f4": 3520})

        # Add diphthong
        acc.add("AY", 0.04, {"f1": 700, "f2": 1500, "f4": 3500})

        stats = acc.get_f4_scoring_stats()
        assert "AE" in stats
        assert "AY" not in stats

    def test_f4_stats_computation(self):
        """Test F4 statistics computation."""
        acc = VowelAccumulator()

        # Add frames with known values
        acc.add("AE", 0.0, {"f1": 700, "f2": 1500, "f4": 3500})
        acc.add("AE", 0.02, {"f1": 700, "f2": 1500, "f4": 3600})  # F4 +100
        acc.add("AE", 0.04, {"f1": 700, "f2": 1500, "f4": 3400})  # F4 -100

        stats = acc.get_f4_scoring_stats()
        assert "AE" in stats

        # F4 mean should be (3500 + 3600 + 3400) / 3 = 3500
        assert abs(stats["AE"]["f4_mean"] - 3500.0) < 0.1

        # F4 SD should be stdev([3500, 3600, 3400]) ≈ 100
        assert abs(stats["AE"]["f4_sd"] - 100.0) < 1.0

        assert stats["AE"]["f1_mean"] == 700.0
        assert stats["AE"]["f2_mean"] == 1500.0
        assert stats["AE"]["n_frames"] == 3

    def test_f4_stats_single_frame_no_sd(self):
        """Test that SD is 0 with a single frame."""
        acc = VowelAccumulator()
        acc.add("AE", 0.0, {"f1": 700, "f2": 1500, "f4": 3500})

        stats = acc.get_f4_scoring_stats()
        assert stats["AE"]["f4_sd"] == 0.0

    def test_get_all_stats_includes_diphthongs(self):
        """Test that get_all_stats() includes both monophthongs and diphthongs."""
        acc = VowelAccumulator()

        acc.add("AE", 0.0, {"f1": 700, "f2": 1500, "f4": 3500})
        acc.add("AY", 0.02, {"f1": 700, "f2": 1500, "f4": 3500})

        stats = acc.get_all_stats()
        assert "AE" in stats
        assert "AY" in stats


class TestResonanceConfidence:
    """Tests for resonance_confidence() calculation."""

    def test_resonance_confidence_no_frames(self):
        """Test confidence with no frames."""
        acc = VowelAccumulator()
        conf = acc.resonance_confidence()
        assert conf == 0.0

    def test_resonance_confidence_frame_component(self):
        """Test frame confidence component (30-frame threshold)."""
        acc = VowelAccumulator()

        # Add 15 monophthong frames (frame_conf = min(1.0, 15/30) = 0.5)
        for i in range(15):
            acc.add("AE", float(i) * 0.02, {"f1": 700, "f2": 1500, "f4": 3500})

        # With 1 vowel category with 15 frames >= 3, vowel_variety_conf = 1/3
        # So overall conf = sqrt(0.5 * 1/3) ≈ 0.408
        conf = acc.resonance_confidence()
        expected = (0.5 * (1.0 / 3.0)) ** 0.5
        assert abs(conf - expected) < 0.01

    def test_resonance_confidence_vowel_variety_component(self):
        """Test vowel variety confidence component (3-category threshold)."""
        acc = VowelAccumulator()

        # Add 30 monophthong frames in 1 category (frame_conf = 1.0)
        for i in range(30):
            acc.add("AE", float(i) * 0.02, {"f1": 700, "f2": 1500, "f4": 3500})

        # With only 1 vowel category and 30 frames >= 3, vowel_variety_conf = 1/3
        # So overall conf = sqrt(1.0 * 1/3) = sqrt(1/3) ≈ 0.577
        conf = acc.resonance_confidence()
        expected = (1.0 * (1.0 / 3.0)) ** 0.5
        assert abs(conf - expected) < 0.01

    def test_resonance_confidence_full_calculation(self):
        """Test full confidence calculation with optimal conditions."""
        acc = VowelAccumulator()

        # Add 30 frames each in 3 vowel categories
        vowels = ["AE", "IH", "UH"]
        for j, vowel in enumerate(vowels):
            for i in range(30):
                ts = j * 30 * 0.02 + i * 0.02
                acc.add(vowel, ts, {"f1": 700, "f2": 1500, "f4": 3500})

        # frame_conf = min(1.0, 90/30) = 1.0
        # vowel_variety_conf = min(1.0, 3/3) = 1.0
        # overall = sqrt(1.0 * 1.0) = 1.0
        conf = acc.resonance_confidence()
        assert abs(conf - 1.0) < 0.001

    def test_resonance_confidence_diphthongs_not_counted(self):
        """Test that diphthongs are not counted for confidence."""
        acc = VowelAccumulator()

        # Add 30 diphthong frames (should not count)
        for i in range(30):
            acc.add("AY", float(i) * 0.02, {"f1": 700, "f2": 1500, "f4": 3500})

        conf = acc.resonance_confidence()
        assert conf == 0.0


class TestPerVowelF4CV:
    """Tests for per_vowel_f4_cv() calculation."""

    def test_per_vowel_f4_cv_no_frames(self):
        """Test with no frames."""
        acc = VowelAccumulator()
        cv = acc.per_vowel_f4_cv()
        assert cv is None

    def test_per_vowel_f4_cv_insufficient_data(self):
        """Test with fewer than 5 frames (insufficient data)."""
        acc = VowelAccumulator()

        for i in range(4):
            acc.add("AE", float(i) * 0.02, {"f1": 700, "f2": 1500, "f4": 3500 + i * 10})

        cv = acc.per_vowel_f4_cv()
        assert cv is None

    def test_per_vowel_f4_cv_with_5_frames(self):
        """Test with exactly 5 frames (meets threshold)."""
        acc = VowelAccumulator()

        # F4 values: [3500, 3510, 3520, 3530, 3540]
        # mean = 3520, SD ≈ 14.14, CV ≈ 0.004
        for i in range(5):
            acc.add("AE", float(i) * 0.02, {"f1": 700, "f2": 1500, "f4": 3500 + i * 10})

        cv = acc.per_vowel_f4_cv()
        assert cv is not None
        assert 0.003 < cv < 0.006  # Loose bounds for CV

    def test_per_vowel_f4_cv_multiple_vowels(self):
        """Test with multiple vowels, average their CVs."""
        acc = VowelAccumulator()

        # Add AE with 5 frames, F4: [3500, 3600, 3700, 3800, 3900]
        # mean = 3700, SD ≈ 158.1, CV ≈ 0.0427
        for i in range(5):
            acc.add("AE", float(i) * 0.02, {"f1": 700, "f2": 1500, "f4": 3500 + i * 100})

        # Add IH with 5 frames, F4: [3600, 3700, 3800, 3900, 4000]
        # mean = 3800, SD ≈ 158.1, CV ≈ 0.0416
        for i in range(5):
            acc.add("IH", float(5 + i) * 0.02, {"f1": 300, "f2": 2000, "f4": 3600 + i * 100})

        cv = acc.per_vowel_f4_cv()
        assert cv is not None
        # Average of two CVs, both around 0.042
        assert 0.03 < cv < 0.05

    def test_per_vowel_f4_cv_zero_mean_defense(self):
        """Test that zero F4 mean doesn't cause division by zero."""
        acc = VowelAccumulator()

        # This shouldn't happen in practice, but let's be defensive
        acc.add("AE", 0.0, {"f1": 700, "f2": 1500, "f4": 0.0})
        acc.add("AE", 0.02, {"f1": 700, "f2": 1500, "f4": 0.0})
        acc.add("AE", 0.04, {"f1": 700, "f2": 1500, "f4": 0.0})
        acc.add("AE", 0.06, {"f1": 700, "f2": 1500, "f4": 0.0})
        acc.add("AE", 0.08, {"f1": 700, "f2": 1500, "f4": 0.0})

        # Should not raise, just skip this vowel
        cv = acc.per_vowel_f4_cv()
        assert cv is None


class TestAccumulatedMeans:
    """Tests for get_accumulated_means()."""

    def test_accumulated_means_empty(self):
        """Test with no frames."""
        acc = VowelAccumulator()
        means = acc.get_accumulated_means()

        assert means["delta_f_hz"] == 0.0
        assert means["f1_mean"] == 0.0
        assert means["f2_mean"] == 0.0
        assert means["h1_h2_mean"] == 0.0

    def test_accumulated_means_from_explicit_delta_f(self):
        """Test that explicit delta_f is used when present."""
        acc = VowelAccumulator()

        acc.add("AE", 0.0, {"f1": 700, "f2": 1500, "delta_f": 100})
        acc.add("AE", 0.02, {"f1": 700, "f2": 1500, "delta_f": 120})

        means = acc.get_accumulated_means()
        assert abs(means["delta_f_hz"] - 110.0) < 0.1

    def test_accumulated_means_f4_as_delta_f_proxy(self):
        """Test that F4 is used as proxy for delta_f when delta_f not present."""
        acc = VowelAccumulator()

        acc.add("AE", 0.0, {"f1": 700, "f2": 1500, "f4": 3500})
        acc.add("AE", 0.02, {"f1": 700, "f2": 1500, "f4": 3600})

        means = acc.get_accumulated_means()
        assert abs(means["delta_f_hz"] - 3550.0) < 0.1

    def test_accumulated_means_h1_h2(self):
        """Test H1-H2 mean calculation."""
        acc = VowelAccumulator()

        acc.add("AE", 0.0, {"f1": 700, "f2": 1500, "f4": 3500, "h1_h2": 5.0})
        acc.add("AE", 0.02, {"f1": 700, "f2": 1500, "f4": 3500, "h1_h2": 7.0})
        acc.add("AE", 0.04, {"f1": 700, "f2": 1500, "f4": 3500, "h1_h2": 9.0})

        means = acc.get_accumulated_means()
        assert abs(means["h1_h2_mean"] - 7.0) < 0.1

    def test_accumulated_means_partial_data(self):
        """Test with some frames missing optional fields."""
        acc = VowelAccumulator()

        acc.add("AE", 0.0, {"f1": 700, "f2": 1500, "f4": 3500})  # No h1_h2
        acc.add("AE", 0.02, {"f1": 700, "f2": 1500, "f4": 3500, "h1_h2": 6.0})
        acc.add("AE", 0.04, {"f1": 700, "f2": 1500, "f4": 3500})  # No h1_h2

        means = acc.get_accumulated_means()
        # Only 1 frame with h1_h2
        assert abs(means["h1_h2_mean"] - 6.0) < 0.1


class TestF0History:
    """Tests for get_f0_history()."""

    def test_f0_history_empty(self):
        """Test with no F0 data."""
        acc = VowelAccumulator()
        f0_list = acc.get_f0_history()
        assert f0_list == []

    def test_f0_history_with_data(self):
        """Test F0 history collection."""
        acc = VowelAccumulator()

        acc.add("AE", 0.0, {"f1": 700, "f2": 1500, "f0": 120.0})
        acc.add("AE", 0.02, {"f1": 700, "f2": 1500, "f0": 122.0})
        acc.add("AE", 0.04, {"f1": 700, "f2": 1500, "f0": 121.0})

        f0_list = acc.get_f0_history()
        assert len(f0_list) == 3
        assert f0_list == [120.0, 122.0, 121.0]

    def test_f0_history_partial(self):
        """Test with some frames missing F0."""
        acc = VowelAccumulator()

        acc.add("AE", 0.0, {"f1": 700, "f2": 1500, "f0": 120.0})
        acc.add("AE", 0.02, {"f1": 700, "f2": 1500})  # No F0
        acc.add("AE", 0.04, {"f1": 700, "f2": 1500, "f0": 121.0})

        f0_list = acc.get_f0_history()
        assert len(f0_list) == 2
        assert f0_list == [120.0, 121.0]


class TestVowelCounts:
    """Tests for get_vowel_counts()."""

    def test_vowel_counts_empty(self):
        """Test with no frames."""
        acc = VowelAccumulator()
        counts = acc.get_vowel_counts()
        assert counts == {}

    def test_vowel_counts_single_vowel(self):
        """Test counts for a single vowel."""
        acc = VowelAccumulator()

        for i in range(5):
            acc.add("AE", float(i) * 0.02, {"f1": 700, "f2": 1500, "f4": 3500})

        counts = acc.get_vowel_counts()
        assert counts == {"AE": 5}

    def test_vowel_counts_multiple_vowels(self):
        """Test counts for multiple vowels."""
        acc = VowelAccumulator()

        for i in range(5):
            acc.add("AE", float(i) * 0.02, {"f1": 700, "f2": 1500, "f4": 3500})
        for i in range(3):
            acc.add("IH", float(5 + i) * 0.02, {"f1": 300, "f2": 2000, "f4": 3600})
        for i in range(2):
            acc.add("UH", float(8 + i) * 0.02, {"f1": 350, "f2": 800, "f4": 3400})

        counts = acc.get_vowel_counts()
        assert counts == {"AE": 5, "IH": 3, "UH": 2}

    def test_vowel_counts_respects_expiry(self):
        """Test that expiry is reflected in vowel counts."""
        acc = VowelAccumulator(window_size_s=10.0)

        # Add 5 frames at time 0
        for i in range(5):
            acc.add("AE", float(i) * 0.02, {"f1": 700, "f2": 1500, "f4": 3500})

        counts = acc.get_vowel_counts()
        assert counts["AE"] == 5

        # Add frame at time 20, expires frames < 10
        acc.add("AE", 20.0, {"f1": 700, "f2": 1500, "f4": 3500})

        counts = acc.get_vowel_counts()
        # Should have no frames < 10, then frame at 20
        assert counts["AE"] == 1


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_realistic_speech_window(self):
        """Test a realistic scenario with varied speech."""
        acc = VowelAccumulator(window_size_s=10.0)

        # Simulate 10 seconds of speech: various vowels with formant variations
        vowels = ["AE", "IH", "UH", "AA", "EH"]
        for j in range(500):  # 500 frames at 50 fps = 10 seconds
            ts = j / 50.0
            vowel = vowels[j % len(vowels)]

            # Add some variation in formants
            f1 = 700 + (j % 50) * 2
            f2 = 1500 + (j % 50) * 5
            f4 = 3500 + (j % 50) * 3

            features = {
                "f1": f1,
                "f2": f2,
                "f4": f4,
                "f0": 120 + (j % 50) * 0.5,
                "h1_h2": 5.0 + (j % 50) * 0.1,
            }
            acc.add(vowel, ts, features)

        # Should have 500 frames in window
        total_frames = sum(acc.get_vowel_counts().values())
        assert total_frames == 500

        # All 5 vowels should appear
        counts = acc.get_vowel_counts()
        assert len(counts) == 5

        # F4 scoring stats should work
        f4_stats = acc.get_f4_scoring_stats()
        assert len(f4_stats) == 5

        # All vowels should have >= 5 frames
        assert all(stat["n_frames"] >= 5 for stat in f4_stats.values())

        # Resonance confidence should be reasonable
        conf = acc.resonance_confidence()
        assert 0.5 < conf <= 1.0

        # Per-vowel F4 CV should be reasonable
        cv = acc.per_vowel_f4_cv()
        assert cv is not None
        assert 0.0 < cv < 0.02  # Small variation in our synthetic data

        # F0 history should have frames
        f0_hist = acc.get_f0_history()
        assert len(f0_hist) == 500

        # Accumulated means should be reasonable
        means = acc.get_accumulated_means()
        assert 700 < means["f1_mean"] < 850
        assert 1500 < means["f2_mean"] < 1700
        assert 3500 < means["delta_f_hz"] < 3700
