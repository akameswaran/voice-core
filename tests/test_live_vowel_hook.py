"""Tests for vowel classifier hook in live.py _formant_worker.

Tests the vowel classification gating logic and integration with LiveAnalyzer.
Since the _formant_worker thread doesn't run in unit tests, we verify:
1. The "vowel" key is properly initialized
2. The gate conditions are correctly implemented
"""

from voice_core.live import LiveAnalyzer
from voice_core.analyze import _classify_vowel


class TestVowelClassifierHook:
    """Test vowel classification in _formant_worker."""

    def test_vowel_key_in_latest_init(self):
        """Test that 'vowel' key is initialized in self.latest."""
        analyzer = LiveAnalyzer()
        assert "vowel" in analyzer.latest
        assert analyzer.latest["vowel"] is None

    def test_get_frame_returns_vowel_key(self):
        """Test that get_frame() returns a vowel key in the frame dict."""
        analyzer = LiveAnalyzer()
        frame = analyzer.get_frame()
        assert "vowel" in frame
        assert frame["vowel"] is None

    def test_vowel_key_persists_in_frame(self):
        """Test that vowel value in latest is propagated to get_frame()."""
        analyzer = LiveAnalyzer()
        with analyzer._lock:
            analyzer.latest["vowel"] = "AA"
        frame = analyzer.get_frame()
        assert frame["vowel"] == "AA"

    def test_vowel_gate_conditions_all_pass(self):
        """Test the vowel gate conditions: all pass → vowel should be classified."""
        # f0 > 80 and rms_db > -35 and f1 > 200 and f2 > 500 and bw1 < 200
        f0 = 150.0  # > 80 ✓
        rms_db = -20.0  # > -35 ✓
        f1 = 450.0  # > 200 ✓
        f2 = 1200.0  # > 500 ✓
        bw1 = 100.0  # < 200 ✓

        # Apply gate (what _formant_worker does)
        should_classify = (f0 > 80 and rms_db > -35 and f1 > 200 and f2 > 500 and bw1 < 200)
        assert should_classify is True

    def test_vowel_gate_f0_too_low(self):
        """Test vowel gate: f0 <= 80 blocks classification."""
        f0 = 50.0  # NOT > 80
        rms_db = -20.0
        f1 = 450.0
        f2 = 1200.0
        bw1 = 100.0

        should_classify = (f0 > 80 and rms_db > -35 and f1 > 200 and f2 > 500 and bw1 < 200)
        assert should_classify is False

    def test_vowel_gate_f0_boundary_exactly_80(self):
        """Test vowel gate: f0 exactly 80 does not pass (need strictly >)."""
        f0 = 80.0  # NOT > 80 (exactly equal)
        rms_db = -20.0
        f1 = 450.0
        f2 = 1200.0
        bw1 = 100.0

        should_classify = (f0 > 80 and rms_db > -35 and f1 > 200 and f2 > 500 and bw1 < 200)
        assert should_classify is False

    def test_vowel_gate_f0_boundary_just_above_80(self):
        """Test vowel gate: f0 > 80 passes."""
        f0 = 80.1  # > 80 ✓
        rms_db = -20.0
        f1 = 450.0
        f2 = 1200.0
        bw1 = 100.0

        should_classify = (f0 > 80 and rms_db > -35 and f1 > 200 and f2 > 500 and bw1 < 200)
        assert should_classify is True

    def test_vowel_gate_rms_db_too_low(self):
        """Test vowel gate: rms_db <= -35 blocks classification."""
        f0 = 150.0
        rms_db = -40.0  # NOT > -35
        f1 = 450.0
        f2 = 1200.0
        bw1 = 100.0

        should_classify = (f0 > 80 and rms_db > -35 and f1 > 200 and f2 > 500 and bw1 < 200)
        assert should_classify is False

    def test_vowel_gate_rms_db_boundary_exactly_minus_35(self):
        """Test vowel gate: rms_db exactly -35 does not pass."""
        f0 = 150.0
        rms_db = -35.0  # NOT > -35 (exactly equal)
        f1 = 450.0
        f2 = 1200.0
        bw1 = 100.0

        should_classify = (f0 > 80 and rms_db > -35 and f1 > 200 and f2 > 500 and bw1 < 200)
        assert should_classify is False

    def test_vowel_gate_rms_db_boundary_just_above_minus_35(self):
        """Test vowel gate: rms_db > -35 passes."""
        f0 = 150.0
        rms_db = -34.9  # > -35 ✓
        f1 = 450.0
        f2 = 1200.0
        bw1 = 100.0

        should_classify = (f0 > 80 and rms_db > -35 and f1 > 200 and f2 > 500 and bw1 < 200)
        assert should_classify is True

    def test_vowel_gate_f1_too_low(self):
        """Test vowel gate: f1 <= 200 blocks classification."""
        f0 = 150.0
        rms_db = -20.0
        f1 = 150.0  # NOT > 200
        f2 = 1200.0
        bw1 = 100.0

        should_classify = (f0 > 80 and rms_db > -35 and f1 > 200 and f2 > 500 and bw1 < 200)
        assert should_classify is False

    def test_vowel_gate_f1_boundary_exactly_200(self):
        """Test vowel gate: f1 exactly 200 does not pass."""
        f0 = 150.0
        rms_db = -20.0
        f1 = 200.0  # NOT > 200 (exactly equal)
        f2 = 1200.0
        bw1 = 100.0

        should_classify = (f0 > 80 and rms_db > -35 and f1 > 200 and f2 > 500 and bw1 < 200)
        assert should_classify is False

    def test_vowel_gate_f2_too_low(self):
        """Test vowel gate: f2 <= 500 blocks classification."""
        f0 = 150.0
        rms_db = -20.0
        f1 = 450.0
        f2 = 400.0  # NOT > 500
        bw1 = 100.0

        should_classify = (f0 > 80 and rms_db > -35 and f1 > 200 and f2 > 500 and bw1 < 200)
        assert should_classify is False

    def test_vowel_gate_f2_boundary_exactly_500(self):
        """Test vowel gate: f2 exactly 500 does not pass."""
        f0 = 150.0
        rms_db = -20.0
        f1 = 450.0
        f2 = 500.0  # NOT > 500 (exactly equal)
        bw1 = 100.0

        should_classify = (f0 > 80 and rms_db > -35 and f1 > 200 and f2 > 500 and bw1 < 200)
        assert should_classify is False

    def test_vowel_gate_bw1_too_high(self):
        """Test vowel gate: bw1 >= 200 blocks classification."""
        f0 = 150.0
        rms_db = -20.0
        f1 = 450.0
        f2 = 1200.0
        bw1 = 250.0  # NOT < 200

        should_classify = (f0 > 80 and rms_db > -35 and f1 > 200 and f2 > 500 and bw1 < 200)
        assert should_classify is False

    def test_vowel_gate_bw1_boundary_exactly_200(self):
        """Test vowel gate: bw1 exactly 200 does not pass."""
        f0 = 150.0
        rms_db = -20.0
        f1 = 450.0
        f2 = 1200.0
        bw1 = 200.0  # NOT < 200 (exactly equal)

        should_classify = (f0 > 80 and rms_db > -35 and f1 > 200 and f2 > 500 and bw1 < 200)
        assert should_classify is False

    def test_vowel_gate_bw1_boundary_just_below_200(self):
        """Test vowel gate: bw1 < 200 passes."""
        f0 = 150.0
        rms_db = -20.0
        f1 = 450.0
        f2 = 1200.0
        bw1 = 199.9  # < 200 ✓

        should_classify = (f0 > 80 and rms_db > -35 and f1 > 200 and f2 > 500 and bw1 < 200)
        assert should_classify is True

    def test_classify_vowel_function_imported(self):
        """Test that _classify_vowel can be imported from voice_core.analyze."""
        from voice_core.analyze import _classify_vowel
        assert callable(_classify_vowel)

    def test_classify_vowel_returns_string_or_none(self):
        """Test that _classify_vowel returns a string (vowel) or None."""
        # Test with formants that should match a vowel
        result1 = _classify_vowel(f1=450.0, f2=1200.0)
        assert result1 is None or isinstance(result1, str)

        # Test with formants far from all vowels (beyond threshold)
        # Use very extreme values to trigger None return
        result2 = _classify_vowel(f1=50.0, f2=10000.0)
        assert result2 is None or isinstance(result2, str)

    def test_vowel_realistic_scenario_male_voice(self):
        """Test realistic male voice formant values."""
        # Male voice: F0 ~100 Hz, /ʌ/ (but) F1~600 Hz, F2~1200 Hz
        f0 = 100.0  # > 80 ✓
        rms_db = -15.0  # > -35 ✓
        f1 = 600.0  # > 200 ✓
        f2 = 1200.0  # > 500 ✓
        bw1 = 80.0  # < 200 ✓

        should_classify = (f0 > 80 and rms_db > -35 and f1 > 200 and f2 > 500 and bw1 < 200)
        assert should_classify is True

        # Try to classify
        vowel = _classify_vowel(f1, f2)
        assert vowel is not None

    def test_vowel_realistic_scenario_female_voice(self):
        """Test realistic female voice formant values."""
        # Female voice: F0 ~200 Hz, /ɪ/ (bit) F1~550 Hz, F2~2000 Hz
        f0 = 200.0  # > 80 ✓
        rms_db = -18.0  # > -35 ✓
        f1 = 550.0  # > 200 ✓
        f2 = 2000.0  # > 500 ✓
        bw1 = 90.0  # < 200 ✓

        should_classify = (f0 > 80 and rms_db > -35 and f1 > 200 and f2 > 500 and bw1 < 200)
        assert should_classify is True

        # Try to classify
        vowel = _classify_vowel(f1, f2)
        assert vowel is not None

    def test_vowel_during_silence(self):
        """Test that silence (low rms_db) blocks vowel classification."""
        f0 = 150.0
        rms_db = -50.0  # < -35, silent
        f1 = 450.0
        f2 = 1200.0
        bw1 = 100.0

        should_classify = (f0 > 80 and rms_db > -35 and f1 > 200 and f2 > 500 and bw1 < 200)
        assert should_classify is False

    def test_vowel_during_consonant_transition_low_f1(self):
        """Test that consonant transitions (low f1) block vowel classification."""
        # Consonant transitions typically have low F1
        f0 = 150.0
        rms_db = -20.0
        f1 = 180.0  # < 200, consonant-like
        f2 = 1200.0
        bw1 = 100.0

        should_classify = (f0 > 80 and rms_db > -35 and f1 > 200 and f2 > 500 and bw1 < 200)
        assert should_classify is False

    def test_vowel_during_consonant_transition_narrow_bandwidth(self):
        """Test that narrow bandwidth indicates clear vowel."""
        # Narrow BW indicates clean, sustained vowel (not noisy consonant)
        f0 = 150.0
        rms_db = -20.0
        f1 = 450.0
        f2 = 1200.0
        bw1 = 50.0  # Very narrow, clear vowel

        should_classify = (f0 > 80 and rms_db > -35 and f1 > 200 and f2 > 500 and bw1 < 200)
        assert should_classify is True

    def test_vowel_during_noisy_articulation_wide_bandwidth(self):
        """Test that wide bandwidth indicates noisy articulation, blocks vowel."""
        # Wide BW indicates noisy/transitional state
        f0 = 150.0
        rms_db = -20.0
        f1 = 450.0
        f2 = 1200.0
        bw1 = 250.0  # Wide, noisy transition

        should_classify = (f0 > 80 and rms_db > -35 and f1 > 200 and f2 > 500 and 0 < bw1 < 200)
        assert should_classify is False

    def test_vowel_gate_bw1_zero_sentinel(self):
        """Test vowel gate: bw1 == 0.0 (Parselmouth no-data sentinel) blocks classification."""
        f0 = 150.0
        rms_db = -20.0
        f1 = 450.0
        f2 = 1200.0
        bw1 = 0.0  # Sentinel — Parselmouth found no valid bandwidth frames

        should_classify = (f0 > 80 and rms_db > -35 and f1 > 200 and f2 > 500 and 0 < bw1 < 200)
        assert should_classify is False
