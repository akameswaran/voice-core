"""Tests for falsetto slip detector in safety_monitor.py."""

from voice_core.safety_monitor import SafetyMonitor


class TestFalettoDetector:
    """Test suite for the falsetto slip detection logic."""

    def test_no_warning_when_f0_jump_below_threshold(self):
        """No warning when F0 jump < 100 Hz even if F4 drops."""
        monitor = SafetyMonitor()

        # Frame 1: establish baseline
        metrics_1 = {
            "f0_hz": 200.0,
            "f1_hz": 700.0,
            "f4_hz": 3500.0,
            "hnr_db": 20.0,
            "rms_db": -20.0,
        }
        warnings_1 = monitor.check(metrics_1)
        assert len(warnings_1) == 0

        # Frame 2: F0 jump of 80 Hz (below 100), F4 drops 250 Hz
        metrics_2 = {
            "f0_hz": 280.0,  # +80 Hz
            "f1_hz": 700.0,
            "f4_hz": 3250.0,  # -250 Hz
            "hnr_db": 20.0,
            "rms_db": -20.0,
        }
        warnings_2 = monitor.check(metrics_2)

        # Should not trigger falsetto warning (F0 jump too small)
        falsetto_warnings = [w for w in warnings_2 if w.type == "falsetto_slip"]
        assert len(falsetto_warnings) == 0

    def test_no_warning_when_f4_drop_below_threshold(self):
        """No warning when F4 drop < 200 Hz even if F0 jumps."""
        monitor = SafetyMonitor()

        # Frame 1: establish baseline
        metrics_1 = {
            "f0_hz": 200.0,
            "f1_hz": 700.0,
            "f4_hz": 3500.0,
            "hnr_db": 20.0,
            "rms_db": -20.0,
        }
        warnings_1 = monitor.check(metrics_1)
        assert len(warnings_1) == 0

        # Frame 2: F0 jump 120 Hz (above threshold), F4 drops 150 Hz (below threshold)
        metrics_2 = {
            "f0_hz": 320.0,  # +120 Hz
            "f1_hz": 700.0,
            "f4_hz": 3350.0,  # -150 Hz
            "hnr_db": 20.0,
            "rms_db": -20.0,
        }
        warnings_2 = monitor.check(metrics_2)

        # Should not trigger falsetto warning (F4 drop too small)
        falsetto_warnings = [w for w in warnings_2 if w.type == "falsetto_slip"]
        assert len(falsetto_warnings) == 0

    def test_warning_emitted_when_both_conditions_met(self):
        """Warning IS emitted when both F0 +120 Hz and F4 -250 Hz."""
        monitor = SafetyMonitor()

        # Frame 1: establish baseline
        metrics_1 = {
            "f0_hz": 200.0,
            "f1_hz": 700.0,
            "f4_hz": 3500.0,
            "hnr_db": 20.0,
            "rms_db": -20.0,
        }
        warnings_1 = monitor.check(metrics_1)
        assert len(warnings_1) == 0

        # Frame 2: F0 jump 120 Hz AND F4 drops 250 Hz
        metrics_2 = {
            "f0_hz": 320.0,  # +120 Hz
            "f1_hz": 700.0,
            "f4_hz": 3250.0,  # -250 Hz
            "hnr_db": 20.0,
            "rms_db": -20.0,
        }
        warnings_2 = monitor.check(metrics_2)

        # Should trigger falsetto warning
        falsetto_warnings = [w for w in warnings_2 if w.type == "falsetto_slip"]
        assert len(falsetto_warnings) == 1
        assert falsetto_warnings[0].level == "warning"
        assert "falsetto" in falsetto_warnings[0].message.lower()

    def test_no_warning_on_first_frame(self):
        """No warning on first frame (no prev_f0/prev_f4 yet)."""
        monitor = SafetyMonitor()

        # Frame 1: no previous state, so no comparison possible
        metrics_1 = {
            "f0_hz": 300.0,
            "f1_hz": 700.0,
            "f4_hz": 3250.0,
            "hnr_db": 20.0,
            "rms_db": -20.0,
        }
        warnings_1 = monitor.check(metrics_1)

        # Should not warn on first frame
        falsetto_warnings = [w for w in warnings_1 if w.type == "falsetto_slip"]
        assert len(falsetto_warnings) == 0

    def test_warning_message_and_type(self):
        """Verify warning type and message content."""
        monitor = SafetyMonitor()

        # Frame 1: baseline
        metrics_1 = {
            "f0_hz": 200.0,
            "f1_hz": 700.0,
            "f4_hz": 3500.0,
            "hnr_db": 20.0,
            "rms_db": -20.0,
        }
        monitor.check(metrics_1)

        # Frame 2: trigger falsetto slip
        metrics_2 = {
            "f0_hz": 330.0,  # +130 Hz
            "f1_hz": 700.0,
            "f4_hz": 3250.0,  # -250 Hz
            "hnr_db": 20.0,
            "rms_db": -20.0,
        }
        warnings_2 = monitor.check(metrics_2)

        falsetto_warnings = [w for w in warnings_2 if w.type == "falsetto_slip"]
        assert len(falsetto_warnings) == 1

        w = falsetto_warnings[0]
        assert w.type == "falsetto_slip"
        assert w.level == "warning"
        assert w.dimension == "health"
        assert "falsetto" in w.message.lower()
        assert "shift" in w.message.lower()

    def test_rate_limiting(self):
        """Verify that falsetto warnings are rate-limited."""
        monitor = SafetyMonitor()

        # Frame 1: baseline
        metrics_1 = {
            "f0_hz": 200.0,
            "f1_hz": 700.0,
            "f4_hz": 3500.0,
            "hnr_db": 20.0,
            "rms_db": -20.0,
        }
        monitor.check(metrics_1)

        # Frame 2: trigger falsetto slip (first warning)
        metrics_2 = {
            "f0_hz": 330.0,
            "f1_hz": 700.0,
            "f4_hz": 3250.0,
            "hnr_db": 20.0,
            "rms_db": -20.0,
        }
        warnings_2 = monitor.check(metrics_2)
        falsetto_warnings_2 = [w for w in warnings_2 if w.type == "falsetto_slip"]
        assert len(falsetto_warnings_2) == 1  # First warning triggers

        # Frame 3: another trigger event within rate-limit window
        metrics_3 = {
            "f0_hz": 430.0,
            "f1_hz": 700.0,
            "f4_hz": 3000.0,
            "hnr_db": 20.0,
            "rms_db": -20.0,
        }
        warnings_3 = monitor.check(metrics_3)
        falsetto_warnings_3 = [w for w in warnings_3 if w.type == "falsetto_slip"]
        # Should be rate-limited, no warning
        assert len(falsetto_warnings_3) == 0

    def test_silent_audio_skipped(self):
        """Silent audio (rms < -60 dB) should skip all checks."""
        monitor = SafetyMonitor()

        # Frame 1: silent audio
        metrics_1 = {
            "f0_hz": 200.0,
            "f1_hz": 700.0,
            "f4_hz": 3500.0,
            "hnr_db": 20.0,
            "rms_db": -80.0,  # Too quiet
        }
        warnings_1 = monitor.check(metrics_1)
        # Should return early, no history recorded
        assert len(warnings_1) == 0

        # Frame 2: loud with trigger event
        metrics_2 = {
            "f0_hz": 330.0,
            "f1_hz": 700.0,
            "f4_hz": 3250.0,
            "hnr_db": 20.0,
            "rms_db": -20.0,
        }
        warnings_2 = monitor.check(metrics_2)
        # Should not warn because frame 1 was skipped (no prev_f0/prev_f4)
        falsetto_warnings = [w for w in warnings_2 if w.type == "falsetto_slip"]
        assert len(falsetto_warnings) == 0

    def test_no_f4_in_metrics(self):
        """Gracefully handle missing f4_hz in metrics."""
        monitor = SafetyMonitor()

        # Frame 1: baseline with f4
        metrics_1 = {
            "f0_hz": 200.0,
            "f1_hz": 700.0,
            "f4_hz": 3500.0,
            "hnr_db": 20.0,
            "rms_db": -20.0,
        }
        monitor.check(metrics_1)

        # Frame 2: missing f4_hz
        metrics_2 = {
            "f0_hz": 330.0,
            "f1_hz": 700.0,
            # No f4_hz
            "hnr_db": 20.0,
            "rms_db": -20.0,
        }
        warnings_2 = monitor.check(metrics_2)

        # Should not crash, no falsetto warning
        falsetto_warnings = [w for w in warnings_2 if w.type == "falsetto_slip"]
        assert len(falsetto_warnings) == 0

    def test_reset_clears_state(self):
        """Verify reset() clears prev_f0 and prev_f4."""
        monitor = SafetyMonitor()

        # Frame 1: establish state
        metrics_1 = {
            "f0_hz": 200.0,
            "f1_hz": 700.0,
            "f4_hz": 3500.0,
            "hnr_db": 20.0,
            "rms_db": -20.0,
        }
        monitor.check(metrics_1)

        # Reset
        monitor.reset()

        # Frame 2 after reset: should not trigger warning (no prev state)
        metrics_2 = {
            "f0_hz": 330.0,
            "f1_hz": 700.0,
            "f4_hz": 3250.0,
            "hnr_db": 20.0,
            "rms_db": -20.0,
        }
        warnings_2 = monitor.check(metrics_2)

        falsetto_warnings = [w for w in warnings_2 if w.type == "falsetto_slip"]
        assert len(falsetto_warnings) == 0  # No warning due to reset

    def test_edge_case_exact_thresholds(self):
        """Test behavior at exact threshold boundaries."""
        monitor = SafetyMonitor()

        # Frame 1: baseline
        metrics_1 = {
            "f0_hz": 200.0,
            "f1_hz": 700.0,
            "f4_hz": 3500.0,
            "hnr_db": 20.0,
            "rms_db": -20.0,
        }
        monitor.check(metrics_1)

        # Frame 2: exactly at thresholds (>100 and >200, not >=)
        # F0 jump = exactly 100 Hz (should NOT trigger, needs >100)
        # F4 drop = exactly 200 Hz (should NOT trigger, needs >200)
        metrics_2 = {
            "f0_hz": 300.0,  # +100 Hz (not > 100)
            "f1_hz": 700.0,
            "f4_hz": 3300.0,  # -200 Hz (not > 200)
            "hnr_db": 20.0,
            "rms_db": -20.0,
        }
        warnings_2 = monitor.check(metrics_2)

        falsetto_warnings = [w for w in warnings_2 if w.type == "falsetto_slip"]
        assert len(falsetto_warnings) == 0  # At boundary, should not trigger

        # Frame 3: just over both thresholds (from Frame 2 baseline)
        metrics_3 = {
            "f0_hz": 300.1,  # +0.1 Hz more from Frame 2 (300.0)
            "f1_hz": 700.0,
            "f4_hz": 3299.9,  # -0.1 Hz more from Frame 2 (3300.0)
            "hnr_db": 20.0,
            "rms_db": -20.0,
        }
        warnings_3 = monitor.check(metrics_3)

        falsetto_warnings_3 = [w for w in warnings_3 if w.type == "falsetto_slip"]
        # Frame 3 is compared to Frame 2, so jump is only 0.1 Hz and drop is 0.1 Hz
        # Neither exceeds threshold, so should not trigger
        assert len(falsetto_warnings_3) == 0

        # Frame 4: now with large jump from Frame 3
        monitor2 = SafetyMonitor()
        m1 = {
            "f0_hz": 200.0,
            "f1_hz": 700.0,
            "f4_hz": 3500.0,
            "hnr_db": 20.0,
            "rms_db": -20.0,
        }
        monitor2.check(m1)

        # Frame 2 with exactly +100.1 Hz and -200.1 Hz (just over boundaries)
        m2 = {
            "f0_hz": 300.1,  # +100.1 Hz
            "f1_hz": 700.0,
            "f4_hz": 3299.9,  # -200.1 Hz
            "hnr_db": 20.0,
            "rms_db": -20.0,
        }
        warnings_m2 = monitor2.check(m2)
        falsetto_warnings_m2 = [w for w in warnings_m2 if w.type == "falsetto_slip"]
        assert len(falsetto_warnings_m2) == 1  # Just over both thresholds
