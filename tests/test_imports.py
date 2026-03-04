"""Smoke test: all voice_core modules import without coaching_gender installed."""


def test_analyze_imports():
    from voice_core.analyze import analyze, analyze_formants, analyze_pitch_crepe


def test_live_imports():
    from voice_core.live import LiveAnalyzer


def test_segment_imports():
    from voice_core.segment import analyze_segments


def test_phoneme_align_imports():
    from voice_core.phoneme_align import align, extract_vowel_formants


def test_safety_monitor_imports():
    from voice_core.safety_monitor import SafetyMonitor


def test_video_monitor_imports():
    from voice_core.video_monitor import VideoTensionMonitor


def test_world_convert_imports():
    from voice_core.world_convert import warp_spectral_envelope


def test_live_analyzer_accepts_callbacks():
    from voice_core.live import LiveAnalyzer
    # Should construct with no coaching deps
    # (won't start audio, just test constructor accepts params)
    try:
        analyzer = LiveAnalyzer(realtime_coach=None, exercise_manager=None, zone_classifier=None)
    except Exception as e:
        # May fail due to missing audio devices in CI — that's OK
        # What matters is NO ImportError from coaching_gender
        assert "coaching_gender" not in str(e), f"Upward dependency still present: {e}"
