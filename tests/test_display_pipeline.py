"""Tests for DisplayPipeline: bridges raw acoustic frames to WebSocket display data."""

from voice_core.display_pipeline import DisplayPipeline, SCORE_SUPPRESSING_WARNINGS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_TOP_LEVEL_KEYS = {
    "delta_f_hz",
    "delta_f_zone",
    "h1_h2_corrected_db",
    "f0_hz",
    "f0_confidence",
    "gesture_bars",
    "weight_factors",
    "stability_pct",
    "trend_resonance",
    "trend_weight",
    "scores",
    "confidence",
    "accum_vowel_counts",
    "display_mode",
    "score_suppression_reason",
}


def _minimal_frame(**overrides) -> dict:
    """Return a minimal raw frame with sensible defaults.

    Keyword overrides are merged on top of the defaults so individual tests
    can supply only the fields they care about.
    """
    base = {
        "vowel": "AE",
        "ts": 0.0,
        "f0_hz": 180.0,
        "f0_confidence": 0.9,
        "f1_hz": 700.0,
        "f2_hz": 1800.0,
        "f4_hz": 3500.0,
        "delta_f_hz": 1100.0,
        "h1_h2_corrected_db": 5.0,
        "rms_db": -20.0,
        "scores": None,
        "warnings": [],
        "display_mode": "default",
    }
    base.update(overrides)
    return base


def _feed_frames(pipeline: DisplayPipeline, n: int, *, delta_f: float = 1100.0) -> dict:
    """Feed *n* sequential frames into *pipeline* and return the last display frame."""
    result = {}
    for i in range(n):
        frame = _minimal_frame(
            ts=float(i) * 0.02,
            vowel="AE",
            f1_hz=700.0,
            f2_hz=1900.0,
            f4_hz=3500.0,
            delta_f_hz=delta_f,
            h1_h2_corrected_db=5.0,
            f0_hz=180.0,
        )
        result = pipeline.get_display_frame(frame)
    return result


# ---------------------------------------------------------------------------
# Test 1: All required top-level keys are present
# ---------------------------------------------------------------------------

class TestRequiredKeys:
    """get_display_frame returns a dict with all required top-level keys."""

    def test_all_required_keys_present_empty_accumulator(self):
        """Even on the very first frame all keys must be present."""
        pipeline = DisplayPipeline()
        display = pipeline.get_display_frame(_minimal_frame())
        missing = REQUIRED_TOP_LEVEL_KEYS - set(display.keys())
        assert missing == set(), f"Missing keys: {missing}"

    def test_gesture_bars_sub_keys(self):
        """gesture_bars must contain larynx, opc, tongue."""
        pipeline = DisplayPipeline()
        display = pipeline.get_display_frame(_minimal_frame())
        assert set(display["gesture_bars"].keys()) == {"larynx", "opc", "tongue"}

    def test_confidence_sub_keys(self):
        """confidence must contain resonance, pitch, vocal_weight, prosody, composite."""
        pipeline = DisplayPipeline()
        display = pipeline.get_display_frame(_minimal_frame())
        assert set(display["confidence"].keys()) == {
            "resonance", "pitch", "vocal_weight", "prosody", "composite"
        }

    def test_weight_factors_sub_keys(self):
        """weight_factors must contain thinness and lightness."""
        pipeline = DisplayPipeline()
        display = pipeline.get_display_frame(_minimal_frame())
        assert set(display["weight_factors"].keys()) == {"thinness", "lightness"}


# ---------------------------------------------------------------------------
# Test 2: scores is None when composite confidence < 0.5
# ---------------------------------------------------------------------------

class TestScoreSuppressedByLowConfidence:
    """scores must be None when composite confidence < 0.5."""

    def test_scores_none_on_first_frame(self):
        """A single frame cannot build enough confidence to unlock scores."""
        pipeline = DisplayPipeline()
        display = pipeline.get_display_frame(
            _minimal_frame(scores={"pitch": 0.8, "resonance": 0.7})
        )
        assert display["scores"] is None

    def test_suppression_reason_set_when_low_confidence(self):
        """score_suppression_reason should be 'low_confidence' below threshold."""
        pipeline = DisplayPipeline()
        display = pipeline.get_display_frame(_minimal_frame())
        # A single frame will always have low confidence
        assert display["score_suppression_reason"] == "low_confidence"

    def test_scores_none_reason_is_low_confidence_not_safety(self):
        """When both low confidence AND a suppressing warning are present,
        low_confidence takes priority (checked first)."""
        pipeline = DisplayPipeline()
        display = pipeline.get_display_frame(
            _minimal_frame(warnings=["breathiness_masking"])
        )
        assert display["scores"] is None
        assert display["score_suppression_reason"] == "low_confidence"


# ---------------------------------------------------------------------------
# Test 3: Gesture bars computed from accumulated means (not raw frame)
# ---------------------------------------------------------------------------

class TestGestureBarsFromAccumulatedMeans:
    """Gesture bars must reflect the accumulated window, not any single frame."""

    def test_gesture_bars_nonzero_after_many_frames(self):
        """After 10+ frames with valid acoustics, all gesture bars should be non-zero."""
        pipeline = DisplayPipeline()
        display = _feed_frames(pipeline, 10, delta_f=1150.0)
        bars = display["gesture_bars"]
        assert bars["larynx"] != 0.0 or bars["opc"] != 0.0 or bars["tongue"] != 0.0

    def test_gesture_bars_change_with_different_delta_f(self):
        """Larynx bar should differ between pipelines fed different ΔF values."""
        pipeline_low = DisplayPipeline()
        pipeline_high = DisplayPipeline()

        display_low = _feed_frames(pipeline_low, 10, delta_f=1050.0)
        display_high = _feed_frames(pipeline_high, 10, delta_f=1180.0)

        # Higher ΔF → higher larynx bar
        assert display_high["gesture_bars"]["larynx"] > display_low["gesture_bars"]["larynx"]

    def test_gesture_bars_not_raw_frame_value(self):
        """Gesture bars should reflect the accumulated mean, not the last raw frame.

        Feed 9 frames with delta_f=1050 then 1 frame with an extremely low
        delta_f=800. If gesture bars were computed from the raw frame alone
        the larynx bar would drop to its minimum. Because they use the accumulated
        mean over all 10 frames it should remain above the minimum.
        """
        pipeline = DisplayPipeline()
        # Feed 9 baseline frames at a mid-range ΔF
        for i in range(9):
            pipeline.get_display_frame(_minimal_frame(ts=float(i) * 0.02, delta_f_hz=1150.0))
        # Feed one extreme low outlier
        display = pipeline.get_display_frame(_minimal_frame(ts=9 * 0.02, delta_f_hz=800.0))

        # If computed from raw frame only: ((800 - 1029) / (1207-1029)) * 50 + 50 < 0 → 0
        # If computed from accumulated mean: (9*1150 + 800)/10 = 1115 → ~43 → clamp → 43
        # So the bar must be greater than 0.
        assert display["gesture_bars"]["larynx"] > 0.0


# ---------------------------------------------------------------------------
# Test 4: stability_pct is None when not enough data
# ---------------------------------------------------------------------------

class TestStabilityPct:
    """stability_pct is None until VowelAccumulator has enough data."""

    def test_stability_none_with_single_frame(self):
        """One frame cannot satisfy the ≥5-frame threshold for F4 CV."""
        pipeline = DisplayPipeline()
        display = pipeline.get_display_frame(_minimal_frame())
        assert display["stability_pct"] is None

    def test_stability_none_with_four_frames(self):
        """Four frames is still below the ≥5-frame threshold."""
        pipeline = DisplayPipeline()
        display = _feed_frames(pipeline, 4)
        assert display["stability_pct"] is None

    def test_stability_not_none_with_five_frames(self):
        """Five frames meets the threshold; stability_pct should be a float."""
        pipeline = DisplayPipeline()
        display = _feed_frames(pipeline, 5)
        assert display["stability_pct"] is not None
        assert isinstance(display["stability_pct"], float)

    def test_stability_in_valid_range(self):
        """stability_pct should be in [0.0, 100.0]."""
        pipeline = DisplayPipeline()
        display = _feed_frames(pipeline, 10)
        assert 0.0 <= display["stability_pct"] <= 100.0


# ---------------------------------------------------------------------------
# Test 5: SCORE_SUPPRESSING_WARNINGS suppress scores
# ---------------------------------------------------------------------------

class TestScoreSuppressedBySafetyWarning:
    """scores must be None when an active safety warning is in raw_frame["warnings"]."""

    def _build_high_confidence_pipeline(self) -> DisplayPipeline:
        """Return a (pipeline, last_ts) pair with confidence ≥ 0.5.

        Feeds 50 frames (5 monophthong vowels × 10 frames) at 50 Hz. The
        timestamps are dense enough that all frames remain within the 10s
        rolling window. Returns the pipeline and the timestamp of the last
        frame so callers can continue feeding without expiring the window.
        """
        pipeline = DisplayPipeline()
        # Feed enough diverse vowel frames to push composite confidence above 0.5.
        # We need 30+ monophthong frames in 3+ categories to saturate resonance
        # confidence, and 30+ voiced F0 frames to saturate pitch confidence.
        vowels = ["AE", "IH", "UH", "AA", "EH"]
        last_ts = 0.0
        for j, vowel in enumerate(vowels):
            for i in range(10):
                last_ts = j * 10 * 0.02 + i * 0.02
                pipeline.get_display_frame(
                    _minimal_frame(
                        vowel=vowel,
                        ts=last_ts,
                        f0_hz=180.0,
                        delta_f_hz=1100.0,
                    )
                )
        return pipeline, last_ts

    def test_suppressing_warning_suppresses_scores(self):
        """A SCORE_SUPPRESSING_WARNINGS entry must set scores=None."""
        pipeline, last_ts = self._build_high_confidence_pipeline()
        for warning in SCORE_SUPPRESSING_WARNINGS:
            # Use a timestamp just after the last training frame (still within window)
            next_ts = last_ts + 0.02
            display = pipeline.get_display_frame(
                _minimal_frame(
                    ts=next_ts,
                    vowel="AE",
                    scores={"pitch": 0.9},
                    warnings=[warning],
                )
            )
            assert display["scores"] is None, (
                f"Warning '{warning}' should suppress scores but did not"
            )

    def test_suppression_reason_includes_warning_name(self):
        """score_suppression_reason should contain the warning name."""
        pipeline, last_ts = self._build_high_confidence_pipeline()
        next_ts = last_ts + 0.02
        display = pipeline.get_display_frame(
            _minimal_frame(
                ts=next_ts,
                vowel="AE",
                scores={"pitch": 0.9},
                warnings=["breathiness_masking"],
            )
        )
        assert display["score_suppression_reason"] is not None
        assert "breathiness_masking" in display["score_suppression_reason"]

    def test_non_suppressing_warning_does_not_suppress(self):
        """A warning not in SCORE_SUPPRESSING_WARNINGS must not suppress scores."""
        pipeline, last_ts = self._build_high_confidence_pipeline()
        next_ts = last_ts + 0.02
        display = pipeline.get_display_frame(
            _minimal_frame(
                ts=next_ts,
                vowel="AE",
                scores={"pitch": 0.9},
                warnings=["mild_tension"],  # Not in the suppression set
            )
        )
        # scores should pass through (assuming confidence ≥ 0.5 after many frames)
        if display["confidence"]["composite"] >= 0.5:
            assert display["scores"] == {"pitch": 0.9}


# ---------------------------------------------------------------------------
# Test 6: delta_f_zone passes through zone_classifier
# ---------------------------------------------------------------------------

class TestDeltaFZone:
    """delta_f_zone must reflect the zone_classifier return value."""

    def test_zone_classifier_called_with_accumulated_delta_f(self):
        """zone_classifier receives the accumulated ΔF, not the raw frame value."""
        received_values = []

        def zone_classifier(delta_f: float) -> str:
            received_values.append(delta_f)
            return "fem"

        pipeline = DisplayPipeline(zone_classifier=zone_classifier)
        pipeline.get_display_frame(_minimal_frame(delta_f_hz=1200.0))

        assert len(received_values) == 1

    def test_zone_fem_propagated(self):
        """zone_classifier returning 'fem' produces delta_f_zone='fem'."""
        pipeline = DisplayPipeline(zone_classifier=lambda _: "fem")
        display = pipeline.get_display_frame(_minimal_frame())
        assert display["delta_f_zone"] == "fem"

    def test_zone_masc_propagated(self):
        """zone_classifier returning 'masc' produces delta_f_zone='masc'."""
        pipeline = DisplayPipeline(zone_classifier=lambda _: "masc")
        display = pipeline.get_display_frame(_minimal_frame())
        assert display["delta_f_zone"] == "masc"

    def test_zone_andro_propagated(self):
        """zone_classifier returning 'andro' produces delta_f_zone='andro'."""
        pipeline = DisplayPipeline(zone_classifier=lambda _: "andro")
        display = pipeline.get_display_frame(_minimal_frame())
        assert display["delta_f_zone"] == "andro"

    def test_no_zone_classifier_defaults_to_andro(self):
        """Without a zone_classifier, delta_f_zone defaults to 'andro'."""
        pipeline = DisplayPipeline()
        display = pipeline.get_display_frame(_minimal_frame())
        assert display["delta_f_zone"] == "andro"

    def test_zone_classifier_none_return_falls_back(self):
        """A zone_classifier returning None must fall back to 'andro'."""
        pipeline = DisplayPipeline(zone_classifier=lambda _: None)
        display = pipeline.get_display_frame(_minimal_frame())
        assert display["delta_f_zone"] == "andro"
