"""DisplayPipeline: bridges raw acoustic frames to WebSocket display data.

Wraps VowelAccumulator and computes confidence-gated display values for
gesture bars, zone badges, stability rings, trend arrows, and partial scores.
Called between get_frame() and WebSocket serialization.
"""

import statistics
from typing import Callable

from voice_core.vowel_accumulator import VowelAccumulator


# Warnings that suppress partial scores
SCORE_SUPPRESSING_WARNINGS = {
    "breathiness_masking",
    "falsetto_slip",
    "false_fold_constriction",
}


def _compute_gesture_bars(
    accum_delta_f: float, accum_f1: float, accum_f2: float
) -> dict:
    """Compute gesture bar percentages from accumulated means.

    All three bars use clamped linear scaling against perceptually meaningful
    ranges. The formulas are intentionally approximate display values, not
    acoustic measurements.

    Args:
        accum_delta_f: Accumulated mean ΔF in Hz.
        accum_f1: Accumulated mean F1 in Hz.
        accum_f2: Accumulated mean F2 in Hz.

    Returns:
        {"larynx": float, "opc": float, "tongue": float} — each in [0, 100].
    """
    # Larynx height proxy: ΔF range [1029, 1207] → [50, 100]
    if accum_delta_f > 0:
        larynx = ((accum_delta_f - 1029) / (1207 - 1029)) * 50 + 50
    else:
        larynx = 50.0

    # Oral pharynx coupling (OPC): F1/ΔF ratio range [0.35, 0.65] → [0, 100]
    if accum_delta_f > 0:
        ratio = accum_f1 / accum_delta_f
        opc = ((ratio - 0.35) / 0.30) * 100
    else:
        opc = 50.0

    # Tongue advancement: F2 range [1800, 2200] → [50, 100]
    tongue = ((accum_f2 - 1800) / (2200 - 1800)) * 50 + 50

    # Clamp to [0, 100]
    larynx = max(0.0, min(100.0, larynx))
    opc = max(0.0, min(100.0, opc))
    tongue = max(0.0, min(100.0, tongue))

    return {"larynx": larynx, "opc": opc, "tongue": tongue}


def _compute_weight_factors(h1_h2: float) -> dict:
    """Compute thinness and lightness from H1-H2 corrected dB.

    Uses a linear mapping from the typical H1-H2 range [-10, +10] dB onto
    [0.0, 1.0]. Both factors use the same formula as a placeholder until
    the exact formula is specified.

    Args:
        h1_h2: Accumulated mean H1-H2 in dB.

    Returns:
        {"thinness": float, "lightness": float} — each in [0.0, 1.0].
    """
    value = min(1.0, max(0.0, (h1_h2 + 10) / 20.0))
    return {"thinness": value, "lightness": value}


def _compute_trend(f0_history: list) -> str:
    """Compute resonance trend from F0 history.

    Compares the second half mean against the first half mean. A difference
    greater than ±10 Hz constitutes an "up" or "down" trend.

    Args:
        f0_history: List of F0 values from the accumulator window.

    Returns:
        "up" | "down" | "stable"
    """
    if len(f0_history) >= 6:
        mid = len(f0_history) // 2
        first_half_mean = statistics.mean(f0_history[:mid])
        second_half_mean = statistics.mean(f0_history[mid:])
        diff = second_half_mean - first_half_mean
        if diff > 10:
            return "up"
        elif diff < -10:
            return "down"
        else:
            return "stable"
    return "stable"


def _stability_cv_to_pct(cv: float | None) -> float | None:
    """Convert coefficient of variation to a stability percentage.

    Lower CV → higher stability. A CV of 0.0 maps to 100%, and a CV of 0.10
    (10%) maps to 0%. Values above 0.10 are clamped to 0%.

    Args:
        cv: Mean F4 coefficient of variation across qualifying vowels,
            or None if insufficient data.

    Returns:
        Stability percentage in [0.0, 100.0], or None.
    """
    if cv is None:
        return None
    # Linear: stability = (1 - cv / 0.10) * 100, clamped to [0, 100]
    stability = (1.0 - cv / 0.10) * 100.0
    return max(0.0, min(100.0, stability))


class DisplayPipeline:
    """Bridges raw acoustic frames to WebSocket display payloads.

    Wraps a VowelAccumulator and computes confidence-gated display values
    including gesture bars, zone classification, stability metrics, trend
    arrows, and optional partial scores.

    Usage:
        pipeline = DisplayPipeline(zone_classifier=my_classifier)
        display_frame = pipeline.get_display_frame(raw_frame)
        # Serialize display_frame and send over WebSocket
    """

    def __init__(
        self,
        zone_classifier: Callable[[float], str] | None = None,
        window_size_s: float = 10.0,
    ):
        """Initialize the display pipeline.

        Args:
            zone_classifier: Optional callable that maps delta_f_hz (float)
                to a zone string "masc" | "andro" | "fem". If None, the zone
                is always reported as "andro".
            window_size_s: Rolling window duration for VowelAccumulator
                (default 10.0 seconds).
        """
        self._zone_classifier = zone_classifier
        self._accumulator = VowelAccumulator(window_size_s=window_size_s)
        self._phrase_boundary = False  # internal silence-detector state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_display_frame(self, raw_frame: dict) -> dict:
        """Compute display values from a raw acoustic frame.

        Feeds the frame into the internal VowelAccumulator then computes
        all display fields. Safe to call at frame rate (50–100 Hz).

        Args:
            raw_frame: Dict produced by get_frame(). Expected keys:
                - "vowel": str | None — ARPABET vowel label
                - "ts": float — timestamp in seconds
                - "f0_hz": float — fundamental frequency
                - "f0_confidence": float — pitch confidence
                - "f1": float — first formant Hz
                - "f2": float — second formant Hz
                - "f4": float — fourth formant Hz
                - "delta_f_hz": float — vocal tract length proxy
                - "h1_h2_corrected_db": float — spectral tilt
                - "rms": float — signal RMS amplitude
                - "scores": dict | None — pre-computed partial scores
                - "warnings": list[str] — active safety warnings
                - "display_mode": str — exercise display hint
                - "weight_factors": dict — passed through if present

        Returns:
            Display frame dict with all required keys for WebSocket
            serialization. See module docstring for the full schema.
        """
        ts = raw_frame.get("ts", 0.0)
        vowel = raw_frame.get("vowel")

        # --- Feed into accumulator ---
        if vowel:
            features = {
                "f1": raw_frame.get("f1_hz", 0.0),
                "f2": raw_frame.get("f2_hz", 0.0),
                "f4": raw_frame.get("f4_hz", 0.0),
                "delta_f": raw_frame.get("delta_f_hz", 0.0),
                "h1_h2": raw_frame.get("h1_h2_corrected_db", 0.0),
                "f0": raw_frame.get("f0_hz", 0.0),
            }
            self._accumulator.add(vowel, ts, features)

        # --- Silence / phrase boundary detection (rms_db is dB, threshold -40 dB) ---
        rms_db = raw_frame.get("rms_db", -60.0)
        self._phrase_boundary = rms_db < -40.0

        # --- Accumulated means (gesture bars use these) ---
        means = self._accumulator.get_accumulated_means()
        accum_delta_f = means["delta_f_hz"]
        accum_f1 = means["f1_mean"]
        accum_f2 = means["f2_mean"]
        accum_h1_h2 = means["h1_h2_mean"]

        # --- Per-frame values (no accumulation) ---
        f0_hz = raw_frame.get("f0_hz", 0.0)
        f0_confidence = raw_frame.get("f0_confidence", 0.0)
        h1_h2_corrected_db = raw_frame.get("h1_h2_corrected_db", 0.0)
        delta_f_hz_raw = raw_frame.get("delta_f_hz", 0.0)

        # --- Zone classification (from accumulated ΔF) ---
        effective_delta_f = accum_delta_f if accum_delta_f > 0 else delta_f_hz_raw
        if self._zone_classifier is not None:
            delta_f_zone = self._zone_classifier(effective_delta_f) or "andro"
        else:
            delta_f_zone = "andro"

        # --- Gesture bars (accumulated means when available, raw frame fallback) ---
        raw_f1 = raw_frame.get("f1_hz", 0.0)
        raw_f2 = raw_frame.get("f2_hz", 0.0)
        bar_delta_f = accum_delta_f if accum_delta_f > 0 else delta_f_hz_raw
        bar_f1 = accum_f1 if accum_f1 > 0 else raw_f1
        bar_f2 = accum_f2 if accum_f2 > 0 else raw_f2
        gesture_bars = _compute_gesture_bars(bar_delta_f, bar_f1, bar_f2)

        # --- Weight factors (from accumulated H1-H2) ---
        weight_factors = _compute_weight_factors(accum_h1_h2)

        # --- Stability (per-vowel F4 CV → ring fill) ---
        cv = self._accumulator.per_vowel_f4_cv()
        stability_pct = _stability_cv_to_pct(cv)

        # --- Trend computation ---
        f0_history = self._accumulator.get_f0_history()
        trend_resonance = _compute_trend(f0_history)
        # H1-H2 history not tracked in VowelAccumulator — stable placeholder
        trend_weight = "stable"

        # --- Confidence ---
        resonance_confidence = self._accumulator.resonance_confidence()
        n_voiced = len([f for f in f0_history if f > 0])
        pitch_confidence = min(1.0, n_voiced / 30.0)
        vocal_weight_confidence = pitch_confidence  # placeholder
        prosody_confidence = pitch_confidence  # placeholder
        composite_confidence = min(
            resonance_confidence,
            pitch_confidence,
            vocal_weight_confidence,
            prosody_confidence,
        )
        confidence = {
            "resonance": resonance_confidence,
            "pitch": pitch_confidence,
            "vocal_weight": vocal_weight_confidence,
            "prosody": prosody_confidence,
            "composite": composite_confidence,
        }

        # --- Score suppression ---
        # Warnings may be dicts (from live.py to_dict()) or plain strings.
        raw_warnings = raw_frame.get("warnings", []) or []
        warning_types = {
            w["type"] if isinstance(w, dict) else w
            for w in raw_warnings
            if (isinstance(w, dict) and "type" in w) or isinstance(w, str)
        }
        safety_active = bool(warning_types & SCORE_SUPPRESSING_WARNINGS)
        score_suppression_reason: str | None = None

        if composite_confidence < 0.5:
            scores = None
            score_suppression_reason = "low_confidence"
        elif safety_active:
            scores = None
            suppressed = sorted(warning_types & SCORE_SUPPRESSING_WARNINGS)
            score_suppression_reason = ",".join(suppressed)
        else:
            raw_scores = raw_frame.get("scores")
            scores = raw_scores if isinstance(raw_scores, dict) else None

        # --- Vowel counts (debug) ---
        accum_vowel_counts = self._accumulator.get_vowel_counts()

        # --- Display mode ---
        display_mode = raw_frame.get("display_mode", "default") or "default"

        return {
            # Widget values — use effective_delta_f (accumulated OR raw fallback)
            # so widgets show data immediately, not only after accumulator fills.
            "delta_f_hz": effective_delta_f,
            "delta_f_zone": delta_f_zone,
            "h1_h2_corrected_db": h1_h2_corrected_db,
            "f0_hz": f0_hz,
            "f0_confidence": f0_confidence,
            "gesture_bars": gesture_bars,
            "weight_factors": weight_factors,
            # Stability / trend
            "stability_pct": stability_pct,
            "trend_resonance": trend_resonance,
            "trend_weight": trend_weight,
            # Partial score (confidence-gated)
            "scores": scores,
            # Confidence and state
            "confidence": confidence,
            "accum_vowel_counts": accum_vowel_counts,
            "display_mode": display_mode,
            "score_suppression_reason": score_suppression_reason,
        }
