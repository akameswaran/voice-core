"""Rolling per-vowel bucket with 10s sliding window for real-time analysis.

VowelAccumulator maintains a rolling 10-second window of vowel frames and computes
statistics for resonance feedback, gesture visualization, and confidence metrics.
"""

import statistics
from collections import defaultdict


MONOPHTHONG_VOWELS = {"AO", "AH", "AA", "EH", "ER", "IH", "IY", "AE", "OW", "UH"}
DIPHTHONG_VOWELS = {"AY", "EY", "OY", "UW"}


class VowelAccumulator:
    """Rolling per-vowel bucket with 10s sliding window.

    Stores vowel frames (vowel label + acoustic features) in a rolling window,
    automatically expiring entries older than 10 seconds. Computes per-vowel
    statistics for F1, F2, F4 and confidence metrics.
    """

    def __init__(self, window_size_s: float = 10.0):
        """Initialize the accumulator.

        Args:
            window_size_s: Size of the rolling window in seconds (default 10.0).
        """
        self.window_size_s = window_size_s
        self._frames = []  # List of (ts, vowel, features) tuples

    def add(self, vowel: str, ts: float, features: dict) -> None:
        """Append a frame to the bucket and expire old entries.

        Args:
            vowel: Vowel label (e.g., "AE", "IH").
            ts: Timestamp in seconds.
            features: Dict with keys: f1, f2, f4 (formant frequencies in Hz).
                Can also contain: delta_f, f0, h1_h2 for optional features.
        """
        # Add the frame
        self._frames.append((ts, vowel, features))

        # Expire frames older than window_size_s
        cutoff_ts = ts - self.window_size_s
        self._frames = [(t, v, f) for t, v, f in self._frames if t >= cutoff_ts]

    def get_f4_scoring_stats(self) -> dict:
        """Return per-vowel stats for monophthongs only (for resonance score input).

        Returns:
            Dict mapping vowel -> {"f4_mean": float, "f4_sd": float,
                "f1_mean": float, "f2_mean": float, "n_frames": int}
        """
        vowel_frames = defaultdict(list)

        for _, vowel, features in self._frames:
            if vowel in MONOPHTHONG_VOWELS:
                vowel_frames[vowel].append(features)

        result = {}
        for vowel, frames_list in vowel_frames.items():
            if frames_list:
                f4_vals = [f.get("f4", 0) for f in frames_list]
                f1_vals = [f.get("f1", 0) for f in frames_list]
                f2_vals = [f.get("f2", 0) for f in frames_list]

                f4_mean = statistics.mean(f4_vals) if f4_vals else 0.0
                f4_sd = statistics.stdev(f4_vals) if len(f4_vals) > 1 else 0.0
                f1_mean = statistics.mean(f1_vals) if f1_vals else 0.0
                f2_mean = statistics.mean(f2_vals) if f2_vals else 0.0

                result[vowel] = {
                    "f4_mean": f4_mean,
                    "f4_sd": f4_sd,
                    "f1_mean": f1_mean,
                    "f2_mean": f2_mean,
                    "n_frames": len(frames_list),
                }

        return result

    def get_all_stats(self) -> dict:
        """Return per-vowel stats for all vowels (for ΔF, gesture bars, confidence).

        Returns:
            Dict mapping vowel -> {"f4_mean": float, "f4_sd": float,
                "f1_mean": float, "f2_mean": float, "n_frames": int}
        """
        vowel_frames = defaultdict(list)

        for _, vowel, features in self._frames:
            vowel_frames[vowel].append(features)

        result = {}
        for vowel, frames_list in vowel_frames.items():
            if frames_list:
                f4_vals = [f.get("f4", 0) for f in frames_list]
                f1_vals = [f.get("f1", 0) for f in frames_list]
                f2_vals = [f.get("f2", 0) for f in frames_list]

                f4_mean = statistics.mean(f4_vals) if f4_vals else 0.0
                f4_sd = statistics.stdev(f4_vals) if len(f4_vals) > 1 else 0.0
                f1_mean = statistics.mean(f1_vals) if f1_vals else 0.0
                f2_mean = statistics.mean(f2_vals) if f2_vals else 0.0

                result[vowel] = {
                    "f4_mean": f4_mean,
                    "f4_sd": f4_sd,
                    "f1_mean": f1_mean,
                    "f2_mean": f2_mean,
                    "n_frames": len(frames_list),
                }

        return result

    def resonance_confidence(self) -> float:
        """Compute geometric mean of frame and vowel variety confidence.

        frame_conf = min(1.0, monophthong_frames / 30.0)
        vowel_variety_conf = min(1.0, mono_vowel_categories_with_3plus_frames / 3.0)
        return (frame_conf * vowel_variety_conf) ** 0.5

        Returns:
            Confidence score in [0, 1].
        """
        # Count monophthong frames
        mono_frames = sum(1 for _, vowel, _ in self._frames
                         if vowel in MONOPHTHONG_VOWELS)
        frame_conf = min(1.0, mono_frames / 30.0)

        # Count monophthong categories with 3+ frames
        vowel_frames = defaultdict(int)
        for _, vowel, _ in self._frames:
            if vowel in MONOPHTHONG_VOWELS:
                vowel_frames[vowel] += 1

        vowel_cats_with_3plus = sum(1 for count in vowel_frames.values() if count >= 3)
        vowel_variety_conf = min(1.0, vowel_cats_with_3plus / 3.0)

        return (frame_conf * vowel_variety_conf) ** 0.5

    def per_vowel_f4_cv(self) -> float | None:
        """Compute mean F4 CV across monophthongs with ≥5 frames.

        CV = coefficient of variation = SD / mean

        Returns:
            Mean F4 CV across qualifying vowels, or None if no vowels qualify.
        """
        stats = self.get_f4_scoring_stats()

        cvs = []
        for _, stat_dict in stats.items():
            if stat_dict["n_frames"] >= 5:
                f4_mean = stat_dict["f4_mean"]
                f4_sd = stat_dict["f4_sd"]
                if f4_mean > 0:  # Avoid division by zero
                    cv = f4_sd / f4_mean
                    cvs.append(cv)

        if cvs:
            return statistics.mean(cvs)
        return None

    def get_accumulated_means(self) -> dict:
        """Compute overall means across all frames in the window.

        Returns:
            {"delta_f_hz": float, "f1_mean": float, "f2_mean": float, "h1_h2_mean": float}
        """
        if not self._frames:
            return {
                "delta_f_hz": 0.0,
                "f1_mean": 0.0,
                "f2_mean": 0.0,
                "h1_h2_mean": 0.0,
            }

        delta_f_vals = []
        f1_vals = []
        f2_vals = []
        h1_h2_vals = []

        for _, _, features in self._frames:
            # delta_f: expected to be populated by live.py's formant worker (Task: vowel classifier hook).
            # Do NOT use F4 as a proxy—F4 is a raw formant (~3500 Hz) while delta_f is a derived
            # vocal tract metric (~1100-1400 Hz), and substituting one would produce wrong gesture values.
            if "delta_f" in features:
                delta_f_vals.append(features["delta_f"])

            if "f1" in features:
                f1_vals.append(features["f1"])
            if "f2" in features:
                f2_vals.append(features["f2"])
            if "h1_h2" in features:
                h1_h2_vals.append(features["h1_h2"])

        return {
            "delta_f_hz": statistics.mean(delta_f_vals) if delta_f_vals else 0.0,
            "f1_mean": statistics.mean(f1_vals) if f1_vals else 0.0,
            "f2_mean": statistics.mean(f2_vals) if f2_vals else 0.0,
            "h1_h2_mean": statistics.mean(h1_h2_vals) if h1_h2_vals else 0.0,
        }

    def get_f0_history(self) -> list[float]:
        """Return list of F0 values from features over the window.

        Returns:
            List of F0 values in Hz (empty if no F0 data).
        """
        f0_vals = []
        for _, _, features in self._frames:
            if "f0" in features:
                f0_vals.append(features["f0"])
        return f0_vals

    def get_vowel_counts(self) -> dict:
        """Return {vowel: n_frames} for all vowels in current window.

        Returns:
            Dict mapping vowel label to frame count.
        """
        counts = defaultdict(int)
        for _, vowel, _ in self._frames:
            counts[vowel] += 1
        return dict(counts)
