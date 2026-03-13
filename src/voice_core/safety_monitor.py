"""Real-time vocal safety monitor — detects strain, constriction, and fatigue.

Provides both:
1. Real-time workers (HNR, jitter/shimmer) that plug into LiveAnalyzer
2. SafetyMonitor class that evaluates the combined metric snapshot and
   produces constructive warnings with remediation exercises.

Safety checks (from realtime-analysis-swarm-prompt.md Task 5):
- Constriction: high F1 + low HNR + elevated jitter → throat tension
- Breathiness masking: H1-H2 very high + low HNR → fake breathiness
- Hypernasality: spectral anomalies between F1 and F2 (basic heuristic)
- Session fatigue: HNR decline / jitter climb over 15+ minutes
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Warning data structures
# ---------------------------------------------------------------------------

@dataclass
class SafetyWarning:
    """A single safety/coaching warning."""
    level: str          # "info", "coaching", "warning", "alert"
    type: str           # e.g. "constriction", "breathiness_masking", "fatigue"
    message: str        # Human-readable constructive feedback
    timestamp: float = field(default_factory=time.time)
    dimension: str = "" # "resonance", "weight", "health", or ""

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "type": self.type,
            "message": self.message,
            "timestamp": self.timestamp,
            "dimension": self.dimension,
        }


# ---------------------------------------------------------------------------
# Thresholds — tuned for voice feminization context
# ---------------------------------------------------------------------------

# HNR thresholds
HNR_HEALTHY = 20.0        # Above this = clean phonation
HNR_CONCERN = 16.0        # Below this + high H1-H2 = breathiness flag
HNR_WARNING = 12.0        # Below this = possible strain
HNR_ALERT = 7.0           # Below this = significant vocal strain

# Jitter/shimmer thresholds (as percentages)
JITTER_CONCERN = 1.5      # Starting to show instability
JITTER_WARNING = 2.5      # Likely strain or fatigue
SHIMMER_CONCERN = 5.0
SHIMMER_WARNING = 8.0

# Breathiness masking
H1_H2_BREATHY = 8.0       # H1-H2 > this with low HNR = breathy masking
H1_H2_LIGHT = 6.0         # H1-H2 in 2-6 with good HNR = healthy light weight
H1_H2_PRESSED = 0.0       # H1-H2 < 0 = pressed/heavy phonation

# Constriction (combined thresholds)
F1_HIGH_THRESH = 800.0    # F1 above this during exercise = possibly squeezed
CONSTRICTION_HNR = 12.0   # HNR below this with high F1 = constriction flag

# Fatigue detection
FATIGUE_WINDOW_MINUTES = 15.0
FATIGUE_HNR_DROP = 3.0    # dB decline over window
FATIGUE_JITTER_RISE = 1.0 # percentage point rise over window

# Rate limiting
MIN_WARNING_INTERVAL = 10.0  # seconds between warnings of same type


# ---------------------------------------------------------------------------
# Real-time HNR worker (plugs into LiveAnalyzer)
# ---------------------------------------------------------------------------

def hnr_worker_fn(live_analyzer):
    """Thread worker: compute HNR via Parselmouth at ~10 Hz.

    Call this from LiveAnalyzer._start_workers() as a daemon thread.
    Updates live_analyzer.latest["hnr_db"].
    """
    import parselmouth
    from parselmouth.praat import call

    while live_analyzer._running:
        try:
            # 200ms window for more stable HNR
            n_samples = int(live_analyzer.sr * 0.2)
            chunk = live_analyzer.ring.read_last(n_samples)

            if np.max(np.abs(chunk)) < 1e-5:
                time.sleep(0.01)
                continue

            snd = parselmouth.Sound(chunk, sampling_frequency=live_analyzer.sr)
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)

            if not np.isnan(hnr):
                with live_analyzer._lock:
                    live_analyzer.latest["hnr_db"] = float(hnr)

        except Exception:
            pass

        time.sleep(0.05)  # ~20 Hz max, CPU-bound


def jitter_shimmer_worker_fn(live_analyzer):
    """Thread worker: compute jitter and shimmer via Parselmouth at ~5 Hz.

    Uses PointProcess for cycle-to-cycle perturbation measurement.
    Updates live_analyzer.latest["jitter_pct"] and ["shimmer_pct"].
    """
    import parselmouth
    from parselmouth.praat import call

    while live_analyzer._running:
        try:
            # 300ms window — need several pitch periods
            n_samples = int(live_analyzer.sr * 0.3)
            chunk = live_analyzer.ring.read_last(n_samples)

            if np.max(np.abs(chunk)) < 1e-5:
                time.sleep(0.01)
                continue

            snd = parselmouth.Sound(chunk, sampling_frequency=live_analyzer.sr)
            point_process = call(snd, "To PointProcess (periodic, cc)", 75.0, 600.0)

            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call(
                [snd, point_process], "Get shimmer (local)",
                0, 0, 0.0001, 0.02, 1.3, 1.6
            )

            jitter_pct = float(jitter * 100) if not np.isnan(jitter) else 0.0
            shimmer_pct = float(shimmer * 100) if not np.isnan(shimmer) else 0.0

            with live_analyzer._lock:
                live_analyzer.latest["jitter_pct"] = jitter_pct
                live_analyzer.latest["shimmer_pct"] = shimmer_pct

        except Exception:
            pass

        time.sleep(0.1)  # ~10 Hz max, relatively expensive


# ---------------------------------------------------------------------------
# SafetyMonitor — evaluates metric snapshots and produces warnings
# ---------------------------------------------------------------------------

class SafetyMonitor:
    """Evaluates real-time metrics for vocal safety issues.

    Tracks metric history for fatigue detection and rate-limits warnings.
    """

    def __init__(self, session_start: Optional[float] = None):
        self.session_start = session_start or time.time()

        # Metric history for fatigue detection (store (timestamp, value) tuples)
        self._hnr_history: deque = deque(maxlen=1000)   # ~15 min at 1 Hz
        self._jitter_history: deque = deque(maxlen=1000)
        self._shimmer_history: deque = deque(maxlen=1000)

        # Rate limiting: last warning time per type
        self._last_warning_time: dict[str, float] = {}

        # Active warnings (cleared when condition resolves)
        self._active_warnings: dict[str, SafetyWarning] = {}

        # Falsetto detector state
        self._prev_f0: Optional[float] = None
        self._prev_f4: Optional[float] = None

    @property
    def session_duration_minutes(self) -> float:
        return (time.time() - self.session_start) / 60.0

    def check(self, metrics: dict) -> list[SafetyWarning]:
        """Evaluate a metric snapshot and return any new warnings.

        Args:
            metrics: Dict with keys like f0_hz, f1_hz, delta_f_hz,
                     h1_h2_db, hnr_db, jitter_pct, shimmer_pct, rms_db.
                     Missing keys are handled gracefully.

        Returns:
            List of SafetyWarning objects (may be empty).
        """
        now = time.time()
        warnings = []

        hnr = metrics.get("hnr_db", None)
        h1_h2 = metrics.get("h1_h2_db", None)
        f1 = metrics.get("f1_hz", 0.0)
        jitter = metrics.get("jitter_pct", None)
        shimmer = metrics.get("shimmer_pct", None)
        f0 = metrics.get("f0_hz", 0.0)
        rms = metrics.get("rms_db", -200.0)

        # Skip all checks if no voiced audio detected — prevents
        # false alarms on silence, startup, and background noise
        if f0 <= 0 or rms < -60:
            return warnings

        # Record history for fatigue detection
        if hnr is not None and hnr != 0:
            self._hnr_history.append((now, hnr))
        if jitter is not None and jitter != 0:
            self._jitter_history.append((now, jitter))
        if shimmer is not None and shimmer != 0:
            self._shimmer_history.append((now, shimmer))

        # --- Check 1: Constriction ---
        if hnr is not None and f1 > 0:
            w = self._check_constriction(f1, hnr, jitter or 0, shimmer or 0)
            if w:
                warnings.append(w)

        # --- Check 2: Breathiness masking ---
        if h1_h2 is not None and hnr is not None:
            w = self._check_breathiness_masking(h1_h2, hnr)
            if w:
                warnings.append(w)

        # --- Check 3: Severe vocal strain ---
        if hnr is not None:
            w = self._check_strain(hnr, jitter or 0, shimmer or 0)
            if w:
                warnings.append(w)

        # --- Check 4: Session fatigue ---
        if self.session_duration_minutes >= FATIGUE_WINDOW_MINUTES:
            w = self._check_fatigue()
            if w:
                warnings.append(w)

        # --- Check 5: F0/formant interference ---
        if f0 > 0 and f1 > 0:
            w = self._check_f0_interference(f0, f1)
            if w:
                warnings.append(w)

        # --- Check 6: Falsetto slip detector ---
        f4 = metrics.get("f4_hz", None)
        if f4 is not None and f0 > 0:
            w = self._check_falsetto_slip(f0, f4)
            if w:
                warnings.append(w)

        # Update previous frame state
        if f0 > 0:
            self._prev_f0 = f0
        if f4 is not None:
            self._prev_f4 = f4

        return warnings

    def _rate_limited(self, warning_type: str) -> bool:
        """Check if we've recently issued this warning type."""
        last = self._last_warning_time.get(warning_type, 0)
        if time.time() - last < MIN_WARNING_INTERVAL:
            return True
        return False

    def _emit(self, level: str, wtype: str, message: str,
              dimension: str = "") -> SafetyWarning:
        """Create a warning and update rate-limiting state."""
        self._last_warning_time[wtype] = time.time()
        w = SafetyWarning(
            level=level, type=wtype, message=message, dimension=dimension
        )
        self._active_warnings[wtype] = w
        return w

    # --- Individual checks ---

    def _check_constriction(self, f1: float, hnr: float,
                            jitter: float, shimmer: float
                            ) -> Optional[SafetyWarning]:
        """High F1 + low HNR + elevated jitter = throat squeezing."""
        if self._rate_limited("constriction"):
            return None

        if f1 > F1_HIGH_THRESH and hnr < CONSTRICTION_HNR:
            if jitter > JITTER_CONCERN or shimmer > SHIMMER_CONCERN:
                return self._emit(
                    "warning", "constriction",
                    "Your voice sounds strained — the brightness may be from "
                    "throat tension rather than resonance shaping. Try relaxing "
                    "your throat and using the whisper siren to find brightness "
                    "without phonation first.",
                    dimension="health"
                )
            elif hnr < HNR_WARNING:
                return self._emit(
                    "warning", "constriction",
                    "High F1 with low harmonics quality — possible constriction. "
                    "Try backing off the brightness slightly and focusing on "
                    "oral cavity shaping (tongue forward, gentle smile) instead "
                    "of throat tension.",
                    dimension="health"
                )
        return None

    def _check_breathiness_masking(self, h1_h2: float, hnr: float
                                   ) -> Optional[SafetyWarning]:
        """Very high H1-H2 + low HNR = using breathiness as a crutch."""
        if self._rate_limited("breathiness_masking"):
            return None

        if h1_h2 > H1_H2_BREATHY and hnr < HNR_CONCERN:
            return self._emit(
                "coaching", "breathiness_masking",
                "You're reducing fold closure to sound lighter, but this is "
                "tiring and unsustainable. Try maintaining full voice contact "
                "while thinning the sound — think 'clear and light' not 'airy'.",
                dimension="weight"
            )
        return None

    def _check_strain(self, hnr: float, jitter: float, shimmer: float
                      ) -> Optional[SafetyWarning]:
        """Severe vocal strain detection."""
        if self._rate_limited("vocal_strain"):
            return None

        if hnr < HNR_ALERT:
            return self._emit(
                "alert", "vocal_strain",
                "High vocal strain detected. Please take a break and do some "
                "gentle humming to relax your voice. If this persists, consider "
                "consulting a speech-language pathologist.",
                dimension="health"
            )

        if jitter > JITTER_WARNING and shimmer > SHIMMER_WARNING:
            return self._emit(
                "warning", "vocal_strain",
                "Your voice is showing instability (jitter and shimmer are "
                "elevated). This usually means fatigue — take a short break "
                "with some gentle lip trills or humming.",
                dimension="health"
            )

        return None

    def _check_fatigue(self) -> Optional[SafetyWarning]:
        """Detect gradual vocal fatigue over the session."""
        if self._rate_limited("session_fatigue"):
            return None

        now = time.time()
        window_start = now - (FATIGUE_WINDOW_MINUTES * 60)

        # Check HNR trend
        recent_hnr = [v for t, v in self._hnr_history if t >= window_start]
        if len(recent_hnr) >= 20:
            first_half = np.mean(recent_hnr[:len(recent_hnr) // 2])
            second_half = np.mean(recent_hnr[len(recent_hnr) // 2:])
            hnr_drop = first_half - second_half

            if hnr_drop >= FATIGUE_HNR_DROP:
                return self._emit(
                    "coaching", "session_fatigue",
                    f"Your voice quality has been declining over the past "
                    f"{int(FATIGUE_WINDOW_MINUTES)} minutes (HNR dropped "
                    f"~{hnr_drop:.0f} dB). Take a 5-minute break — do some "
                    f"gentle humming, drink water, and stretch your neck.",
                    dimension="health"
                )

        # Check jitter trend
        recent_jitter = [v for t, v in self._jitter_history if t >= window_start]
        if len(recent_jitter) >= 20:
            first_half = np.mean(recent_jitter[:len(recent_jitter) // 2])
            second_half = np.mean(recent_jitter[len(recent_jitter) // 2:])
            jitter_rise = second_half - first_half

            if jitter_rise >= FATIGUE_JITTER_RISE:
                return self._emit(
                    "coaching", "session_fatigue",
                    "Your vocal stability is decreasing — signs of fatigue. "
                    "Take a break, hydrate, and do gentle warm-down exercises "
                    "(lip trills, soft humming).",
                    dimension="health"
                )

        return None

    def _check_f0_interference(self, f0: float, f1: float
                               ) -> Optional[SafetyWarning]:
        """Warn when F0 is high enough to compromise formant tracking."""
        if self._rate_limited("f0_interference"):
            return None

        ratio = f0 / f1 if f1 > 0 else 0
        if ratio > 0.8:
            return self._emit(
                "info", "f0_interference",
                "Your pitch is high enough that formant tracking is less "
                "reliable. For more accurate resonance feedback, try this "
                "exercise at a slightly lower pitch.",
                dimension="resonance"
            )
        elif f0 > 300 and ratio > 0.5:
            # Only warn once — less urgent than full interference
            if "f0_interference" not in self._active_warnings:
                return self._emit(
                    "info", "f0_interference",
                    "Pitch is above 300 Hz — formant measurements have "
                    "reduced accuracy. ΔF readings may be less stable.",
                    dimension="resonance"
                )
        return None

    def _check_falsetto_slip(self, f0: float, f4: float
                             ) -> Optional[SafetyWarning]:
        """Detect falsetto slip: sudden upward F0 jump with concurrent F4 drop."""
        if self._rate_limited("falsetto_slip"):
            return None

        # Need previous frame data to detect sudden jumps
        if self._prev_f0 is None or self._prev_f4 is None:
            return None

        f0_jump = f0 - self._prev_f0
        f4_drop = self._prev_f4 - f4

        # Both conditions must be met: F0 up >100 Hz AND F4 down >200 Hz
        if f0_jump > 100 and f4_drop > 200:
            return self._emit(
                "warning", "falsetto_slip",
                "Sudden falsetto shift detected. Your pitch jumped while your "
                "resonance dropped — this suggests unintended register shift. "
                "Try centering your voice and maintaining steady breath support.",
                dimension="health"
            )

        return None

    def get_status(self) -> dict:
        """Return current safety status summary."""
        return {
            "session_minutes": round(self.session_duration_minutes, 1),
            "active_warnings": [
                w.to_dict() for w in self._active_warnings.values()
            ],
            "hnr_samples": len(self._hnr_history),
            "jitter_samples": len(self._jitter_history),
        }

    def clear_warning(self, warning_type: str):
        """Clear an active warning (e.g., when user acknowledges it)."""
        self._active_warnings.pop(warning_type, None)

    def reset(self):
        """Reset all state for a new session."""
        self.session_start = time.time()
        self._hnr_history.clear()
        self._jitter_history.clear()
        self._shimmer_history.clear()
        self._last_warning_time.clear()
        self._active_warnings.clear()
        self._prev_f0 = None
        self._prev_f4 = None
