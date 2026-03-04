"""Video-based tension monitoring — evaluates body posture features for strain alerts.

Receives feature vectors from browser-side MediaPipe (face/pose landmarks)
and generates tension alerts matching the SafetyWarning shape used elsewhere.

Features expected from video_features.js:
- shoulder_elevation_delta: normalized upward shift from baseline
- shoulder_asymmetry: normalized abs(left.y - right.y)
- forward_head_offset: normalized ear-shoulder x displacement
- tension_composite: weighted combination (0-1)
- face_detected, pose_detected: booleans
"""

import time
from collections import deque
from dataclasses import dataclass


@dataclass
class VideoAlert:
    """A tension/posture alert from video analysis."""
    level: str          # "info" | "coaching" | "warning"
    type: str           # alert identifier
    message: str        # coaching text
    dimension: str      # always "tension" for video alerts
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "type": self.type,
            "message": self.message,
            "dimension": self.dimension,
            "timestamp": self.timestamp,
        }


# Thresholds
_ELEVATION_COACHING = 0.15      # shoulder rise (normalized) triggers coaching
_ASYMMETRY_INFO = 0.12          # shoulder asymmetry triggers info
_FORWARD_HEAD_COACHING = 0.15   # forward head offset triggers coaching
_TENSION_WARNING = 0.7          # composite tension triggers warning
_TREND_RISE_THRESHOLD = 0.2     # tension increase over 5 min window


class VideoTensionMonitor:
    """Evaluates video features for tension alerts."""

    def __init__(self):
        self._baseline: dict | None = None
        self._history: deque[dict] = deque(maxlen=900)  # ~1 min at 15 Hz
        self._long_history: deque[dict] = deque(maxlen=4500)  # ~5 min at 15 Hz
        self._last_alert_time: dict[str, float] = {}
        self._alert_cooldown = 15.0  # seconds between same alert type

    def set_baseline(self, features: dict):
        """Set the session baseline for comparison."""
        self._baseline = dict(features)

    def check(self, features: dict) -> list[VideoAlert]:
        """Evaluate features and return any triggered alerts."""
        if not features.get("pose_detected"):
            return []

        self._history.append({**features, "_ts": time.time()})
        self._long_history.append({**features, "_ts": time.time()})

        now = time.time()
        alerts = []

        # Shoulder elevation rising
        elevation = abs(features.get("shoulder_elevation_delta", 0))
        if elevation > _ELEVATION_COACHING:
            alert = self._maybe_alert(
                "shoulder_elevation", "coaching", now,
                "Shoulders creeping up — drop them down and breathe. "
                "Tension in the shoulders often couples with throat tension."
            )
            if alert:
                alerts.append(alert)

        # Shoulder asymmetry
        asymmetry = features.get("shoulder_asymmetry", 0)
        if asymmetry > _ASYMMETRY_INFO:
            alert = self._maybe_alert(
                "shoulder_asymmetry", "info", now,
                "One shoulder higher than the other — try to level out. "
                "Asymmetric posture can create uneven tension."
            )
            if alert:
                alerts.append(alert)

        # Forward head posture
        forward = abs(features.get("forward_head_offset", 0))
        if forward > _FORWARD_HEAD_COACHING:
            alert = self._maybe_alert(
                "forward_head", "coaching", now,
                "Head drifting forward — bring your ears back over your shoulders. "
                "Forward head posture strains the neck and can affect voice production."
            )
            if alert:
                alerts.append(alert)

        # Tension composite high
        tension = features.get("tension_composite", 0)
        if tension > _TENSION_WARNING:
            alert = self._maybe_alert(
                "tension_high", "warning", now,
                "Overall body tension is high — take a moment to relax. "
                "Drop shoulders, unclench jaw, let your arms hang loose."
            )
            if alert:
                alerts.append(alert)

        # Tension trend rising over 5 min
        if len(self._long_history) > 300:  # at least ~20s of data
            trend_alert = self._check_tension_trend(now)
            if trend_alert:
                alerts.append(trend_alert)

        return alerts

    def _check_tension_trend(self, now: float) -> VideoAlert | None:
        """Check if tension has been trending upward over the last 5 minutes."""
        history = list(self._long_history)
        if len(history) < 60:
            return None

        # Compare first quarter vs last quarter
        quarter = len(history) // 4
        early = [h.get("tension_composite", 0) for h in history[:quarter]]
        late = [h.get("tension_composite", 0) for h in history[-quarter:]]

        if not early or not late:
            return None

        early_avg = sum(early) / len(early)
        late_avg = sum(late) / len(late)
        rise = late_avg - early_avg

        if rise > _TREND_RISE_THRESHOLD:
            return self._maybe_alert(
                "tension_trend", "coaching", now,
                "Tension has been gradually climbing over the session — "
                "good time to take a break and reset your posture."
            )

        return None

    def _maybe_alert(self, alert_type: str, level: str, now: float,
                     message: str) -> VideoAlert | None:
        """Create alert if not in cooldown."""
        last = self._last_alert_time.get(alert_type, 0)
        if now - last < self._alert_cooldown:
            return None

        self._last_alert_time[alert_type] = now
        return VideoAlert(
            level=level,
            type=alert_type,
            message=message,
            dimension="tension",
            timestamp=now,
        )

    def reset(self):
        """Reset state for a new session."""
        self._baseline = None
        self._history.clear()
        self._long_history.clear()
        self._last_alert_time.clear()
