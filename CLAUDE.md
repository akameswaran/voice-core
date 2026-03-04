# voice-core — Shared Acoustic Analysis Engine

Domain-neutral acoustic analysis, real-time audio processing, and vocal safety
monitoring. Open source. Used as a dependency by voice coaching apps.

## Key principle

This package contains ZERO domain-specific logic. No gender thresholds, no
Spanish phoneme targets, no singing pitch accuracy rules. It provides primitives
that apps compose into coaching systems.

## Modules

- `analyze.py` — Parselmouth formant extraction, CREPE pitch, spectral moments, HNR, jitter, shimmer, CPP, H1-H2, ΔF, gesture z-scores, sibilant centroid, speech rate, vowel classification.
- `live.py` — Real-time audio capture via sounddevice. Accepts optional coach/exercise/zone callbacks via dependency injection.
- `segment.py` — Silence detection, per-segment analysis. Accepts optional score_fn.
- `phoneme_align.py` — Montreal Forced Aligner wrapper.
- `safety_monitor.py` — Vocal health: constriction, breathiness, fatigue.
- `video_monitor.py` — MediaPipe tension monitoring.
- `world_convert.py` — WORLD vocoder wrapper.
- `data/vowel_norms.json` — Per-vowel F1/F2/F3 population norms (not gendered).

## Dependency injection pattern

```python
from voice_core.live import LiveAnalyzer
analyzer = LiveAnalyzer(
    realtime_coach=MyCoach(),
    exercise_manager=MyExerciseManager(),
    zone_classifier=my_zone_classifier,
)
```

If callbacks are None, features are silently disabled.
