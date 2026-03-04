# voice-core

Domain-neutral acoustic analysis and real-time voice processing engine.

Provides pitch tracking (CREPE), formant extraction (Parselmouth), spectral analysis,
vocal safety monitoring, and real-time audio capture — without any domain-specific
coaching logic.

## Installation

```bash
pip install -e .
```

## Usage

```python
from voice_core.analyze import analyze

# Analyze a WAV file
result = analyze("recording.wav")

# Real-time analysis with dependency injection
from voice_core.live import LiveAnalyzer

analyzer = LiveAnalyzer(
    realtime_coach=my_coach,        # optional
    exercise_manager=my_exercises,  # optional
    zone_classifier=my_classifier,  # optional
)
analyzer.start()
frame = analyzer.get_frame()
analyzer.stop()
```

## License

MIT
