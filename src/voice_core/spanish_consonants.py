"""Rioplatense Spanish consonant analysis.

Classifies productions of consonants that distinguish Rioplatense from
other Spanish dialects: sheísmo (LL/Y -> /ʃ/), tap-r vs approximant-r,
and s-aspiration.
"""

import numpy as np
import librosa


def classify_sheismo(y: np.ndarray, sr: int,
                     min_duration: float = 0.03) -> dict:
    """Classify an LL/Y segment as sheísmo (/ʃ/) or yeísmo (/j/).

    Sheísmo: LL/Y -> /ʃ/ (Rioplatense fricative, energy at 2.5-4 kHz)
    Yeísmo: LL/Y -> /j/ (glide, low-frequency formant structure)

    Args:
        y: Audio segment containing the LL/Y production.
        sr: Sample rate.
        min_duration: Minimum segment duration in seconds.

    Returns:
        Dict with {classification: "sheismo"|"yeismo"|"unknown",
                    confidence: 0-1, spectral_centroid_hz: float,
                    high_freq_ratio: float}
    """
    if len(y) / sr < min_duration:
        return {"classification": "unknown", "confidence": 0.0,
                "spectral_centroid_hz": 0.0, "high_freq_ratio": 0.0}

    n_fft = min(2048, len(y))
    hop_length = n_fft // 4

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Energy in bands
    high_mask = freqs >= 2500
    low_mask = (freqs >= 100) & (freqs < 2500)

    high_energy = np.sum(S[high_mask, :] ** 2)
    low_energy = np.sum(S[low_mask, :] ** 2)
    total_energy = high_energy + low_energy + 1e-10

    high_freq_ratio = float(high_energy / total_energy)

    # Spectral centroid of the full segment
    centroid_frames = librosa.feature.spectral_centroid(
        S=S, freq=freqs)[0]
    centroid_hz = float(np.mean(centroid_frames)) if len(centroid_frames) > 0 else 0.0

    # Classification logic:
    # /ʃ/ (sheismo): high_freq_ratio > 0.4, centroid > 2500 Hz
    # /j/ (yeismo): low-freq dominant, centroid < 1500 Hz
    if high_freq_ratio > 0.4 and centroid_hz > 2500:
        classification = "sheismo"
        confidence = min(1.0, high_freq_ratio * 1.5)
    elif high_freq_ratio < 0.2 or centroid_hz < 1500:
        classification = "yeismo"
        confidence = min(1.0, (1.0 - high_freq_ratio) * 1.2)
    else:
        classification = "intermediate"
        confidence = 0.5

    return {
        "classification": classification,
        "confidence": round(confidence, 3),
        "spectral_centroid_hz": round(centroid_hz, 1),
        "high_freq_ratio": round(high_freq_ratio, 3),
    }
