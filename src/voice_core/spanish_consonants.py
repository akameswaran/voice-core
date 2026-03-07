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


def classify_tap_r(y: np.ndarray, sr: int,
                   min_duration: float = 0.03,
                   closure_max_ms: float = 40.0,
                   closure_threshold_db: float = -20.0) -> dict:
    """Classify an r-position segment as tap /ɾ/ or approximant /ɹ/.

    Spanish tap-r has a brief (~20-30ms) amplitude dip from tongue closure.
    English approximant-r has continuous sound with no closure.

    Args:
        y: Audio segment containing the r production.
        sr: Sample rate.
        min_duration: Minimum segment duration in seconds.
        closure_max_ms: Maximum closure duration to count as tap (ms).
        closure_threshold_db: dB threshold below peak for closure detection.

    Returns:
        Dict with {classification: "tap"|"approximant"|"unknown",
                    has_closure: bool, closure_duration_ms: float,
                    confidence: float}
    """
    if len(y) / sr < min_duration:
        return {"classification": "unknown", "has_closure": False,
                "closure_duration_ms": 0.0, "confidence": 0.0}

    # Compute amplitude envelope (RMS in short windows)
    frame_length = int(0.005 * sr)  # 5ms frames
    hop = frame_length // 2
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]

    if len(rms) < 3:
        return {"classification": "unknown", "has_closure": False,
                "closure_duration_ms": 0.0, "confidence": 0.0}

    # Convert to dB
    rms_db = 20 * np.log10(rms + 1e-10)
    peak_db = np.max(rms_db)

    # Find frames below closure threshold (relative to peak)
    closure_frames = rms_db < (peak_db + closure_threshold_db)

    # Find contiguous closure regions
    has_closure = False
    closure_ms = 0.0

    if np.any(closure_frames):
        # Find longest contiguous run of closure frames
        diffs = np.diff(closure_frames.astype(int))
        starts = np.where(diffs == 1)[0] + 1
        ends = np.where(diffs == -1)[0] + 1

        # Handle edge cases
        if closure_frames[0]:
            starts = np.concatenate([[0], starts])
        if closure_frames[-1]:
            ends = np.concatenate([ends, [len(closure_frames)]])

        if len(starts) > 0 and len(ends) > 0:
            lengths = ends[:len(starts)] - starts[:len(ends)]
            if len(lengths) > 0:
                longest = np.max(lengths)
                closure_ms = float(longest * hop / sr * 1000)
                has_closure = 5.0 < closure_ms < closure_max_ms

    if has_closure:
        classification = "tap"
        confidence = min(1.0, 0.5 + closure_ms / 40.0)
    else:
        classification = "approximant"
        rms_cv = float(np.std(rms) / (np.mean(rms) + 1e-10))
        confidence = min(1.0, max(0.5, 1.0 - rms_cv))

    return {
        "classification": classification,
        "has_closure": has_closure,
        "closure_duration_ms": round(closure_ms, 1),
        "confidence": round(confidence, 3),
    }
