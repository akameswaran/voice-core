"""Core acoustic analysis module.

Produces comprehensive acoustic analysis across pitch, formants/resonance,
voice quality, articulation, and prosody. Domain-agnostic — coaches provide
interpretation and scoring.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call

warnings.filterwarnings("ignore", category=UserWarning)

# Speed of sound for vocal tract length estimation (cm/s)
SPEED_OF_SOUND_CM = 34300

# ──────────────────────────────────────────────────────────
# Vowel norms for phoneme-aware gesture z-scoring
# ──────────────────────────────────────────────────────────
_VOWEL_NORMS_PATH = Path(__file__).parent / "data" / "vowel_norms.json"
_VOWEL_NORMS = None  # Lazy loaded


def _get_vowel_norms():
    """Lazy-load vowel norms from reference/vowel_norms.json."""
    global _VOWEL_NORMS
    if _VOWEL_NORMS is None:
        with open(_VOWEL_NORMS_PATH) as f:
            _VOWEL_NORMS = json.load(f)["vowels"]
    return _VOWEL_NORMS


# ──────────────────────────────────────────────────────────
# Hillenbrand (1995) female speaker F1 norms for OPC detection
# Source: Hillenbrand, Getty, Clark & Wheeler (1995) JASA 97(5):3099-3111
# ──────────────────────────────────────────────────────────
_HILLENBRAND_FEMALE_F1 = {
    "ih": {"mean": 483, "std": 56},   # /ɪ/ (bit)
    "uh": {"mean": 753, "std": 83},   # /ʌ/ (but)
    "oo": {"mean": 459, "std": 78},   # /u/ (boot)
    "ah": {"mean": 936, "std": 104},  # /ɑ/ (hot)
    "ee": {"mean": 437, "std": 62},   # /i/ (beat)
    "eh": {"mean": 731, "std": 82},   # /ɛ/ (bet)
    "aw": {"mean": 781, "std": 98},   # /ɔ/ (caught)
}

# ARPABET (from _classify_vowel) → Hillenbrand lowercase label
# Only the 7 OPC target vowels are mapped; others return None
_ARPABET_TO_HILLENBRAND = {
    "IH": "ih",
    "AH": "uh",  # maps both /ʌ/ (but) and /ə/ (schwa); schwa F1 ~500 Hz may slightly inflate uh z-scores
    "UW": "oo",
    "AA": "ah",
    "IY": "ee",
    "EH": "eh",
    "AO": "aw",
}


def _classify_vowel(f1: float, f2: float, norms: dict | None = None) -> str | None:
    """Classify a formant frame into the nearest vowel category.

    Uses Euclidean distance in F1×F2 space (Hz-normalized) against vowel
    centroids from reference/vowel_norms.json. Returns vowel key (e.g. "AA")
    or None if the frame is too far from any vowel (likely a consonant).
    """
    if norms is None:
        norms = _get_vowel_norms()
    # Normalize F1 and F2 by typical range to give equal weight
    # F1 range ~250-900 Hz, F2 range ~800-2500 Hz
    F1_SCALE = 500.0
    F2_SCALE = 1000.0

    best_vowel = None
    best_dist = float("inf")

    for vowel, ref in norms.items():
        d = ((f1 - ref["f1_mean"]) / F1_SCALE) ** 2 + \
            ((f2 - ref["f2_mean"]) / F2_SCALE) ** 2
        if d < best_dist:
            best_dist = d
            best_vowel = vowel

    # Reject if too far from any vowel (consonant frames, silence, etc.)
    # Threshold: ~1.5 normalized units → ~750 Hz F1 or ~1500 Hz F2 off
    if best_dist > 2.25:
        return None
    return best_vowel


def _compute_gesture_zscores(f1_vals: list, f2_vals: list, f3_vals: list, norms: dict | None = None) -> dict:
    """Compute phoneme-aware gesture z-scores from per-frame formant values.

    For each frame, detect which vowel is being produced (from F1+F2 position),
    then compute z-scores against that vowel's reference norms. Average across
    all vowel frames to get overall gesture scores.

    Returns dict with:
        f1_zscore: F1 deviation from per-vowel norm (positive = above norm)
        f2_zscore: F2 deviation from per-vowel norm (positive = above norm)
        f3_zscore: F3 deviation from per-vowel norm (positive = above norm)
        n_vowel_frames: how many frames were classified as vowels
        vowel_distribution: count per detected vowel
    """
    if norms is None:
        norms = _get_vowel_norms()

    f1_zscores = []
    f2_zscores = []
    f3_zscores = []
    vowel_counts = {}

    n = min(len(f1_vals), len(f2_vals), len(f3_vals))
    for i in range(n):
        f1, f2, f3 = f1_vals[i], f2_vals[i], f3_vals[i]
        if f1 <= 0 or f2 <= 0 or f3 <= 0:
            continue

        vowel = _classify_vowel(f1, f2, norms=norms)
        if vowel is None:
            continue

        ref = norms[vowel]
        vowel_counts[vowel] = vowel_counts.get(vowel, 0) + 1

        # Z-score: (measured - population_mean) / population_std
        # Positive z-score = higher than average = more feminine direction
        if ref["f1_std"] > 0:
            f1_zscores.append((f1 - ref["f1_mean"]) / ref["f1_std"])
        if ref["f2_std"] > 0:
            f2_zscores.append((f2 - ref["f2_mean"]) / ref["f2_std"])
        if ref["f3_std"] > 0:
            f3_zscores.append((f3 - ref["f3_mean"]) / ref["f3_std"])

    return {
        "f1_zscore": float(np.mean(f1_zscores)) if f1_zscores else 0.0,
        "f2_zscore": float(np.mean(f2_zscores)) if f2_zscores else 0.0,
        "f3_zscore": float(np.mean(f3_zscores)) if f3_zscores else 0.0,
        "n_vowel_frames": sum(vowel_counts.values()),
        "vowel_distribution": vowel_counts,
    }


def _compute_per_vowel_zscores(
    f1_vals, f2_vals, f3_vals, f4_vals,
    population_norms: dict | None = None,
    vowel_mapping: dict | None = None,
) -> dict:
    """Compute per-vowel F1 z-scores against Hillenbrand (1995) female norms.

    For each frame, classifies the vowel from (F1, F2), then buckets F1 and F4
    values by the detected vowel. Only the 7 OPC-relevant vowels are tracked
    (ih, uh, oo, ah, ee, eh, aw).

    Args:
        f1_vals: Per-frame F1 frequencies (Hz). Frame-aligned with f2/f3/f4.
        f2_vals: Per-frame F2 frequencies (Hz).
        f3_vals: Per-frame F3 frequencies (Hz, unused but kept for signature consistency).
        f4_vals: Per-frame F4 frequencies (Hz). 0 = unavailable for that frame.

    Returns:
        Dict keyed by Hillenbrand vowel label, e.g.:
        {"ih": {"f1_zscore": 3.7, "f1_mean_hz": 849.0, "f4_mean_hz": 3200.0,
                "n_frames": 15}, ...}
        Only vowels with >= 1 classified frame are included.
    """
    if population_norms is None:
        population_norms = _HILLENBRAND_FEMALE_F1
    if vowel_mapping is None:
        vowel_mapping = _ARPABET_TO_HILLENBRAND
    n = min(len(f1_vals), len(f2_vals))
    f3_len = len(f3_vals)
    f4_len = len(f4_vals)

    vowel_f1: dict[str, list] = {}
    vowel_f2: dict[str, list] = {}
    vowel_f3: dict[str, list] = {}
    vowel_f4: dict[str, list] = {}

    for i in range(n):
        f1, f2 = f1_vals[i], f2_vals[i]
        if f1 <= 0 or f2 <= 0:
            continue

        arpabet = _classify_vowel(f1, f2)
        if arpabet is None:
            continue

        label = vowel_mapping.get(arpabet)
        if label is None:
            continue  # Vowel not in the 7 target vowels

        vowel_f1.setdefault(label, []).append(f1)
        vowel_f2.setdefault(label, []).append(f2)

        if i < f3_len and f3_vals[i] > 0:
            vowel_f3.setdefault(label, []).append(f3_vals[i])
        if i < f4_len and f4_vals[i] > 0:
            vowel_f4.setdefault(label, []).append(f4_vals[i])

    result = {}
    for label, f1_list in vowel_f1.items():
        f1_mean = float(np.mean(f1_list))
        pnorm = population_norms[label]
        z = (f1_mean - pnorm["mean"]) / pnorm["std"]
        f2_mean = float(np.mean(vowel_f2[label])) if vowel_f2.get(label) else 0.0
        f3_mean = float(np.mean(vowel_f3[label])) if vowel_f3.get(label) else 0.0
        f4_mean = float(np.mean(vowel_f4[label])) if vowel_f4.get(label) else 0.0

        entry: dict = {
            "f1_zscore": round(z, 3),
            "f1_mean_hz": round(f1_mean, 1),
            "f4_mean_hz": round(f4_mean, 1),
            "n_frames": len(f1_list),
        }
        # Per-vowel ΔF (requires at least 3 valid formants)
        df = _compute_delta_f(f1_mean, f2_mean, f3_mean, f4_mean)
        if df > 0:
            entry["delta_f_hz"] = round(df, 1)

        result[label] = entry

    return result


def _compute_per_vowel_source(
    f1_vals: list, f2_vals: list,
    h1_h2_per_frame: list[tuple[float, float, float]],
    snd: parselmouth.Sound | None = None,
    frame_step: float = 0.02,
) -> dict:
    """Bucket per-frame H1-H2, spectral tilt, and CPP by vowel identity.

    Args:
        f1_vals: Per-frame F1 frequencies (Hz) from formant extraction.
        f2_vals: Per-frame F2 frequencies (Hz).
        h1_h2_per_frame: List of (time_s, h1_h2_raw, h1_h2_corrected) from
            _estimate_h1_h2(return_per_frame=True).
        snd: Parselmouth Sound object for per-vowel tilt/CPP. Optional.
        frame_step: Time step between formant frames (s). Default 0.02.

    Returns:
        Dict keyed by Hillenbrand vowel label:
        {"ih": {"h1_h2_raw_db": 3.2, "h1_h2_corrected_db": 2.8,
                "spectral_tilt_db_oct": -5.1, "cpp_db": 14.2,
                "n_frames": 45}, ...}
    """
    if not h1_h2_per_frame or not f1_vals or not f2_vals:
        return {}

    # Build time→vowel_label lookup from formant frames
    n_formant = min(len(f1_vals), len(f2_vals))
    time_to_vowel: dict[int, str] = {}  # frame_index → vowel_label
    for i in range(n_formant):
        f1, f2 = f1_vals[i], f2_vals[i]
        if f1 <= 0 or f2 <= 0:
            continue
        arpabet = _classify_vowel(f1, f2)
        if arpabet is None:
            continue
        label = _ARPABET_TO_HILLENBRAND.get(arpabet)
        if label is not None:
            time_to_vowel[i] = label

    # Bucket H1-H2 per-frame values by vowel
    vowel_raw: dict[str, list] = {}
    vowel_corrected: dict[str, list] = {}
    vowel_times: dict[str, list] = {}  # frame times for tilt/CPP extraction

    for t, raw, corrected in h1_h2_per_frame:
        # Map H1-H2 frame time to nearest formant frame index
        idx = int(round(t / frame_step))
        label = time_to_vowel.get(idx)
        if label is None:
            continue
        vowel_raw.setdefault(label, []).append(raw)
        vowel_corrected.setdefault(label, []).append(corrected)
        vowel_times.setdefault(label, []).append(t)

    # Compute per-vowel spectral tilt and CPP if Sound is provided
    vowel_tilt: dict[str, float] = {}
    vowel_cpp: dict[str, float | None] = {}
    if snd is not None:
        sr = int(snd.sampling_frequency)
        y = snd.values[0]
        half_frame = int(frame_step * sr / 2)

        for label, times in vowel_times.items():
            # Per-vowel spectral tilt: extract frames, average power spectrum, fit
            frames_audio = []
            for t in times:
                center = int(t * sr)
                start = max(0, center - half_frame)
                end = min(len(y), center + half_frame)
                if end > start:
                    frames_audio.append(y[start:end])

            if frames_audio:
                # Spectral tilt from averaged power spectrum
                n_fft = 2048
                avg_spectrum = np.zeros(n_fft // 2 + 1)
                for chunk in frames_audio:
                    if len(chunk) < n_fft:
                        chunk = np.pad(chunk, (0, n_fft - len(chunk)))
                    S = np.abs(np.fft.rfft(chunk[:n_fft] * np.hanning(min(len(chunk), n_fft)))) ** 2
                    avg_spectrum[:len(S)] += S
                avg_spectrum /= len(frames_audio)

                freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
                mask = (freqs >= 100) & (freqs <= 5000) & (avg_spectrum > 0)
                if np.sum(mask) >= 5:
                    log_f = np.log2(freqs[mask])
                    log_p = 10 * np.log10(avg_spectrum[mask] + 1e-20)
                    coeffs = np.polyfit(log_f, log_p, 1)
                    vowel_tilt[label] = round(float(coeffs[0]), 2)

            # Per-vowel CPP: need >= 10 frames (200ms) for reliable cepstral analysis
            if len(times) >= 10:
                segments = []
                for t in sorted(times):
                    center = int(t * sr)
                    start = max(0, center - half_frame)
                    end = min(len(y), center + half_frame)
                    if end > start:
                        segments.append(y[start:end])
                if segments:
                    concat = np.concatenate(segments)
                    temp_snd = parselmouth.Sound(concat, sampling_frequency=sr)
                    vowel_cpp[label] = round(_estimate_cpp(temp_snd), 2)

    result = {}
    for label in vowel_raw:
        raw_list = vowel_raw[label]
        corr_list = vowel_corrected.get(label, [])
        entry = {
            "h1_h2_raw_db": round(float(np.mean(raw_list)), 2),
            "h1_h2_corrected_db": round(float(np.mean(corr_list)), 2) if corr_list else 0.0,
            "n_frames": len(raw_list),
        }
        if label in vowel_tilt:
            entry["spectral_tilt_db_oct"] = vowel_tilt[label]
        if label in vowel_cpp:
            entry["cpp_db"] = vowel_cpp[label]
        result[label] = entry

    return result


def _load_audio(wav_path: str) -> tuple[np.ndarray, int]:
    """Load audio file, convert to mono float32, peak-normalize to -3 dBFS."""
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    y = y.astype(np.float32)
    peak = np.max(np.abs(y))
    if peak > 0 and peak < 0.5:
        # Normalize quiet recordings to -3 dBFS so formant/pitch analysis works well
        target = 10 ** (-3 / 20)  # ~0.708
        y = y * (target / peak)
    return y, sr


def _resample(y: np.ndarray, sr_orig: int, sr_target: int) -> np.ndarray:
    """Resample audio if needed."""
    if sr_orig == sr_target:
        return y
    return librosa.resample(y, orig_sr=sr_orig, target_sr=sr_target)


def analyze_pitch_crepe(y: np.ndarray, sr: int, device: str = "cuda:0") -> dict:
    """Analyze pitch using CREPE on GPU.

    Tries torchcrepe (PyTorch, GPU-native) first, falls back to crepe (TensorFlow),
    then to Parselmouth if neither is available.
    """
    # CREPE expects 16kHz mono audio
    y_16k = _resample(y, sr, 16000)

    try:
        # Prefer torchcrepe (PyTorch-native, explicit GPU control)
        import torch
        import torchcrepe

        audio_tensor = torch.tensor(y_16k, dtype=torch.float32).unsqueeze(0).to(device)
        frequency, confidence = torchcrepe.predict(
            audio_tensor, 16000,
            hop_length=160,  # 10ms at 16kHz
            fmin=50.0,
            fmax=550.0,
            model="full",
            decoder=torchcrepe.decode.viterbi,
            return_periodicity=True,
            batch_size=1024,
            device=device,
        )
        frequency = frequency.squeeze(0).cpu().numpy()
        confidence = confidence.squeeze(0).cpu().numpy()
        time_arr = np.arange(len(frequency)) * 0.01  # 10ms steps
    except ImportError:
        try:
            # Fall back to crepe (TensorFlow-based, auto GPU)
            import crepe
            time_arr, frequency, confidence, _ = crepe.predict(
                y_16k, 16000, model_capacity="full",
                viterbi=True, step_size=10,
            )
        except (ImportError, ModuleNotFoundError):
            # Final fallback: Parselmouth pitch tracking
            snd_16k = parselmouth.Sound(y_16k, sampling_frequency=16000)
            pitch_obj = call(snd_16k, "To Pitch (ac)", 0.0, 50.0, 15, False, 0.03, 0.45, 0.01, 0.35, 0.14, 550.0)
            n_frames = call(pitch_obj, "Get number of frames")
            frequency, confidence, time_arr = [], [], []
            for i in range(1, n_frames + 1):
                t = call(pitch_obj, "Get time from frame number", i)
                f0 = call(pitch_obj, "Get value in frame", i, "Hertz")
                time_arr.append(t)
                if np.isnan(f0) or f0 == 0:
                    frequency.append(0.0)
                    confidence.append(0.0)
                else:
                    frequency.append(f0)
                    confidence.append(1.0)
            frequency = np.array(frequency)
            confidence = np.array(confidence)
            time_arr = np.array(time_arr)

    # Filter by confidence threshold
    mask = confidence > 0.5
    voiced_f0 = frequency[mask]

    if len(voiced_f0) == 0:
        return {
            "f0_mean_hz": 0.0,
            "f0_median_hz": 0.0,
            "f0_std_hz": 0.0,
            "f0_min_hz": 0.0,
            "f0_max_hz": 0.0,
            "f0_contour_hz": [],
            "f0_contour_time_s": [],
            "f0_confidence": [],
            "voiced_fraction": 0.0,
        }

    return {
        "f0_mean_hz": float(np.mean(voiced_f0)),
        "f0_median_hz": float(np.median(voiced_f0)),
        "f0_std_hz": float(np.std(voiced_f0)),
        "f0_min_hz": float(np.min(voiced_f0)),
        "f0_max_hz": float(np.max(voiced_f0)),
        "f0_contour_hz": frequency.tolist(),
        "f0_contour_time_s": time_arr.tolist(),
        "f0_confidence": confidence.tolist(),
        "voiced_fraction": float(mask.sum() / len(mask)),
    }


def _score_formant_track(formant, snd: parselmouth.Sound,
                         ceiling_hz: float = 5500) -> float:
    """Score formant track plausibility for ceiling selection.

    Evaluates:
    - Frame-to-frame smoothness of each formant track
    - F1 < F2 < F3 < F4 ordering consistency
    - Bandwidth reasonableness (BW > 0 and BW < 500 Hz)
    - Penalizes NaN/0 gaps in tracks
    - Penalizes F4 squeezed against the ceiling (sign ceiling is too low)

    Returns a float score, higher = better.
    """
    n_frames = call(formant, "Get number of frames")
    if n_frames < 2:
        return 0.0

    # Collect per-frame formant values and bandwidths
    frame_data = []  # list of (f1, f2, f3, f4) per frame
    frame_bws = []   # list of (bw1, bw2, bw3) per frame
    for i in range(1, n_frames + 1):
        t = call(formant, "Get time from frame number", i)
        fs = []
        bws = []
        for n in range(1, 5):
            f = call(formant, "Get value at time", n, t, "Hertz", "Linear")
            fs.append(f if not np.isnan(f) else 0.0)
        for n in range(1, 4):
            bw = call(formant, "Get bandwidth at time", n, t, "Hertz", "Linear")
            bws.append(bw if not np.isnan(bw) else 0.0)
        frame_data.append(tuple(fs))
        frame_bws.append(tuple(bws))

    score = 0.0
    total_checks = 0

    # 1. Smoothness: low variance between adjacent frames for each formant
    for formant_idx in range(4):
        vals = [fd[formant_idx] for fd in frame_data if fd[formant_idx] > 0]
        if len(vals) >= 2:
            diffs = np.abs(np.diff(vals))
            # Reward low frame-to-frame jumps relative to the formant frequency
            mean_val = np.mean(vals)
            if mean_val > 0:
                smoothness = 1.0 - min(1.0, np.mean(diffs) / (mean_val * 0.1))
                score += max(0.0, smoothness)
            total_checks += 1

    # 2. F1 < F2 < F3 < F4 ordering consistency
    ordering_ok = 0
    ordering_total = 0
    for fd in frame_data:
        if all(f > 0 for f in fd):
            ordering_total += 1
            if fd[0] < fd[1] < fd[2] < fd[3]:
                ordering_ok += 1
    if ordering_total > 0:
        score += 2.0 * (ordering_ok / ordering_total)  # Weight ordering heavily
        total_checks += 2

    # 3. Bandwidth reasonableness (BW > 0 and BW < 500 Hz)
    bw_ok = 0
    bw_total = 0
    for bws in frame_bws:
        for bw in bws:
            bw_total += 1
            if 0 < bw < 500:
                bw_ok += 1
    if bw_total > 0:
        score += (bw_ok / bw_total)
        total_checks += 1

    # 4. Penalize NaN/0 gaps
    gap_count = 0
    total_values = 0
    for fd in frame_data:
        for f in fd:
            total_values += 1
            if f <= 0:
                gap_count += 1
    if total_values > 0:
        gap_penalty = gap_count / total_values
        score -= gap_penalty
        total_checks += 1

    # 5. Penalize F4 squeezed against ceiling — indicates ceiling is too low
    # and formants are being compressed downward.
    f4_vals = [fd[3] for fd in frame_data if fd[3] > 0]
    if f4_vals:
        f4_mean = np.mean(f4_vals)
        # If F4 is above 85% of the ceiling, the ceiling is likely too low
        f4_ratio = f4_mean / ceiling_hz
        if f4_ratio > 0.85:
            # Penalty scales from 0 at 0.85 to -1.0 at 0.95+
            squeeze_penalty = min(1.0, (f4_ratio - 0.85) / 0.10)
            score -= squeeze_penalty
        total_checks += 1

    return score


def _compute_delta_f(f1: float, f2: float, f3: float, f4: float) -> float:
    """Compute formant dispersion (delta_f) via regression.

    Uses the uniform tube model: F_n = (2n-1) * delta_f / 2
    Fits formant frequencies against x = [1, 3, 5, 7] (the (2n-1) coefficients).
    delta_f = 2 * slope of regression.

    Returns delta_f in Hz, or 0.0 if fewer than 3 valid formants.
    """
    formants = [(1, f1), (3, f2), (5, f3), (7, f4)]
    valid = [(x, f) for x, f in formants if f > 0]
    if len(valid) < 3:
        return 0.0

    x_vals = np.array([v[0] for v in valid], dtype=float)
    y_vals = np.array([v[1] for v in valid], dtype=float)

    # Linear regression: y = slope * x + intercept
    # F_n = (2n-1) * delta_f/2, so slope = delta_f/2
    coeffs = np.polyfit(x_vals, y_vals, 1)
    slope = coeffs[0]
    delta_f = 2.0 * slope
    return float(delta_f)


def analyze_formants(snd: parselmouth.Sound, f0_mean_hz: float = 0.0,
                     formant_ceiling_hz: float | None = None) -> dict:
    """Analyze formant frequencies using F0-informed ceiling selection.

    Dual-ceiling approach validated on TIMIT (630 speakers):
    - ΔF/VTL: F0-adaptive ceiling (d=4.44, 97.6% accuracy).
      F0>200→5500, F0 160-200→adaptive 5000-5500, else→adaptive 4500-5500.
    - Gesture z-scores: Fixed 5000 Hz + 5/95 percentile trimming (F1 d=+0.79,
      F2 d=+2.16, F3 d=+2.19 vs F1 d=-0.79 at 5500).

    Extracts F1-F4 means, bandwidths (BW1-BW3), formant dispersion (delta_f),
    gesture z-scores, and F0-formant interference detection.

    Args:
        snd: Parselmouth Sound object.
        f0_mean_hz: Mean F0 from pitch analysis, for F0 interference detection.
        formant_ceiling_hz: If provided, skip adaptive selection and use this
            fixed ceiling. Useful for vocoder-warped audio where the adaptive
            selector can pick a misleading ceiling.
    """
    # --- Ceiling selection ---
    if formant_ceiling_hz is not None:
        # Fixed ceiling — skip adaptive selection
        best_ceiling = formant_ceiling_hz
        best_formant = call(snd, "To Formant (burg)", 0.0, 5,
                            float(formant_ceiling_hz), 0.025, 50.0)
        best_score = _score_formant_track(best_formant, snd,
                                              ceiling_hz=formant_ceiling_hz)
    else:
        # F0-informed ceiling selection.
        # Praat recommends 5000 Hz for male and 5500 Hz for female speakers.
        # The adaptive plausibility scorer is biased toward lower ceilings
        # (lower = smoother tracks but compressed formants), so for voices
        # with clearly feminine F0, use a fixed 5500 Hz ceiling directly.
        if f0_mean_hz > 200:
            # Clearly feminine pitch range — fixed 5500 Hz (Praat recommendation)
            best_ceiling = 5500
            best_formant = call(snd, "To Formant (burg)", 0.0, 5, 5500.0, 0.025, 50.0)
            best_score = _score_formant_track(best_formant, snd, ceiling_hz=5500)
        elif f0_mean_hz > 160:
            # Ambiguous range — adaptive between 5000-5500
            ceilings = [5000, 5200, 5500]
            best_ceiling = ceilings[-1]
            best_score = -float("inf")
            best_formant = None
            for ceiling in ceilings:
                formant_obj = call(snd, "To Formant (burg)", 0.0, 5, float(ceiling), 0.025, 50.0)
                score = _score_formant_track(formant_obj, snd, ceiling_hz=ceiling)
                if score >= best_score:
                    best_score = score
                    best_ceiling = ceiling
                    best_formant = formant_obj
        else:
            # Masculine range or unknown (f0_mean_hz == 0) — adaptive
            ceilings = [4500, 4800, 5000, 5200, 5500]
            best_ceiling = ceilings[-1]
            best_score = -float("inf")
            best_formant = None
            for ceiling in ceilings:
                formant_obj = call(snd, "To Formant (burg)", 0.0, 5, float(ceiling), 0.025, 50.0)
                score = _score_formant_track(formant_obj, snd, ceiling_hz=ceiling)
                if score >= best_score:
                    best_score = score
                    best_ceiling = ceiling
                    best_formant = formant_obj

    formant = best_formant

    n_frames = call(formant, "Get number of frames")
    f1_vals, f2_vals, f3_vals, f4_vals = [], [], [], []
    bw1_vals, bw2_vals, bw3_vals, bw4_vals = [], [], [], []

    for i in range(1, n_frames + 1):
        t = call(formant, "Get time from frame number", i)
        f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
        f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
        f3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")
        f4 = call(formant, "Get value at time", 4, t, "Hertz", "Linear")

        # Extract bandwidths
        bw1 = call(formant, "Get bandwidth at time", 1, t, "Hertz", "Linear")
        bw2 = call(formant, "Get bandwidth at time", 2, t, "Hertz", "Linear")
        bw3 = call(formant, "Get bandwidth at time", 3, t, "Hertz", "Linear")
        bw4 = call(formant, "Get bandwidth at time", 4, t, "Hertz", "Linear")

        if not np.isnan(f1) and f1 > 0:
            f1_vals.append(f1)
        if not np.isnan(f2) and f2 > 0:
            f2_vals.append(f2)
        if not np.isnan(f3) and f3 > 0:
            f3_vals.append(f3)
        f4_vals.append(f4 if not np.isnan(f4) and f4 > 0 else 0.0)

        if not np.isnan(bw1) and bw1 > 0:
            bw1_vals.append(bw1)
        if not np.isnan(bw2) and bw2 > 0:
            bw2_vals.append(bw2)
        if not np.isnan(bw3) and bw3 > 0:
            bw3_vals.append(bw3)
        bw4_vals.append(bw4 if not np.isnan(bw4) and bw4 > 0 else 0.0)

    f1_mean = float(np.mean(f1_vals)) if f1_vals else 0.0
    f2_mean = float(np.mean(f2_vals)) if f2_vals else 0.0
    f3_mean = float(np.mean(f3_vals)) if f3_vals else 0.0
    _f4_nonzero = [v for v in f4_vals if v > 0]
    f4_mean = float(np.mean(_f4_nonzero)) if _f4_nonzero else 0.0

    bw1_mean = float(np.mean(bw1_vals)) if bw1_vals else 0.0
    bw2_mean = float(np.mean(bw2_vals)) if bw2_vals else 0.0
    bw3_mean = float(np.mean(bw3_vals)) if bw3_vals else 0.0
    _bw4_nonzero = [v for v in bw4_vals if v > 0]
    bw4_mean = float(np.mean(_bw4_nonzero)) if _bw4_nonzero else 0.0

    # Formant spacing
    spacing_f2_f1 = f2_mean - f1_mean if f1_mean and f2_mean else 0.0
    spacing_f3_f2 = f3_mean - f2_mean if f2_mean and f3_mean else 0.0
    avg_spacing = (spacing_f2_f1 + spacing_f3_f2) / 2 if spacing_f2_f1 and spacing_f3_f2 else 0.0

    # Estimate vocal tract length from formant spacing
    # VTL ≈ c / (2 * avg_formant_spacing) where spacing ≈ c / (2 * L)
    # Using average of F1-F4 to estimate
    formant_means = [f for f in [f1_mean, f2_mean, f3_mean, f4_mean] if f > 0]
    if len(formant_means) >= 3:
        # VTL estimate from formant frequencies: F_n ≈ (2n-1) * c / (4 * L)
        # Use regression across formants for better estimate
        vtl_estimates = []
        for n, fn in enumerate(formant_means, 1):
            vtl_est = (2 * n - 1) * SPEED_OF_SOUND_CM / (4 * fn)
            vtl_estimates.append(vtl_est)
        vtl_cm = float(np.mean(vtl_estimates))
    else:
        vtl_cm = 0.0

    # --- True delta_f via regression ---
    delta_f = _compute_delta_f(f1_mean, f2_mean, f3_mean, f4_mean)

    # --- Gesture z-scores (phoneme-aware, dual-ceiling) ---
    # TIMIT validation (630 speakers) shows per-formant accuracy varies by ceiling:
    #   At 5500 Hz: F1 d=-0.79 (INVERTED!), F2 d=+1.36, F3 d=+0.66
    #   At 5000 Hz + trim: F1 d=+0.79, F2 d=+2.16, F3 d=+2.19
    # So we use fixed 5000 Hz ceiling + 5th/95th percentile trimming for gesture
    # z-scores, while keeping the F0-adaptive ceiling for ΔF/VTL (d=4.44).
    if best_ceiling != 5000:
        gesture_formant = call(snd, "To Formant (burg)", 0.0, 5, 5000.0, 0.025, 50.0)
        g_n_frames = call(gesture_formant, "Get number of frames")
        g_f1, g_f2, g_f3, g_f4 = [], [], [], []
        for i in range(1, g_n_frames + 1):
            t = call(gesture_formant, "Get time from frame number", i)
            gf1 = call(gesture_formant, "Get value at time", 1, t, "Hertz", "Linear")
            gf2 = call(gesture_formant, "Get value at time", 2, t, "Hertz", "Linear")
            gf3 = call(gesture_formant, "Get value at time", 3, t, "Hertz", "Linear")
            gf4 = call(gesture_formant, "Get value at time", 4, t, "Hertz", "Linear")
            if not np.isnan(gf1) and gf1 > 0:
                g_f1.append(gf1)
            if not np.isnan(gf2) and gf2 > 0:
                g_f2.append(gf2)
            if not np.isnan(gf3) and gf3 > 0:
                g_f3.append(gf3)
            g_f4.append(gf4 if not np.isnan(gf4) and gf4 > 0 else 0.0)
    else:
        g_f1, g_f2, g_f3 = f1_vals, f2_vals, f3_vals
        g_f4 = f4_vals

    # Trim outliers (5th/95th percentile) for cleaner gesture signals
    def _trim(vals):
        if len(vals) < 10:
            return vals
        arr = np.array(vals)
        p5, p95 = np.percentile(arr, [5, 95])
        return arr[(arr >= p5) & (arr <= p95)].tolist()

    g_f1_trim = _trim(g_f1)
    g_f2_trim = _trim(g_f2)
    g_f3_trim = _trim(g_f3)

    gesture_zscores = _compute_gesture_zscores(g_f1_trim, g_f2_trim, g_f3_trim)
    f1_zscore = gesture_zscores["f1_zscore"]
    f2_zscore = gesture_zscores["f2_zscore"]
    f3_zscore = gesture_zscores["f3_zscore"]

    # Intentionally use untrimmed lists for per-vowel computation: more frames
    # improves vowel classification reliability; outlier clipping would discard
    # valid frames from less-common vowels.
    per_vowel_zscores = _compute_per_vowel_zscores(g_f1, g_f2, g_f3, g_f4)

    # --- Formant amplitude ratios ---
    formant_amplitudes = _compute_formant_amplitudes(snd, formant, f0_mean_hz)

    # --- F0-formant interference detection ---
    f0_interference_flag = False
    f0_interference_severity = 0.0
    if f0_mean_hz > 0 and f1_mean > 0:
        ratio = f0_mean_hz / f1_mean
        f0_interference_flag = ratio > 0.5
        f0_interference_severity = float(max(0, min(1, (ratio - 0.3) / 0.4)))

    return {
        "f1_mean_hz": f1_mean,
        "f2_mean_hz": f2_mean,
        "f3_mean_hz": f3_mean,
        "f4_mean_hz": f4_mean,
        "formant_spacing_f2_f1_hz": spacing_f2_f1,
        "formant_spacing_f3_f2_hz": spacing_f3_f2,
        "formant_spacing_avg_hz": avg_spacing,
        "vocal_tract_length_cm": vtl_cm,
        "f1_values": f1_vals,
        "f2_values": f2_vals,
        # New keys
        "delta_f_hz": delta_f,
        "delta_f_method": "regression",
        "bw1_mean_hz": bw1_mean,
        "bw2_mean_hz": bw2_mean,
        "bw3_mean_hz": bw3_mean,
        "bw4_mean_hz": round(bw4_mean, 1),
        "formant_ceiling_used_hz": best_ceiling,
        "formant_ceiling_score": float(best_score),
        "f1_gesture_zscore": f1_zscore,   # Positive = F1 higher than norm for this vowel
        "f2_gesture_zscore": f2_zscore,   # Positive = F2 higher than norm for this vowel
        "f3_gesture_zscore": f3_zscore,   # Positive = F3 higher than norm for this vowel
        "gesture_vowel_frames": gesture_zscores["n_vowel_frames"],
        "gesture_vowel_distribution": gesture_zscores["vowel_distribution"],
        "per_vowel_zscores": per_vowel_zscores,
        "formant_amplitudes": {
            k: v for k, v in formant_amplitudes.items() if k != "per_frame"
        },
        "formant_amplitude_per_frame": formant_amplitudes.get("per_frame", []),
        # Backward compat: old residual keys mapped from z-scores
        "f1_residual_hz": f1_zscore,
        "f2_residual_hz": f2_zscore,
        "f3_residual_hz": f3_zscore,
        "f0_interference": f0_interference_flag,
        "f0_interference_severity": f0_interference_severity,
    }


# ──────────────────────────────────────────────────────────
# Formant amplitude ratios (spectral energy at formant locations)
# ──────────────────────────────────────────────────────────

def _compute_formant_amplitudes(
    snd: parselmouth.Sound,
    formant_obj,
    f0_mean_hz: float,
) -> dict:
    """Extract formant amplitudes and amplitude ratios from spectral peaks.

    Measures the amplitude (dB) at each formant frequency (F1-F4) in the
    spectrum for each frame. Returns session means and per-frame data.
    Uses UNCORRECTED amplitudes (combined source+filter output).

    Returns:
        {
            "a1_mean_db": float, "a2_mean_db": float,
            "a3_mean_db": float, "a4_mean_db": float,
            "a2_a1_db": float, "a4_a1_db": float, "a4_a2_db": float,
            "f1_prominence_db": float,
            "per_frame": [{"time_s", "f1_hz", "a1_db", ..., "a4_db", "f1_prom_db"}, ...],
        }
    """
    duration = snd.get_total_duration()
    n_frames = call(formant_obj, "Get number of frames")
    frame_step = duration / max(1, n_frames)
    half_win = max(0.025, 2.5 / max(f0_mean_hz, 75.0))  # ≥2.5 pitch periods

    a1_vals, a2_vals, a3_vals, a4_vals = [], [], [], []
    prom_vals = []
    per_frame = []

    for i in range(1, n_frames + 1):
        t = call(formant_obj, "Get time from frame number", i)
        f1 = call(formant_obj, "Get value at time", 1, t, "Hertz", "Linear")
        f2 = call(formant_obj, "Get value at time", 2, t, "Hertz", "Linear")
        f3 = call(formant_obj, "Get value at time", 3, t, "Hertz", "Linear")
        f4 = call(formant_obj, "Get value at time", 4, t, "Hertz", "Linear")

        if np.isnan(f1) or f1 <= 0 or np.isnan(f2) or f2 <= 0:
            continue

        # Extract short segment and compute spectrum
        start_t = max(0, t - half_win)
        end_t = min(duration, t + half_win)
        segment = snd.extract_part(start_t, end_t,
                                   parselmouth.WindowShape.HAMMING, 1.0, False)
        spectrum = segment.to_spectrum()

        # Measure amplitude at each formant frequency (band energy in narrow window)
        def _amp_at(freq):
            if np.isnan(freq) or freq <= 0:
                return None
            bw = max(50.0, freq * 0.05)  # ±5% of formant freq, min 50 Hz
            energy = call(spectrum, "Get band energy", freq - bw, freq + bw)
            return 10 * np.log10(max(energy, 1e-20))

        a1 = _amp_at(f1)
        a2 = _amp_at(f2)
        a3 = _amp_at(f3)
        a4 = _amp_at(f4)

        if a1 is None or a2 is None:
            continue

        a1_vals.append(a1)
        a2_vals.append(a2)
        if a3 is not None:
            a3_vals.append(a3)
        if a4 is not None:
            a4_vals.append(a4)

        # F1 prominence: amplitude at F1 minus minimum between F1 and F2
        mid_freq = (f1 + f2) / 2.0
        n_steps = 5
        min_amp = a1
        for step in range(1, n_steps + 1):
            probe = f1 + (mid_freq - f1) * step / n_steps
            probe_amp = _amp_at(probe)
            if probe_amp is not None and probe_amp < min_amp:
                min_amp = probe_amp
        f1_prom = a1 - min_amp

        prom_vals.append(f1_prom)

        frame_data = {
            "time_s": round(float(t), 4),
            "f1_hz": round(float(f1), 1), "a1_db": round(a1, 2),
            "f2_hz": round(float(f2), 1), "a2_db": round(a2, 2),
        }
        if a3 is not None:
            frame_data["f3_hz"] = round(float(f3), 1)
            frame_data["a3_db"] = round(a3, 2)
        if a4 is not None:
            frame_data["f4_hz"] = round(float(f4), 1)
            frame_data["a4_db"] = round(a4, 2)
        frame_data["f1_prom_db"] = round(f1_prom, 2)
        per_frame.append(frame_data)

    a1_m = float(np.mean(a1_vals)) if a1_vals else 0.0
    a2_m = float(np.mean(a2_vals)) if a2_vals else 0.0
    a3_m = float(np.mean(a3_vals)) if a3_vals else 0.0
    a4_m = float(np.mean(a4_vals)) if a4_vals else 0.0

    return {
        "a1_mean_db": round(a1_m, 2),
        "a2_mean_db": round(a2_m, 2),
        "a3_mean_db": round(a3_m, 2),
        "a4_mean_db": round(a4_m, 2),
        "a2_a1_db": round(a2_m - a1_m, 2) if a1_vals and a2_vals else 0.0,
        "a4_a1_db": round(a4_m - a1_m, 2) if a1_vals and a4_vals else 0.0,
        "a4_a2_db": round(a4_m - a2_m, 2) if a2_vals and a4_vals else 0.0,
        "f1_prominence_db": round(float(np.mean(prom_vals)), 2) if prom_vals else 0.0,
        "per_frame": per_frame,
    }


def _compute_per_vowel_amplitudes(
    f1_vals: list, f2_vals: list,
    amplitude_per_frame: list[dict],
    frame_step: float = 0.02,
) -> dict:
    """Bucket per-frame formant amplitudes by vowel identity.

    Same pattern as _compute_per_vowel_source(): classify each frame by F1/F2,
    bucket amplitude data, compute per-vowel means.

    Returns:
        {"ih": {"a2_a1_db": float, "a4_a1_db": float,
                "f1_prominence_db": float, "n_frames": int}, ...}
    """
    n = min(len(f1_vals), len(f2_vals), len(amplitude_per_frame))
    buckets: dict[str, list] = {}

    for i in range(n):
        f1, f2 = f1_vals[i], f2_vals[i]
        if f1 <= 0 or f2 <= 0:
            continue
        arpabet = _classify_vowel(f1, f2)
        if arpabet is None:
            continue
        label = _ARPABET_TO_HILLENBRAND.get(arpabet)
        if label is None:
            continue
        buckets.setdefault(label, []).append(amplitude_per_frame[i])

    result = {}
    for label, frames in buckets.items():
        a1s = [f["a1_db"] for f in frames if "a1_db" in f]
        a2s = [f["a2_db"] for f in frames if "a2_db" in f]
        a4s = [f.get("a4_db") for f in frames if f.get("a4_db") is not None]
        proms = [f["f1_prom_db"] for f in frames if "f1_prom_db" in f]

        a1_m = float(np.mean(a1s)) if a1s else 0.0
        a2_m = float(np.mean(a2s)) if a2s else 0.0
        a4_m = float(np.mean(a4s)) if a4s else 0.0

        result[label] = {
            "a2_a1_db": round(a2_m - a1_m, 2) if a1s and a2s else 0.0,
            "a4_a1_db": round(a4_m - a1_m, 2) if a1s and a4s else 0.0,
            "f1_prominence_db": round(float(np.mean(proms)), 2) if proms else 0.0,
            "n_frames": len(frames),
        }

    return result


def _iseli_correction(f_hz: float, formant_freqs: list, formant_bws: list) -> float:
    """Compute vocal tract transfer function magnitude correction in dB at frequency f_hz.

    Based on Iseli, Shue & Alwan (2007). Computes the difference between the
    transfer function magnitude at f_hz and at each formant center, summed
    across all formants. Used to remove vocal tract influence from harmonic
    amplitude measurements.

    Args:
        f_hz: Frequency at which to compute the correction (e.g., F0, 2*F0).
        formant_freqs: List of formant center frequencies [F1, F2, F3, ...].
        formant_bws: List of formant bandwidths [BW1, BW2, BW3, ...].

    Returns:
        Correction in dB to subtract from the raw harmonic amplitude.
    """
    correction_db = 0.0
    for fi, bi in zip(formant_freqs, formant_bws):
        if fi <= 0 or bi <= 0:
            continue
        num = (fi**2 + bi**2)**2
        denom = (f_hz**2 - fi**2)**2 + (f_hz * bi)**2
        h_at_f = 10 * np.log10(num / denom) if denom > 0 else 0.0
        denom_center = bi**2 * fi**2
        h_at_center = 10 * np.log10(num / denom_center) if denom_center > 0 else 0.0
        correction_db += (h_at_f - h_at_center)
    return correction_db


def analyze_voice_quality(snd: parselmouth.Sound, f3_mean_hz: float = 0.0,
                          formant_freqs: list | None = None,
                          formant_bws: list | None = None,
                          f1_values: list | None = None,
                          f2_values: list | None = None,
                          formant_amplitude_per_frame: list | None = None) -> dict:
    """Analyze voice quality: HNR, H1-H2, jitter, shimmer, CPP, spectral tilt, CQ, H1-A3.

    Args:
        snd: Parselmouth Sound object.
        f3_mean_hz: Mean F3 frequency from formant analysis (needed for H1-A3).
        formant_freqs: List of formant center frequencies [F1, F2, F3] for Iseli correction.
        formant_bws: List of formant bandwidths [BW1, BW2, BW3] for Iseli correction.
        f1_values: Per-frame F1 values for per-vowel source bucketing.
        f2_values: Per-frame F2 values for per-vowel source bucketing.
        formant_amplitude_per_frame: Per-frame formant amplitudes for per-vowel bucketing.
    """
    # Point process for jitter/shimmer
    pitch_obj = call(snd, "To Pitch", 0.0, 75.0, 600.0)
    point_process = call(snd, "To PointProcess (periodic, cc)", 75.0, 600.0)

    # HNR
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)

    # Jitter
    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

    # Shimmer
    shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    # H1-H2: difference between first and second harmonic amplitudes
    # Request per-frame data when we have formant values for vowel bucketing
    want_per_frame = f1_values is not None and f2_values is not None
    h1_h2_result = _estimate_h1_h2(
        snd, pitch_obj, formant_freqs=formant_freqs, formant_bws=formant_bws,
        return_per_frame=want_per_frame,
    )
    if want_per_frame:
        h1_h2_raw, h1_h2_corrected, h1_h2_per_frame = h1_h2_result
    else:
        h1_h2_raw, h1_h2_corrected = h1_h2_result
        h1_h2_per_frame = []

    # Per-vowel source measures (H1-H2, tilt, CPP bucketed by vowel identity)
    per_vowel_source = {}
    if want_per_frame and h1_h2_per_frame:
        per_vowel_source = _compute_per_vowel_source(
            f1_values, f2_values, h1_h2_per_frame, snd=snd,
        )

    # Per-vowel amplitude ratios (formant energy distribution per vowel)
    per_vowel_amplitudes = {}
    if want_per_frame and formant_amplitude_per_frame:
        per_vowel_amplitudes = _compute_per_vowel_amplitudes(
            f1_values, f2_values, formant_amplitude_per_frame,
        )

    # Vocal fry detection
    fry_data = _detect_vocal_fry(snd, pitch_obj)
    fry_proportion = fry_data["fry_proportion"]
    fry_times = fry_data["fry_frame_times"]

    # CPP: Cepstral Peak Prominence (raw = all frames, clean = fry excluded)
    cpp_raw = _estimate_cpp(snd)
    cpp_clean = _cpp_excluding_frames(snd, fry_times) if fry_times else cpp_raw

    # HNR: raw = all frames, clean = fry excluded
    hnr_raw = float(hnr) if not np.isnan(hnr) else 0.0
    hnr_clean = _hnr_excluding_frames(snd, fry_times) if fry_times else hnr_raw

    # Spectral tilt via librosa — voiced frames only
    spectral_tilt = _estimate_spectral_tilt(snd, pitch_obj=pitch_obj)

    # H1-A3: difference between first harmonic and third formant amplitude
    h1_a3_raw, h1_a3_corrected = _estimate_h1_a3(
        snd, pitch_obj, f3_mean_hz,
        formant_freqs=formant_freqs, formant_bws=formant_bws
    )

    # CQ: closed quotient estimate from harmonic relationships
    cq = _estimate_closed_quotient(snd, pitch_obj)

    # Use fry-excluded values as primary when fry is present
    result = {
        "hnr_db": hnr_clean,
        "hnr_db_raw": hnr_raw,
        "h1_h2_db": h1_h2_raw,
        "h1_h2_db_corrected": h1_h2_corrected,
        "cpp_db": cpp_clean,
        "cpp_db_raw": cpp_raw,
        "jitter_local": float(jitter * 100) if not np.isnan(jitter) else 0.0,  # as percent
        "shimmer_local": float(shimmer * 100) if not np.isnan(shimmer) else 0.0,  # as percent
        "spectral_tilt_db_per_octave": spectral_tilt,
        "h1_a3_db": h1_a3_raw,
        "h1_a3_db_corrected": h1_a3_corrected,
        "closed_quotient": cq,
        "fry_proportion": round(fry_proportion, 3),
    }
    if per_vowel_source:
        result["per_vowel_source"] = per_vowel_source
    if per_vowel_amplitudes:
        result["per_vowel_amplitudes"] = per_vowel_amplitudes
    return result


def _estimate_h1_h2(snd: parselmouth.Sound, pitch_obj,
                    formant_freqs: list | None = None,
                    formant_bws: list | None = None,
                    return_per_frame: bool = False):
    """Estimate H1-H2 (breathiness) from harmonic amplitudes.

    Args:
        return_per_frame: If True, return a 3-tuple with per-frame lists appended.

    Returns:
        (raw_mean, corrected_mean) — if no formant data, corrected_mean = raw_mean.
        If return_per_frame: (raw_mean, corrected_mean, per_frame_list)
            where per_frame_list = [(time_s, h1_h2_raw, h1_h2_corrected), ...]
    """
    duration = snd.get_total_duration()
    n_frames = max(1, int(duration / 0.02))
    h1_h2_raw_vals = []
    h1_h2_corrected_vals = []
    per_frame = []
    has_formants = formant_freqs is not None and formant_bws is not None

    for i in range(n_frames):
        t = (i + 0.5) * 0.02
        f0 = call(pitch_obj, "Get value at time", t, "Hertz", "Linear")
        if np.isnan(f0) or f0 <= 0:
            continue

        # Extract spectrum at this time point
        start_t = max(0, t - 0.025)
        end_t = min(duration, t + 0.025)
        segment = snd.extract_part(start_t, end_t, parselmouth.WindowShape.HAMMING, 1.0, False)
        spectrum = segment.to_spectrum()

        # Pitch-synchronous peak picking within ±(F0/2) Hz of each harmonic.
        # VoiceSauce method: avoids the F0-dependent bias of band energy
        # integration (wider harmonic spacing at high F0 inflates band energy).
        half_f0 = f0 / 2
        spec_freqs = np.array(spectrum.xs())
        spec_amp = np.abs(spectrum.values[0] + 1j * spectrum.values[1])
        spec_amp_db = 20 * np.log10(spec_amp + 1e-10)

        h1_mask = (spec_freqs >= f0 - half_f0) & (spec_freqs <= f0 + half_f0)
        h2_mask = (spec_freqs >= 2 * f0 - half_f0) & (spec_freqs <= 2 * f0 + half_f0)

        if h1_mask.sum() > 0 and h2_mask.sum() > 0:
            h1_db = float(np.max(spec_amp_db[h1_mask]))
            h2_db = float(np.max(spec_amp_db[h2_mask]))
            raw_val = h1_db - h2_db
            h1_h2_raw_vals.append(raw_val)

            corrected_val = raw_val
            if has_formants:
                h1_corrected = h1_db - _iseli_correction(f0, formant_freqs, formant_bws)
                h2_corrected = h2_db - _iseli_correction(2 * f0, formant_freqs, formant_bws)
                corrected_val = h1_corrected - h2_corrected
                h1_h2_corrected_vals.append(corrected_val)

            if return_per_frame:
                per_frame.append((t, raw_val, corrected_val))

    raw_mean = float(np.mean(h1_h2_raw_vals)) if h1_h2_raw_vals else 0.0
    if has_formants and h1_h2_corrected_vals:
        corrected_mean = float(np.mean(h1_h2_corrected_vals))
    else:
        corrected_mean = raw_mean

    if return_per_frame:
        return (raw_mean, corrected_mean, per_frame)
    return (raw_mean, corrected_mean)


def _estimate_h1_a3(snd: parselmouth.Sound, pitch_obj, f3_mean_hz: float,
                    formant_freqs: list | None = None,
                    formant_bws: list | None = None) -> tuple[float, float]:
    """Estimate H1-A3: amplitude difference between first harmonic and F3 region.

    Higher H1-A3 means more energy at F0 relative to F3, indicating lighter
    vocal fold mass (less high-frequency harmonic energy from heavy contact).

    Returns:
        (raw_mean, corrected_mean) — if no formant data, corrected_mean = raw_mean.
    """
    if f3_mean_hz <= 0:
        return (0.0, 0.0)

    duration = snd.get_total_duration()
    n_frames = max(1, int(duration / 0.02))
    h1_a3_raw_vals = []
    h1_a3_corrected_vals = []
    has_formants = formant_freqs is not None and formant_bws is not None

    for i in range(n_frames):
        t = (i + 0.5) * 0.02
        f0 = call(pitch_obj, "Get value at time", t, "Hertz", "Linear")
        if np.isnan(f0) or f0 <= 0:
            continue

        start_t = max(0, t - 0.025)
        end_t = min(duration, t + 0.025)
        segment = snd.extract_part(start_t, end_t, parselmouth.WindowShape.HAMMING, 1.0, False)
        spectrum = segment.to_spectrum()

        # Pitch-synchronous peak picking within ±(F0/2) Hz of each target.
        # Consistent with _estimate_h1_h2 — avoids F0-dependent band-energy bias.
        half_f0 = f0 / 2
        spec_freqs = np.array(spectrum.xs())
        spec_amp = np.abs(spectrum.values[0] + 1j * spectrum.values[1])
        spec_amp_db = 20 * np.log10(spec_amp + 1e-10)

        h1_mask = (spec_freqs >= f0 - half_f0) & (spec_freqs <= f0 + half_f0)
        a3_mask = (spec_freqs >= f3_mean_hz - half_f0) & (spec_freqs <= f3_mean_hz + half_f0)

        if h1_mask.sum() > 0 and a3_mask.sum() > 0:
            h1_db = float(np.max(spec_amp_db[h1_mask]))
            a3_db = float(np.max(spec_amp_db[a3_mask]))
            h1_a3_raw_vals.append(h1_db - a3_db)

            if has_formants:
                h1_corrected = h1_db - _iseli_correction(f0, formant_freqs, formant_bws)
                a3_corrected = a3_db - _iseli_correction(f3_mean_hz, formant_freqs, formant_bws)
                h1_a3_corrected_vals.append(h1_corrected - a3_corrected)

    raw_mean = float(np.mean(h1_a3_raw_vals)) if h1_a3_raw_vals else 0.0
    if has_formants and h1_a3_corrected_vals:
        corrected_mean = float(np.mean(h1_a3_corrected_vals))
    else:
        corrected_mean = raw_mean
    return (raw_mean, corrected_mean)


def _estimate_closed_quotient(snd: parselmouth.Sound, pitch_obj) -> float:
    """Estimate closed quotient (CQ) from harmonic amplitude relationships.

    CQ is the fraction of each glottal cycle where the vocal folds are closed.
    Lower CQ = lighter vocal fold contact = more feminine.

    Without EGG, we estimate CQ from the H1-H2-H4 relationship:
    - Higher H1 relative to H2/H4 suggests more open quotient (lower CQ)
    - More equal harmonics suggest higher CQ (heavier contact)

    Uses the approximation: CQ ≈ 0.5 + 0.3 * sigmoid(H2+H4-2*H1)
    which maps typical harmonic patterns to the ~0.3-0.6 range.
    """
    duration = snd.get_total_duration()
    n_frames = max(1, int(duration / 0.02))
    cq_vals = []

    for i in range(n_frames):
        t = (i + 0.5) * 0.02
        f0 = call(pitch_obj, "Get value at time", t, "Hertz", "Linear")
        if np.isnan(f0) or f0 <= 0:
            continue

        start_t = max(0, t - 0.025)
        end_t = min(duration, t + 0.025)
        segment = snd.extract_part(start_t, end_t, parselmouth.WindowShape.HAMMING, 1.0, False)
        spectrum = segment.to_spectrum()

        # Get amplitudes at H1, H2, H4
        h1_amp = call(spectrum, "Get band energy", f0 * 0.9, f0 * 1.1)
        h2_amp = call(spectrum, "Get band energy", f0 * 1.8, f0 * 2.2)
        h4_amp = call(spectrum, "Get band energy", f0 * 3.6, f0 * 4.4)

        if h1_amp > 0 and h2_amp > 0 and h4_amp > 0:
            h1_db = 10 * np.log10(h1_amp)
            h2_db = 10 * np.log10(h2_amp)
            h4_db = 10 * np.log10(h4_amp)

            # Higher harmonics relative to H1 = higher CQ
            harmonic_balance = (h2_db + h4_db) / 2 - h1_db
            # Sigmoid mapping to 0.3-0.6 range
            cq = 0.45 + 0.15 * (2 / (1 + np.exp(-harmonic_balance / 3)) - 1)
            cq_vals.append(float(np.clip(cq, 0.2, 0.8)))

    return float(np.mean(cq_vals)) if cq_vals else 0.0


# ──────────────────────────────────────────────────────────
# Vocal fry (creaky voice) detection
# ──────────────────────────────────────────────────────────

def _detect_vocal_fry(snd: parselmouth.Sound, pitch_obj,
                      f0_threshold: float = 80.0,
                      jitter_threshold: float = 0.05,
                      min_consecutive: int = 3,
                      frame_step: float = 0.01) -> dict:
    """Detect vocal fry frames using F0 < threshold AND high local jitter.

    A frame is fry if: F0 < f0_threshold Hz AND local jitter > jitter_threshold.
    Only segments of >= min_consecutive frames (>= 30ms at 10ms step) count.

    Returns:
        {
            "fry_mask": list[bool] — per-frame fry flags,
            "fry_proportion": float — fraction of voiced frames that are fry,
            "fry_frame_times": list[float] — center times of fry frames,
            "n_fry_frames": int,
            "n_voiced_frames": int,
        }
    """
    duration = snd.get_total_duration()
    n_frames = max(1, int(duration / frame_step))

    # Get per-frame F0
    f0_per_frame = []
    for i in range(n_frames):
        t = (i + 0.5) * frame_step
        f0 = call(pitch_obj, "Get value at time", t, "Hertz", "Linear")
        f0_per_frame.append(float(f0) if not np.isnan(f0) else 0.0)

    # Get per-frame local jitter via point process period differences
    point_process = call(snd, "To PointProcess (periodic, cc)", 50.0, 600.0)
    n_periods = call(point_process, "Get number of periods", 0, 0, 0.0001, 0.04, 1.3)

    # Build period array for jitter calculation
    periods = []
    for i in range(1, int(n_periods) + 1):
        t_start = call(point_process, "Get time from index", i)
        t_end = call(point_process, "Get time from index", i + 1)
        if t_end > 0:
            periods.append((float(t_start), float(t_end - t_start)))

    # Compute local jitter per frame: ratio of period perturbation to local mean
    # Map periods to frames via time
    jitter_per_frame = [0.0] * n_frames
    for idx in range(1, len(periods) - 1):
        t_center, p_center = periods[idx]
        _, p_prev = periods[idx - 1]
        _, p_next = periods[idx + 1]
        local_mean = (p_prev + p_center + p_next) / 3.0
        if local_mean > 0:
            local_jitter = abs(p_center - p_prev) / local_mean
            frame_idx = min(n_frames - 1, int(t_center / frame_step))
            # Take max jitter if multiple periods map to same frame
            jitter_per_frame[frame_idx] = max(jitter_per_frame[frame_idx], local_jitter)

    # Mark candidate fry frames: low F0 AND high jitter
    candidates = []
    for i in range(n_frames):
        f0 = f0_per_frame[i]
        is_low_f0 = 0 < f0 < f0_threshold
        is_high_jitter = jitter_per_frame[i] > jitter_threshold
        candidates.append(is_low_f0 and is_high_jitter)

    # Filter: only keep runs of >= min_consecutive frames
    fry_mask = [False] * n_frames
    run_start = None
    for i in range(n_frames + 1):
        if i < n_frames and candidates[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_len = i - run_start
                if run_len >= min_consecutive:
                    for j in range(run_start, i):
                        fry_mask[j] = True
                run_start = None

    fry_frame_times = [
        (i + 0.5) * frame_step for i in range(n_frames) if fry_mask[i]
    ]
    n_voiced = sum(1 for f0 in f0_per_frame if f0 > 0)
    n_fry = sum(fry_mask)

    return {
        "fry_mask": fry_mask,
        "fry_proportion": float(n_fry / n_voiced) if n_voiced > 0 else 0.0,
        "fry_frame_times": fry_frame_times,
        "n_fry_frames": n_fry,
        "n_voiced_frames": n_voiced,
    }


def _hnr_excluding_frames(snd: parselmouth.Sound,
                           exclude_times: list[float],
                           frame_step: float = 0.01) -> float:
    """Compute HNR excluding specific time frames (e.g., fry frames).

    Extracts per-frame harmonicity values from Praat's Harmonicity object
    and averages only the non-excluded frames.
    """
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
    n_frames = call(harmonicity, "Get number of frames")
    # Quantize exclude times to frame indices for O(1) lookup
    exclude_indices = set(round(t / frame_step) for t in exclude_times)

    vals = []
    for i in range(1, int(n_frames) + 1):
        t = call(harmonicity, "Get time from frame number", i)
        frame_idx = round(float(t) / frame_step)
        if frame_idx in exclude_indices:
            continue
        val = call(harmonicity, "Get value in frame", i)
        if not np.isnan(val) and val > -200:  # Praat uses -200 for silence
            vals.append(float(val))

    return float(np.mean(vals)) if vals else 0.0


def _cpp_excluding_frames(snd: parselmouth.Sound,
                           exclude_times: list[float],
                           frame_step: float = 0.01) -> float:
    """Compute CPP excluding fry frames by zeroing them in the signal."""
    if not exclude_times:
        return _estimate_cpp(snd)

    # Zero out fry regions in a copy of the signal
    y = snd.values[0].copy()
    sr = int(snd.sampling_frequency)
    half_frame = int(frame_step * sr / 2)

    for t in exclude_times:
        center = int(t * sr)
        start = max(0, center - half_frame)
        end = min(len(y), center + half_frame)
        y[start:end] = 0.0

    clean_snd = parselmouth.Sound(y, sampling_frequency=sr)
    return _estimate_cpp(clean_snd)


def _estimate_cpp(snd: parselmouth.Sound) -> float:
    """Estimate Cepstral Peak Prominence."""
    try:
        # Use Praat's built-in CPP via PowerCepstrogram
        cepstrogram = call(snd, "To PowerCepstrogram", 60.0, 0.002, 5000.0, 50)
        cpps = call(cepstrogram, "Get CPPS", False, 0.02, 0.0005, 60.0, 330.0, 0.05, "Parabolic", 0.001, 0.0, "Exponential decay", "Robust slow")
        return float(cpps) if not np.isnan(cpps) else 0.0
    except Exception:
        # Fallback: manual cepstral analysis
        y = snd.values[0]
        sr = int(snd.sampling_frequency)
        frame_len = int(0.04 * sr)
        hop = int(0.01 * sr)
        cpp_vals = []

        for start in range(0, len(y) - frame_len, hop):
            frame = y[start:start + frame_len]
            windowed = frame * np.hamming(len(frame))
            spectrum = np.fft.rfft(windowed)
            log_spectrum = np.log(np.abs(spectrum) + 1e-10)
            cepstrum = np.fft.irfft(log_spectrum)

            # Find peak in quefrency range corresponding to 60-330 Hz
            q_low = int(sr / 330)
            q_high = int(sr / 60)
            q_high = min(q_high, len(cepstrum) - 1)

            if q_low < q_high:
                search_region = np.abs(cepstrum[q_low:q_high])
                peak_val = np.max(search_region)
                # CPP = peak - regression line (simplified as peak prominence)
                baseline = np.mean(search_region)
                if baseline > 0:
                    cpp_vals.append(20 * np.log10(peak_val / baseline))

        return float(np.mean(cpp_vals)) if cpp_vals else 0.0


def _estimate_spectral_tilt(snd: parselmouth.Sound, pitch_obj=None) -> float:
    """Estimate spectral tilt in dB/octave from voiced frames only.

    Args:
        snd: Parselmouth Sound object.
        pitch_obj: Praat Pitch object for voiced frame selection. When provided,
            only voiced frames are included in the average, preventing silence
            and unvoiced segments from systematically flattening the tilt.
    """
    y = snd.values[0]
    sr = int(snd.sampling_frequency)

    # Compute power spectrum
    n_fft = 2048
    hop_length = n_fft // 4  # 512 samples
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2

    # Gate to voiced frames only when pitch object is available
    if pitch_obj is not None:
        n_time_frames = S.shape[1]
        voiced_mask = np.zeros(n_time_frames, dtype=bool)
        for i in range(n_time_frames):
            t = i * hop_length / sr
            f0 = call(pitch_obj, "Get value at time", t, "Hertz", "Linear")
            voiced_mask[i] = not (np.isnan(f0) or f0 <= 0)
        if voiced_mask.sum() >= 2:
            avg_spectrum = np.mean(S[:, voiced_mask], axis=1)
        else:
            avg_spectrum = np.mean(S, axis=1)
    else:
        avg_spectrum = np.mean(S, axis=1)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Fit log-log regression (spectral tilt)
    # Only use frequencies 100-5000 Hz
    mask = (freqs >= 100) & (freqs <= 5000)
    if mask.sum() < 2:
        return 0.0

    log_freqs = np.log2(freqs[mask])
    log_power = 10 * np.log10(avg_spectrum[mask] + 1e-10)

    # Linear regression in log-freq space gives dB/octave
    coeffs = np.polyfit(log_freqs, log_power, 1)
    return float(coeffs[0])  # slope = dB per octave


def analyze_articulation(y: np.ndarray, sr: int, formant_data: dict) -> dict:
    """/s/ centroid, vowel space area, speech rate."""
    s_centroid = _estimate_sibilant_centroid(y, sr)
    vowel_space = _estimate_vowel_space(formant_data)
    speech_rate = _estimate_speech_rate(y, sr)

    return {
        "s_centroid_hz": s_centroid,
        "vowel_space_area_bark2": vowel_space,
        "speech_rate_syl_per_sec": speech_rate,
    }


def _estimate_sibilant_centroid(y: np.ndarray, sr: int) -> float:
    """Estimate /s/ centroid by finding high-frequency energy segments.

    Sibilants (/s/, /z/, /sh/) have concentrated energy above 3kHz.
    We detect segments where high-freq energy dominates and compute
    the spectral centroid of those segments.
    """
    hop_length = 512
    n_fft = 2048

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # High-frequency band (3kHz+) vs low band
    high_mask = freqs >= 3000
    low_mask = (freqs >= 100) & (freqs < 3000)

    high_energy = np.sum(S[high_mask, :] ** 2, axis=0)
    low_energy = np.sum(S[low_mask, :] ** 2, axis=0)
    total_energy = high_energy + low_energy + 1e-10

    # Sibilant frames: high freq dominates
    ratio = high_energy / total_energy
    sibilant_frames = ratio > 0.6  # high-freq dominant

    if sibilant_frames.sum() < 3:
        return 0.0

    # Compute spectral centroid of sibilant frames only, in the high band
    sibilant_spectra = S[:, sibilant_frames]
    centroids = []
    for frame_idx in range(sibilant_spectra.shape[1]):
        frame = sibilant_spectra[high_mask, frame_idx]
        high_freqs = freqs[high_mask]
        total = np.sum(frame)
        if total > 0:
            centroid = np.sum(high_freqs * frame) / total
            centroids.append(centroid)

    return float(np.mean(centroids)) if centroids else 0.0


def _estimate_vowel_space(formant_data: dict) -> float:
    """Estimate vowel space area from F1/F2 distributions.

    Uses the spread of F1 and F2 values as a proxy for vowel space area.
    Converts to Bark scale for perceptual relevance.
    """
    f1_vals = formant_data.get("f1_values", [])
    f2_vals = formant_data.get("f2_values", [])

    if len(f1_vals) < 10 or len(f2_vals) < 10:
        return 0.0

    def hz_to_bark(f):
        return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)

    f1_bark = np.array([hz_to_bark(f) for f in f1_vals])
    f2_bark = np.array([hz_to_bark(f) for f in f2_vals])

    # Vowel space area as the spread (IQR-based)
    f1_range = np.percentile(f1_bark, 90) - np.percentile(f1_bark, 10)
    f2_range = np.percentile(f2_bark, 90) - np.percentile(f2_bark, 10)

    return float(f1_range * f2_range)


def _estimate_speech_rate(y: np.ndarray, sr: int) -> float:
    """Estimate syllables per second using onset detection as proxy."""
    # Use onset detection as an approximation for syllable nuclei
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=512,
        backtrack=False, units="time",
    )

    duration = len(y) / sr
    if duration < 0.5:
        return 0.0

    # Rough filter: syllable nuclei are typically 100-400ms apart
    if len(onsets) < 2:
        return 0.0

    # Count onsets as syllable proxy
    n_syllables = len(onsets)
    return float(n_syllables / duration)


def analyze_prosody(pitch_data: dict) -> dict:
    """Analyze prosody: F0 variability, range in semitones, contour dynamics."""
    f0_contour = np.array(pitch_data.get("f0_contour_hz", []))
    confidence = np.array(pitch_data.get("f0_confidence", []))

    if len(f0_contour) == 0:
        return {
            "f0_cv": 0.0,
            "pitch_range_semitones": 0.0,
            "n_rises": 0,
            "n_falls": 0,
            "rise_fall_ratio": 0.0,
        }

    # Filter to voiced frames
    mask = confidence > 0.5 if len(confidence) == len(f0_contour) else np.ones(len(f0_contour), dtype=bool)
    voiced = f0_contour[mask]
    voiced = voiced[voiced > 0]

    if len(voiced) < 5:
        return {
            "f0_cv": 0.0,
            "pitch_range_semitones": 0.0,
            "n_rises": 0,
            "n_falls": 0,
            "rise_fall_ratio": 0.0,
        }

    # Coefficient of variation
    f0_cv = float(np.std(voiced) / np.mean(voiced))

    # Pitch range in semitones
    f0_min = np.percentile(voiced, 5)  # 5th percentile to avoid outliers
    f0_max = np.percentile(voiced, 95)
    if f0_min > 0:
        pitch_range_st = 12 * np.log2(f0_max / f0_min)
    else:
        pitch_range_st = 0.0

    # Count rises and falls in smoothed F0 contour
    # Smooth to remove micro-variations
    kernel_size = min(11, len(voiced) // 2 * 2 + 1)
    if kernel_size >= 3:
        smoothed = np.convolve(voiced, np.ones(kernel_size) / kernel_size, mode="valid")
    else:
        smoothed = voiced

    diffs = np.diff(smoothed)
    n_rises = int(np.sum(diffs > 0))
    n_falls = int(np.sum(diffs < 0))
    rise_fall_ratio = float(n_rises / max(n_falls, 1))

    return {
        "f0_cv": f0_cv,
        "pitch_range_semitones": float(pitch_range_st),
        "n_rises": n_rises,
        "n_falls": n_falls,
        "rise_fall_ratio": rise_fall_ratio,
    }


def check_recording_quality(audio: np.ndarray, sr: int) -> dict:
    """Check recording quality before analysis.

    Returns:
        dict with:
            snr_db: Estimated signal-to-noise ratio (dB). Computed as the ratio
                between the mean RMS of the loudest quartile and quietest
                quartile of 25ms frames.
            clipping_fraction: Fraction of samples with |amplitude| > 0.99.
            voiced_fraction: Estimated fraction of frames above 10% of peak RMS
                (rough voiced activity proxy, no pitch tracking needed).
            quality_ok: True if all thresholds are met (SNR > 15 dB,
                clipping < 1%, voiced_fraction > 30%).
    """
    # Clipping detection
    clipping_fraction = float(np.mean(np.abs(audio) > 0.99))

    # SNR estimate: loudest quartile RMS vs quietest quartile RMS
    frame_len = int(sr * 0.025)  # 25ms frames
    hop = max(1, frame_len // 2)
    n_frames = max(1, (len(audio) - frame_len) // hop + 1)
    frame_rms = []
    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start + frame_len]
        if len(frame) < frame_len:
            continue
        frame_rms.append(float(np.sqrt(np.mean(frame ** 2))))

    if len(frame_rms) >= 4:
        frame_rms_sorted = sorted(frame_rms)
        q = max(1, len(frame_rms_sorted) // 4)
        noise_floor = float(np.mean(frame_rms_sorted[:q]))
        signal_peak = float(np.mean(frame_rms_sorted[-q:]))
        snr_db = float(20 * np.log10(signal_peak / noise_floor)) if noise_floor > 0 else 60.0
    else:
        snr_db = 0.0

    # Voiced fraction: frames louder than 10% of peak RMS
    if frame_rms:
        peak_rms = max(frame_rms)
        threshold = peak_rms * 0.1
        voiced_fraction = float(sum(1 for r in frame_rms if r > threshold) / len(frame_rms))
    else:
        voiced_fraction = 0.0

    quality_ok = (
        clipping_fraction < 0.01
        and snr_db > 15.0
        and voiced_fraction > 0.3
    )

    return {
        "snr_db": round(snr_db, 1),
        "clipping_fraction": round(clipping_fraction, 4),
        "voiced_fraction": round(voiced_fraction, 3),
        "quality_ok": bool(quality_ok),
    }


def analyze(wav_path: str, output_path: str | None = None, crepe_device: str = "cuda:0") -> dict:
    """Run full acoustic analysis on a WAV file.

    Args:
        wav_path: Path to WAV file (any sample rate, mono or stereo).
        output_path: Optional path to save analysis.json. If None, saves
            next to the WAV file as analysis.json.
        crepe_device: CUDA device for CREPE pitch analysis.

    Returns:
        Dict with analysis results organized by category.
    """
    wav_path = str(Path(wav_path).resolve())
    y, sr = _load_audio(wav_path)

    # 0. Recording quality gate
    recording_quality = check_recording_quality(y, sr)

    # Parselmouth Sound object (at native sample rate)
    snd = parselmouth.Sound(y, sampling_frequency=sr)

    # 1. Pitch analysis (CREPE on GPU)
    print("  Analyzing pitch (CREPE)...")
    pitch_data = analyze_pitch_crepe(y, sr, device=crepe_device)

    # 2. Formant analysis (Parselmouth)
    print("  Analyzing formants (Parselmouth)...")
    formant_data = analyze_formants(snd, f0_mean_hz=pitch_data["f0_mean_hz"])

    # 3. Voice quality (Parselmouth)
    print("  Analyzing voice quality...")
    formant_freqs = [formant_data.get("f1_mean_hz", 0), formant_data.get("f2_mean_hz", 0), formant_data.get("f3_mean_hz", 0)]
    formant_bws = [formant_data.get("bw1_mean_hz", 0), formant_data.get("bw2_mean_hz", 0), formant_data.get("bw3_mean_hz", 0)]
    quality_data = analyze_voice_quality(
        snd, f3_mean_hz=formant_data.get("f3_mean_hz", 0.0),
        formant_freqs=formant_freqs, formant_bws=formant_bws,
        f1_values=formant_data.get("f1_values"),
        f2_values=formant_data.get("f2_values"),
        formant_amplitude_per_frame=formant_data.get("formant_amplitude_per_frame"),
    )

    # 4. Articulation
    print("  Analyzing articulation...")
    articulation_data = analyze_articulation(y, sr, formant_data)

    # 5. Prosody (derived from pitch data)
    print("  Analyzing prosody...")
    prosody_data = analyze_prosody(pitch_data)

    # Clean up formant data for JSON output (remove raw value lists)
    formant_output = {k: v for k, v in formant_data.items() if k not in ("f1_values", "f2_values")}

    result = {
        "source_file": wav_path,
        "sample_rate": sr,
        "duration_s": float(len(y) / sr),
        "quality": recording_quality,
        "formant_ceiling_used_hz": formant_data.get("formant_ceiling_used_hz", 0),
        "pitch": pitch_data,
        "formants": formant_output,
        "voice_quality": quality_data,
        "articulation": articulation_data,
        "prosody": prosody_data,
    }

    # Save output
    if output_path is None:
        output_path = str(Path(wav_path).parent / "analysis.json")

    # Strip contour data for the saved JSON (large arrays)
    save_result = _strip_contours(result)
    with open(output_path, "w") as f:
        json.dump(save_result, f, indent=2)

    print(f"  Analysis saved to {output_path}")
    return result


def _strip_contours(result: dict) -> dict:
    """Create a copy with large contour arrays replaced by summaries."""
    import copy
    out = copy.deepcopy(result)

    # Replace pitch contour arrays with just length info
    if "pitch" in out:
        contour_len = len(out["pitch"].get("f0_contour_hz", []))
        out["pitch"].pop("f0_contour_hz", None)
        out["pitch"].pop("f0_contour_time_s", None)
        out["pitch"].pop("f0_confidence", None)
        out["pitch"]["contour_frames"] = contour_len

    # Strip large per-frame arrays from formants
    if "formants" in out:
        out["formants"].pop("f1_values", None)
        out["formants"].pop("f2_values", None)
        out["formants"].pop("formant_amplitude_per_frame", None)

    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <wav_file> [output_json] [--device cuda:0]")
        sys.exit(1)

    wav_file = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    device = "cuda:0"
    for i, arg in enumerate(sys.argv):
        if arg == "--device" and i + 1 < len(sys.argv):
            device = sys.argv[i + 1]

    result = analyze(wav_file, out_file, crepe_device=device)
    print(f"\nPitch: {result['pitch']['f0_mean_hz']:.1f} Hz (mean)")
    print(f"F1: {result['formants']['f1_mean_hz']:.0f} Hz, F2: {result['formants']['f2_mean_hz']:.0f} Hz")
    print(f"HNR: {result['voice_quality']['hnr_db']:.1f} dB")
    print(f"F0 CV: {result['prosody']['f0_cv']:.3f}")
