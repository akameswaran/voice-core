"""Research-grade acoustic analysis -- all metrics, no latency constraint.

Composes the same building blocks as the production analyzer (analyze.py)
but adds experimental metrics (LPC spectral envelopes, inter-formant valley
depth) and preserves full per-frame data.

Usage:
    # Single file
    python -m voice_core.research recording.wav

    # Single file with output path
    python -m voice_core.research recording.wav -o results.json

    # Directory of WAVs
    python -m voice_core.research /path/to/recordings/ -o results/

    # Compare two result files
    python -m voice_core.research --compare a.json b.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import parselmouth

from voice_core.analyze import (
    _load_audio,
    analyze_pitch_crepe,
    analyze_formants,
    analyze_voice_quality,
    analyze_articulation,
    analyze_prosody,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ──────────────────────────────────────────────────────────
# LPC spectral envelope
# ──────────────────────────────────────────────────────────

def _compute_lpc_envelope(
    y: np.ndarray, sr: int, order: int = 14, n_fft: int = 2048
) -> dict:
    """Compute LPC spectral envelope.

    Extracts the middle 1.5 seconds of audio to avoid onset/offset artifacts,
    applies pre-emphasis, computes LPC coefficients via the autocorrelation
    method (Levinson-Durbin), and derives the spectral envelope.

    Returns:
        {
            "frequencies_hz": list[float],  # Frequency axis
            "magnitude_db": list[float],    # Envelope in dB (normalized to 0 dB peak)
            "order": int,
            "f1_prominence_db": float,      # F1 peak - F1-F2 valley (OPC indicator)
        }
    """
    from scipy.signal import freqz, lfilter
    from scipy.linalg import solve_toeplitz

    # Extract middle 1.5 seconds to avoid onset/offset artifacts
    total_samples = len(y)
    target_samples = int(1.5 * sr)
    if total_samples > target_samples:
        start = (total_samples - target_samples) // 2
        y_segment = y[start : start + target_samples]
    else:
        y_segment = y.copy()

    # Pre-emphasis filter (coefficient 0.97)
    y_emph = np.asarray(lfilter([1.0, -0.97], [1.0], y_segment), dtype=np.float64)

    # Compute autocorrelation via numpy
    n = len(y_emph)
    # Pad for FFT-based autocorrelation (faster than direct)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    Y = np.fft.rfft(y_emph, n=fft_size)
    autocorr_full = np.fft.irfft(Y * np.conj(Y))
    autocorr = autocorr_full[:order + 1].real

    # Guard against silence / DC-only signal
    if autocorr[0] <= 0:
        n_bins = n_fft // 2 + 1
        return {
            "frequencies_hz": np.linspace(0, sr / 2, n_bins).tolist(),
            "magnitude_db": [0.0] * n_bins,
            "order": order,
            "f1_prominence_db": 0.0,
        }

    # Solve Toeplitz system (Levinson-Durbin) for LPC coefficients
    # autocorr[0] * a[0] + autocorr[1] * a[1] + ... = -autocorr[1:]
    # Using scipy's solve_toeplitz which implements Levinson-Durbin
    r = autocorr / autocorr[0]  # Normalize
    try:
        a_coeffs = solve_toeplitz(r[:order], -r[1 : order + 1])
    except np.linalg.LinAlgError:
        n_bins = n_fft // 2 + 1
        return {
            "frequencies_hz": np.linspace(0, sr / 2, n_bins).tolist(),
            "magnitude_db": [0.0] * n_bins,
            "order": order,
            "f1_prominence_db": 0.0,
        }

    # Full LPC polynomial: [1, a1, a2, ..., a_order]
    a_full = np.concatenate(([1.0], a_coeffs))

    # Compute frequency response
    w, h = freqz(np.array([1.0]), a_full, worN=n_fft // 2 + 1, fs=sr)
    freq_hz = np.asarray(w)
    magnitude = np.abs(np.asarray(h))

    # Convert to dB, normalize to 0 dB peak
    magnitude_db = 20.0 * np.log10(magnitude + 1e-20)
    peak_db = np.max(magnitude_db)
    magnitude_db_norm = magnitude_db - peak_db

    # Compute F1 prominence: find first spectral peak, then valley to F2 region
    f1_prominence = _lpc_f1_prominence(freq_hz, magnitude_db_norm)

    return {
        "frequencies_hz": freq_hz.tolist(),
        "magnitude_db": magnitude_db_norm.tolist(),
        "order": order,
        "f1_prominence_db": f1_prominence,
    }


def _lpc_f1_prominence(frequencies: np.ndarray, magnitude_db: np.ndarray) -> float:
    """Find F1 prominence from LPC envelope: F1 peak minus the F1-F2 valley.

    Searches for the first spectral peak in the 200-800 Hz range (F1),
    then finds the minimum between that peak and 1500 Hz (the F1-F2 valley).
    """
    from scipy.signal import find_peaks

    # Search for peaks in the F1 region (200-800 Hz)
    f1_mask = (frequencies >= 200) & (frequencies <= 800)
    if not np.any(f1_mask):
        return 0.0

    f1_indices = np.where(f1_mask)[0]
    f1_magnitudes = magnitude_db[f1_indices]

    # Find peaks in the F1 region
    peaks, _properties = find_peaks(f1_magnitudes, height=-30)
    if len(peaks) == 0:
        return 0.0

    # Use the tallest peak in the F1 region
    best_peak_local = peaks[np.argmax(f1_magnitudes[peaks])]
    f1_peak_idx = f1_indices[best_peak_local]
    f1_peak_db = magnitude_db[f1_peak_idx]
    f1_peak_freq = frequencies[f1_peak_idx]

    # Find valley between F1 peak and F2 region (~800-1500 Hz)
    valley_mask = (frequencies > f1_peak_freq) & (frequencies <= 1500)
    if not np.any(valley_mask):
        return 0.0

    valley_indices = np.where(valley_mask)[0]
    valley_min_db = np.min(magnitude_db[valley_indices])

    prominence = f1_peak_db - valley_min_db
    return round(float(prominence), 2)


# ──────────────────────────────────────────────────────────
# Inter-formant valley depth
# ──────────────────────────────────────────────────────────

def _compute_valley_depth(
    lpc_envelope: dict, formant_data: dict
) -> dict:
    """Compute inter-formant valley depths from the LPC envelope.

    For each pair of adjacent formants (F1-F2, F2-F3, F3-F4), finds the
    minimum in the LPC envelope between them. Valley depth equals the average
    of the two formant peak amplitudes minus the valley minimum.

    Returns:
        {
            "f1_f2_valley_db": float,  # Depth of valley between F1 and F2
            "f2_f3_valley_db": float,  # Depth of valley between F2 and F3
            "f3_f4_valley_db": float,  # Depth of valley between F3 and F4
            "mean_valley_db": float,
        }
    """
    frequencies = np.array(lpc_envelope["frequencies_hz"])
    magnitude_db = np.array(lpc_envelope["magnitude_db"])

    f1 = formant_data.get("f1_mean_hz", 0)
    f2 = formant_data.get("f2_mean_hz", 0)
    f3 = formant_data.get("f3_mean_hz", 0)
    f4 = formant_data.get("f4_mean_hz", 0)

    def _valley_between(low_freq: float, high_freq: float) -> float:
        """Find valley depth between two formant frequencies."""
        if low_freq <= 0 or high_freq <= 0 or low_freq >= high_freq:
            return 0.0

        # Get amplitude at each formant frequency
        low_idx = np.argmin(np.abs(frequencies - low_freq))
        high_idx = np.argmin(np.abs(frequencies - high_freq))
        low_amp = magnitude_db[low_idx]
        high_amp = magnitude_db[high_idx]

        # Find the minimum between the two formants
        if low_idx >= high_idx:
            return 0.0
        valley_region = magnitude_db[low_idx:high_idx + 1]
        valley_min = np.min(valley_region)

        # Valley depth = average of the two formant peaks minus the valley
        avg_peak = (low_amp + high_amp) / 2.0
        depth = avg_peak - valley_min
        return round(float(max(0.0, depth)), 2)

    f1_f2 = _valley_between(f1, f2)
    f2_f3 = _valley_between(f2, f3)
    f3_f4 = _valley_between(f3, f4)

    valid_depths = [d for d in [f1_f2, f2_f3, f3_f4] if d > 0]
    mean_depth = round(float(np.mean(valid_depths)), 2) if valid_depths else 0.0

    return {
        "f1_f2_valley_db": f1_f2,
        "f2_f3_valley_db": f2_f3,
        "f3_f4_valley_db": f3_f4,
        "mean_valley_db": mean_depth,
    }


# ──────────────────────────────────────────────────────────
# Correlation analysis (optional)
# ──────────────────────────────────────────────────────────

def _compute_correlations(result: dict) -> dict:
    """Compute pairwise correlations between key per-frame contours.

    Uses pitch contour, formant value tracks, and formant amplitude tracks
    when available. Returns Pearson r values for interpretable pairs.
    """
    correlations = {}

    f0_contour = result.get("pitch", {}).get("f0_contour_hz", [])
    f1_values = result.get("formants", {}).get("f1_values", [])
    f2_values = result.get("formants", {}).get("f2_values", [])

    def _pearson(a: list, b: list, label: str) -> None:
        """Compute Pearson r for two lists, adding to correlations if valid."""
        min_len = min(len(a), len(b))
        if min_len < 10:
            return
        arr_a = np.array(a[:min_len], dtype=float)
        arr_b = np.array(b[:min_len], dtype=float)
        # Filter to frames where both are non-zero
        mask = (arr_a > 0) & (arr_b > 0)
        if mask.sum() < 10:
            return
        arr_a = arr_a[mask]
        arr_b = arr_b[mask]
        if np.std(arr_a) == 0 or np.std(arr_b) == 0:
            return
        r = float(np.corrcoef(arr_a, arr_b)[0, 1])
        if not np.isnan(r):
            correlations[label] = round(r, 4)

    _pearson(f1_values, f2_values, "f1_f2")

    # Correlate F0 with F1/F2 if contour lengths are comparable
    if f0_contour and f1_values:
        _pearson(f0_contour, f1_values, "f0_f1")
    if f0_contour and f2_values:
        _pearson(f0_contour, f2_values, "f0_f2")

    return correlations


# ──────────────────────────────────────────────────────────
# Main research analysis
# ──────────────────────────────────────────────────────────

def research_analyze(
    wav_path: str,
    f0_mean_hz: float | None = None,
    crepe_device: str = "cuda:0",
    lpc_order: int = 14,
    include_lpc: bool = True,
    include_correlations: bool = False,
) -> dict:
    """Run research-grade acoustic analysis on a WAV file.

    Composes the same pipeline as ``analyze()`` but preserves all per-frame
    data (contours, amplitudes, formant tracks) and adds experimental
    metrics (LPC spectral envelope, inter-formant valley depth).

    The returned dict is a strict superset of what ``analyze()`` returns:
    all standard pipeline fields are present, plus ``lpc_envelope``,
    ``valley_depth``, and optionally ``correlations``.

    Args:
        wav_path: Path to WAV file (any sample rate, mono or stereo).
        f0_mean_hz: Pre-computed mean F0 in Hz.  If provided, CREPE is
            skipped and Praat pitch is used as a lightweight fallback for
            the pitch contour.  Useful when re-analyzing files where F0 is
            already known and you want to avoid the GPU cost.
        crepe_device: CUDA device string for CREPE pitch analysis.
        lpc_order: Order of the LPC model (default 14, suitable for
            16 kHz--48 kHz audio).
        include_lpc: Whether to compute the LPC spectral envelope and
            valley depth metrics.  Set False to save time when only the
            standard pipeline output is needed.
        include_correlations: Whether to compute pairwise Pearson
            correlations between per-frame contours.

    Returns:
        Dict with all standard analysis categories (pitch, formants,
        voice_quality, articulation, prosody) plus research-only keys.
    """
    wav_path = str(Path(wav_path).resolve())
    y, sr = _load_audio(wav_path)

    # Parselmouth Sound object (at native sample rate)
    snd = parselmouth.Sound(y, sampling_frequency=sr)

    # ── 1. Pitch analysis ──────────────────────────────────
    if f0_mean_hz is not None:
        # Caller supplied F0 — skip CREPE, use Praat for the contour
        print("  Using supplied F0 mean; Praat pitch for contour...")
        pitch_data = _praat_pitch_fallback(snd, f0_mean_hz)
    else:
        print("  Analyzing pitch (CREPE)...")
        try:
            pitch_data = analyze_pitch_crepe(y, sr, device=crepe_device)
        except Exception as exc:
            print(f"  CREPE failed ({exc}), falling back to Praat pitch...")
            pitch_data = _praat_pitch_fallback(snd)

    f0_mean = pitch_data["f0_mean_hz"]

    # ── 2. Formant analysis ────────────────────────────────
    print("  Analyzing formants (Parselmouth)...")
    formant_data = analyze_formants(snd, f0_mean_hz=f0_mean)

    # ── 3. Voice quality ───────────────────────────────────
    print("  Analyzing voice quality...")
    formant_freqs = [
        formant_data.get("f1_mean_hz", 0),
        formant_data.get("f2_mean_hz", 0),
        formant_data.get("f3_mean_hz", 0),
    ]
    formant_bws = [
        formant_data.get("bw1_mean_hz", 0),
        formant_data.get("bw2_mean_hz", 0),
        formant_data.get("bw3_mean_hz", 0),
    ]
    quality_data = analyze_voice_quality(
        snd,
        f3_mean_hz=formant_data.get("f3_mean_hz", 0.0),
        formant_freqs=formant_freqs,
        formant_bws=formant_bws,
        f1_values=formant_data.get("f1_values"),
        f2_values=formant_data.get("f2_values"),
        formant_amplitude_per_frame=formant_data.get("formant_amplitude_per_frame"),
    )

    # ── 4. Articulation ────────────────────────────────────
    print("  Analyzing articulation...")
    articulation_data = analyze_articulation(y, sr, formant_data)

    # ── 5. Prosody ─────────────────────────────────────────
    print("  Analyzing prosody...")
    prosody_data = analyze_prosody(pitch_data)

    # ── 6. LPC envelope + valley depth (research-only) ────
    lpc_envelope = None
    valley_depth = None
    if include_lpc:
        print("  Computing LPC spectral envelope...")
        lpc_envelope = _compute_lpc_envelope(y, sr, order=lpc_order)
        valley_depth = _compute_valley_depth(lpc_envelope, formant_data)

    # ── 7. Correlations (optional) ─────────────────────────
    # Build a preliminary result so _compute_correlations can access it
    correlations = None

    # ── Assemble result ────────────────────────────────────
    # Preserve ALL per-frame data — do NOT strip contours
    result = {
        "source_file": wav_path,
        "sample_rate": sr,
        "duration_s": float(len(y) / sr),
        "pitch": pitch_data,
        "formants": formant_data,  # Includes f1_values, f2_values, formant_amplitude_per_frame
        "voice_quality": quality_data,
        "articulation": articulation_data,
        "prosody": prosody_data,
    }

    # Research-only keys
    if lpc_envelope is not None:
        result["lpc_envelope"] = lpc_envelope
    if valley_depth is not None:
        result["valley_depth"] = valley_depth

    if include_correlations:
        print("  Computing correlations...")
        correlations = _compute_correlations(result)
        result["correlations"] = correlations

    return result


def _praat_pitch_fallback(
    snd: parselmouth.Sound, f0_mean_override: float | None = None
) -> dict:
    """Lightweight pitch analysis using Praat (no GPU).

    Used when CREPE is unavailable or when the caller pre-supplies f0_mean_hz.
    Produces the same dict shape as ``analyze_pitch_crepe``.
    """
    from parselmouth.praat import call

    pitch_obj = call(
        snd, "To Pitch (ac)", 0.0, 50.0, 15, False, 0.03, 0.45, 0.01, 0.35, 0.14, 550.0
    )
    n_frames = call(pitch_obj, "Get number of frames")
    frequency, confidence, time_arr = [], [], []
    for i in range(1, int(n_frames) + 1):
        t = call(pitch_obj, "Get time from frame number", i)
        f0 = call(pitch_obj, "Get value in frame", i, "Hertz")
        time_arr.append(float(t))
        if np.isnan(f0) or f0 == 0:
            frequency.append(0.0)
            confidence.append(0.0)
        else:
            frequency.append(float(f0))
            confidence.append(1.0)

    frequency = np.array(frequency)
    confidence = np.array(confidence)
    time_arr = np.array(time_arr)

    mask = confidence > 0.5
    voiced_f0 = frequency[mask]

    if len(voiced_f0) == 0:
        return {
            "f0_mean_hz": f0_mean_override or 0.0,
            "f0_median_hz": 0.0,
            "f0_std_hz": 0.0,
            "f0_min_hz": 0.0,
            "f0_max_hz": 0.0,
            "f0_contour_hz": [],
            "f0_contour_time_s": [],
            "f0_confidence": [],
            "voiced_fraction": 0.0,
        }

    result = {
        "f0_mean_hz": f0_mean_override or float(np.mean(voiced_f0)),
        "f0_median_hz": float(np.median(voiced_f0)),
        "f0_std_hz": float(np.std(voiced_f0)),
        "f0_min_hz": float(np.min(voiced_f0)),
        "f0_max_hz": float(np.max(voiced_f0)),
        "f0_contour_hz": frequency.tolist(),
        "f0_contour_time_s": time_arr.tolist(),
        "f0_confidence": confidence.tolist(),
        "voiced_fraction": float(mask.sum() / len(mask)),
    }
    return result


# ──────────────────────────────────────────────────────────
# Batch processing
# ──────────────────────────────────────────────────────────

def research_analyze_batch(
    wav_dir: str,
    output_dir: str | None = None,
    crepe_device: str = "cuda:0",
    **kwargs,
) -> dict[str, dict]:
    """Run research analysis on all WAV files in a directory.

    Args:
        wav_dir: Directory containing WAV files.
        output_dir: If provided, saves per-file JSON results here.
        crepe_device: CUDA device for CREPE.
        **kwargs: Forwarded to ``research_analyze`` (e.g. lpc_order,
            include_lpc, include_correlations).

    Returns:
        Dict keyed by filename (without extension), values are analysis dicts.
    """
    wav_path = Path(wav_dir)
    if not wav_path.is_dir():
        raise ValueError(f"Not a directory: {wav_path}")

    wav_files = sorted(wav_path.glob("*.wav"))
    if not wav_files:
        print(f"  No WAV files found in {wav_dir}")
        return {}

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = None

    results: dict[str, dict] = {}
    total = len(wav_files)

    for idx, wav_file in enumerate(wav_files, 1):
        stem = wav_file.stem
        print(f"\n[{idx}/{total}] Analyzing {wav_file.name}...")
        try:
            result = research_analyze(
                str(wav_file), crepe_device=crepe_device, **kwargs
            )
            results[stem] = result

            if out_path is not None:
                json_path = out_path / f"{stem}.json"
                _save_result(result, str(json_path))
                print(f"  Saved to {json_path}")
        except Exception as exc:
            print(f"  ERROR analyzing {wav_file.name}: {exc}")
            results[stem] = {"error": str(exc)}

    return results


# ──────────────────────────────────────────────────────────
# Comparison
# ──────────────────────────────────────────────────────────

def compare_results(
    a: dict, b: dict, metrics: list[str] | None = None
) -> dict:
    """Compare two research analysis results.

    For each metric, computes the absolute and relative delta.  When
    ``metrics`` is None, compares all shared top-level numeric fields and
    all numeric fields within the standard sub-dicts (pitch, formants,
    voice_quality, articulation, prosody).

    Args:
        a: First analysis result dict.
        b: Second analysis result dict.
        metrics: Optional list of dot-separated metric paths to compare
            (e.g. ``["pitch.f0_mean_hz", "formants.delta_f_hz"]``).  If
            None, compares all shared numeric fields.

    Returns:
        Dict with per-metric deltas and summary statistics::

            {
                "deltas": {
                    "pitch.f0_mean_hz": {"a": 180.0, "b": 210.0,
                                          "delta": 30.0, "pct": 16.7},
                    ...
                },
                "summary": {
                    "n_compared": int,
                    "largest_pct_change": str,
                    "largest_abs_change": str,
                }
            }
    """

    def _get_nested(d: dict, path: str):
        """Retrieve a value from a nested dict using dot-separated path."""
        parts = path.split(".")
        current = d
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    def _collect_numeric_paths(d: dict, prefix: str = "") -> dict[str, float]:
        """Recursively collect all numeric leaf values with dot paths."""
        paths: dict[str, float] = {}
        for key, val in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                paths[full_key] = float(val)
            elif isinstance(val, dict):
                # Only recurse into standard sub-dicts (not huge arrays)
                if key in ("pitch", "formants", "voice_quality",
                           "articulation", "prosody", "valley_depth",
                           "lpc_envelope"):
                    paths.update(_collect_numeric_paths(val, full_key))
        return paths

    if metrics is not None:
        # Compare only the specified metrics
        paths_to_compare = metrics
    else:
        # Auto-discover all shared numeric paths
        a_paths = _collect_numeric_paths(a)
        b_paths = _collect_numeric_paths(b)
        paths_to_compare = sorted(set(a_paths.keys()) & set(b_paths.keys()))

    deltas: dict[str, dict] = {}
    for path in paths_to_compare:
        val_a = _get_nested(a, path)
        val_b = _get_nested(b, path)
        if val_a is None or val_b is None:
            continue
        if not isinstance(val_a, (int, float)) or not isinstance(val_b, (int, float)):
            continue
        va = float(val_a)
        vb = float(val_b)
        delta = vb - va
        pct = (delta / va * 100.0) if va != 0 else (0.0 if delta == 0 else float("inf"))
        deltas[path] = {
            "a": round(va, 4),
            "b": round(vb, 4),
            "delta": round(delta, 4),
            "pct": round(pct, 2),
        }

    # Summary statistics
    largest_pct = ""
    largest_abs = ""
    max_pct = 0.0
    max_abs = 0.0
    for path, d in deltas.items():
        abs_pct = abs(d["pct"]) if d["pct"] != float("inf") else 0.0
        abs_delta = abs(d["delta"])
        if abs_pct > max_pct:
            max_pct = abs_pct
            largest_pct = path
        if abs_delta > max_abs:
            max_abs = abs_delta
            largest_abs = path

    return {
        "deltas": deltas,
        "summary": {
            "n_compared": len(deltas),
            "largest_pct_change": largest_pct,
            "largest_abs_change": largest_abs,
        },
    }


# ──────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────

def _save_result(result: dict, json_path: str) -> None:
    """Save analysis result to JSON, converting numpy types."""

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, cls=_NumpyEncoder)


# ──────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Research-grade voice analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input", nargs="?", help="WAV file or directory of WAV files"
    )
    parser.add_argument(
        "-o", "--output", help="Output JSON path (single file) or directory (batch)"
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("A", "B"),
        help="Compare two result JSON files"
    )
    parser.add_argument(
        "--device", default="cuda:0", help="CREPE device (default: cuda:0)"
    )
    parser.add_argument(
        "--no-lpc", action="store_true", help="Skip LPC envelope computation"
    )
    parser.add_argument(
        "--no-crepe", action="store_true",
        help="Skip CREPE pitch analysis (use Praat F0 instead)"
    )
    parser.add_argument(
        "--correlations", action="store_true",
        help="Include pairwise contour correlations"
    )
    parser.add_argument(
        "--lpc-order", type=int, default=14,
        help="LPC model order (default: 14)"
    )

    args = parser.parse_args()

    # ── Compare mode ──
    if args.compare:
        a_path, b_path = args.compare
        with open(a_path) as f:
            data_a = json.load(f)
        with open(b_path) as f:
            data_b = json.load(f)

        comparison = compare_results(data_a, data_b)
        print(f"\nComparing {a_path} vs {b_path}")
        print(f"  {comparison['summary']['n_compared']} metrics compared")
        print(f"  Largest % change: {comparison['summary']['largest_pct_change']}")
        print(f"  Largest abs change: {comparison['summary']['largest_abs_change']}")
        print()

        # Print top deltas sorted by absolute percentage change
        sorted_deltas = sorted(
            comparison["deltas"].items(),
            key=lambda x: abs(x[1]["pct"]) if x[1]["pct"] != float("inf") else 0,
            reverse=True,
        )
        for metric, d in sorted_deltas[:20]:
            print(f"  {metric:45s}  {d['a']:>10.2f} -> {d['b']:>10.2f}"
                  f"  ({d['delta']:+.2f}, {d['pct']:+.1f}%)")

        if args.output:
            _save_result(comparison, args.output)
            print(f"\nFull comparison saved to {args.output}")
        sys.exit(0)

    # ── Require input for non-compare modes ──
    if not args.input:
        parser.error("Input WAV file or directory is required (or use --compare)")

    input_path = Path(args.input)

    # Build common kwargs
    device = "__no_crepe__" if args.no_crepe else args.device
    analyze_kwargs = {
        "include_lpc": not args.no_lpc,
        "include_correlations": args.correlations,
        "lpc_order": args.lpc_order,
        "crepe_device": device,
    }

    # ── Directory mode ──
    if input_path.is_dir():
        print(f"Batch analysis: {input_path}")
        results = research_analyze_batch(
            str(input_path),
            output_dir=args.output,
            **analyze_kwargs,
        )
        print(f"\nDone. Analyzed {len(results)} files.")
        # Print summary
        for name, res in results.items():
            if "error" in res:
                print(f"  {name}: ERROR - {res['error']}")
            else:
                f0 = res.get("pitch", {}).get("f0_mean_hz", 0)
                df = res.get("formants", {}).get("delta_f_hz", 0)
                print(f"  {name}: F0={f0:.1f} Hz, dF={df:.1f} Hz")
        sys.exit(0)

    # ── Single file mode ──
    if not input_path.is_file():
        parser.error(f"File not found: {input_path}")

    print(f"Analyzing: {input_path}")

    result = research_analyze(str(input_path), **analyze_kwargs)

    # Print summary
    pitch = result["pitch"]
    formants = result["formants"]
    vq = result["voice_quality"]
    prosody = result["prosody"]

    print(f"\n{'='*60}")
    print(f"  File:     {input_path.name}")
    print(f"  Duration: {result['duration_s']:.2f}s @ {result['sample_rate']} Hz")
    print(f"{'='*60}")
    print(f"  Pitch:      F0 mean={pitch['f0_mean_hz']:.1f} Hz, "
          f"median={pitch['f0_median_hz']:.1f} Hz, "
          f"std={pitch['f0_std_hz']:.1f} Hz")
    print(f"  Formants:   F1={formants['f1_mean_hz']:.0f}, "
          f"F2={formants['f2_mean_hz']:.0f}, "
          f"F3={formants['f3_mean_hz']:.0f}, "
          f"F4={formants['f4_mean_hz']:.0f} Hz")
    print(f"  Delta-F:    {formants['delta_f_hz']:.1f} Hz, "
          f"VTL={formants['vocal_tract_length_cm']:.1f} cm")
    print(f"  Quality:    HNR={vq['hnr_db']:.1f} dB, "
          f"H1-H2={vq['h1_h2_db']:.1f} dB (raw), "
          f"CPP={vq['cpp_db']:.1f} dB")
    print(f"  Prosody:    CV={prosody['f0_cv']:.3f}, "
          f"range={prosody['pitch_range_semitones']:.1f} st")

    if "lpc_envelope" in result:
        lpc = result["lpc_envelope"]
        print(f"  LPC:        order={lpc['order']}, "
              f"F1 prominence={lpc['f1_prominence_db']:.1f} dB")
    if "valley_depth" in result:
        vd = result["valley_depth"]
        print(f"  Valleys:    F1-F2={vd['f1_f2_valley_db']:.1f}, "
              f"F2-F3={vd['f2_f3_valley_db']:.1f}, "
              f"F3-F4={vd['f3_f4_valley_db']:.1f} dB "
              f"(mean={vd['mean_valley_db']:.1f})")
    if "correlations" in result:
        print(f"  Correlations: {result['correlations']}")

    # Save output
    output_path = args.output or str(input_path.with_suffix(".research.json"))
    _save_result(result, output_path)
    print(f"\n  Results saved to {output_path}")
