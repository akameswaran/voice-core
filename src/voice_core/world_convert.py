"""WORLD vocoder wrapper for voice conversion experiments.

Uses pyworld (WORLD vocoder) for decomposed voice manipulation:
pitch shifting, spectral envelope warping, tilt modification,
and aperiodicity (breathiness) control.
"""

import numpy as np
import pyworld as pw
import soundfile as sf
from scipy.interpolate import interp1d


def load_wav(wav_path, sr=None):
    """Load WAV file. If sr given, resample."""
    y, file_sr = sf.read(wav_path, dtype='float64')
    if y.ndim > 1:
        y = y.mean(axis=1)  # mono
    if sr is not None and sr != file_sr:
        from librosa import resample
        y = resample(y, orig_sr=file_sr, target_sr=sr)
        return y, sr
    return y, file_sr


def analyze(wav_path, sr=None):
    """Extract WORLD parameters: F0, spectral envelope, aperiodicity.

    Returns (f0, sp, ap, t, sr) where:
      f0: fundamental frequency contour (numpy array)
      sp: spectral envelope (2D array: frames × freq_bins)
      ap: aperiodicity (2D array: same shape as sp)
      t: time axis (numpy array)
      sr: sample rate used
    """
    y, actual_sr = load_wav(wav_path, sr)
    # DIO for F0 estimation, then StoneMask for refinement
    f0, t = pw.dio(y, actual_sr)
    f0 = pw.stonemask(y, f0, t, actual_sr)
    # CheapTrick for spectral envelope
    sp = pw.cheaptrick(y, f0, t, actual_sr)
    # D4C for aperiodicity
    ap = pw.d4c(y, f0, t, actual_sr)
    return f0, sp, ap, t, actual_sr


def shift_pitch(f0, semitones):
    """Shift pitch by given semitones. Preserves unvoiced frames (f0==0)."""
    f0_shifted = f0.copy()
    voiced = f0_shifted > 0
    f0_shifted[voiced] *= 2.0 ** (semitones / 12.0)
    return f0_shifted


def shift_pitch_hz(f0, delta_hz):
    """Shift pitch by absolute Hz amount. Preserves unvoiced frames."""
    f0_shifted = f0.copy()
    voiced = f0_shifted > 0
    f0_shifted[voiced] += delta_hz
    f0_shifted[voiced] = np.maximum(f0_shifted[voiced], 50.0)  # floor at 50 Hz
    return f0_shifted


def warp_spectral_envelope(sp, warp_factor):
    """Warp spectral envelope along frequency axis.

    warp_factor > 1.0 = shorter vocal tract = higher formants (more feminine)
    warp_factor < 1.0 = longer vocal tract = lower formants (more masculine)

    Uses cubic interpolation on each frame independently.
    """
    if abs(warp_factor - 1.0) < 1e-6:
        return sp.copy()

    n_frames, n_bins = sp.shape
    sp_warped = np.zeros_like(sp)

    # Original frequency indices (normalized 0..1)
    orig_freqs = np.arange(n_bins) / (n_bins - 1)
    # Warped frequency indices — where to sample from
    warped_freqs = orig_freqs / warp_factor

    for i in range(n_frames):
        interp_func = interp1d(
            orig_freqs, sp[i], kind='cubic',
            bounds_error=False, fill_value=(sp[i, 0], sp[i, -1])
        )
        sp_warped[i] = interp_func(warped_freqs)

    return sp_warped


def modify_spectral_tilt(sp, tilt_db_per_octave, sr):
    """Modify spectral tilt (slope) of the spectral envelope.

    Negative tilt_db_per_octave = steeper rolloff (more feminine/lighter)
    Positive = flatter spectrum (more masculine/heavier)

    Applied as a frequency-dependent gain relative to a reference (1000 Hz).
    """
    if abs(tilt_db_per_octave) < 1e-6:
        return sp.copy()

    n_frames, n_bins = sp.shape
    # Frequency axis for the spectral envelope
    freqs = np.linspace(0, sr / 2, n_bins)

    # Avoid log(0): start from a small positive frequency
    ref_hz = 1000.0
    gain_db = np.zeros(n_bins)
    valid = freqs > 0
    # Octaves relative to reference
    octaves = np.zeros(n_bins)
    octaves[valid] = np.log2(freqs[valid] / ref_hz)
    gain_db = octaves * tilt_db_per_octave

    # Convert dB gain to linear (sp is power spectrum, so dB/10)
    gain_linear = 10.0 ** (gain_db / 10.0)

    sp_modified = sp.copy()
    for i in range(n_frames):
        sp_modified[i] *= gain_linear

    return sp_modified


def modify_aperiodicity(ap, breathiness_delta, center_hz=3000, sr=44100):
    """Modify aperiodicity to adjust breathiness.

    breathiness_delta > 0 = more breathy (increase aperiodicity)
    breathiness_delta < 0 = less breathy

    Applied with a Gaussian window centered at center_hz to avoid
    affecting low frequencies (which would make it sound broken).
    """
    if abs(breathiness_delta) < 1e-6:
        return ap.copy()

    n_frames, n_bins = ap.shape
    freqs = np.linspace(0, sr / 2, n_bins)

    # Gaussian window centered at center_hz, width ~2000 Hz
    sigma_hz = 2000.0
    window = np.exp(-0.5 * ((freqs - center_hz) / sigma_hz) ** 2)

    ap_modified = ap.copy()
    for i in range(n_frames):
        ap_modified[i] = np.clip(ap_modified[i] + breathiness_delta * window, 0.0, 1.0)

    return ap_modified


def synthesize(f0, sp, ap, sr):
    """Synthesize waveform from WORLD parameters.

    Returns numpy array (float64) of the synthesized waveform.
    """
    return pw.synthesize(f0, sp, ap, sr)


def save_wav(wav_path, y, sr):
    """Save waveform to WAV file."""
    sf.write(wav_path, y, sr, subtype='PCM_16')


def convert(wav_path, output_path, pitch_semitones=0, pitch_hz=0,
            warp_factor=1.0, tilt_db_per_octave=0.0,
            breathiness_delta=0.0, breathiness_center_hz=3000):
    """Full conversion pipeline: load → analyze → modify → synthesize → save.

    Args:
        wav_path: Input WAV file path
        output_path: Output WAV file path
        pitch_semitones: Pitch shift in semitones (applied first)
        pitch_hz: Additional pitch shift in Hz (applied after semitones)
        warp_factor: Spectral envelope warp (>1 = higher formants)
        tilt_db_per_octave: Spectral tilt change (negative = steeper/lighter)
        breathiness_delta: Aperiodicity change (positive = breathier)
        breathiness_center_hz: Center frequency for breathiness Gaussian

    Returns:
        dict with conversion parameters and output path
    """
    f0, sp, ap, t, sr = analyze(wav_path)

    # Apply modifications
    if pitch_semitones != 0:
        f0 = shift_pitch(f0, pitch_semitones)
    if pitch_hz != 0:
        f0 = shift_pitch_hz(f0, pitch_hz)
    if warp_factor != 1.0:
        sp = warp_spectral_envelope(sp, warp_factor)
    if tilt_db_per_octave != 0:
        sp = modify_spectral_tilt(sp, tilt_db_per_octave, sr)
    if breathiness_delta != 0:
        ap = modify_aperiodicity(ap, breathiness_delta, breathiness_center_hz, sr)

    # Synthesize and save
    y = synthesize(f0, sp, ap, sr)
    save_wav(output_path, y, sr)

    return {
        "input": wav_path,
        "output": output_path,
        "sr": sr,
        "pitch_semitones": pitch_semitones,
        "pitch_hz": pitch_hz,
        "warp_factor": warp_factor,
        "tilt_db_per_octave": tilt_db_per_octave,
        "breathiness_delta": breathiness_delta,
    }
