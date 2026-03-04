"""Phoneme-aligned formant analysis using Montreal Forced Aligner.

Provides per-vowel formant extraction with known phoneme identity,
enabling reliable articulatory gesture decomposition (OPC, tongue
fronting, lip spread) that isn't contaminated by vowel distribution
differences between recordings.

Requires:
  - MFA conda env: conda create -n mfa -c conda-forge montreal-forced-aligner
  - Models: mfa model download {acoustic,dictionary,g2p} english_mfa
  - praatio: pip install praatio (for TextGrid parsing)
"""

import json
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import parselmouth
from parselmouth import praat
from praatio import textgrid

# MFA IPA phone labels → ARPA vowel categories (matching vowel_norms.json)
IPA_TO_ARPA = {
    "iː": "IY", "i": "IY", "ɪ": "IH", "ej": "EY", "eː": "EY",
    "ɛ": "EH", "æ": "AE", "ɑ": "AA", "ɒ": "AA", "a": "AH",
    "ɐ": "AH", "ʌ": "AH", "ə": "AH", "ɔ": "AO",
    "oʊ": "OW", "ow": "OW", "əw": "OW",
    "ʊ": "UW", "uː": "UW", "u": "UW",
    "aj": "AY", "aɪ": "AY", "aw": "AW",
    "ɔj": "OY", "ɔɪ": "OY",
    "ɜː": "ER", "ɝ": "ER", "ɚ": "ER",
}

VOWEL_NORMS_PATH = Path(__file__).parent / "data" / "vowel_norms.json"
MFA_CONDA_ENV = "mfa"

_vowel_norms_cache = None


def _get_vowel_norms() -> dict:
    global _vowel_norms_cache
    if _vowel_norms_cache is None:
        with open(VOWEL_NORMS_PATH) as f:
            _vowel_norms_cache = json.load(f)["vowels"]
    return _vowel_norms_cache


def align(wav_path: str, transcript: str,
          output_textgrid: Optional[str] = None) -> str:
    """Run MFA forced alignment on a single audio file.

    Args:
        wav_path: Path to WAV file.
        transcript: Orthographic transcript of the audio.
        output_textgrid: Output TextGrid path. If None, writes next to WAV.

    Returns:
        Path to the output TextGrid file.
    """
    wav_path = str(Path(wav_path).resolve())
    if output_textgrid is None:
        output_textgrid = wav_path.replace(".wav", ".TextGrid")

    # Write transcript to temp file next to audio
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                      dir=Path(wav_path).parent,
                                      delete=False) as f:
        f.write(transcript)
        txt_path = f.name

    try:
        result = subprocess.run(
            ["conda", "run", "-n", MFA_CONDA_ENV,
             "mfa", "align_one",
             wav_path, txt_path,
             "english_mfa", "english_mfa",
             output_textgrid],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"MFA alignment failed: {result.stderr[:500]}")
    finally:
        os.unlink(txt_path)

    return output_textgrid


def extract_vowel_formants(wav_path: str, textgrid_path: str,
                           formant_ceiling: int = 5500,
                           min_vowel_dur: float = 0.03) -> dict:
    """Extract F1/F2/F3 at vowel midpoints from MFA-aligned audio.

    Args:
        wav_path: Path to WAV file.
        textgrid_path: Path to MFA TextGrid output.
        formant_ceiling: Formant ceiling for Burg algorithm.
        min_vowel_dur: Minimum vowel duration to include (seconds).

    Returns:
        Dict mapping ARPA vowel category to list of (f1, f2, f3) tuples.
    """
    snd = parselmouth.Sound(wav_path)
    tg = textgrid.openTextgrid(textgrid_path, includeEmptyIntervals=False)
    phones = tg.getTier("phones")

    results = defaultdict(list)
    for start, end, label in phones.entries:
        if label not in IPA_TO_ARPA:
            continue
        dur = end - start
        if dur < min_vowel_dur:
            continue

        cat = IPA_TO_ARPA[label]
        mid = (start + end) / 2
        margin = min(0.025, dur / 4)

        segment = snd.extract_part(
            max(mid - margin, start), min(mid + margin, end))
        formant = praat.call(
            segment, "To Formant (burg)", 0.0, 5,
            formant_ceiling, 0.025, 50.0)
        t_mid = segment.duration / 2

        f1 = praat.call(formant, "Get value at time",
                        1, t_mid, "Hertz", "Linear")
        f2 = praat.call(formant, "Get value at time",
                        2, t_mid, "Hertz", "Linear")
        f3 = praat.call(formant, "Get value at time",
                        3, t_mid, "Hertz", "Linear")

        if f1 > 0 and f2 > 0 and f3 > 0:
            results[cat].append((f1, f2, f3))

    return dict(results)


def compute_gesture_zscores(vowel_formants: dict) -> dict:
    """Compute per-vowel and overall gesture z-scores against norms.

    Args:
        vowel_formants: Output of extract_vowel_formants().

    Returns:
        Dict with:
          per_vowel: {category: {n, f1, f2, f3, f1z, f2z, f3z}}
          overall: {f1z, f2z, f3z, n_vowels, n_categories}
          gestures: {opc, tongue_fronting, lip_spread} with score and tip
    """
    norms = _get_vowel_norms()
    per_vowel = {}
    all_f1z, all_f2z, all_f3z = [], [], []

    for cat in sorted(vowel_formants.keys()):
        vals = vowel_formants[cat]
        if len(vals) < 2 or cat not in norms:
            continue

        f1m = float(np.mean([v[0] for v in vals]))
        f2m = float(np.mean([v[1] for v in vals]))
        f3m = float(np.mean([v[2] for v in vals]))

        norm = norms[cat]
        f1z = (f1m - norm["f1_mean"]) / norm["f1_std"]
        f2z = (f2m - norm["f2_mean"]) / norm["f2_std"]
        f3z = (f3m - norm["f3_mean"]) / norm["f3_std"]

        per_vowel[cat] = {
            "n": len(vals),
            "f1": round(f1m, 1), "f2": round(f2m, 1), "f3": round(f3m, 1),
            "f1z": round(f1z, 3), "f2z": round(f2z, 3), "f3z": round(f3z, 3),
        }
        all_f1z.append(f1z)
        all_f2z.append(f2z)
        all_f3z.append(f3z)

    if not all_f1z:
        return {"per_vowel": {}, "overall": {}, "gestures": {}}

    f1z_mean = float(np.mean(all_f1z))
    f2z_mean = float(np.mean(all_f2z))
    f3z_mean = float(np.mean(all_f3z))

    n_vowels = sum(v["n"] for v in per_vowel.values())

    # Map to articulatory gestures
    gestures = {
        "opc": {
            "zscore": round(f1z_mean, 3),
            "label": "OPC / Pharynx narrowing",
            "direction": "F1 relative to vowel norms",
            "interpretation": (
                "engaged" if f1z_mean > 0.3 else
                "minimal" if f1z_mean > -0.3 else
                "wide (wrong direction)"
            ),
        },
        "tongue_fronting": {
            "zscore": round(f2z_mean, 3),
            "label": "Tongue fronting",
            "direction": "F2 relative to vowel norms",
            "interpretation": (
                "strong fronting" if f2z_mean > 0.8 else
                "moderate fronting" if f2z_mean > 0.3 else
                "minimal" if f2z_mean > -0.3 else
                "retracted (wrong direction)"
            ),
        },
        "lip_spread": {
            "zscore": round(f3z_mean, 3),
            "label": "Lip spread / front cavity",
            "direction": "F3 relative to vowel norms",
            "interpretation": (
                "spreading" if f3z_mean > 0.3 else
                "neutral" if f3z_mean > -0.3 else
                "rounding (darkening)"
            ),
        },
    }

    return {
        "per_vowel": per_vowel,
        "overall": {
            "f1z": round(f1z_mean, 3),
            "f2z": round(f2z_mean, 3),
            "f3z": round(f3z_mean, 3),
            "n_vowels": n_vowels,
            "n_categories": len(per_vowel),
        },
        "gestures": gestures,
    }


def compare_gesture_zscores(baseline: dict, target: dict) -> dict:
    """Compare gesture z-scores between two recordings.

    Args:
        baseline: Output of compute_gesture_zscores() for baseline (e.g. masc).
        target: Output of compute_gesture_zscores() for target (e.g. femme).

    Returns:
        Dict with delta z-scores and articulatory interpretation.
    """
    b = baseline.get("overall", {})
    t = target.get("overall", {})

    if not b or not t:
        return {"error": "Missing data for comparison"}

    d_f1z = t.get("f1z", 0) - b.get("f1z", 0)
    d_f2z = t.get("f2z", 0) - b.get("f2z", 0)
    d_f3z = t.get("f3z", 0) - b.get("f3z", 0)

    # Per-vowel deltas for shared vowels
    per_vowel_delta = {}
    b_vowels = baseline.get("per_vowel", {})
    t_vowels = target.get("per_vowel", {})
    for cat in sorted(set(b_vowels) & set(t_vowels)):
        bv = b_vowels[cat]
        tv = t_vowels[cat]
        per_vowel_delta[cat] = {
            "df1": round(tv["f1"] - bv["f1"], 1),
            "df2": round(tv["f2"] - bv["f2"], 1),
            "df3": round(tv["f3"] - bv["f3"], 1),
            "df1z": round(tv["f1z"] - bv["f1z"], 3),
            "df2z": round(tv["f2z"] - bv["f2z"], 3),
            "df3z": round(tv["f3z"] - bv["f3z"], 3),
        }

    # Priorities
    priorities = []
    if d_f1z < 0.3:
        priorities.append({
            "gesture": "OPC",
            "delta_z": round(d_f1z, 2),
            "tip": ("Not engaging enough. Practice big_dog_small_dog "
                    "focusing on pharynx narrowing, not just larynx raise."),
        })
    if d_f2z < 0.5:
        priorities.append({
            "gesture": "Tongue fronting",
            "delta_z": round(d_f2z, 2),
            "tip": ("Not fronting enough. Practice with /i/ and /e/ vowels, "
                    "tongue tip behind lower teeth, body pushed forward."),
        })
    if d_f3z < 0:
        priorities.append({
            "gesture": "Lip position",
            "delta_z": round(d_f3z, 2),
            "tip": ("Rounding/darkening. Try a gentle smile posture "
                    "(lip spread) while speaking."),
        })

    return {
        "delta_overall": {
            "f1z": round(d_f1z, 3),
            "f2z": round(d_f2z, 3),
            "f3z": round(d_f3z, 3),
        },
        "per_vowel_delta": per_vowel_delta,
        "interpretation": {
            "opc": (
                "engaging" if d_f1z > 0.3 else
                "minimal change" if abs(d_f1z) < 0.3 else
                "disengaging"
            ),
            "tongue": (
                "strong fronting" if d_f2z > 0.8 else
                "moderate fronting" if d_f2z > 0.3 else
                "minimal change" if abs(d_f2z) < 0.3 else
                "retracting"
            ),
            "lips": (
                "spreading" if d_f3z > 0.3 else
                "neutral" if abs(d_f3z) < 0.3 else
                "rounding"
            ),
        },
        "priorities": priorities,
    }


def analyze_recording(wav_path: str, transcript: str,
                      cache_dir: Optional[str] = None) -> dict:
    """Full pipeline: align + extract + z-score a single recording.

    Args:
        wav_path: Path to WAV file.
        transcript: Orthographic text of what was spoken.
        cache_dir: Directory to cache TextGrid files. If None, uses temp dir.

    Returns:
        Output of compute_gesture_zscores().
    """
    wav_path = str(Path(wav_path).resolve())

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        stem = Path(wav_path).stem
        tg_path = os.path.join(cache_dir, f"{stem}.TextGrid")
    else:
        tg_path = wav_path.replace(".wav", ".TextGrid")

    if not os.path.exists(tg_path):
        align(wav_path, transcript, tg_path)

    vowel_formants = extract_vowel_formants(wav_path, tg_path)
    return compute_gesture_zscores(vowel_formants)
