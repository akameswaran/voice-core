"""Segment-level analysis for voice feminization scoring.

Splits audio on silence/phrase boundaries, creates 2-5 second analysis windows,
runs the full analysis + scoring pipeline on each segment, and identifies
segments with notable score changes.
"""

import json
import os
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf


def detect_segments(wav_path: str, min_duration: float = 2.0,
                    max_duration: float = 5.0,
                    silence_thresh_db: float = -25.0,
                    min_silence_len: float = 0.3) -> list[dict]:
    """Detect phrase segments based on silence boundaries.

    Args:
        wav_path: Path to WAV file.
        min_duration: Minimum segment duration in seconds.
        max_duration: Maximum segment duration in seconds.
        silence_thresh_db: Silence threshold in dB below peak RMS.
        min_silence_len: Minimum silence duration to split on (seconds).

    Returns:
        List of dicts with start_s, end_s, duration_s for each segment.
    """
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    duration = len(y) / sr

    # Use librosa's RMS energy to detect silence
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max(rms))

    # Frames below threshold = silence
    is_silent = rms_db < silence_thresh_db
    frame_duration = hop_length / sr

    # Find silence regions
    silence_regions = []
    in_silence = False
    silence_start = 0

    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            silence_start = i
            in_silence = True
        elif not silent and in_silence:
            silence_dur = (i - silence_start) * frame_duration
            if silence_dur >= min_silence_len:
                mid_time = (silence_start + i) / 2 * frame_duration
                silence_regions.append(mid_time)
            in_silence = False

    # Build segments from silence boundaries
    boundaries = [0.0] + silence_regions + [duration]
    raw_segments = []

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        dur = end - start
        if dur >= min_duration:
            raw_segments.append({"start_s": start, "end_s": end, "duration_s": dur})

    # Split segments that are too long
    segments = []
    for seg in raw_segments:
        if seg["duration_s"] <= max_duration:
            segments.append(seg)
        else:
            # Split into chunks of max_duration
            start = seg["start_s"]
            while start < seg["end_s"]:
                end = min(start + max_duration, seg["end_s"])
                dur = end - start
                if dur >= min_duration:
                    segments.append({"start_s": round(start, 3), "end_s": round(end, 3), "duration_s": round(dur, 3)})
                start = end

    # Merge segments that are too short with neighbors
    if not segments:
        # No good segments found — treat whole file as one segment
        if duration >= 1.0:
            segments = [{"start_s": 0.0, "end_s": duration, "duration_s": duration}]

    return segments


def extract_segment_audio(wav_path: str, start_s: float, end_s: float,
                          output_path: str) -> str:
    """Extract a segment from a WAV file and save it."""
    y, sr = librosa.load(wav_path, sr=None, mono=True,
                         offset=start_s, duration=end_s - start_s)
    sf.write(output_path, y, sr)
    return output_path


def analyze_segments(wav_path: str, output_dir: str | None = None,
                     crepe_device: str = "cuda:0",
                     min_duration: float = 2.0,
                     max_duration: float = 5.0,
                     score_fn=None) -> dict:
    """Run full analysis + scoring on each segment of an audio file.

    Args:
        wav_path: Path to WAV file.
        output_dir: Directory to save segment results. Defaults to
            segments/ next to the WAV file.
        crepe_device: CUDA device for CREPE.
        min_duration: Minimum segment duration.
        max_duration: Maximum segment duration.
        score_fn: Optional scoring callable(analysis_dict, output_path=str) -> dict.
            If None, scoring is skipped.

    Returns:
        Dict with segments list, summary stats, and notable changes.
    """
    from voice_core.analyze import analyze

    wav_path = str(Path(wav_path).resolve())

    if output_dir is None:
        output_dir = str(Path(wav_path).parent / "segments")
    os.makedirs(output_dir, exist_ok=True)

    # Detect segments
    print("  Detecting phrase segments...")
    segments = detect_segments(wav_path, min_duration=min_duration, max_duration=max_duration)
    print(f"  Found {len(segments)} segments")

    results = []
    for i, seg in enumerate(segments):
        seg_id = f"seg_{i:03d}"
        seg_wav = os.path.join(output_dir, f"{seg_id}.wav")
        seg_analysis = os.path.join(output_dir, f"{seg_id}_analysis.json")
        seg_scores = os.path.join(output_dir, f"{seg_id}_scores.json")

        print(f"\n  Segment {i+1}/{len(segments)}: {seg['start_s']:.1f}s - {seg['end_s']:.1f}s")

        # Extract segment audio
        extract_segment_audio(wav_path, seg["start_s"], seg["end_s"], seg_wav)

        # Run analysis
        analysis_result = analyze(seg_wav, output_path=seg_analysis, crepe_device=crepe_device)

        # Run scoring if score_fn provided
        result_entry = {
            "segment_id": seg_id,
            "start_s": seg["start_s"],
            "end_s": seg["end_s"],
            "duration_s": seg["duration_s"],
            "scores": {},
            "composite": {},
        }
        if score_fn is not None:
            score_result = score_fn(analysis_result, output_path=seg_scores)
            result_entry["scores"] = score_result["sub_scores"]
            result_entry["composite"] = score_result["composite"]

        results.append(result_entry)

    # Find notable changes between consecutive segments
    notable_changes = _find_notable_changes(results)

    # Summary statistics
    summary = _compute_segment_summary(results)

    output = {
        "n_segments": len(results),
        "segments": results,
        "notable_changes": notable_changes,
        "summary": summary,
    }

    # Save combined output
    combined_path = os.path.join(output_dir, "segments.json")
    with open(combined_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Segment analysis saved to {combined_path}")

    return output


def _find_notable_changes(segments: list[dict], threshold: float = 15.0) -> list[dict]:
    """Identify segments where scores change significantly from the previous segment.

    A change of >15 points in any sub-score is considered notable.
    """
    changes = []

    for i in range(1, len(segments)):
        prev = segments[i - 1]
        curr = segments[i]

        for category in ["pitch", "resonance", "vocal_weight", "voice_quality", "articulation", "prosody"]:
            prev_score = prev["scores"].get(category, {}).get("score", 0)
            curr_score = curr["scores"].get(category, {}).get("score", 0)
            delta = curr_score - prev_score

            if abs(delta) >= threshold:
                direction = "increased" if delta > 0 else "decreased"
                changes.append({
                    "segment_from": prev["segment_id"],
                    "segment_to": curr["segment_id"],
                    "time_s": curr["start_s"],
                    "category": category,
                    "prev_score": prev_score,
                    "curr_score": curr_score,
                    "delta": round(delta, 1),
                    "direction": direction,
                    "description": f"{category} {direction} by {abs(delta):.0f} pts at {curr['start_s']:.1f}s",
                })

    # Sort by absolute delta (biggest changes first)
    changes.sort(key=lambda x: abs(x["delta"]), reverse=True)
    return changes


def _compute_segment_summary(segments: list[dict]) -> dict:
    """Compute summary statistics across segments."""
    if not segments:
        return {}

    categories = ["pitch", "resonance", "vocal_weight", "voice_quality", "articulation", "prosody"]
    summary = {}

    for cat in categories:
        scores = [s["scores"].get(cat, {}).get("score", 0) for s in segments]
        if scores:
            summary[cat] = {
                "mean": round(float(np.mean(scores)), 1),
                "std": round(float(np.std(scores)), 1),
                "min": round(float(np.min(scores)), 1),
                "max": round(float(np.max(scores)), 1),
                "range": round(float(np.max(scores) - np.min(scores)), 1),
            }

    # Composite summary (only if scoring was performed)
    full_composites = [s["composite"].get("full", 0) for s in segments if s.get("composite")]
    tech_composites = [s["composite"].get("pitch_excluded", 0) for s in segments if s.get("composite")]

    if full_composites:
        summary["composite_full"] = {
            "mean": round(float(np.mean(full_composites)), 1),
            "std": round(float(np.std(full_composites)), 1),
        }
    if tech_composites:
        summary["composite_technique"] = {
            "mean": round(float(np.mean(tech_composites)), 1),
            "std": round(float(np.std(tech_composites)), 1),
        }

        # Best and worst segments by technique score
        best_idx = int(np.argmax(tech_composites))
        worst_idx = int(np.argmin(tech_composites))

        summary["best_segment"] = {
            "segment_id": segments[best_idx]["segment_id"],
            "time_s": segments[best_idx]["start_s"],
            "technique_score": tech_composites[best_idx],
        }
        summary["worst_segment"] = {
            "segment_id": segments[worst_idx]["segment_id"],
            "time_s": segments[worst_idx]["start_s"],
            "technique_score": tech_composites[worst_idx],
        }

    return summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python segment.py <wav_file> [output_dir]")
        sys.exit(1)

    wav_file = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None

    result = analyze_segments(wav_file, output_dir=out_dir)
    print(f"\n--- Segment Summary ---")
    print(f"  {result['n_segments']} segments analyzed")
    if result["notable_changes"]:
        print(f"  {len(result['notable_changes'])} notable score changes:")
        for change in result["notable_changes"][:5]:
            print(f"    {change['description']}")
    if result.get("summary", {}).get("best_segment"):
        best = result["summary"]["best_segment"]
        print(f"  Best technique: {best['technique_score']:.1f} at {best['time_s']:.1f}s")
