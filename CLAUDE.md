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



## Session Management

### On Session Start
1. Check `.session-reports/` for the most recent report
2. If it contains **CLAUDE.md Suggestions**: briefly tell the user (e.g., "Last session has 2 CLAUDE.md suggestions — want to review?")
3. If it contains **Open Items**: briefly mention them
4. Do NOT read the full report aloud unless asked — just flag what needs attention

### Before Ending a Session
Before finishing any substantive session (i.e., you wrote code, made plans, or made decisions), you MUST:
1. Run the `/session-report` command (or generate the report manually following that command's format)
2. Review `docs/plans/` and `docs/research/` for stale documents — add stale paths to `.archive-queue`
3. Confirm the report was written before signing off

If the session was trivial (quick question, no code changes), skip the report.

### File Organization
- **`docs/plans/`** — Active plans and design documents only. Name files descriptively: `feature-name-plan.md`
- **`docs/research/`** — Feature research, organized by topic subfolder: `docs/research/auth-providers/`
- **`docs/screenshots/`** — Debug screenshots from headless dev. Auto-archived after 14 days.
- **`.session-reports/`** — Session reports. Git-tracked. Do not delete.
- **`.project-archive/`** — Archived stale documents. Git-excluded.

### Archive Policy
- Files in `.project-archive/` preserve their original directory structure
- Do NOT reference archived files unless the user explicitly says to
- To archive a file mid-session: add its relative path to `.archive-queue` (processed by the Stop hook)
- When in doubt, archive rather than delete — nothing is lost

### CLAUDE.md Updates
- NEVER auto-update CLAUDE.md. Always propose changes in session reports under "CLAUDE.md Suggestions"
- Format suggestions as concrete diffs so the user can review and apply
- On session start, if pending suggestions exist, flag them for review
