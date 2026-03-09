# Shared Review Component & Analyses Table Design

**Date:** 2026-03-09
**Status:** Approved

## Problem

All three coaching apps need the ability to review saved practice sessions and evaluations. The basic experience — list sessions, play recordings, see results, track progress — is the same across apps. The specific metrics differ (femme: resonance/pitch scores, spanish: transcription accuracy, singing: pitch accuracy/vibrato).

Currently:
- Femme has a review page with filesystem-based storage (WAV + sidecar JSON files)
- Spanish and singing don't persist anything
- None of the apps use voice_user's `sessions` or `recordings` tables
- voice_user has no concept of analysis results

## Design

### 1. voice_user: analyses table

New table added as migration 2:

```sql
CREATE TABLE analyses (
    id           TEXT PRIMARY KEY,    -- UUIDv7
    recording_id TEXT NOT NULL,
    analyzer     TEXT NOT NULL,       -- e.g. "femme_scoring", "whisper_es", "pitch_accuracy"
    version      TEXT,                -- optional, e.g. "v2", "crepe-0.0.14"
    created_at   TEXT NOT NULL,       -- UTC ISO
    results      JSON NOT NULL,       -- app-specific analysis results
    FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
);
CREATE INDEX idx_analyses_recording ON analyses(recording_id, analyzer);
```

No user_id column — querying by user requires join through recordings. This is intentional: recording_id → analyses is the typical access pattern, and denormalizing user_id creates consistency risk.

New functions in `analyses.py`:
- `save_analysis(conn, recording_id, analyzer, results, version=None) -> AnalysisRecord`
- `get_latest_analysis(conn, recording_id, analyzer) -> AnalysisRecord | None`
- `list_analyses(conn, recording_id, analyzer=None) -> list[AnalysisRecord]`
- `delete_analyses(conn, recording_id, analyzer=None)`

All follow existing patterns: no auto-commit, UUIDv7 IDs, UTC timestamps, typed dict records.

### 2. voice-core: shared review component

New file `frontend/review.js` — reusable review shell that apps configure.

**Shared component handles:**
- Session list (date, duration, type) with filtering
- Recording list per session with audio playback
- Delete recording with confirmation
- Progress-over-time chart scaffold (app provides data extraction)

**App provides via config object:**
- `analyzerKey` — which analyzer type to fetch
- `renderAnalysis(analysisRecord, containerEl)` — render app-specific scores into a DOM element
- `extractProgressPoint(analysisRecord) -> {label, value} | null` — for progress chart
- `formatSessionSummary(session) -> string` (optional)

**Required API contract (all apps must implement):**
- `GET /api/sessions?user_id=X` — list sessions (voice_user.list_sessions)
- `GET /api/recordings?session_id=X` — list recordings (voice_user.list_recordings)
- `GET /api/recordings/{id}/audio` — serve audio file
- `DELETE /api/recordings/{id}` — delete recording + file (voice_user.delete_recording)
- `GET /api/analyses/{recording_id}?analyzer=X` — latest analysis (voice_user.get_latest_analysis)
- `GET /api/analyses/{recording_id}` — all analyses for recording (voice_user.list_analyses)

**Shared component does NOT handle:**
- App-specific scoring cards, word grids, pitch displays
- Evaluation workflows (femme's multi-recording eval is its own thing)
- Any domain logic

### 3. Storage conventions

- **recordings.metadata** (JSON) — universal recording metadata: mic label, activity tag, app-specific context (e.g. phrase_id for spanish). NOT analysis results.
- **analyses.results** (JSON) — app-specific analysis output. Multiple analyses per recording, multiple analyzer types supported.
- **Audio files** — managed by voice_user's `save_recording()` / `write_audio_file()`. Stored at `data/recordings/{user_id}/`.

### 4. Integration testing

**voice_user unit tests (before individual app work):**
- Analyses CRUD: save, get_latest, list, delete
- CASCADE: deleting recording removes all its analyses
- Multiple analyzer types on one recording
- Migration 2 applied cleanly on existing v1 databases

**Cross-repo smoke tests:**
- Shared asset checksums match across all 3 ports
- User API works on all 3 apps
- Recording + analysis persistence (after apps are wired)

### 5. Per-app integration (separate sessions)

Each app needs:
1. Wire recording flow to use voice_user (save_recording on stop)
2. Wire session lifecycle (create_session on start, end_session on stop)
3. Save analysis results via save_analysis after processing
4. Add the standard API endpoints listed above
5. Create a review page that imports shared review.js and provides app-specific config
6. Migrate femme off filesystem sidecar files (optional, can coexist initially)
