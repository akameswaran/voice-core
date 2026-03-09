# App Integration Prompts

Prepared prompts for wiring each coaching app to use voice_user for persistence and the shared review component.

**Prerequisites:** voice-user has the `analyses` table (migration 2), voice-core has `frontend/review.js`.

---

## Spanish Voice Coach

Open a session in `/home/ak/Projects/VoiceCoaches/spanish-voice-coach/`.

```
Wire spanish-voice-coach to persist practice results and add a review page.

## Context

voice_user (installed as editable dep) now has:
- `sessions` table: create_session, end_session, list_sessions
- `recordings` table: save_recording, list_recordings, delete_recording, get_recording
- `analyses` table (NEW): save_analysis, get_latest_analysis, list_analyses, delete_analyses
- All functions take a conn, do NOT auto-commit. Use `with conn:` for transactions.
- IDs are UUIDv7 strings. Timestamps are UTC ISO.

voice-core has a shared review component at `/static/core/review.js`:
- `import { initReview } from '/static/core/review.js'`
- Config: { mountTo, analyzerKey, renderAnalysis(analyses, el), extractProgressPoint(analysis), formatSessionSummary(session) }
- Requires these API endpoints: GET /api/sessions, GET /api/recordings, GET /api/recordings/{id}/audio, DELETE /api/recordings/{id}, GET /api/analyses/{recording_id}

## What to build

1. **Persist recordings**: In the 3-WS adapter (`/api/live/save`), after Whisper analysis completes:
   - Create a session (session_type="practice") if not already active
   - Save the audio via voice_user.save_recording()
   - Save the Whisper result via voice_user.save_analysis(conn, recording_id, "whisper_es", result)
   - Store phrase_id in recordings.metadata for linking back to the phrase

2. **Standard API endpoints**: Add thin wrappers around voice_user functions:
   - GET /api/sessions?user_id=X → voice_user.list_sessions
   - GET /api/recordings?session_id=X or ?user_id=X → voice_user.list_recordings
   - GET /api/recordings/{id}/audio → serve WAV from data/recordings/
   - DELETE /api/recordings/{id} → voice_user.delete_recording
   - GET /api/analyses/{recording_id}?analyzer=X → voice_user.get_latest_analysis or list_analyses

3. **Review page**: Create frontend/review.html that imports the shared review component.
   - analyzerKey: "whisper_es"
   - renderAnalysis: show heard text, word comparison grid (reuse displayResults logic from practice.js)
   - extractProgressPoint: return { label: 'Accuracy', value: matched/total * 100 }
   - Add /review route to server.py

4. **Nav link**: Add "Review" to the nav links so users can navigate to it.

## DB path
data/spanish_coach.db (already initialized by voice_user.init_db)

## Recordings dir
data/recordings/ (voice_user.save_recording creates user subdirs)

## Don't change
- The existing single-WS converse flow — it stays as-is
- practice.js UI — it keeps showing immediate results
- The phrase corpus or TTS system
```

---

## Singing Voice Coach

Open a session in `/home/ak/Projects/VoiceCoaches/singing-voice-coach/`.

```
Wire singing-voice-coach to persist practice results and add a review page.

## Context

voice_user (installed as editable dep) now has:
- `sessions` table: create_session, end_session, list_sessions
- `recordings` table: save_recording, list_recordings, delete_recording, get_recording
- `analyses` table (NEW): save_analysis, get_latest_analysis, list_analyses, delete_analyses
- All functions take a conn, do NOT auto-commit. Use `with conn:` for transactions.
- IDs are UUIDv7 strings. Timestamps are UTC ISO.

voice-core has a shared review component at `/static/core/review.js`:
- `import { initReview } from '/static/core/review.js'`
- Config: { mountTo, analyzerKey, renderAnalysis(analyses, el), extractProgressPoint(analysis), formatSessionSummary(session) }
- Requires these API endpoints: GET /api/sessions, GET /api/recordings, GET /api/recordings/{id}/audio, DELETE /api/recordings/{id}, GET /api/analyses/{recording_id}

## What to build

1. **Persist recordings**: Wire the existing LiveAnalyzer flow:
   - Add POST /api/live/save endpoint (currently missing — RC expects it)
   - On save: collect accumulated frames from LiveAnalyzer, save audio via voice_user.save_recording()
   - Save pitch analysis via voice_user.save_analysis(conn, recording_id, "pitch_tracking", results)
   - Results should include: avg_pitch_hz, min_hz, max_hz, pitch_samples (array of {time, hz} or summary stats)
   - Create/end sessions around recording lifecycle

2. **Standard API endpoints**: Add thin wrappers around voice_user functions:
   - GET /api/sessions?user_id=X → voice_user.list_sessions
   - GET /api/recordings?session_id=X or ?user_id=X → voice_user.list_recordings
   - GET /api/recordings/{id}/audio → serve WAV from data/recordings/
   - DELETE /api/recordings/{id} → voice_user.delete_recording
   - GET /api/analyses/{recording_id}?analyzer=X → voice_user.get_latest_analysis or list_analyses

3. **Review page**: Create frontend/review.html that imports the shared review component.
   - analyzerKey: "pitch_tracking"
   - renderAnalysis: show avg pitch, range, sample count (reuse the session-stats format from practice.js)
   - extractProgressPoint: return { label: 'Avg Pitch', value: avg_pitch_hz }
   - Add /review route to server.py

4. **Nav link**: Add "Review" to the nav links so users can navigate to it.

## DB path
data/singing_coach.db (already initialized by voice_user.init_db)

## Recordings dir
data/recordings/ (voice_user.save_recording creates user subdirs)

## Server currently has
- POST /api/live/start-browser — starts LiveAnalyzer
- POST /api/live/stop — stops analyzer (but does NOT save)
- WS /ws/live — streams frames
- WS /ws/live/audio — receives PCM
- No save endpoint — this must be added

## Audio format note
LiveAnalyzer accumulates audio internally. After stopping, you can access
the buffer. Check voice_core.live.LiveAnalyzer for how to get the recorded
audio data after a session. The browser sends Int16 PCM (128 samples = 256 bytes
per AudioWorklet frame).

## Don't change
- practice.js live pitch display — it stays as-is
- The PitchStrip or note display widgets
- The zone classifier config
```

---

## Femme Voice Coach (migration)

Open a session in `/home/ak/Projects/VoiceCoaches/femme-voice-coach/`.

```
Migrate femme-voice-coach to use voice_user for session/recording persistence and the shared review component.

## Context

voice_user now has sessions, recordings, and analyses tables. Femme currently uses filesystem-based storage:
- data/recordings/{user_id}/*.wav — audio files
- data/recordings/{user_id}/*.analysis.json — scoring results
- data/recordings/{user_id}/*.coaching.json — real-time coaching messages
- data/recordings/{user_id}/*.meta.json — metadata (mic, activity, name)

voice-core has a shared review component at `/static/core/review.js`.

## What to build

1. **Wire recording persistence to voice_user**:
   - In routes_live.py POST /api/live/save: after saving WAV, also call voice_user.save_recording() to register in DB
   - Save analysis results via voice_user.save_analysis(conn, recording_id, "femme_scoring", scores)
   - Save coaching messages via voice_user.save_analysis(conn, recording_id, "femme_coaching", coaching_data)
   - Store activity tag, mic label in recordings.metadata

2. **Standard API endpoints**: Add/modify to match the shared review contract:
   - GET /api/sessions?user_id=X → voice_user.list_sessions
   - GET /api/recordings?session_id=X or ?user_id=X → voice_user.list_recordings
   - GET /api/recordings/{id}/audio → serve WAV (can keep existing endpoint, add id-based lookup)
   - DELETE /api/recordings/{id} → voice_user.delete_recording (replaces filename-based delete)
   - GET /api/analyses/{recording_id}?analyzer=X → voice_user.get_latest_analysis

3. **Migrate review page to shared component**:
   - Keep review.html but replace the recordings browser section with shared review.js
   - analyzerKey: "femme_scoring"
   - renderAnalysis: show score badges (Tech/Res/Wt/F0) — move rendering logic from current review.js
   - extractProgressPoint: return { label: 'Technique', value: scores.technique }
   - Keep the Chart.js composite/sub-score charts as femme-specific additions above the shared review

4. **Dual-write period**: During migration, keep writing .analysis.json sidecar files alongside DB records so existing review.js still works. Once shared review is validated, remove sidecar file writes.

## Important
- This is the most complex migration of the three — take it in stages
- Don't break the existing review page during migration
- The evaluation flow (multi-recording evaluations) is femme-specific and stays separate
```
