# Shared Review & Analyses Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an analyses table to voice-user, build a shared review component in voice-core, and prepare prompts for wiring spanish and singing apps.

**Architecture:** voice-user gets a new `analyses` table (migration 2) with CRUD functions following existing patterns. voice-core gets `frontend/review.js` — a configurable review shell that apps import and customize with app-specific renderers. Each app implements a thin API layer mapping voice-user functions to HTTP endpoints.

**Tech Stack:** Python/SQLite (voice-user), vanilla JS (voice-core frontend), FastAPI (app servers)

**Design doc:** `docs/plans/2026-03-09-shared-review-and-analyses-design.md`

---

## Part 1: voice-user — analyses table

All work in `/home/ak/Projects/VoiceCoaches/voice-user/`.

### Task 1: Add AnalysisRecord type

**Files:**
- Modify: `src/voice_user/analyses.py` (create new)

**Step 1: Create analyses.py with the TypedDict**

```python
"""Analysis storage — multiple analysis results per recording."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import TypedDict

from uuid_utils import uuid7


class AnalysisRecord(TypedDict):
    id: str
    recording_id: str
    analyzer: str
    version: str | None
    created_at: str
    results: dict
```

**Step 2: Commit**

```bash
git add src/voice_user/analyses.py
git commit -m "feat: add AnalysisRecord type"
```

---

### Task 2: Write failing tests for analyses CRUD

**Files:**
- Create: `tests/test_analyses.py`

**Step 1: Write the test file**

Follow existing test patterns from `test_recordings.py` and `test_sessions.py`. Use the same fixtures.

```python
"""Tests for analysis CRUD operations."""

import pytest
import numpy as np

from voice_user import (
    init_db,
    create_user,
    create_session,
    save_recording,
    save_analysis,
    get_latest_analysis,
    list_analyses,
    delete_analyses,
    AnalysisRecord,
)


@pytest.fixture
def conn(tmp_path):
    c = init_db(tmp_path / "test.db")
    yield c
    c.close()


@pytest.fixture
def user(conn):
    return create_user(conn, name="TestUser")


@pytest.fixture
def session(conn, user):
    return create_session(conn, user["id"], "practice")


@pytest.fixture
def rec_dir(tmp_path):
    d = tmp_path / "recordings"
    d.mkdir()
    return d


@pytest.fixture
def sample_audio():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr


@pytest.fixture
def recording(conn, user, session, rec_dir, sample_audio):
    audio, sr = sample_audio
    with conn:
        return save_recording(conn, user["id"], audio, sr, rec_dir, session_id=session["id"])


# ── save_analysis ────────────────────────────────────────────


def test_save_analysis_returns_record(conn, recording):
    results = {"score": 85, "detail": {"pitch": 90, "resonance": 80}}
    with conn:
        a = save_analysis(conn, recording["id"], "femme_scoring", results)
    assert a["recording_id"] == recording["id"]
    assert a["analyzer"] == "femme_scoring"
    assert a["results"] == results
    assert a["version"] is None
    assert a["id"]  # UUIDv7
    assert a["created_at"]  # ISO timestamp


def test_save_analysis_with_version(conn, recording):
    results = {"accuracy": 0.95}
    with conn:
        a = save_analysis(conn, recording["id"], "whisper_es", results, version="large-v3")
    assert a["version"] == "large-v3"
    assert a["analyzer"] == "whisper_es"


def test_save_analysis_multiple_types(conn, recording):
    with conn:
        a1 = save_analysis(conn, recording["id"], "pitch_accuracy", {"cents_off": 5})
        a2 = save_analysis(conn, recording["id"], "vibrato_detect", {"rate_hz": 5.5})
    assert a1["id"] != a2["id"]
    assert a1["analyzer"] == "pitch_accuracy"
    assert a2["analyzer"] == "vibrato_detect"


def test_save_analysis_multiple_versions(conn, recording):
    with conn:
        a1 = save_analysis(conn, recording["id"], "femme_scoring", {"score": 70}, version="v1")
        a2 = save_analysis(conn, recording["id"], "femme_scoring", {"score": 85}, version="v2")
    assert a1["id"] != a2["id"]


# ── get_latest_analysis ──────────────────────────────────────


def test_get_latest_analysis(conn, recording):
    with conn:
        save_analysis(conn, recording["id"], "femme_scoring", {"score": 70}, version="v1")
        save_analysis(conn, recording["id"], "femme_scoring", {"score": 85}, version="v2")
    latest = get_latest_analysis(conn, recording["id"], "femme_scoring")
    assert latest is not None
    assert latest["results"]["score"] == 85
    assert latest["version"] == "v2"


def test_reanalysis_returns_newer(conn, recording):
    """Re-running analysis on same recording with updated code should return the new result."""
    with conn:
        save_analysis(conn, recording["id"], "femme_scoring", {"score": 70, "model": "old"}, version="v1")
    # Simulate re-analysis with updated pipeline
    with conn:
        save_analysis(conn, recording["id"], "femme_scoring", {"score": 82, "model": "new"}, version="v1")
    latest = get_latest_analysis(conn, recording["id"], "femme_scoring")
    assert latest["results"]["score"] == 82
    assert latest["results"]["model"] == "new"
    # Both analyses still exist
    all_analyses = list_analyses(conn, recording["id"], analyzer="femme_scoring")
    assert len(all_analyses) == 2


def test_get_latest_analysis_wrong_analyzer(conn, recording):
    with conn:
        save_analysis(conn, recording["id"], "femme_scoring", {"score": 85})
    result = get_latest_analysis(conn, recording["id"], "whisper_es")
    assert result is None


def test_get_latest_analysis_no_analyses(conn, recording):
    result = get_latest_analysis(conn, recording["id"], "femme_scoring")
    assert result is None


# ── list_analyses ────────────────────────────────────────────


def test_list_analyses_all(conn, recording):
    with conn:
        save_analysis(conn, recording["id"], "femme_scoring", {"score": 85})
        save_analysis(conn, recording["id"], "whisper_es", {"text": "hello"})
    analyses = list_analyses(conn, recording["id"])
    assert len(analyses) == 2


def test_list_analyses_filtered(conn, recording):
    with conn:
        save_analysis(conn, recording["id"], "femme_scoring", {"score": 85})
        save_analysis(conn, recording["id"], "whisper_es", {"text": "hello"})
    analyses = list_analyses(conn, recording["id"], analyzer="femme_scoring")
    assert len(analyses) == 1
    assert analyses[0]["analyzer"] == "femme_scoring"


def test_list_analyses_empty(conn, recording):
    analyses = list_analyses(conn, recording["id"])
    assert analyses == []


# ── delete_analyses ──────────────────────────────────────────


def test_delete_analyses_by_analyzer(conn, recording):
    with conn:
        save_analysis(conn, recording["id"], "femme_scoring", {"score": 85})
        save_analysis(conn, recording["id"], "whisper_es", {"text": "hello"})
    with conn:
        delete_analyses(conn, recording["id"], analyzer="femme_scoring")
    remaining = list_analyses(conn, recording["id"])
    assert len(remaining) == 1
    assert remaining[0]["analyzer"] == "whisper_es"


def test_delete_analyses_all(conn, recording):
    with conn:
        save_analysis(conn, recording["id"], "femme_scoring", {"score": 85})
        save_analysis(conn, recording["id"], "whisper_es", {"text": "hello"})
    with conn:
        delete_analyses(conn, recording["id"])
    assert list_analyses(conn, recording["id"]) == []


# ── CASCADE on recording delete ──────────────────────────────


def test_cascade_delete_recording(conn, recording, rec_dir):
    with conn:
        save_analysis(conn, recording["id"], "femme_scoring", {"score": 85})
    from voice_user import delete_recording
    with conn:
        delete_recording(conn, recording["id"], rec_dir)
    # Analysis should be gone — recording FK cascades
    # We can't list_analyses on a deleted recording, but we can
    # verify directly via SQL
    row = conn.execute(
        "SELECT COUNT(*) FROM analyses WHERE recording_id = ?",
        (recording["id"],)
    ).fetchone()
    assert row[0] == 0


# ── Migration on existing DB ────────────────────────────────


def test_migration_v2_on_existing_v1(tmp_path):
    """Simulate a v1 DB and verify migration 2 adds analyses table."""
    conn = init_db(tmp_path / "old.db")
    # init_db should have run both migrations
    # Verify analyses table exists
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='analyses'"
    ).fetchone()
    assert tables is not None
    # Verify schema_version is 2
    ver = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    assert ver[0] == 2
    conn.close()
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/ak/Projects/VoiceCoaches/voice-user && uv run pytest tests/test_analyses.py -v`
Expected: ImportError — `save_analysis`, `get_latest_analysis`, `list_analyses`, `delete_analyses`, `AnalysisRecord` not found.

**Step 3: Commit**

```bash
git add tests/test_analyses.py
git commit -m "test: add failing tests for analyses CRUD"
```

---

### Task 3: Add migration 2 (analyses table)

**Files:**
- Modify: `src/voice_user/db.py`

**Step 1: Add migration 2 to the `_run_migrations` function**

In `db.py`, find the migrations list and add migration 2 after the existing migration 1. The pattern is:

```python
# After existing migration 1 block:

if current_version < 2:
    conn.executescript("""
        CREATE TABLE analyses (
            id           TEXT PRIMARY KEY,
            recording_id TEXT NOT NULL,
            analyzer     TEXT NOT NULL,
            version      TEXT,
            created_at   TEXT NOT NULL,
            results      JSON NOT NULL,
            FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
        );
        CREATE INDEX idx_analyses_recording ON analyses(recording_id, analyzer);
    """)
    conn.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
        (2, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
```

**Step 2: Run migration test only**

Run: `cd /home/ak/Projects/VoiceCoaches/voice-user && uv run pytest tests/test_analyses.py::test_migration_v2_on_existing_v1 -v`
Expected: PASS (table exists, version is 2)

**Step 3: Commit**

```bash
git add src/voice_user/db.py
git commit -m "feat: add analyses table migration (v2)"
```

---

### Task 4: Implement analyses CRUD functions

**Files:**
- Modify: `src/voice_user/analyses.py`

**Step 1: Add CRUD functions to analyses.py**

Append after the AnalysisRecord TypedDict:

```python
def _row_to_analysis(row: tuple) -> AnalysisRecord:
    results = row[5]
    if isinstance(results, str):
        results = json.loads(results)
    return AnalysisRecord(
        id=row[0],
        recording_id=row[1],
        analyzer=row[2],
        version=row[3],
        created_at=row[4],
        results=results,
    )


def save_analysis(
    conn: sqlite3.Connection,
    recording_id: str,
    analyzer: str,
    results: dict,
    version: str | None = None,
) -> AnalysisRecord:
    """Save an analysis result for a recording. Does NOT commit."""
    aid = str(uuid7())
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO analyses (id, recording_id, analyzer, version, created_at, results)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (aid, recording_id, analyzer, version, now, json.dumps(results)),
    )
    return AnalysisRecord(
        id=aid,
        recording_id=recording_id,
        analyzer=analyzer,
        version=version,
        created_at=now,
        results=results,
    )


def get_latest_analysis(
    conn: sqlite3.Connection,
    recording_id: str,
    analyzer: str,
) -> AnalysisRecord | None:
    """Get the most recent analysis of a given type for a recording."""
    row = conn.execute(
        """SELECT id, recording_id, analyzer, version, created_at, results
           FROM analyses
           WHERE recording_id = ? AND analyzer = ?
           ORDER BY created_at DESC LIMIT 1""",
        (recording_id, analyzer),
    ).fetchone()
    return _row_to_analysis(row) if row else None


def list_analyses(
    conn: sqlite3.Connection,
    recording_id: str,
    analyzer: str | None = None,
) -> list[AnalysisRecord]:
    """List analyses for a recording, optionally filtered by analyzer type."""
    if analyzer:
        rows = conn.execute(
            """SELECT id, recording_id, analyzer, version, created_at, results
               FROM analyses
               WHERE recording_id = ? AND analyzer = ?
               ORDER BY created_at""",
            (recording_id, analyzer),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT id, recording_id, analyzer, version, created_at, results
               FROM analyses
               WHERE recording_id = ?
               ORDER BY created_at""",
            (recording_id,),
        ).fetchall()
    return [_row_to_analysis(r) for r in rows]


def delete_analyses(
    conn: sqlite3.Connection,
    recording_id: str,
    analyzer: str | None = None,
) -> None:
    """Delete analyses for a recording. If analyzer given, only that type. Does NOT commit."""
    if analyzer:
        conn.execute(
            "DELETE FROM analyses WHERE recording_id = ? AND analyzer = ?",
            (recording_id, analyzer),
        )
    else:
        conn.execute(
            "DELETE FROM analyses WHERE recording_id = ?",
            (recording_id,),
        )
```

**Step 2: Export from `__init__.py`**

Add to `src/voice_user/__init__.py`:

```python
from .analyses import (
    AnalysisRecord,
    save_analysis,
    get_latest_analysis,
    list_analyses,
    delete_analyses,
)
```

**Step 3: Run all analyses tests**

Run: `cd /home/ak/Projects/VoiceCoaches/voice-user && uv run pytest tests/test_analyses.py -v`
Expected: All PASS

**Step 4: Run full test suite to verify no regressions**

Run: `cd /home/ak/Projects/VoiceCoaches/voice-user && uv run pytest -v`
Expected: All PASS (existing tests unaffected)

**Step 5: Commit**

```bash
git add src/voice_user/analyses.py src/voice_user/__init__.py
git commit -m "feat: implement analyses CRUD functions"
```

---

## Part 2: voice-core — shared review component

All work in `/home/ak/Projects/VoiceCoaches/voice-core/`.

### Task 5: Create shared review.js

**Files:**
- Create: `frontend/review.js`

**Step 1: Write the shared review component**

This is a configurable ES module. Apps import it and call `initReview(config)`. It renders into a mount element.

```javascript
/**
 * Shared Review Component — reusable session/recording browser.
 *
 * Usage:
 *   import { initReview } from '/static/core/review.js';
 *   initReview({
 *     mountTo: '#review-mount',
 *     analyzerKey: 'femme_scoring',
 *     renderAnalysis: (analysis, el) => { ... },
 *     extractProgressPoint: (analysis) => ({ label: '...', value: 85 }),
 *     formatSessionSummary: (session) => '...',  // optional
 *   });
 *
 * Required API endpoints (app must implement):
 *   GET  /api/sessions?user_id=X
 *   GET  /api/recordings?session_id=X
 *   GET  /api/recordings/{id}/audio
 *   DELETE /api/recordings/{id}
 *   GET  /api/analyses/{recording_id}?analyzer=X   (latest)
 *   GET  /api/analyses/{recording_id}               (all)
 */

import { listUsers } from './user_api.js';

// ── Active user helper ─────────────────────────────────────
function _activeUserId() {
    return localStorage.getItem('activeUserId') || '';
}

function _headers() {
    return { 'X-User-Id': _activeUserId() };
}

// ── Data fetching ──────────────────────────────────────────

async function fetchSessions() {
    const uid = _activeUserId();
    if (!uid) return [];
    const res = await fetch(`/api/sessions?user_id=${encodeURIComponent(uid)}`, { headers: _headers() });
    if (!res.ok) return [];
    return res.json();
}

async function fetchRecordings(sessionId) {
    const res = await fetch(`/api/recordings?session_id=${encodeURIComponent(sessionId)}`, { headers: _headers() });
    if (!res.ok) return [];
    return res.json();
}

async function fetchRecordingsByUser() {
    const uid = _activeUserId();
    if (!uid) return [];
    const res = await fetch(`/api/recordings?user_id=${encodeURIComponent(uid)}`, { headers: _headers() });
    if (!res.ok) return [];
    return res.json();
}

async function fetchAnalyses(recordingId, analyzer) {
    const url = analyzer
        ? `/api/analyses/${recordingId}?analyzer=${encodeURIComponent(analyzer)}`
        : `/api/analyses/${recordingId}`;
    const res = await fetch(url, { headers: _headers() });
    if (!res.ok) return [];
    const data = await res.json();
    return Array.isArray(data) ? data : [data];
}

async function deleteRecording(recordingId) {
    const res = await fetch(`/api/recordings/${recordingId}`, {
        method: 'DELETE',
        headers: _headers(),
    });
    return res.ok;
}

// ── Formatting helpers ─────────────────────────────────────

function formatDate(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

function formatTime(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}

function formatDuration(seconds) {
    if (!seconds) return '--';
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

// ── Main init ──────────────────────────────────────────────

export function initReview(config) {
    const {
        mountTo,
        analyzerKey,
        renderAnalysis,
        extractProgressPoint,
        formatSessionSummary,
    } = config;

    const mount = typeof mountTo === 'string' ? document.querySelector(mountTo) : mountTo;
    if (!mount) {
        console.error('[review] Mount element not found:', mountTo);
        return;
    }

    const state = {
        sessions: [],
        recordings: [],
        selectedSessionId: null,
        audioEl: null,
        playingRecId: null,
    };

    // ── Render shell ───────────────────────────────────────
    mount.innerHTML = `
        <div class="vc-review">
            <div class="vc-review-progress" id="vc-review-progress">
                <h3 class="vc-review-section-title">Progress</h3>
                <div class="vc-review-chart-area" id="vc-review-chart"></div>
            </div>
            <div class="vc-review-sessions">
                <div class="vc-review-header">
                    <h3 class="vc-review-section-title">Sessions</h3>
                    <span class="vc-review-count" id="vc-review-session-count"></span>
                </div>
                <div id="vc-review-session-list" class="vc-review-list"></div>
                <div id="vc-review-sessions-empty" class="vc-review-empty" style="display:none">
                    <p>No sessions yet. Start practicing!</p>
                </div>
            </div>
            <div class="vc-review-recordings">
                <div class="vc-review-header">
                    <h3 class="vc-review-section-title">Recordings</h3>
                    <span class="vc-review-count" id="vc-review-rec-count"></span>
                </div>
                <div id="vc-review-rec-list" class="vc-review-list"></div>
                <div id="vc-review-recs-empty" class="vc-review-empty" style="display:none">
                    <p>Select a session to see recordings.</p>
                </div>
            </div>
        </div>
    `;

    const sessionList = mount.querySelector('#vc-review-session-list');
    const sessionCount = mount.querySelector('#vc-review-session-count');
    const sessionsEmpty = mount.querySelector('#vc-review-sessions-empty');
    const recList = mount.querySelector('#vc-review-rec-list');
    const recCount = mount.querySelector('#vc-review-rec-count');
    const recsEmpty = mount.querySelector('#vc-review-recs-empty');
    const chartArea = mount.querySelector('#vc-review-chart');

    // Shared audio element
    state.audioEl = document.createElement('audio');
    state.audioEl.preload = 'none';

    // ── Session list rendering ─────────────────────────────
    async function loadSessions() {
        state.sessions = await fetchSessions();
        sessionCount.textContent = `${state.sessions.length} sessions`;
        sessionsEmpty.style.display = state.sessions.length ? 'none' : '';
        sessionList.innerHTML = '';

        state.sessions.forEach(s => {
            const card = document.createElement('div');
            card.className = 'vc-review-card' + (s.id === state.selectedSessionId ? ' selected' : '');
            const summary = formatSessionSummary ? formatSessionSummary(s) : (s.session_type || 'practice');
            card.innerHTML = `
                <div class="vc-review-card-header">
                    <span class="vc-review-card-date">${formatDate(s.started_at)}</span>
                    <span class="vc-review-card-time">${formatTime(s.started_at)}</span>
                </div>
                <div class="vc-review-card-body">
                    <span class="vc-review-card-type">${summary}</span>
                    <span class="vc-review-card-duration">${formatDuration(s.duration_s)}</span>
                </div>
            `;
            card.addEventListener('click', () => selectSession(s.id));
            sessionList.appendChild(card);
        });

        // Load progress chart if extractProgressPoint provided
        if (extractProgressPoint && state.sessions.length > 0) {
            await loadProgress();
        }
    }

    async function selectSession(sessionId) {
        state.selectedSessionId = sessionId;
        // Re-render session list to update selection
        sessionList.querySelectorAll('.vc-review-card').forEach(c => c.classList.remove('selected'));
        const idx = state.sessions.findIndex(s => s.id === sessionId);
        if (idx >= 0) sessionList.children[idx]?.classList.add('selected');
        await loadRecordings(sessionId);
    }

    // ── Recording list rendering ───────────────────────────
    async function loadRecordings(sessionId) {
        state.recordings = await fetchRecordings(sessionId);
        recCount.textContent = `${state.recordings.length} recordings`;
        recsEmpty.style.display = state.recordings.length ? 'none' : '';
        recList.innerHTML = '';

        for (const rec of state.recordings) {
            const card = document.createElement('div');
            card.className = 'vc-review-card vc-review-rec-card';
            card.dataset.recId = rec.id;

            card.innerHTML = `
                <div class="vc-review-card-header">
                    <span class="vc-review-card-date">${formatDate(rec.recorded_at)}</span>
                    <span class="vc-review-card-duration">${formatDuration(rec.duration_s)}</span>
                </div>
                <div class="vc-review-rec-controls">
                    <button class="vc-review-btn vc-review-btn-play" data-rec-id="${rec.id}" title="Play">&#9654;</button>
                    <button class="vc-review-btn vc-review-btn-delete" data-rec-id="${rec.id}" title="Delete">&#128465;</button>
                </div>
                <div class="vc-review-analysis" id="vc-review-analysis-${rec.id}"></div>
            `;

            // Play button
            card.querySelector('.vc-review-btn-play').addEventListener('click', (e) => {
                e.stopPropagation();
                togglePlay(rec.id);
            });

            // Delete button
            card.querySelector('.vc-review-btn-delete').addEventListener('click', async (e) => {
                e.stopPropagation();
                if (!confirm('Delete this recording?')) return;
                if (await deleteRecording(rec.id)) {
                    await loadRecordings(sessionId);
                }
            });

            recList.appendChild(card);

            // Load and render analysis
            if (renderAnalysis) {
                const analyses = await fetchAnalyses(rec.id, analyzerKey);
                const container = card.querySelector(`#vc-review-analysis-${rec.id}`);
                if (analyses.length > 0) {
                    renderAnalysis(analyses, container);
                }
            }
        }
    }

    // ── Audio playback ─────────────────────────────────────
    function togglePlay(recId) {
        if (state.playingRecId === recId) {
            state.audioEl.pause();
            state.playingRecId = null;
            updatePlayButtons();
            return;
        }
        state.audioEl.src = `/api/recordings/${recId}/audio`;
        state.audioEl.play();
        state.playingRecId = recId;
        updatePlayButtons();
        state.audioEl.onended = () => {
            state.playingRecId = null;
            updatePlayButtons();
        };
    }

    function updatePlayButtons() {
        recList.querySelectorAll('.vc-review-btn-play').forEach(btn => {
            const id = btn.dataset.recId;
            btn.textContent = id === state.playingRecId ? '\u23F8' : '\u25B6';
        });
    }

    // ── Progress chart (simple — no Chart.js dependency) ──
    async function loadProgress() {
        if (!extractProgressPoint) return;

        // Gather all recordings across all sessions for progress
        const allRecs = await fetchRecordingsByUser();
        const points = [];

        for (const rec of allRecs) {
            const analyses = await fetchAnalyses(rec.id, analyzerKey);
            if (analyses.length > 0) {
                const point = extractProgressPoint(analyses[analyses.length - 1]);
                if (point) {
                    points.push({ date: rec.recorded_at, ...point });
                }
            }
        }

        if (points.length < 2) {
            chartArea.innerHTML = '<p class="vc-review-dim">Need more recordings for progress chart.</p>';
            return;
        }

        // Render simple text-based progress summary (apps can override with Chart.js)
        const first = points[0];
        const last = points[points.length - 1];
        const delta = last.value - first.value;
        const sign = delta >= 0 ? '+' : '';
        chartArea.innerHTML = `
            <div class="vc-review-progress-summary">
                <div class="vc-review-progress-stat">
                    <span class="vc-review-progress-label">Latest</span>
                    <span class="vc-review-progress-value">${last.value.toFixed(1)}</span>
                </div>
                <div class="vc-review-progress-stat">
                    <span class="vc-review-progress-label">Change</span>
                    <span class="vc-review-progress-value" style="color: ${delta >= 0 ? 'var(--fem, #4ade80)' : 'var(--masc, #f87171)'}">${sign}${delta.toFixed(1)}</span>
                </div>
                <div class="vc-review-progress-stat">
                    <span class="vc-review-progress-label">Sessions</span>
                    <span class="vc-review-progress-value">${points.length}</span>
                </div>
            </div>
        `;
    }

    // ── Kick off ───────────────────────────────────────────
    loadSessions();

    // Return API for programmatic control
    return { loadSessions, selectSession };
}
```

**Step 2: Commit**

```bash
git add frontend/review.js
git commit -m "feat: add shared review component"
```

---

### Task 6: Add review.css styles to theme.css

**Files:**
- Modify: `frontend/theme.css`

**Step 1: Append review component styles**

Add to the end of `theme.css`:

```css
/* ── Shared Review Component ───────────────────────────────── */
.vc-review { max-width: 1100px; margin: 0 auto; padding: 24px 16px; }
.vc-review-section-title { font-size: 1.1rem; font-weight: 700; margin: 0; }
.vc-review-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.vc-review-count { font-size: 0.85rem; color: var(--text-dim); }
.vc-review-list { display: flex; flex-direction: column; gap: 8px; margin-bottom: 24px; }
.vc-review-empty { text-align: center; padding: 40px 20px; color: var(--text-dim); }
.vc-review-dim { color: var(--text-dim); font-size: 0.9rem; }

.vc-review-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    cursor: pointer;
    transition: border-color 0.15s;
}
.vc-review-card:hover { border-color: var(--accent); }
.vc-review-card.selected { border-color: var(--accent); background: var(--surface2, var(--surface)); }

.vc-review-card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
.vc-review-card-date { font-size: 0.85rem; font-weight: 600; }
.vc-review-card-time { font-size: 0.8rem; color: var(--text-dim); }
.vc-review-card-body { display: flex; justify-content: space-between; align-items: center; }
.vc-review-card-type { font-size: 0.9rem; }
.vc-review-card-duration { font-size: 0.8rem; color: var(--text-dim); }

.vc-review-rec-controls { display: flex; gap: 8px; margin-top: 8px; }
.vc-review-btn {
    background: none;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 4px 10px;
    color: var(--text);
    cursor: pointer;
    font-size: 0.85rem;
}
.vc-review-btn:hover { border-color: var(--accent); color: var(--accent); }
.vc-review-btn-delete:hover { border-color: var(--masc, #f87171); color: var(--masc, #f87171); }

.vc-review-analysis { margin-top: 8px; }

.vc-review-progress { margin-bottom: 24px; }
.vc-review-chart-area { min-height: 60px; }
.vc-review-progress-summary { display: flex; gap: 24px; padding: 16px 0; }
.vc-review-progress-stat { display: flex; flex-direction: column; align-items: center; }
.vc-review-progress-label { font-size: 0.8rem; color: var(--text-dim); margin-bottom: 4px; }
.vc-review-progress-value { font-size: 1.3rem; font-weight: 700; }

.vc-review-sessions { margin-bottom: 24px; }
```

**Step 2: Commit**

```bash
git add frontend/theme.css
git commit -m "feat: add review component styles to theme.css"
```

---

## Part 3: Integration prompts for individual apps

### Task 7: Write app integration prompts

**Files:**
- Create: `docs/plans/2026-03-09-app-integration-prompts.md`

**Step 1: Write the prompts document**

These prompts are for opening a new Claude session in each app's repo to wire up persistence and review.

````markdown
# App Integration Prompts

Prepared prompts for wiring each coaching app to use voice_user for persistence and the shared review component.

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
````

**Step 2: Commit**

```bash
git add docs/plans/2026-03-09-app-integration-prompts.md
git commit -m "docs: add integration prompts for spanish, singing, femme"
```

---

## Summary

| Task | Repo | What |
|------|------|------|
| 1 | voice-user | AnalysisRecord TypedDict |
| 2 | voice-user | Failing tests for analyses CRUD |
| 3 | voice-user | Migration 2 (analyses table) |
| 4 | voice-user | Implement CRUD + export + run all tests |
| 5 | voice-core | Shared review.js component |
| 6 | voice-core | Review styles in theme.css |
| 7 | voice-core | App integration prompts doc |
