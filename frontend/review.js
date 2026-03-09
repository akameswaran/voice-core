/**
 * Shared Review Component — reusable session/recording browser.
 *
 * Usage:
 *   import { initReview } from '/static/core/review.js';
 *   initReview({
 *     mountTo: '#review-mount',
 *     analyzerKey: 'femme_scoring',
 *     renderAnalysis: (analyses, el) => { ... },
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

        // Load progress if extractProgressPoint provided
        if (extractProgressPoint && state.sessions.length > 0) {
            await loadProgress();
        }
    }

    async function selectSession(sessionId) {
        state.selectedSessionId = sessionId;
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

            card.querySelector('.vc-review-btn-play').addEventListener('click', (e) => {
                e.stopPropagation();
                togglePlay(rec.id);
            });

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

    // ── Progress summary ───────────────────────────────────
    async function loadProgress() {
        if (!extractProgressPoint) return;

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

    return { loadSessions, selectSession };
}
