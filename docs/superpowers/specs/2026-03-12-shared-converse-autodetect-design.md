# Shared Converse Component + Auto-Detect VAD — Design Spec

**Date:** 2026-03-12
**Status:** Draft (pending user review)
**Repos affected:** voice-core, spanish-voice-coach, femme-voice-coach

---

## Overview

Two tightly related features:

1. **`ConversationEngine`** extracted to `voice_core` as a configurable shared backend — Spanish and femme each inject their own prompt, topics, and analysis function.
2. **Auto-detect VAD** added as a standalone `vad.js` utility in voice-core — used by `RecordingControl` (single-session) and `converse.js` (multi-turn) independently. After mic check establishes a noise floor, silence detection drives turn submission automatically. Default-on for Converse, off for practice/eval.

---

## 1. Backend: `voice_core/converse.py`

### `ConversationEngine`

```python
class ConversationEngine:
    def __init__(
        self,
        system_prompt: str,
        topics: list[dict],              # [{id, label, description}]
        llm_config: dict,                # {url, model, api_key}
        analysis_fn: Callable[[Path], Awaitable[dict]] | None = None,
        tts_fn: Callable[[str], Awaitable[Path]] | None = None,
        analysis_ready_fn: Callable[[str, dict], None] | None = None,
        max_history: int = 40,
    ): ...

    async def start(self, topic_id: str) -> dict:
        # Returns {opening_text, audio_url?, turn_id}

    async def process_turn(self, transcript: str, audio_path: Path) -> dict:
        # transcript: pre-computed by WS handler (e.g. faster_whisper) — engine never transcribes
        # audio_path: passed to analysis_fn only
        # Returns {response_text, audio_url?, turn_id}
        # Dispatches analysis_fn(audio_path) as background task

    async def end(self) -> dict:
        # Returns {turns}
        # duration_s is computed by the WS handler (tracks session start time)
```

**Responsibilities:**
- LLM call (OpenAI-compatible API, configurable url/model/api_key)
- Conversation history management (rolling `max_history` non-system messages)
- Transcript extraction from audio (via voice-core transcription)
- Async dispatch of `analysis_fn(audio_path)` as a background task
- When analysis completes, calls `analysis_ready_fn(turn_id, results)` — registered at construction by the WS handler
- Optional TTS via `tts_fn(text) → audio_path`

**Not responsible for:** WebSocket transport, user storage, app-specific routing.

---

## 2. Frontend: `voice-core/frontend/vad.js`

Standalone voice activity detection utility, used independently by `RecordingControl` and `converse.js`.

### API

```javascript
class VoiceActivityDetector extends EventTarget {
  constructor(analyserNode, opts = {}) {
    // opts: thresholdDb, silenceMs (default 2000), onsetMs (default 150)
  }

  async calibrate(durationMs = 2000)
  // Samples ambient RMS over durationMs, sets this.noiseFloorDb
  // threshold = noiseFloorDb + 8 dB headroom
  // Fallback if never calibrated: -35 dBFS

  arm()     // enable silence detection
  disarm()  // suppress silence detection (e.g. during AI audio playback)
  destroy() // stop interval, clean up
}
```

**Events:**

| Event | Detail | When |
|-------|--------|------|
| `voicedetected` | `{rmsDb}` | Voice onset confirmed (sustained > onsetMs) |
| `silencedetected` | `{silenceMs}` | Silence window exceeded |
| `silenceprogress` | `{pct: 0-100}` | Ticks during silence window (80ms interval) |

**Noise floor fallback:** Threshold defaults to `-35 dBFS` if `calibrate()` was never called.

**Implementation:** Runs on the existing `AnalyserNode` at 80ms ticks. No new audio graph nodes.

---

## 3. Frontend: Auto-Detect in `RecordingControl`

`RecordingControl` uses `VoiceActivityDetector` internally when `autoDetect: true`.

### New constructor options

```javascript
new RecordingControl(containerEl, {
  autoDetect: false,           // default off; Converse does not use RC
  autoDetectSilenceMs: 2000,
  autoDetectOnsetMs: 150,
})
```

### Behavior when `autoDetect: true`

- `checkLevel()` calls `vad.calibrate()` to sample noise floor
- When recording is active, VAD is armed; `silencedetected` → calls `this.stop()`
- New public method **`armAutoDetect()`**: re-arms VAD after a turn (for practice flows that use RC in auto-detect mode)
- `silenceprogress` events → RC updates `--vc-rc-silence-pct` CSS custom property and adds `vc-rc-silence-counting` class for countdown bar styling

### Zero overhead when `autoDetect: false`

No VAD instance is created. No behavior change for practice or eval.

---

## 4. Frontend: Shared `converse.js`

**Location:** `voice-core/frontend/converse.js`

Converse manages its **own mic lifecycle** independently of `RecordingControl`. The existing Spanish implementation confirms this pattern — each turn creates a new server-side `AudioSession` while the client mic stays open for the full session. `RecordingControl` is a single-session tool; Converse is multi-turn.

### Mic lifecycle in Converse

```
session start → openMic() [AudioContext + AnalyserNode + AudioWorklet]
  turn N: startAudioStream() → user speaks → stopAudioStream() → send audio via /ws/live/audio
  ...
session end → closeMic()
```

`converse.js` reuses `audio_worklet.js` (already shared in voice-core/frontend/) and creates its own `VoiceActivityDetector` instance on the same `AnalyserNode`.

`converse.js` uses a **single WebSocket** (`/ws/live`) for all JSON control messages (matching the existing Spanish pattern). Audio is streamed separately on `/ws/live/audio` as binary PCM — the same two-socket subset of the 3-socket pattern.

### Init API

```javascript
import { initConverse } from '/static/core/converse.js';

initConverse({
  mountEl: document.getElementById('converse-mount'),
  topics: [{id, label}],
  analysisFields: [
    {key: 'gender_score', label: 'Gender',    format: 'score_100'},
    {key: 'resonance',    label: 'Resonance', format: 'score_100'},
    {key: 'vocal_weight', label: 'Weight',    format: 'score_100'},
  ],
  wsBasePath: '/ws/live',
  audioSpeed: 0.85,
  autoDetect: true,
  silenceMs: 2000,
  onsetMs: 150,
  userIdFn: () => localStorage.getItem('activeUserId'),
})
```

### Turn state machine

```
idle
  → [start topic] → opening_playing
opening_playing
  → [audio ended] → vad.arm() → armed
armed
  → [voicedetected] → speaking
speaking
  → [silencedetected / manual stop] → stopAudioStream() → send converse:turn_done → processing
processing
  → [converse:user_heard] → render user bubble with '--' analysis badges
  → [converse:response + audio] → playing
  → [converse:analysis] → update badges in-place (any time after user_heard, matched by turn_id)
playing
  → [audio ended] → vad.arm() → armed
```

### Auto-detect toggle

Rendered inside the component above the chat log:

```
[●] Auto-detect
```

- Default matches `autoDetect` init option
- **Toggle ON:** calls `vad.calibrate()` (2s sample), then arms — forces fresh noise floor
- **Toggle OFF:** disarms VAD; manual Start/Stop buttons appear
- State persisted in `localStorage` keyed per app (`vc_converse_autodetect_{appKey}`)

### Silence countdown bar

Thin progress bar driven by `silenceprogress` events. Fades out when voice is detected.

### Chat bubble analysis badges

- Render as `--` (dimmed) when `converse:user_heard` arrives
- Update in-place when `converse:analysis` arrives
- Format `score_100`: numeric + color bar (green ≥70, amber 40–69, red <40)

### App-specific chrome stays per-app

Page layout, nav, topic selector, and domain-specific UI (Spanish dictionary etc.) stay in each app's `converse.html`. Shared component mounts into `#converse-mount`.

---

## 5. WebSocket Protocol

The existing `/ws/live` socket is **already bidirectional** in Spanish — confirmed in `spanish_coach/server.py` where both server-push metric frames and client JSON messages (`converse:start`, `converse:turn_done`) are handled on the same socket. The spec follows this established pattern.

| Direction | Type | Payload |
|-----------|------|---------|
| client→server | `converse:start` | `{topic_id}` |
| server→client | `converse:opening` | `{text, audio_url?, turn_id}` |
| client→server | `converse:turn_done` | — |
| server→client | `converse:user_heard` | `{transcript, turn_id}` |
| server→client | `converse:response` | `{text, audio_url?, turn_id}` |
| server→client | `converse:analysis` | `{turn_id, results: dict}` |
| client→server | `converse:end` | — |
| server→client | `converse:ended` | `{turns, duration_s}` |

`converse:analysis` is async — no ordering guarantee vs. `converse:response`. Client matches by `turn_id`.

---

## 6. App Integration

### Spanish voice coach (Phase 3 — migration)

**Backend:** Current `ConversationEngine` in `conversation.py` replaced by `voice_core.converse.ConversationEngine`.

**Async migration required:** Spanish's existing LLM calls use `httpx.Client` (synchronous). The new engine is fully async (`httpx.AsyncClient`). Phase 3 includes porting the LLM call pattern and updating the WS handler. This is non-trivial porting, not a drop-in replacement.

Spanish injects: existing system prompt, 13 topics, async pronunciation scorer as `analysis_fn`, XTTS as `tts_fn`, WS send closure as `analysis_ready_fn`.

**Frontend:** Spanish `converse.js` migrated to use `initConverse()`. Spanish-specific UI (word confidence badges, dictionary tooltips, cheat box) stays in `spanish/frontend/converse-ext.js` layered on top after mount.

**Protocol rename:** Spanish currently uses `converse:pronunciation` for async analysis results. Phase 3 renames this to `converse:analysis` (the shared protocol). Frontend and backend must be updated together.

**Implementation note:** The `analysis_ready_fn` closure captures the active WebSocket. Guard against WS disconnect between turn submission and analysis completion — check `ws.client_state` before sending.

**Risk:** Spanish converse works today. Validate Phase 2 before touching it.

### Femme voice coach (Phase 2 — greenfield)

**Backend:** New `routes_converse.py`. `ConversationEngine` with:
- `system_prompt` — friendly conversationalist, no coaching persona
- `topics` — daily life, work, social situations, hobbies, relationships
- `analysis_fn` — `femme_coach.scoring.score()` → `{gender_score, resonance, vocal_weight}`
- `tts_fn` — existing Kokoro infrastructure
- `analysis_ready_fn` — WS send closure

**Frontend:** New `femme/frontend/converse.html`. Nav updated: Practice / Converse / Review.

---

## 7. Implementation Phases

### Phase 1 — `vad.js` + RecordingControl auto-detect (voice-core only)
- `VoiceActivityDetector` class (calibrate, arm, disarm, events)
- Wire into RecordingControl when `autoDetect: true`
- CSS countdown hook
- No app changes required

### Phase 2 — Femme Converse (greenfield, lower risk)
- `voice_core/converse.py` ConversationEngine
- `voice-core/frontend/converse.js` shared component
- Femme `routes_converse.py` + WS handler
- Femme `converse.html`

### Phase 3 — Spanish migration (after Phase 2 validated)
- Port Spanish ConversationEngine to async
- Replace Spanish WS handler with shared ConversationEngine
- Migrate Spanish `converse.js` to `initConverse()` + `converse-ext.js`
- Regression test press-to-talk + auto-detect paths

---

## 8. Out of Scope

- ML-based VAD (Silero, WebRTC VAD) — calibrated RMS sufficient for v1
- Multi-language support in femme converse
- Pronunciation word-level analysis for femme (session-level only)
- Singing converse
