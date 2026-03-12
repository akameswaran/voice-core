# Shared Converse Component + Auto-Detect VAD — Design Spec

**Date:** 2026-03-12
**Status:** Approved
**Repos affected:** voice-core, spanish-voice-coach, femme-voice-coach

---

## Overview

Two tightly related features:

1. **`ConversationEngine`** extracted to `voice_core` as a configurable shared backend — Spanish and femme each inject their own prompt, topics, and analysis function.
2. **Auto-detect VAD** added to `RecordingControl` as an opt-in mode — after mic check establishes a noise floor, silence detection drives turn submission automatically. Default-on for Converse, off for practice/eval.

---

## 1. Backend: `voice_core/converse.py`

### `ConversationEngine`

```python
class ConversationEngine:
    def __init__(
        self,
        system_prompt: str,
        topics: list[dict],          # [{id, label, description}]
        llm_config: dict,            # {url, model, api_key}
        analysis_fn: Callable[[Path], Awaitable[dict]] | None = None,
        tts_fn: Callable[[str], Awaitable[Path]] | None = None,
        max_history: int = 40,
    ): ...

    async def start(self, topic_id: str) -> dict:
        # Returns {opening_text, audio_url?}

    async def process_turn(self, audio_path: Path) -> dict:
        # Returns {transcript, response_text, audio_url?}
        # Dispatches analysis_fn as background task; results delivered
        # via on_analysis_ready callback when complete

    async def end(self) -> dict:
        # Returns {turns, duration_s}
```

**Responsibilities:**
- LLM call (OpenAI-compatible API, configurable url/model)
- Conversation history management (rolling `max_history` non-system messages)
- Transcript extraction from audio (via voice-core transcription)
- Async dispatch of `analysis_fn(audio_path)` — does not block response
- Optional TTS via `tts_fn(text) → audio_path`

**Not responsible for:** WebSocket transport, user storage, app-specific routing. Those stay in each app's server layer.

### Analysis callback pattern

`process_turn` dispatches `analysis_fn` as a background task. When results are ready, the engine calls `on_analysis_ready(turn_id, results)` — a callback the WebSocket handler registers. The handler forwards results to the client via the existing WS connection. This keeps conversation latency independent of analysis latency.

---

## 2. Frontend: Auto-Detect VAD in `RecordingControl`

### New constructor options

```javascript
new RecordingControl(containerEl, {
  autoDetect: false,               // default off; Converse sets true
  autoDetectThresholdDb: null,     // null = noise_floor + 8 dB (dynamic)
  autoDetectSilenceMs: 2000,       // 2s silence → auto-submit turn
  autoDetectOnsetMs: 150,          // voice must be present 150ms to count
})
```

### Noise floor calibration

During `checkLevel()`, when `autoDetect: true`, RecordingControl samples ambient RMS over 2 seconds and stores `this._noiseFloorDb`. Dynamic threshold = `_noiseFloorDb + 8`. This replaces any previous session's noise floor.

If `autoDetectThresholdDb` is explicitly set, it overrides the dynamic calculation.

### Turn gate: `armAutoDetect()`

New public method. Must be called by the app after AI audio finishes playing. Until called, silence detection is suppressed — prevents AI speaker bleed from triggering a false turn.

Auto-detect state machine (when `autoDetect: true` and armed):
```
armed → [voice onset > 150ms] → speaking
speaking → [silence > 2000ms] → this.stop() dispatched
speaking → [manual stop] → stop() as normal
```

The gate resets to `unarmed` automatically when `stop()` is called.

### New events

| Event | Detail | When |
|-------|--------|------|
| `voicedetected` | `{rmsDb}` | Voice onset confirmed |
| `silencedetected` | `{silenceMs}` | Silence timeout approaching |
| `autodetectarmed` | — | `armAutoDetect()` called |

### Visual silence countdown

During the 2s silence window, RecordingControl updates a `--vc-rc-silence-pct` CSS custom property (0→100) on its container, and adds class `vc-rc-silence-counting`. Apps can style a countdown bar using this hook without coupling to internal RC state.

### Implementation notes

- Reuses existing `_analyser` AnalyserNode — no new audio graph nodes
- RMS sampling runs in the existing `_levelInterval` tick (80ms)
- All VAD logic is off when `autoDetect: false` — zero overhead for practice/eval

---

## 3. Frontend: Shared `converse.js`

**Location:** `voice-core/frontend/converse.js`

### Init API

```javascript
import { initConverse } from '/static/core/converse.js';

initConverse({
  mountEl: document.getElementById('converse-mount'),
  topics: [{id, label}],
  analysisFields: [
    // Rendered under each user turn as async badges
    {key: 'gender_score', label: 'Gender',    format: 'score_100'},
    {key: 'resonance',    label: 'Resonance', format: 'score_100'},
    {key: 'vocal_weight', label: 'Weight',    format: 'score_100'},
  ],
  wsBasePath: '/ws/live',
  audioSpeed: 0.85,
  autoDetect: true,
})
```

### Turn state machine

```
idle → [start topic] → armed
armed → [voicedetected] → speaking
speaking → [silencedetected / manual stop] → processing
processing → [response received] → playing
playing → [audio ended] → armAutoDetect() → armed
```

### Chat bubble rendering

- **AI turns:** text bubble, no analysis
- **User turns:** text bubble + analysis badge row
  - Badges render immediately as `--` (dimmed)
  - Update in-place when WS delivers `converse:analysis` message
  - Format `score_100`: colored bar + numeric (0–100), color-coded green/amber/red

### Auto-detect toggle

A small toggle rendered inside the component above the chat log:

```
[●] Auto-detect  ← toggles on/off
```

- Default: on (matches `autoDetect` init option)
- **Toggling ON:** calls `rc.checkLevel()` to force mic recheck and recalibrate noise floor before arming
- **Toggling OFF:** disables VAD; Start/Stop buttons appear for manual control
- Toggle state persists in `localStorage` per app key

### Silence countdown bar

A thin progress bar above the chat input, visible only during the 2s silence window. Uses the `--vc-rc-silence-pct` CSS hook from RecordingControl. Fades out when user speaks again.

### App-specific chrome stays per-app

Each app's `converse.html` owns: page layout, nav, topic selector (populated from `/api/topics`), and any domain-specific UI (Spanish dictionary panel, etc.). The shared component mounts into `#converse-mount` and owns only the chat area.

---

## 4. App Integration

### Spanish voice coach (migration)

**Backend:** Current `conversation.py` `ConversationEngine` class replaced by `voice_core.converse.ConversationEngine`. Spanish injects:
- `system_prompt` — existing Spanish teacher/conversation partner prompt (unchanged)
- `topics` — existing 13 categories
- `analysis_fn` — existing `_analyze_converse_turn` (pronunciation scorer)
- `tts_fn` — existing XTTS call

**Frontend:** `converse.js` replaced with `initConverse()` from voice-core. `analysisFields` set to pronunciation fields. Spanish-specific UI (word confidence badges, dictionary tooltips, cheat box) stays in a thin `spanish/frontend/converse-ext.js` that extends the shared component via DOM manipulation after mount.

**Risk:** Spanish converse.js is 720 lines and currently works. Migration must be done carefully with regression testing against the existing press-to-talk flow before enabling auto-detect.

### Femme voice coach (new feature)

**Backend:** New `src/femme_coach/server/routes_converse.py`. `ConversationEngine` configured with:
- `system_prompt` — friendly conversationalist, no coaching persona, natural chat
- `topics` — initial set: daily life, work, social situations, hobbies, relationships
- `analysis_fn` — calls `femme_coach.scoring.score()` on the saved audio, returns `{gender_score, resonance, vocal_weight}`
- `tts_fn` — reuses existing Kokoro TTS infrastructure

**Frontend:** New `femme/frontend/converse.html` with `initConverse()`, `analysisFields` = gender/resonance/weight.

**Nav:** Add "Converse" link to femme nav alongside Practice / Review.

---

## 5. WebSocket Protocol (shared)

New message types added to the existing `/ws/live` protocol (both apps):

| Direction | Type | Payload |
|-----------|------|---------|
| client→server | `converse:start` | `{topic_id}` |
| server→client | `converse:opening` | `{text, audio_url?, turn_id}` |
| client→server | `converse:turn_done` | — (sent after auto-silence or manual stop) |
| server→client | `converse:user_heard` | `{transcript, turn_id}` |
| server→client | `converse:response` | `{text, audio_url?, turn_id}` |
| server→client | `converse:analysis` | `{turn_id, results: dict}` |
| client→server | `converse:end` | — |
| server→client | `converse:ended` | `{turns, duration_s}` |

`converse:analysis` is sent asynchronously after `converse:user_heard` — no ordering guarantee relative to `converse:response`. The client matches by `turn_id`.

---

## 6. Implementation Phases

### Phase 1 — VAD in RecordingControl (voice-core)
- Noise floor calibration in `checkLevel()`
- Silence detection + onset gating
- `armAutoDetect()` method
- CSS countdown hook
- Auto-detect toggle component
- No app changes needed — ships as voice-core update

### Phase 2 — Femme Converse (new feature)
- `voice_core/converse.py` ConversationEngine
- `voice-core/frontend/converse.js` shared component
- Femme `routes_converse.py` + WS handler
- Femme `converse.html`

### Phase 3 — Spanish migration
- Replace Spanish `conversation.py` with shared ConversationEngine
- Migrate Spanish `converse.js` to use `initConverse()`
- Regression test press-to-talk + auto-detect paths

Femme is greenfield (Phase 2) so it's lower risk than Spanish migration (Phase 3). Do Phase 2 first to validate the shared component, then migrate Spanish.

---

## 7. Out of Scope

- ML-based VAD (Silero, WebRTC VAD) — RMS + noise floor is sufficient for v1
- Multi-language support in femme converse
- Pronunciation word-level analysis for femme (only session-level acoustic scores)
- Singing converse
