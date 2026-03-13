# Shared Converse Component + Auto-Detect VAD — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract Converse as a shared voice-core component, add auto-detect VAD to RecordingControl, and build femme-voice-coach Converse as the first consumer.

**Architecture:** `VoiceActivityDetector` (vad.js) is a standalone frontend utility reused by both RecordingControl (single-session auto-stop) and converse.js (multi-turn turn detection). `ConversationEngine` in voice-core backend is configured via dependency injection — each app injects its system prompt, topics, analysis function, and TTS function. Femme adds a dedicated `/ws/converse` WebSocket endpoint to keep the existing 60Hz metrics stream on `/ws/live` unchanged.

**Tech Stack:** Python 3.13, FastAPI, asyncio, httpx (async), faster-whisper, JavaScript ESM, AudioWorklet, WebSocket

**Spec:** `voice-core/docs/superpowers/specs/2026-03-12-shared-converse-autodetect-design.md`

---

## Chunk 1: Phase 1 — vad.js + RecordingControl Auto-Detect

### Task 1: Create `voice-core/frontend/vad.js`

**Files:**
- Create: `voice-core/frontend/vad.js`

- [ ] **Step 1: Create vad.js with VoiceActivityDetector class**

```javascript
// voice-core/frontend/vad.js
/**
 * VoiceActivityDetector — RMS-based voice/silence detection.
 *
 * Usage:
 *   const vad = new VoiceActivityDetector(analyserNode, { silenceMs: 2000 });
 *   await vad.calibrate();  // sample noise floor (call during mic check)
 *   vad.arm();              // enable detection
 *   vad.addEventListener('silencedetected', () => stopTurn());
 *   vad.addEventListener('silenceprogress', e => updateBar(e.detail.pct));
 *   vad.disarm();           // suppress (e.g. while AI audio plays)
 *   vad.destroy();          // cleanup
 */

const VAD_TICK_MS = 80;
const NOISE_FLOOR_HEADROOM_DB = 8;
const FALLBACK_THRESHOLD_DB = -35;

export class VoiceActivityDetector extends EventTarget {
  constructor(analyserNode, opts = {}) {
    super();
    this._analyser = analyserNode;
    this._buf = new Float32Array(analyserNode.fftSize);
    this._silenceMs = opts.silenceMs ?? 2000;
    this._onsetMs = opts.onsetMs ?? 150;
    this._thresholdDb = opts.thresholdDb ?? null; // null = dynamic from noise floor
    this._noiseFloorDb = null;
    this._armed = false;
    this._speaking = false;
    this._silenceStart = null;
    this._onsetStart = null;
    this._interval = null;
  }

  get thresholdDb() {
    if (this._thresholdDb !== null) return this._thresholdDb;
    if (this._noiseFloorDb !== null) return this._noiseFloorDb + NOISE_FLOOR_HEADROOM_DB;
    return FALLBACK_THRESHOLD_DB;
  }

  /** Sample ambient RMS for durationMs, set noise floor. Returns noise floor dBFS. */
  async calibrate(durationMs = 2000) {
    const samples = [];
    const start = Date.now();
    return new Promise((resolve) => {
      const tick = setInterval(() => {
        const db = this._rmsDb();
        samples.push(db);
        if (Date.now() - start >= durationMs) {
          clearInterval(tick);
          this._noiseFloorDb = samples.reduce((a, b) => a + b, 0) / samples.length;
          resolve(this._noiseFloorDb);
        }
      }, VAD_TICK_MS);
    });
  }

  /** Enable silence/voice detection. Call after calibrate() and when ready for user input. */
  arm() {
    this._armed = true;
    this._speaking = false;
    this._silenceStart = null;
    this._onsetStart = null;
    if (!this._interval) {
      this._interval = setInterval(() => this._tick(), VAD_TICK_MS);
    }
  }

  /** Suppress detection (e.g. while AI audio is playing). Does not destroy interval. */
  disarm() {
    this._armed = false;
    this._speaking = false;
    this._silenceStart = null;
    this._onsetStart = null;
  }

  /** Stop the tick interval and clean up. */
  destroy() {
    if (this._interval) {
      clearInterval(this._interval);
      this._interval = null;
    }
    this._armed = false;
  }

  _rmsDb() {
    this._analyser.getFloatTimeDomainData(this._buf);
    let sum = 0;
    for (let i = 0; i < this._buf.length; i++) sum += this._buf[i] * this._buf[i];
    const rms = Math.sqrt(sum / this._buf.length);
    return 20 * Math.log10(Math.max(rms, 1e-10));
  }

  _tick() {
    if (!this._armed) return;
    const db = this._rmsDb();
    const threshold = this.thresholdDb;
    const now = Date.now();

    if (db >= threshold) {
      // Voice present
      this._silenceStart = null;
      if (!this._speaking) {
        if (!this._onsetStart) this._onsetStart = now;
        if (now - this._onsetStart >= this._onsetMs) {
          this._speaking = true;
          this._onsetStart = null;
          this.dispatchEvent(new CustomEvent('voicedetected', { detail: { rmsDb: db } }));
        }
      }
    } else {
      // Silence
      this._onsetStart = null;
      if (this._speaking) {
        if (!this._silenceStart) this._silenceStart = now;
        const silenceMs = now - this._silenceStart;
        const pct = Math.min(100, (silenceMs / this._silenceMs) * 100);
        this.dispatchEvent(new CustomEvent('silenceprogress', { detail: { pct } }));
        if (silenceMs >= this._silenceMs) {
          this._speaking = false;
          this._silenceStart = null;
          this._armed = false; // auto-disarm after trigger
          this.dispatchEvent(new CustomEvent('silencedetected', { detail: { silenceMs } }));
        }
      }
    }
  }
}
```

- [ ] **Step 2: Smoke-test in browser console**

Open any page that imports voice-core static files. In devtools console:
```javascript
const ctx = new AudioContext();
await ctx.resume();
const osc = ctx.createOscillator();
const analyser = ctx.createAnalyser();
osc.connect(analyser);
osc.start();
const { VoiceActivityDetector } = await import('/static/core/vad.js');
const vad = new VoiceActivityDetector(analyser, { silenceMs: 500, onsetMs: 50 });
const floor = await vad.calibrate(1000);
console.log('noise floor:', floor, 'dBFS  threshold:', vad.thresholdDb, 'dBFS');
vad.arm();
vad.addEventListener('voicedetected', e => console.log('voice!', e.detail));
vad.addEventListener('silencedetected', e => console.log('silence!', e.detail));
vad.addEventListener('silenceprogress', e => console.log('pct:', e.detail.pct));
// wait 3+ seconds of silence — should fire silencedetected
```

Expected: noise floor ~-50 to -70 dBFS (quiet room), threshold ~8 dB above that, `silencedetected` fires after ~500ms silence.

- [ ] **Step 3: Commit**

```bash
cd /home/ak/Projects/VoiceCoaches/voice-core
git add frontend/vad.js
git commit -m "feat: add VoiceActivityDetector (vad.js) to voice-core frontend"
```

---

### Task 2: Wire VAD into RecordingControl

**Files:**
- Modify: `voice-core/frontend/recording_control.js`

- [ ] **Step 1: Add autoDetect options to constructor defaults**

In `_opts` defaults (around line 70), add:
```javascript
autoDetect: false,
autoDetectSilenceMs: 2000,
autoDetectOnsetMs: 150,
```

Add instance variables after `this._lastSavedFilename = null;`:
```javascript
this._vad = null; // VoiceActivityDetector, created if autoDetect: true
```

- [ ] **Step 2: Initialize VAD in `_openMic()` (called by checkLevel and start)**

In `_openMic()`, after the AnalyserNode is set up, add:
```javascript
// Create VAD if auto-detect enabled
if (this._opts.autoDetect && !this._vad) {
  import('./vad.js').then(({ VoiceActivityDetector }) => {
    this._vad = new VoiceActivityDetector(this._analyser, {
      silenceMs: this._opts.autoDetectSilenceMs,
      onsetMs: this._opts.autoDetectOnsetMs,
    });
    this._vad.addEventListener('silencedetected', () => {
      if (this._state === 'recording') this.stop();
    });
    this._vad.addEventListener('silenceprogress', (e) => {
      this._container.style.setProperty('--vc-rc-silence-pct', e.detail.pct);
      this._container.classList.toggle('vc-rc-silence-counting', e.detail.pct > 0);
    });
  });
}
```

- [ ] **Step 3: Calibrate noise floor in `checkLevel()`**

In `checkLevel()`, after `this._startLevelMeter()`:
```javascript
// Calibrate VAD noise floor if auto-detect enabled
if (this._opts.autoDetect && this._vad) {
  this._vad.calibrate(2000).then(floor => {
    console.log('[vc-rc] VAD noise floor:', floor.toFixed(1), 'dBFS threshold:', this._vad.thresholdDb.toFixed(1), 'dBFS');
  });
}
```

- [ ] **Step 4: Arm VAD when recording starts, disarm when stopped**

In `start()`, after `this._setState('recording')`:
```javascript
if (this._opts.autoDetect && this._vad) this._vad.arm();
```

In `stop()`, before `this._setState('processing')`:
```javascript
if (this._vad) this._vad.disarm();
```

In `cancel()`, add:
```javascript
if (this._vad) this._vad.disarm();
```

- [ ] **Step 5: Add `armAutoDetect()` public method**

After the `get stream()` getter:
```javascript
/** Re-arm VAD after a turn (e.g. after AI audio ends in converse mode). */
armAutoDetect() {
  if (this._opts.autoDetect && this._vad) this._vad.arm();
}
```

- [ ] **Step 6: Destroy VAD in `_closeMic()`**

In `_closeMic()` (find by searching "closeMic"), at the top of the method:
```javascript
if (this._vad) { this._vad.destroy(); this._vad = null; }
```

- [ ] **Step 7: Manual test auto-detect in browser**

Navigate to `https://akpersistant:9000/practice`. Open devtools console. Temporarily test by running:
```javascript
// Patch an existing RC instance to use autoDetect (dev only)
// Verify: after setting autoDetect, checkLevel calibrates, recording auto-stops after 2s silence
```
Or create a minimal test HTML page that passes `autoDetect: true` to RecordingControl.

Expected: after speaking and then going silent for 2s, recording stops automatically. Countdown CSS class `vc-rc-silence-counting` is added during the silence window.

- [ ] **Step 8: Commit**

```bash
cd /home/ak/Projects/VoiceCoaches/voice-core
git add frontend/recording_control.js
git commit -m "feat: add auto-detect VAD mode to RecordingControl"
```

---

## Chunk 2: Phase 2 Backend — ConversationEngine

### Task 3: Create `voice_core/converse.py`

**Files:**
- Create: `voice-core/src/voice_core/converse.py`
- Create: `voice-core/tests/test_converse.py`

- [ ] **Step 1: Write failing tests first**

Create `voice-core/tests/test_converse.py`:
```python
"""Tests for voice_core.converse.ConversationEngine."""
import asyncio
import json
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest

# Add voice-core src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voice_core.converse import ConversationEngine


TOPICS = [
    {"id": "daily", "label": "Daily Life", "description": "Everyday conversations"},
    {"id": "work", "label": "Work", "description": "Professional topics"},
]

LLM_CONFIG = {
    "url": "http://localhost:11434/v1",
    "model": "qwen2.5",
    "api_key": "test",
}


@pytest.fixture
def engine():
    return ConversationEngine(
        system_prompt="You are a friendly conversation partner.",
        topics=TOPICS,
        llm_config=LLM_CONFIG,
    )


@pytest.fixture
def mock_llm_response():
    return {
        "choices": [{"message": {"content": "That sounds wonderful! Tell me more."}}]
    }


class TestConversationEngineInit:
    def test_topics_stored(self, engine):
        assert engine.topics == TOPICS

    def test_initial_history_empty(self, engine):
        assert engine._history == []

    def test_turn_count_zero(self, engine):
        assert engine._turn_count == 0

    def test_no_analysis_fn_by_default(self, engine):
        assert engine._analysis_fn is None

    def test_no_tts_fn_by_default(self, engine):
        assert engine._tts_fn is None

    def test_no_analysis_ready_fn_by_default(self, engine):
        assert engine._analysis_ready_fn is None


class TestConversationEngineStart:
    @pytest.mark.asyncio
    async def test_start_calls_llm(self, engine, mock_llm_response):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value.json = MagicMock(return_value=mock_llm_response)
            mock_post.return_value.raise_for_status = MagicMock()
            result = await engine.start("daily")
        mock_post.assert_called_once()
        call_body = json.loads(mock_post.call_args.kwargs["content"])
        assert call_body["model"] == "qwen2.5"

    @pytest.mark.asyncio
    async def test_start_returns_opening_text(self, engine, mock_llm_response):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value.json = MagicMock(return_value=mock_llm_response)
            mock_post.return_value.raise_for_status = MagicMock()
            result = await engine.start("daily")
        assert result["opening_text"] == "That sounds wonderful! Tell me more."
        assert "turn_id" in result

    @pytest.mark.asyncio
    async def test_start_adds_to_history(self, engine, mock_llm_response):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value.json = MagicMock(return_value=mock_llm_response)
            mock_post.return_value.raise_for_status = MagicMock()
            await engine.start("daily")
        # System prompt + assistant opening
        assert len(engine._history) == 2
        assert engine._history[0]["role"] == "system"
        assert engine._history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_start_no_audio_url_without_tts_fn(self, engine, mock_llm_response):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value.json = MagicMock(return_value=mock_llm_response)
            mock_post.return_value.raise_for_status = MagicMock()
            result = await engine.start("daily")
        assert result.get("audio_url") is None


class TestConversationEngineProcessTurn:
    @pytest.fixture
    async def started_engine(self, engine, mock_llm_response):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value.json = MagicMock(return_value=mock_llm_response)
            mock_post.return_value.raise_for_status = MagicMock()
            await engine.start("daily")
        return engine

    @pytest.mark.asyncio
    async def test_process_turn_returns_response(self, started_engine, mock_llm_response, tmp_path):
        audio = tmp_path / "turn.wav"
        audio.write_bytes(b"fake wav")
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value.json = MagicMock(return_value=mock_llm_response)
            mock_post.return_value.raise_for_status = MagicMock()
            result = await started_engine.process_turn("Hello there!", audio)
        assert result["response_text"] == "That sounds wonderful! Tell me more."
        assert "turn_id" in result

    @pytest.mark.asyncio
    async def test_process_turn_adds_user_and_assistant_to_history(self, started_engine, mock_llm_response, tmp_path):
        audio = tmp_path / "turn.wav"
        audio.write_bytes(b"fake wav")
        history_before = len(started_engine._history)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value.json = MagicMock(return_value=mock_llm_response)
            mock_post.return_value.raise_for_status = MagicMock()
            await started_engine.process_turn("Hello there!", audio)
        assert len(started_engine._history) == history_before + 2
        assert started_engine._history[-2]["role"] == "user"
        assert started_engine._history[-2]["content"] == "Hello there!"
        assert started_engine._history[-1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_process_turn_dispatches_analysis_async(self, tmp_path, mock_llm_response):
        analysis_calls = []
        async def mock_analysis(audio_path):
            analysis_calls.append(audio_path)
            return {"gender_score": 72, "resonance": 65}

        ready_calls = []
        def mock_ready(turn_id, results):
            ready_calls.append((turn_id, results))

        eng = ConversationEngine(
            system_prompt="You are friendly.",
            topics=TOPICS,
            llm_config=LLM_CONFIG,
            analysis_fn=mock_analysis,
            analysis_ready_fn=mock_ready,
        )
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value.json = MagicMock(return_value=mock_llm_response)
            mock_post.return_value.raise_for_status = MagicMock()
            await eng.start("daily")
            audio = tmp_path / "turn.wav"
            audio.write_bytes(b"fake wav")
            result = await eng.process_turn("Hi", audio)
        # Give background task time to run
        await asyncio.sleep(0.05)
        assert len(analysis_calls) == 1
        assert len(ready_calls) == 1
        assert ready_calls[0][0] == result["turn_id"]
        assert ready_calls[0][1]["gender_score"] == 72

    @pytest.mark.asyncio
    async def test_history_trimmed_to_max(self, tmp_path, mock_llm_response):
        eng = ConversationEngine(
            system_prompt="Be friendly.",
            topics=TOPICS,
            llm_config=LLM_CONFIG,
            max_history=6,  # system + 2 turns * 2 messages = 5 max non-system
        )
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value.json = MagicMock(return_value=mock_llm_response)
            mock_post.return_value.raise_for_status = MagicMock()
            await eng.start("daily")
            for i in range(5):
                audio = tmp_path / f"t{i}.wav"
                audio.write_bytes(b"fake")
                await eng.process_turn(f"Turn {i}", audio)
        # Should not exceed max_history + 1 (system prompt)
        assert len(eng._history) <= 7  # 1 system + max_history


class TestConversationEngineEnd:
    @pytest.mark.asyncio
    async def test_end_returns_turn_count(self, engine, mock_llm_response, tmp_path):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value.json = MagicMock(return_value=mock_llm_response)
            mock_post.return_value.raise_for_status = MagicMock()
            await engine.start("daily")
            audio = tmp_path / "t.wav"
            audio.write_bytes(b"x")
            await engine.process_turn("Hello", audio)
            result = await engine.end()
        assert result["turns"] == 1

    @pytest.mark.asyncio
    async def test_end_clears_history(self, engine, mock_llm_response):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value.json = MagicMock(return_value=mock_llm_response)
            mock_post.return_value.raise_for_status = MagicMock()
            await engine.start("daily")
            await engine.end()
        assert engine._history == []
        assert engine._turn_count == 0
```

- [ ] **Step 2: Run tests — confirm they fail**

```bash
cd /home/ak/Projects/VoiceCoaches/voice-core
uv run pytest tests/test_converse.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'voice_core.converse'`

- [ ] **Step 3: Implement ConversationEngine**

Create `voice-core/src/voice_core/converse.py`:
```python
"""Shared ConversationEngine for multi-turn AI conversation coaching.

Apps inject:
  system_prompt    — AI persona (fully app-controlled)
  topics           — [{id, label, description}] for frontend topic picker
  llm_config       — {url, model, api_key} (OpenAI-compatible)
  analysis_fn      — async(audio_path) → dict, called after each user turn
  tts_fn           — async(text) → Path, returns audio file path
  analysis_ready_fn — (turn_id, results) → None, called when analysis completes
  max_history      — rolling window of non-system messages (default 40)

The WS handler owns: transcription, session timing, user storage, WebSocket transport.
The engine owns: LLM calls, history management, async analysis dispatch, TTS.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Awaitable, Callable


class ConversationEngine:
    def __init__(
        self,
        system_prompt: str,
        topics: list[dict],
        llm_config: dict,
        analysis_fn: Callable[[Path], Awaitable[dict]] | None = None,
        tts_fn: Callable[[str], Awaitable[Path]] | None = None,
        analysis_ready_fn: Callable[[str, dict], None] | None = None,
        max_history: int = 40,
    ):
        self._system_prompt = system_prompt
        self.topics = topics
        self._llm_config = llm_config
        self._analysis_fn = analysis_fn
        self._tts_fn = tts_fn
        self._analysis_ready_fn = analysis_ready_fn
        self._max_history = max_history
        self._history: list[dict] = []
        self._turn_count: int = 0

    async def start(self, topic_id: str) -> dict:
        """Begin a conversation on the given topic. Returns opening message."""
        topic = next((t for t in self.topics if t["id"] == topic_id), None)
        topic_label = topic["label"] if topic else topic_id

        self._history = [{"role": "system", "content": self._system_prompt}]
        self._turn_count = 0

        opening = await self._llm_call(
            f"Please start a conversation about: {topic_label}. "
            "Keep your opening short — one or two sentences."
        )
        turn_id = str(uuid.uuid4())
        self._history.append({"role": "assistant", "content": opening})

        result: dict = {"opening_text": opening, "turn_id": turn_id}
        if self._tts_fn:
            audio_path = await self._tts_fn(opening)
            result["audio_url"] = str(audio_path)

        return result

    async def process_turn(self, transcript: str, audio_path: Path) -> dict:
        """Process a user turn. transcript is pre-computed by the WS handler."""
        self._history.append({"role": "user", "content": transcript})
        self._trim_history()

        response_text = await self._llm_call(None)  # None = use history as-is
        turn_id = str(uuid.uuid4())
        self._turn_count += 1
        self._history.append({"role": "assistant", "content": response_text})

        result: dict = {"response_text": response_text, "turn_id": turn_id}
        if self._tts_fn:
            audio_path_tts = await self._tts_fn(response_text)
            result["audio_url"] = str(audio_path_tts)

        # Dispatch analysis as background task
        if self._analysis_fn and self._analysis_ready_fn:
            asyncio.create_task(
                self._run_analysis(turn_id, audio_path)
            )

        return result

    async def end(self) -> dict:
        """End the conversation. Returns {turns}."""
        turns = self._turn_count
        self._history = []
        self._turn_count = 0
        return {"turns": turns}

    async def _run_analysis(self, turn_id: str, audio_path: Path) -> None:
        try:
            results = await self._analysis_fn(audio_path)
            self._analysis_ready_fn(turn_id, results)
        except Exception as e:
            print(f"[converse] analysis failed for turn {turn_id}: {e}")

    async def _llm_call(self, user_message: str | None) -> str:
        """Call the LLM. If user_message is not None, it's appended to history first."""
        import httpx

        messages = list(self._history)
        if user_message is not None:
            messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self._llm_config["model"],
            "messages": messages,
            "temperature": 0.8,
        }
        headers = {"Authorization": f"Bearer {self._llm_config['api_key']}"}
        url = self._llm_config["url"].rstrip("/") + "/chat/completions"

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, content=json.dumps(payload), headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return data["choices"][0]["message"]["content"].strip()

    def _trim_history(self) -> None:
        """Keep system prompt + last max_history non-system messages."""
        system = [m for m in self._history if m["role"] == "system"]
        non_system = [m for m in self._history if m["role"] != "system"]
        if len(non_system) > self._max_history:
            non_system = non_system[-self._max_history:]
        self._history = system + non_system
```

- [ ] **Step 4: Run tests — confirm they pass**

```bash
cd /home/ak/Projects/VoiceCoaches/voice-core
uv run pytest tests/test_converse.py -v
```

Expected: All tests PASS. (The httpx.AsyncClient.post mock intercepts the LLM calls.)

- [ ] **Step 5: Commit**

```bash
git add src/voice_core/converse.py tests/test_converse.py
git commit -m "feat: add ConversationEngine to voice_core"
```

---

## Chunk 3: Phase 2 Frontend + Femme Integration

### Task 4: Create `voice-core/frontend/converse.js`

**Files:**
- Create: `voice-core/frontend/converse.js`

- [ ] **Step 1: Create converse.js shared component**

```javascript
// voice-core/frontend/converse.js
/**
 * initConverse — shared multi-turn conversation UI component.
 *
 * Usage:
 *   import { initConverse } from '/static/core/converse.js';
 *   initConverse({
 *     mountEl: document.getElementById('converse-mount'),
 *     topics: [{id, label}],
 *     analysisFields: [{key, label, format}],  // 'score_100' → 0-100 colored badge
 *     wsPath: '/ws/converse',                   // app-specific WS endpoint
 *     audioWsPath: '/ws/converse/audio',         // PCM upload path
 *     audioSpeed: 0.85,
 *     autoDetect: true,
 *     silenceMs: 2000,
 *     onsetMs: 150,
 *     userIdFn: () => localStorage.getItem('activeUserId'),
 *   });
 */

import { VoiceActivityDetector } from './vad.js';

export function initConverse(opts = {}) {
  const {
    mountEl,
    topics = [],
    analysisFields = [],
    wsPath = '/ws/live',
    audioWsPath = '/ws/live/audio',
    audioSpeed = 0.85,
    autoDetect = true,
    silenceMs = 2000,
    onsetMs = 150,
    userIdFn = () => localStorage.getItem('activeUserId'),
  } = opts;

  if (!mountEl) throw new Error('initConverse: mountEl is required');

  // ── State ────────────────────────────────────────────────────────
  let state = 'idle'; // idle | opening_playing | armed | speaking | processing | playing
  let ws = null;
  let audioWs = null;
  let audioCtx = null;
  let analyserNode = null;
  let workletNode = null;
  let mediaStream = null;
  let vad = null;
  let autoDetectEnabled = autoDetect;
  let currentTopicId = null;

  // Load auto-detect preference from localStorage
  const LS_KEY = 'vc_converse_autodetect';
  const saved = localStorage.getItem(LS_KEY);
  if (saved !== null) autoDetectEnabled = saved === 'true';

  // ── Render ───────────────────────────────────────────────────────
  mountEl.innerHTML = `
    <div class="vc-converse">
      <div class="vc-converse-header">
        <select id="vc-topic-select">
          ${topics.map(t => `<option value="${_esc(t.id)}">${_esc(t.label)}</option>`).join('')}
        </select>
        <button class="btn btn-primary btn-sm" id="vc-converse-start">Start</button>
        <button class="btn btn-ghost btn-sm" id="vc-converse-end" style="display:none">End</button>
        <label class="vc-autodetect-toggle" title="Auto-detect turns by voice activity">
          <input type="checkbox" id="vc-autodetect-cb" ${autoDetectEnabled ? 'checked' : ''}>
          <span>Auto-detect</span>
        </label>
      </div>
      <div class="vc-silence-bar-wrap" id="vc-silence-bar-wrap" style="display:none">
        <div class="vc-silence-bar" id="vc-silence-bar"></div>
      </div>
      <div class="vc-chat-log" id="vc-chat-log"></div>
      <div class="vc-converse-controls" id="vc-converse-controls" style="display:none">
        <button class="btn btn-danger btn-sm" id="vc-speak-btn">&#9679; Speak</button>
        <button class="btn btn-ghost btn-sm" id="vc-stop-btn" style="display:none">&#9632; Done</button>
        <span class="vc-converse-status dim" id="vc-status"></span>
      </div>
    </div>`;

  const $ = id => document.getElementById(id);
  const topicSel = $('vc-topic-select');
  const startBtn = $('vc-converse-start');
  const endBtn = $('vc-converse-end');
  const autoDetectCb = $('vc-autodetect-cb');
  const chatLog = $('vc-chat-log');
  const controls = $('vc-converse-controls');
  const speakBtn = $('vc-speak-btn');
  const stopBtn = $('vc-stop-btn');
  const statusEl = $('vc-status');
  const silenceBarWrap = $('vc-silence-bar-wrap');
  const silenceBar = $('vc-silence-bar');

  // ── Event handlers ───────────────────────────────────────────────
  startBtn.addEventListener('click', startSession);
  endBtn.addEventListener('click', endSession);
  speakBtn.addEventListener('click', startTurn);
  stopBtn.addEventListener('click', stopTurn);

  autoDetectCb.addEventListener('change', async () => {
    autoDetectEnabled = autoDetectCb.checked;
    localStorage.setItem(LS_KEY, autoDetectEnabled);
    if (autoDetectEnabled && vad) {
      // Force recalibration on enable
      setStatus('Calibrating mic...');
      await vad.calibrate(2000);
      setStatus('');
      if (state === 'armed') vad.arm();
    }
    _updateManualControls();
  });

  // ── Core flow ────────────────────────────────────────────────────
  async function startSession() {
    currentTopicId = topicSel.value;
    await _openMic();

    if (autoDetectEnabled) {
      setStatus('Calibrating mic...');
      await vad.calibrate(2000);
      setStatus('');
    }

    const userId = userIdFn();
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}${wsPath}?user_id=${userId}`);
    ws.addEventListener('open', () => {
      ws.send(JSON.stringify({ type: 'converse:start', topic_id: currentTopicId }));
    });
    ws.addEventListener('message', _onWsMessage);
    ws.addEventListener('close', _onWsClose);

    audioWs = new WebSocket(`${proto}//${location.host}${audioWsPath}?user_id=${userId}`);

    startBtn.style.display = 'none';
    endBtn.style.display = '';
    controls.style.display = '';
    setState('opening_playing'); // waits for converse:opening
  }

  async function endSession() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'converse:end' }));
    }
    _cleanup();
  }

  function startTurn() {
    if (!audioWs || audioWs.readyState !== WebSocket.OPEN) return;
    if (vad && autoDetectEnabled) vad.arm();
    setState('speaking');
    speakBtn.style.display = 'none';
    stopBtn.style.display = '';
    _startAudioStream();
  }

  function stopTurn() {
    _stopAudioStream();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'converse:turn_done' }));
    }
    if (vad) vad.disarm();
    setState('processing');
    speakBtn.style.display = '';
    stopBtn.style.display = 'none';
    setStatus('Listening...');
  }

  // ── WebSocket message handler ─────────────────────────────────────
  function _onWsMessage(e) {
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }

    if (msg.type === 'converse:opening') {
      _addBubble('assistant', msg.text);
      if (msg.audio_url) _playAudio(msg.audio_url, () => _armForTurn());
      else _armForTurn();

    } else if (msg.type === 'converse:user_heard') {
      setStatus('');
      _addBubble('user', msg.transcript, msg.turn_id, analysisFields);

    } else if (msg.type === 'converse:response') {
      _addBubble('assistant', msg.text);
      if (msg.audio_url) _playAudio(msg.audio_url, () => _armForTurn());
      else _armForTurn();

    } else if (msg.type === 'converse:analysis') {
      _updateAnalysisBadges(msg.turn_id, msg.results);

    } else if (msg.type === 'converse:ended') {
      _addSystemMsg(`Session ended — ${msg.turns} turn${msg.turns !== 1 ? 's' : ''}`);
      _cleanup();

    } else if (msg.type === 'converse:error') {
      setStatus(`Error: ${msg.message}`);
      _armForTurn();
    }
  }

  function _onWsClose() {
    if (state !== 'idle') _cleanup();
  }

  function _armForTurn() {
    setState('armed');
    setStatus('');
    if (autoDetectEnabled && vad) {
      vad.arm();
    }
    _updateManualControls();
  }

  // ── Audio stream management ───────────────────────────────────────
  async function _openMic() {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    audioCtx = new AudioContext({ sampleRate: 16000 });
    await audioCtx.audioWorklet.addModule('/static/core/audio_worklet.js');
    const source = audioCtx.createMediaStreamSource(mediaStream);
    analyserNode = audioCtx.createAnalyser();
    analyserNode.fftSize = 2048;
    workletNode = new AudioWorkletNode(audioCtx, 'pcm-processor');
    source.connect(analyserNode);
    analyserNode.connect(workletNode);

    // VAD
    vad = new VoiceActivityDetector(analyserNode, { silenceMs, onsetMs });
    vad.addEventListener('silencedetected', () => {
      if (state === 'speaking') stopTurn();
    });
    vad.addEventListener('silenceprogress', (e) => {
      silenceBarWrap.style.display = '';
      silenceBar.style.width = e.detail.pct + '%';
    });
    vad.addEventListener('voicedetected', () => {
      silenceBarWrap.style.display = 'none';
      silenceBar.style.width = '0';
    });
  }

  function _startAudioStream() {
    if (!workletNode) return;
    workletNode.port.onmessage = (e) => {
      if (audioWs && audioWs.readyState === WebSocket.OPEN && state === 'speaking') {
        audioWs.send(e.data);
      }
    };
  }

  function _stopAudioStream() {
    if (workletNode) workletNode.port.onmessage = null;
  }

  function _cleanup() {
    if (ws) { ws.close(); ws = null; }
    if (audioWs) { audioWs.close(); audioWs = null; }
    if (vad) { vad.destroy(); vad = null; }
    if (audioCtx) { audioCtx.close(); audioCtx = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
    workletNode = null;
    analyserNode = null;
    setState('idle');
    startBtn.style.display = '';
    endBtn.style.display = 'none';
    controls.style.display = 'none';
    silenceBarWrap.style.display = 'none';
    setStatus('');
  }

  // ── Audio playback ───────────────────────────────────────────────
  function _playAudio(url, onEnded) {
    setState('playing');
    if (vad) vad.disarm();
    const audio = new Audio(url);
    audio.playbackRate = audioSpeed;
    audio.addEventListener('ended', () => { if (onEnded) onEnded(); });
    audio.addEventListener('error', () => { if (onEnded) onEnded(); });
    audio.play().catch(() => { if (onEnded) onEnded(); });
  }

  // ── Chat bubbles ─────────────────────────────────────────────────
  function _addBubble(role, text, turnId = null, fields = []) {
    const div = document.createElement('div');
    div.className = `vc-bubble vc-bubble-${role}`;
    div.innerHTML = `<div class="vc-bubble-text">${_esc(text)}</div>`;

    if (role === 'user' && fields.length > 0 && turnId) {
      const badges = document.createElement('div');
      badges.className = 'vc-analysis-badges';
      badges.dataset.turnId = turnId;
      for (const f of fields) {
        const badge = document.createElement('span');
        badge.className = 'vc-badge vc-badge-pending';
        badge.dataset.key = f.key;
        badge.innerHTML = `<span class="vc-badge-label">${_esc(f.label)}</span> <span class="vc-badge-value">--</span>`;
        badges.appendChild(badge);
      }
      div.appendChild(badges);
    }

    chatLog.appendChild(div);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  function _addSystemMsg(text) {
    const div = document.createElement('div');
    div.className = 'vc-system-msg dim';
    div.textContent = text;
    chatLog.appendChild(div);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  function _updateAnalysisBadges(turnId, results) {
    const badgeRow = chatLog.querySelector(`.vc-analysis-badges[data-turn-id="${turnId}"]`);
    if (!badgeRow) return;
    for (const f of analysisFields) {
      const badge = badgeRow.querySelector(`[data-key="${f.key}"]`);
      if (!badge) continue;
      const val = results[f.key];
      if (val == null) continue;
      const valueEl = badge.querySelector('.vc-badge-value');
      if (f.format === 'score_100') {
        const n = Math.round(val);
        valueEl.textContent = n;
        badge.classList.remove('vc-badge-pending');
        badge.classList.add(n >= 70 ? 'vc-badge-good' : n >= 40 ? 'vc-badge-mid' : 'vc-badge-low');
      } else {
        valueEl.textContent = val;
        badge.classList.remove('vc-badge-pending');
      }
    }
  }

  // ── Helpers ──────────────────────────────────────────────────────
  function setState(s) { state = s; }
  function setStatus(t) { statusEl.textContent = t; }

  function _updateManualControls() {
    const showManual = !autoDetectEnabled && (state === 'armed' || state === 'speaking');
    speakBtn.style.display = (showManual && state === 'armed') ? '' : 'none';
    stopBtn.style.display = (showManual && state === 'speaking') ? '' : 'none';
  }

  function _esc(s) {
    return String(s).replace(/[&<>"']/g, c =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }
}
```

- [ ] **Step 2: Add converse.css to voice-core/frontend/theme.css (append)**

Add to end of `voice-core/frontend/theme.css`:
```css
/* ── Converse Component ────────────────────────────────────── */
.vc-converse { display: flex; flex-direction: column; gap: 10px; }
.vc-converse-header { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.vc-autodetect-toggle { display: flex; align-items: center; gap: 6px; font-size: 0.9rem;
  color: var(--text-dim); cursor: pointer; margin-left: auto; }
.vc-autodetect-toggle input { accent-color: var(--accent); }
.vc-silence-bar-wrap { height: 3px; background: var(--border); border-radius: 2px; }
.vc-silence-bar { height: 100%; background: var(--accent); border-radius: 2px;
  transition: width 0.08s linear; }
.vc-chat-log { display: flex; flex-direction: column; gap: 8px;
  max-height: 400px; overflow-y: auto; padding: 4px 0; }
.vc-bubble { max-width: 80%; padding: 10px 14px; border-radius: 12px;
  font-size: 0.95rem; line-height: 1.5; }
.vc-bubble-assistant { background: var(--surface-2); align-self: flex-start; }
.vc-bubble-user { background: rgba(232,125,62,0.12); border: 1px solid rgba(232,125,62,0.3);
  align-self: flex-end; }
.vc-analysis-badges { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 6px; }
.vc-badge { font-size: 0.8rem; padding: 2px 8px; border-radius: 10px;
  border: 1px solid var(--border); display: flex; gap: 4px; }
.vc-badge-pending { color: var(--text-dim); }
.vc-badge-good { border-color: var(--fem); color: var(--fem); }
.vc-badge-mid { border-color: #f0a500; color: #f0a500; }
.vc-badge-low { border-color: var(--masc); color: var(--masc); }
.vc-system-msg { font-size: 0.85rem; text-align: center; padding: 4px 0; }
.vc-converse-controls { display: flex; align-items: center; gap: 10px; }
.vc-converse-status { font-size: 0.9rem; }
```

- [ ] **Step 3: Commit**

```bash
cd /home/ak/Projects/VoiceCoaches/voice-core
git add frontend/converse.js frontend/theme.css
git commit -m "feat: add shared converse.js component to voice-core frontend"
```

---

### Task 5: Femme Backend — ConversationEngine wiring

**Files:**
- Modify: `femme-voice-coach/pyproject.toml` — add faster-whisper, httpx deps
- Create: `femme-voice-coach/src/femme_coach/server/routes_converse.py`
- Modify: `femme-voice-coach/src/femme_coach/server/app.py` — register router + WS
- Modify: `femme-voice-coach/src/femme_coach/server/routes_pages.py` — add /converse page route

- [ ] **Step 1: Add dependencies to pyproject.toml**

In `femme-voice-coach/pyproject.toml`, add to `dependencies`:
```toml
"faster-whisper",
"httpx",
```

Then sync:
```bash
cd /home/ak/Projects/VoiceCoaches/femme-voice-coach
uv sync
```

Expected: packages install without error.

- [ ] **Step 2: Create `routes_converse.py`**

```python
# femme-voice-coach/src/femme_coach/server/routes_converse.py
"""Converse feature: WebSocket handler + topic API for femme-voice-coach."""
from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from voice_core.converse import ConversationEngine

router = APIRouter()

# ── LLM / TTS config ─────────────────────────────────────────────────────────
# Reads from env vars with sensible defaults for local Ollama
import os

_LLM_CONFIG = {
    "url": os.getenv("CONVERSE_LLM_URL", "http://localhost:11434/v1"),
    "model": os.getenv("CONVERSE_LLM_MODEL", "qwen2.5:7b"),
    "api_key": os.getenv("CONVERSE_LLM_API_KEY", "ollama"),
}

_SYSTEM_PROMPT = """You are a warm, friendly conversation partner having a casual chat.
Keep responses concise (2-3 sentences max). Ask follow-up questions to keep the
conversation flowing. You are NOT a coach — do not comment on how the user speaks.
Just have a natural conversation."""

_TOPICS = [
    {"id": "daily", "label": "Daily Life", "description": "Everyday conversations"},
    {"id": "work", "label": "Work & Career", "description": "Professional topics"},
    {"id": "social", "label": "Social Situations", "description": "Meeting people, events"},
    {"id": "hobbies", "label": "Hobbies", "description": "Interests and pastimes"},
    {"id": "relationships", "label": "Relationships", "description": "Friends, family, connections"},
]


# ── Lazy-loaded models ────────────────────────────────────────────────────────
_whisper_model = None

def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        print("[converse] Whisper model loaded")
    return _whisper_model


# ── API routes ────────────────────────────────────────────────────────────────
@router.get("/api/converse/topics")
async def converse_topics():
    return _TOPICS


@router.get("/api/converse/audio/{filename}")
async def converse_audio(filename: str):
    from fastapi.responses import FileResponse
    from femme_coach.server.config import DATA_DIR
    path = DATA_DIR / "converse_audio" / filename
    if not path.exists():
        from fastapi import HTTPException
        raise HTTPException(404)
    return FileResponse(str(path), media_type="audio/mpeg")


# ── Analysis function ─────────────────────────────────────────────────────────
async def _femme_analysis(audio_path: Path) -> dict:
    """Run femme scoring on a converse turn recording. Returns gender/resonance/weight."""
    try:
        import voice_core.analyze as vc
        from femme_coach.scoring.score import score

        analysis = vc.analyze(str(audio_path))
        result = score(analysis, exercise_type="integration")
        scores = result.get("scores", {})
        return {
            "gender_score": scores.get("full", 0),
            "resonance": scores.get("resonance", 0),
            "vocal_weight": scores.get("vocal_weight", 0),
        }
    except Exception as e:
        print(f"[converse] femme analysis error: {e}")
        return {}


# ── TTS function ──────────────────────────────────────────────────────────────
async def _femme_tts(text: str) -> Path:
    """Generate TTS for AI response. Returns path to MP3 file."""
    try:
        from femme_coach.server.config import DATA_DIR
        import uuid

        audio_dir = DATA_DIR / "converse_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        out_path = audio_dir / f"{uuid.uuid4()}.mp3"

        # Use existing Kokoro TTS infrastructure
        from femme_coach.server.routes_tts import _generate_kokoro
        await _generate_kokoro(text, out_path)
        return out_path
    except Exception as e:
        print(f"[converse] TTS error: {e}")
        return None


# ── Per-connection state ──────────────────────────────────────────────────────
class _ConverseSession:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.engine: ConversationEngine | None = None
        self.started_at: float | None = None
        self.pcm_buf: bytearray = bytearray()
        self.sample_rate: int = 16000

    def make_engine(self) -> ConversationEngine:
        def on_analysis_ready(turn_id: str, results: dict):
            # Guard: don't send if WS is closed
            try:
                asyncio.create_task(
                    self.ws.send_json({"type": "converse:analysis", "turn_id": turn_id, "results": results})
                )
            except Exception:
                pass

        return ConversationEngine(
            system_prompt=_SYSTEM_PROMPT,
            topics=_TOPICS,
            llm_config=_LLM_CONFIG,
            analysis_fn=_femme_analysis,
            tts_fn=_femme_tts,
            analysis_ready_fn=on_analysis_ready,
        )


# ── WebSocket endpoint ────────────────────────────────────────────────────────
@router.websocket("/ws/converse")
async def ws_converse(websocket: WebSocket, user_id: str = Query(None)):
    await websocket.accept()
    session = _ConverseSession(websocket)

    try:
        while True:
            msg = await websocket.receive()

            if "text" in msg:
                data = json.loads(msg["text"])
                msg_type = data.get("type", "")

                if msg_type == "converse:start":
                    topic_id = data.get("topic_id", "daily")
                    session.engine = session.make_engine()
                    session.started_at = time.time()
                    result = await session.engine.start(topic_id)
                    payload = {"type": "converse:opening", "text": result["opening_text"],
                               "turn_id": result["turn_id"]}
                    if "audio_url" in result and result["audio_url"]:
                        payload["audio_url"] = "/api/converse/audio/" + Path(result["audio_url"]).name
                    await websocket.send_json(payload)

                elif msg_type == "converse:turn_done":
                    if not session.engine or not session.pcm_buf:
                        await websocket.send_json({"type": "converse:error", "message": "No audio received"})
                        continue

                    # Save buffered PCM to WAV
                    wav_path = await _save_pcm_to_wav(bytes(session.pcm_buf), session.sample_rate)
                    session.pcm_buf = bytearray()

                    # Check duration
                    duration_s = len(bytes(session.pcm_buf)) / (session.sample_rate * 2)
                    if duration_s < 1.0:
                        await websocket.send_json({"type": "converse:error", "message": "Too short, try again"})
                        continue

                    # Transcribe
                    transcript = await asyncio.get_event_loop().run_in_executor(
                        None, _transcribe, wav_path
                    )
                    if not transcript.strip():
                        await websocket.send_json({"type": "converse:error", "message": "Nothing heard, try again"})
                        continue

                    await websocket.send_json({"type": "converse:user_heard",
                                               "transcript": transcript, "turn_id": str(id(wav_path))})

                    result = await session.engine.process_turn(transcript, wav_path)
                    payload = {"type": "converse:response", "text": result["response_text"],
                               "turn_id": result["turn_id"]}
                    if result.get("audio_url"):
                        payload["audio_url"] = "/api/converse/audio/" + Path(result["audio_url"]).name
                    await websocket.send_json(payload)

                elif msg_type == "converse:end":
                    turns = 0
                    if session.engine:
                        result = await session.engine.end()
                        turns = result["turns"]
                    duration_s = round(time.time() - session.started_at, 1) if session.started_at else 0
                    await websocket.send_json({"type": "converse:ended", "turns": turns,
                                               "duration_s": duration_s})
                    break

            elif "bytes" in msg:
                # PCM audio from AudioWorklet (Int16, 128 samples = 256 bytes per chunk)
                session.pcm_buf.extend(msg["bytes"])

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws/converse] error: {e}")


@router.websocket("/ws/converse/audio")
async def ws_converse_audio(websocket: WebSocket, user_id: str = Query(None)):
    """Dedicated binary audio upload socket for converse (mirrors /ws/live/audio pattern)."""
    # Audio is sent here, but the converse session state is on /ws/converse.
    # This endpoint is a pass-through — the main /ws/converse handles bytes too.
    # Provided for initConverse() audioWsPath compatibility; redirect bytes to /ws/converse.
    # In practice, initConverse sends audio directly on the control WS in the femme setup.
    await websocket.accept()
    try:
        while True:
            await websocket.receive_bytes()  # discard — audio goes via /ws/converse
    except WebSocketDisconnect:
        pass


# ── Helpers ───────────────────────────────────────────────────────────────────
async def _save_pcm_to_wav(pcm_bytes: bytes, sample_rate: int) -> Path:
    """Save raw Int16 PCM bytes to a temporary WAV file."""
    import wave, struct
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # Int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return Path(tmp.name)


def _transcribe(wav_path: Path) -> str:
    model = _get_whisper()
    segments, _ = model.transcribe(str(wav_path), beam_size=3)
    return " ".join(s.text.strip() for s in segments).strip()
```

- [ ] **Step 3: Register router and page route**

In `femme-voice-coach/src/femme_coach/server/app.py`, add:
```python
from femme_coach.server import routes_converse
# ... after other includes:
app.include_router(routes_converse.router)
```

In `femme-voice-coach/src/femme_coach/server/routes_pages.py`, add:
```python
@router.get("/converse", response_class=HTMLResponse)
async def converse_page():
    return (STATIC_DIR / "converse.html").read_text()
```

- [ ] **Step 4: Write a basic backend test**

Create `femme-voice-coach/tests/test_converse_route.py`:
```python
"""Smoke tests for femme converse backend."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "voice-core" / "src"))


class TestFemmeAnalysisFn:
    """Test _femme_analysis returns expected keys even on bad input."""
    @pytest.mark.asyncio
    async def test_returns_dict_on_bad_audio(self, tmp_path):
        from femme_coach.server.routes_converse import _femme_analysis
        bad = tmp_path / "bad.wav"
        bad.write_bytes(b"not a wav")
        result = await _femme_analysis(bad)
        assert isinstance(result, dict)
        # On failure, returns empty dict — no exception raised
        # On success, has these keys:
        for key in ["gender_score", "resonance", "vocal_weight"]:
            assert key in result or result == {}


class TestTopicsEndpoint:
    def test_topics_not_empty(self):
        from femme_coach.server.routes_converse import _TOPICS
        assert len(_TOPICS) > 0
        for t in _TOPICS:
            assert "id" in t
            assert "label" in t
```

Run:
```bash
cd /home/ak/Projects/VoiceCoaches/femme-voice-coach
uv run pytest tests/test_converse_route.py -v
```

Expected: PASS (analysis returns empty dict on bad wav, topics list is non-empty).

- [ ] **Step 5: Commit backend**

```bash
cd /home/ak/Projects/VoiceCoaches/femme-voice-coach
git add src/femme_coach/server/routes_converse.py \
        src/femme_coach/server/app.py \
        src/femme_coach/server/routes_pages.py \
        tests/test_converse_route.py \
        pyproject.toml
git commit -m "feat: add femme converse backend (ConversationEngine + WS handler)"
```

---

### Task 6: Femme Frontend — converse.html

**Files:**
- Create: `femme-voice-coach/frontend/converse.html`

- [ ] **Step 1: Create converse.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Femme Voice Coach — Converse</title>
<link rel="stylesheet" href="/static/core/theme.css">
<link rel="stylesheet" href="/static/app.css">
<style>
  #converse-mount { max-width: 640px; margin: 0 auto; padding: 16px; }
</style>
</head>
<body>

<script type="module">
import { initNav } from '/static/core/nav.js';
initNav({
  appName: 'Femme Voice Coach',
  links: [
    { label: 'Practice', href: '/practice' },
    { label: 'Converse', href: '/converse' },
    { label: 'Review',   href: '/review' },
  ],
});
</script>

<div id="converse-mount"></div>

<script type="module">
import { initConverse } from '/static/core/converse.js';

const topics = await fetch('/api/converse/topics').then(r => r.json()).catch(() => []);

initConverse({
  mountEl: document.getElementById('converse-mount'),
  topics,
  analysisFields: [
    { key: 'gender_score', label: 'Gender',    format: 'score_100' },
    { key: 'resonance',    label: 'Resonance', format: 'score_100' },
    { key: 'vocal_weight', label: 'Weight',    format: 'score_100' },
  ],
  wsPath: '/ws/converse',
  audioWsPath: '/ws/converse',  // femme sends audio on the same socket as control
  audioSpeed: 0.85,
  autoDetect: true,
  silenceMs: 2000,
  onsetMs: 150,
  userIdFn: () => localStorage.getItem('activeUserId'),
});
</script>
</body>
</html>
```

- [ ] **Step 2: Update practice.html nav to add Converse link**

In `femme-voice-coach/frontend/practice.html`, find the `initNav` call and add Converse:
```javascript
initNav({
  appName: 'Femme Voice Coach',
  links: [
    { label: 'Practice', href: '/practice' },
    { label: 'Converse', href: '/converse' },
    { label: 'Review',   href: '/review' },
  ],
});
```

Do the same in `review.html`.

- [ ] **Step 3: Restart server and smoke-test**

```bash
pkill -f "python -m femme_coach" 2>/dev/null; sleep 1
cd /home/ak/Projects/VoiceCoaches/femme-voice-coach
nohup uv run python -m femme_coach > /tmp/femme-server.log 2>&1 &
sleep 3
curl -sk https://akpersistant:9000/api/converse/topics | python3 -c "import sys,json; print(json.load(sys.stdin))"
```

Expected: list of 5 topics printed. No errors in `/tmp/femme-server.log`.

Navigate to `https://akpersistant:9000/converse` in browser. Expected: Converse page loads, topic dropdown shows 5 topics, Auto-detect toggle visible.

- [ ] **Step 4: End-to-end test**

1. Open `https://akpersistant:9000/converse`
2. Select a topic, click Start
3. Expected: "Calibrating mic..." status, then AI opening message appears
4. Speak for 3+ seconds, then go silent for 2s
5. Expected: turn auto-submits, transcript appears under your bubble with `--` badges
6. Expected: AI responds with text (and audio if TTS configured)
7. Expected: after ~5s, `--` badges update with gender/resonance/weight scores
8. Click "End" — expected: "Session ended — N turns" system message

- [ ] **Step 5: Commit**

```bash
cd /home/ak/Projects/VoiceCoaches/femme-voice-coach
git add frontend/converse.html frontend/practice.html frontend/review.html
git commit -m "feat: add femme converse page and update nav"
```

---

### Task 7: Push both repos

- [ ] **Step 1: Push voice-core**

```bash
cd /home/ak/Projects/VoiceCoaches/voice-core
git checkout main
git merge feat/spanish-stress
GH_TOKEN=REDACTED_PAT git push
```

- [ ] **Step 2: Push femme-voice-coach**

```bash
cd /home/ak/Projects/VoiceCoaches/femme-voice-coach
GH_TOKEN=REDACTED_PAT git push
```

---

## Notes for Phase 3 (Spanish Migration — separate plan)

Write a separate plan after Phase 2 is validated in production. Key items:
- Port `httpx.Client` → `httpx.AsyncClient` in `spanish_coach/conversation.py`
- Replace Spanish `ConversationEngine` with `voice_core.converse.ConversationEngine`
- Migrate `spanish/frontend/converse.js` → `initConverse()` + `converse-ext.js` for word badges/dictionary
- Rename `converse:pronunciation` → `converse:analysis` in Spanish WS handler + frontend
- Full regression test of existing Spanish press-to-talk flow before enabling auto-detect
