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
 *     audioWsPath: '/ws/converse',               // PCM upload path (can be same as wsPath)
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
  let currentAudio = null;
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
    try {
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
    } catch (err) {
      console.error('[converse] startSession failed:', err);
      setStatus('Mic error — check permissions');
      _cleanup();
    }
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
    if (currentAudio) { currentAudio.pause(); currentAudio = null; }
    if (ws) { ws.close(); ws = null; }
    if (audioWs) { audioWs.close(); audioWs = null; }
    if (vad) { vad.destroy(); vad = null; }
    if (audioCtx) { audioCtx.close(); audioCtx = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
    _stopAudioStream();
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
    currentAudio = new Audio(url);
    currentAudio.playbackRate = audioSpeed;
    currentAudio.addEventListener('ended', () => { currentAudio = null; if (onEnded) onEnded(); });
    currentAudio.addEventListener('error', () => { currentAudio = null; if (onEnded) onEnded(); });
    currentAudio.play().catch(() => { currentAudio = null; if (onEnded) onEnded(); });
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
