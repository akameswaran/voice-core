/**
 * RecordingControl — Unified mic/camera/recording control component.
 *
 * State machine:
 *   IDLE ──checkLevel()──► MIC_OPEN ──start()──► RECORDING ──stop()──► PROCESSING ──(done)──► IDLE
 *     │                       │                                                                 ▲
 *     └───────start()─────────┼─────────────────────────────────────────────────────────────────┘
 *                             └──stopCheck()──► IDLE
 *
 * WebSocket channels (3 separate connections):
 *   Metrics:  ws(s)://host{wsBasePath}?user_id={id}         — server→client JSON frames
 *   Audio:    ws(s)://host{wsBasePath}/audio?user_id={id}    — client→server binary PCM
 *   Video:    ws(s)://host{wsBasePath}/video?user_id={id}    — client→server JSON features
 *
 * Audio capture:
 *   Primary:   AudioWorklet (Int16 PCM via PCMProcessor)
 *   Fallback:  ScriptProcessorNode (Float32 PCM) for older browsers
 *
 * Events (dispatched on this via EventTarget):
 *   statechange  {state, previous}
 *   frame        {...metrics from server}
 *   levelupdate  {rmsDb, peakDb}
 *   camerachange {active, videoElement}
 *   calibration  {extractor, onDone}
 *   error        {message, phase}
 *   saved        {filename}
 */

import { wsUrl, apiFetch } from './user_api.js';

// ─── Constants ──────────────────────────────────────────────────────────────

const LEVEL_POLL_MS = 80;
const LEVEL_ZONES = {
    tooQuiet: -40,
    quiet:    -30,
    good:     -6,
    // above -6 = too hot
};

const VAD_NOISE_EMA_ALPHA = 0.05;   // slow noise floor adaptation
const VAD_NOISE_MARGIN_DB = 10;     // speech must be this far above noise floor
const VAD_NOISE_FLOOR_MIN = -60;    // don't let noise estimate drift below this
const VAD_NOISE_FLOOR_MAX = -20;    // don't let noise estimate climb above this

const LS_PREFIX = 'vc_rc_';
const LS_KEYS = {
    source:          LS_PREFIX + 'source',
    serverDevice:    LS_PREFIX + 'server_device',
    browserDevice:   LS_PREFIX + 'browser_device',
    channel:         LS_PREFIX + 'channel',
    cameraCheckbox:  LS_PREFIX + 'camera_checkbox',
    vadThreshold:    LS_PREFIX + 'vad_threshold',
};

// ─── RecordingControl ───────────────────────────────────────────────────────

export class RecordingControl extends EventTarget {

    /**
     * @param {HTMLElement} containerEl — mount point for the control bar
     * @param {Object} opts
     * @param {'stream'|'record'} opts.mode — stream for live, record for capture-then-upload
     * @param {boolean} opts.showSourceSelect — server vs browser mic toggle
     * @param {boolean} opts.showDeviceSelect — browser mic picker
     * @param {boolean} opts.showChannelSelect — L/R channel for stereo interfaces
     * @param {boolean} opts.showCamera — camera toggle + calibrate button
     * @param {boolean} opts.showDeleteButton — delete last recording button
     * @param {function(): string} opts.activityTagFn — returns current activity label
     * @param {'server'|'browser'} opts.defaultSource
     * @param {number} opts.defaultChannel
     * @param {boolean} opts.cameraDefault — camera toggle initial state
     * @param {string} opts.wsBasePath — base path for WebSocket URLs
     */
    constructor(containerEl, opts = {}) {
        super();
        this._container = containerEl;
        this._opts = {
            mode: 'stream',
            showSourceSelect: false,
            showDeviceSelect: false,
            showChannelSelect: false,
            showCamera: true,
            showDeleteButton: false,
            activityTagFn: null,
            defaultSource: 'browser',
            defaultChannel: 1,
            cameraDefault: false,
            wsBasePath: '/ws/live',
            vadEnabled: false,
            vadSilenceMs: 1200,
            vadThresholdDb: -35,
            vadMinSpeechMs: 500,
            ...opts,
        };

        // Restore saved preferences from localStorage
        this._restorePrefs();

        this._state = 'idle';     // idle | mic_open | recording | processing
        this._currentFrame = null;

        // VAD state — restore calibrated threshold from localStorage
        const savedThreshold = localStorage.getItem(LS_KEYS.vadThreshold);
        if (savedThreshold && this._opts.vadEnabled) {
            this._opts.vadThresholdDb = parseFloat(savedThreshold);
            console.log(`[vc-rc:vad] restored calibrated threshold: ${this._opts.vadThresholdDb.toFixed(1)} dBFS`);
        }
        this._vadSpeechDetected = false;
        this._vadSilenceStart = null;
        this._vadSpeechStart = null;
        this._vadSpeechFrames = 0;

        // Adaptive noise floor — seeded from calibration or default
        this._vadNoiseFloor = this._opts.vadThresholdDb;  // start from static threshold

        // Mic manager state
        this._audioStream = null;  // MediaStream
        this._audioCtx = null;     // AudioContext
        this._sourceNode = null;   // MediaStreamSourceNode
        this._analyser = null;     // AnalyserNode for level metering
        this._analyserBuf = null;  // Float32Array for analyser data
        this._audioProcessor = null; // AudioWorkletNode or ScriptProcessorNode
        this._useWorklet = false;  // true if AudioWorklet was loaded successfully

        // WebSocket connections (3-socket pattern)
        this._metricsWs = null;    // server→client JSON frames
        this._audioWs = null;      // client→server binary PCM
        this._videoWs = null;      // client→server video features

        this._isStarting = false;  // guard against concurrent start() calls

        // Camera manager state
        this._videoExtractor = null;
        this._cameraActive = false;
        this._calibrated = false;

        // Video recording (debug: save raw WebM alongside WAV)
        this._mediaRecorder = null;
        this._videoChunks = [];

        // Level meter state
        this._levelInterval = null;
        this._peakDb = -Infinity;
        this._noiseFloorSamples = [];

        // DOM refs (filled by _render)
        this._els = {};

        // Track last saved recording filename (for delete button)
        this._lastSavedFilename = null;

        // Audio debug counter
        this._audioDbgCount = 0;

        this._render();
        this._bindEvents();
        this._loadDevices();
        this._initVideo();
    }

    // ─── Public API ─────────────────────────────────────────────────────

    get state() { return this._state; }
    get currentFrame() { return this._currentFrame; }
    /** Whether this user has saved mic preferences (for first-run detection) */
    get hasSavedPrefs() {
        try { return !!(localStorage.getItem(LS_KEYS.source) || localStorage.getItem(LS_KEYS.browserDevice)); }
        catch { return false; }
    }

    get stream() { return this._audioStream; }

    /** IDLE → MIC_OPEN: open mic + level meter, no server session */
    async checkLevel() {
        if (this._state !== 'idle') return;
        try {
            await this._openMic();
            this._startLevelMeter();
            this._setState('mic_open');
        } catch (e) {
            this._emitError(e.message || String(e), 'checkLevel');
        }
    }

    /** MIC_OPEN → IDLE: close mic, stop level meter, calibrate VAD */
    stopCheck() {
        if (this._state !== 'mic_open') return;

        // Calibrate VAD threshold from noise floor samples
        if (this._opts.vadEnabled && this._noiseFloorSamples.length >= 10) {
            const sorted = [...this._noiseFloorSamples].sort((a, b) => a - b);
            // Use 25th percentile as noise floor (ignores speech peaks)
            const p25 = sorted[Math.floor(sorted.length * 0.25)];
            const margin = 12; // dB above noise floor
            const calibrated = Math.min(p25 + margin, -20);
            this._opts.vadThresholdDb = calibrated;
            this._vadNoiseFloor = calibrated;  // seed adaptive tracker
            try { localStorage.setItem(LS_KEYS.vadThreshold, calibrated.toFixed(1)); } catch (e) { /* ignore */ }
            console.log(`[vc-rc:vad] calibrated threshold: ${calibrated.toFixed(1)} dBFS (noise floor p25: ${p25.toFixed(1)} dBFS)`);
        }

        this._stopLevelMeter();
        this._closeMic();
        this._setState('idle');
    }

    /** IDLE or MIC_OPEN → RECORDING */
    async start() {
        if (this._isStarting) return;
        const prev = this._state;
        if (prev !== 'idle' && prev !== 'mic_open') return;
        this._isStarting = true;
        try {
            if (prev === 'idle') {
                const source = this._getSource();
                if (source === 'browser') {
                    await this._openMic();
                }
            }
            // mic is now open (either from checkLevel or just opened)
            await this._startServerSession();
            this._connectMetricsWs();
            this._stopLevelMeter(); // stop local meter, server frames take over
            this._lastSavedFilename = null; // clear for new session
            this._setState('recording');

            // Reset VAD state for new recording
            this._vadSpeechDetected = false;
            this._vadSilenceStart = null;
            this._vadSpeechStart = null;
            this._vadSpeechFrames = 0;
            // Seed adaptive noise floor from calibrated value (or default)
            this._vadNoiseFloor = this._opts.vadThresholdDb;

            // Start camera if enabled
            if (this._opts.showCamera && this._els.cameraToggle?.checked) {
                this._startCamera().then(() => {
                    this._startVideoRecording();
                }).catch(e =>
                    console.warn('[vc-rc] Camera start failed:', e));
            }
        } catch (e) {
            this._emitError(e.message || String(e), 'start');
            // Roll back to previous state
            if (prev === 'idle') {
                this._closeMic();
                this._setState('idle');
            }
        } finally {
            this._isStarting = false;
        }
    }

    /** RECORDING → PROCESSING → IDLE */
    async stop() {
        if (this._state !== 'recording') return;

        this._disconnectMetricsWs();
        this._stopAudioStreaming();
        this._closeMic();

        // Transition to processing
        this._setState('processing');

        // Auto-save audio recording
        try {
            const tag = this._opts.activityTagFn?.() || '';
            const params = new URLSearchParams();
            if (tag) params.set('activity', tag);
            const res = await apiFetch(`/api/live/save?${params}`, { method: 'POST' });
            const data = await res.json();
            if (data.recording) {
                this._lastSavedFilename = data.recording;
                this.dispatchEvent(new CustomEvent('saved', {
                    detail: { filename: data.recording },
                }));
            }
        } catch (e) {
            console.warn('[vc-rc] Auto-save failed:', e);
        }

        // Stop camera + save video
        await this._stopCamera();

        try {
            await apiFetch('/api/live/stop', { method: 'POST' });
        } catch (e) { /* ignore */ }

        this._currentFrame = null;
        this._setState('idle');
    }

    /** Cancel recording without saving */
    cancel() {
        this._disconnectMetricsWs();
        this._stopAudioStreaming();
        this._closeMic();
        this._stopCamera();
        this._currentFrame = null;
        this._setState('idle');
    }

    /** Delete the last auto-saved recording */
    async deleteLastRecording() {
        if (!this._lastSavedFilename) return;
        try {
            await fetch(`/api/recordings/${encodeURIComponent(this._lastSavedFilename)}`, {
                method: 'DELETE',
            });
            const deleted = this._lastSavedFilename;
            this._lastSavedFilename = null;
            this._updateUI();
            this.dispatchEvent(new CustomEvent('deleted', {
                detail: { recording: deleted },
            }));
        } catch (e) {
            this._emitError('Delete failed: ' + (e.message || e), 'delete');
        }
    }

    /** Preview camera + run calibration (works without recording) */
    async calibrateCamera() {
        if (!this._videoExtractor) {
            this._emitError('Camera not available — MediaPipe not loaded', 'calibrate');
            return;
        }
        try {
            if (!this._cameraActive) {
                await this._videoExtractor.startCamera();
                this._cameraActive = true;
                const videoEl = this._videoExtractor.getVideoElement();
                this.dispatchEvent(new CustomEvent('camerachange', {
                    detail: { active: true, videoElement: videoEl },
                }));
            }
            this._runCalibration();
        } catch (e) {
            this._emitError('Camera failed: ' + (e.message || e), 'calibrate');
        }
    }

    /** Full teardown */
    dispose() {
        this.cancel();
        this._stopLevelMeter();
        this._closeMic();
        this._stopCamera();
        this._disconnectMetricsWs();
        if (this._videoExtractor) {
            this._videoExtractor.stopCamera();
            this._videoExtractor = null;
        }
        this._container.innerHTML = '';
    }

    // ─── State Management ───────────────────────────────────────────────

    _setState(newState) {
        const prev = this._state;
        if (prev === newState) return;
        this._state = newState;
        this._updateUI();
        this.dispatchEvent(new CustomEvent('statechange', {
            detail: { state: newState, previous: prev },
        }));
    }

    // ─── Mic Manager ────────────────────────────────────────────────────

    async _openMic() {
        if (this._audioStream) return; // already open
        const source = this._getSource();
        if (source === 'server') return; // no local mic for server mode

        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error(
                'Browser mic requires HTTPS. Access via https://' +
                location.hostname + ':' + location.port
            );
        }

        const constraints = {
            autoGainControl: false,
            noiseSuppression: false,
            echoCancellation: false,
        };
        const deviceId = this._els.browserDeviceSelect?.value;
        if (deviceId) {
            constraints.deviceId = { exact: deviceId };
        }

        this._audioStream = await navigator.mediaDevices.getUserMedia({
            audio: constraints,
        });

        // Re-enumerate devices now that permission is granted
        this._loadBrowserDevices();

        this._audioCtx = new AudioContext();
        if (this._audioCtx.state === 'suspended') {
            await this._audioCtx.resume();
        }

        // Try AudioWorklet first, fall back to ScriptProcessor
        try {
            await this._audioCtx.audioWorklet.addModule(
                new URL('./audio_worklet.js', import.meta.url).href
            );
            this._useWorklet = true;
        } catch (e) {
            console.warn('[vc-rc] AudioWorklet unavailable, falling back to ScriptProcessor:', e);
            this._useWorklet = false;
        }

        this._sourceNode = this._audioCtx.createMediaStreamSource(this._audioStream);

        // AnalyserNode for level metering (always available when mic open)
        this._analyser = this._audioCtx.createAnalyser();
        this._analyser.fftSize = 2048;
        this._analyserBuf = new Float32Array(this._analyser.fftSize);
        this._sourceNode.connect(this._analyser);
    }

    _closeMic() {
        this._stopAudioStreaming();
        if (this._audioWs) {
            this._audioWs.onclose = null;
            this._audioWs.close();
            this._audioWs = null;
        }
        if (this._sourceNode) {
            this._sourceNode.disconnect();
            this._sourceNode = null;
        }
        if (this._analyser) {
            this._analyser = null;
            this._analyserBuf = null;
        }
        if (this._audioCtx) {
            this._audioCtx.close();
            this._audioCtx = null;
        }
        if (this._audioStream) {
            this._audioStream.getTracks().forEach(t => t.stop());
            this._audioStream = null;
        }
    }

    // ─── Audio Streaming (AudioWorklet or ScriptProcessor → WebSocket) ──

    _setupAudioStreaming() {
        if (!this._audioCtx || !this._sourceNode) return;

        if (this._useWorklet) {
            // AudioWorklet path: sends Int16 PCM
            this._audioProcessor = new AudioWorkletNode(this._audioCtx, 'pcm-processor');
            this._audioProcessor.port.onmessage = (e) => {
                if (this._audioWs && this._audioWs.readyState === WebSocket.OPEN) {
                    this._audioWs.send(e.data);
                }
                // VAD: check energy of this PCM chunk
                if (this._opts.vadEnabled && this._state === 'recording') {
                    this._checkVad(e.data);
                }
            };
        } else {
            // ScriptProcessor fallback: sends Float32 PCM
            const nChannels = this._sourceNode.channelCount || 2;
            this._audioProcessor = this._audioCtx.createScriptProcessor(
                2048, nChannels, 1
            );
            const useChannel = parseInt(this._els.channelSelect?.value) || 0;

            this._audioProcessor.onaudioprocess = (e) => {
                if (!this._audioWs || this._audioWs.readyState !== WebSocket.OPEN) return;
                const numCh = e.inputBuffer.numberOfChannels;
                const ch = Math.min(useChannel, numCh - 1);

                if (this._audioDbgCount++ % 100 === 0) {
                    const peaks = [];
                    for (let c = 0; c < numCh; c++) {
                        const buf = e.inputBuffer.getChannelData(c);
                        let peak = 0;
                        for (let i = 0; i < buf.length; i++) {
                            const a = Math.abs(buf[i]);
                            if (a > peak) peak = a;
                        }
                        peaks.push(peak);
                    }
                    const info = peaks.map((p, i) =>
                        `ch${i}: ${p.toFixed(4)} (${p > 0 ? (20*Math.log10(p)).toFixed(1) : '-inf'} dBFS)`
                    ).join(', ');
                    console.log(`[vc-rc:audio] ${info}, using ch${ch}, sr: ${this._audioCtx.sampleRate}`);
                }

                const float32 = e.inputBuffer.getChannelData(ch);
                // Convert Float32 → Int16 to match AudioWorklet format
                const int16 = new Int16Array(float32.length);
                for (let i = 0; i < float32.length; i++) {
                    int16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32768));
                }
                this._audioWs.send(int16.buffer);
                if (this._opts.vadEnabled && this._state === 'recording') {
                    this._checkVad(int16.buffer);
                }
            };
        }

        this._sourceNode.connect(this._audioProcessor);
        this._audioProcessor.connect(this._audioCtx.destination);
    }

    _stopAudioStreaming() {
        if (this._audioProcessor) {
            this._audioProcessor.disconnect();
            this._audioProcessor = null;
        }
    }

    _checkVad(pcmBuffer) {
        const samples = new Int16Array(pcmBuffer);
        let sumSq = 0;
        for (let i = 0; i < samples.length; i++) {
            const s = samples[i] / 32768;
            sumSq += s * s;
        }
        const rms = Math.sqrt(sumSq / samples.length);
        const db = rms > 0 ? 20 * Math.log10(rms) : -100;
        const now = performance.now();

        // Adaptive threshold: noise floor + margin
        const speechThreshold = this._vadNoiseFloor + VAD_NOISE_MARGIN_DB;

        if (db > speechThreshold) {
            // Speech detected — count consecutive speech frames
            this._vadSpeechFrames = (this._vadSpeechFrames || 0) + 1;
            if (this._vadSpeechFrames >= 3 && !this._vadSpeechDetected) {
                this._vadSpeechStart = now;
                this._vadSpeechDetected = true;
                console.log(`[vc-rc:vad] speech onset (floor=${this._vadNoiseFloor.toFixed(1)}, thresh=${speechThreshold.toFixed(1)}, db=${db.toFixed(1)})`);
            }
            this._vadSilenceStart = null;
        } else {
            // Below speech threshold — update noise floor estimate
            const clamped = Math.max(VAD_NOISE_FLOOR_MIN, Math.min(VAD_NOISE_FLOOR_MAX, db));
            this._vadNoiseFloor = VAD_NOISE_EMA_ALPHA * clamped + (1 - VAD_NOISE_EMA_ALPHA) * this._vadNoiseFloor;

            if (this._vadSpeechDetected) {
                this._vadSpeechFrames = 0;
                if (!this._vadSilenceStart) {
                    this._vadSilenceStart = now;
                }
                const speechDuration = now - (this._vadSpeechStart || now);
                const silenceDuration = now - this._vadSilenceStart;
                if (speechDuration >= this._opts.vadMinSpeechMs &&
                    silenceDuration >= this._opts.vadSilenceMs) {
                    console.log(`[vc-rc:vad] auto-stop after ${Math.round(silenceDuration)}ms silence (floor=${this._vadNoiseFloor.toFixed(1)})`);
                    this.stop();
                }
            }
        }
    }

    // ─── Level Meter ────────────────────────────────────────────────────

    _startLevelMeter() {
        this._peakDb = -Infinity;
        this._noiseFloorSamples = [];
        this._showLevelMeter(true);
        if (this._levelInterval) clearInterval(this._levelInterval);
        this._levelInterval = setInterval(() => this._pollLevel(), LEVEL_POLL_MS);
    }

    _stopLevelMeter() {
        if (this._levelInterval) {
            clearInterval(this._levelInterval);
            this._levelInterval = null;
        }
        this._showLevelMeter(false);
    }

    _pollLevel() {
        let rmsDb;

        if (this._analyser && this._analyserBuf) {
            // Local AnalyserNode metering
            this._analyser.getFloatTimeDomainData(this._analyserBuf);
            let sum = 0;
            for (let i = 0; i < this._analyserBuf.length; i++) {
                sum += this._analyserBuf[i] * this._analyserBuf[i];
            }
            const rms = Math.sqrt(sum / this._analyserBuf.length);
            rmsDb = 20 * Math.log10(Math.max(rms, 1e-10));
        } else if (this._currentFrame) {
            // Fallback: use server-reported RMS from frames
            rmsDb = this._currentFrame.rms_db;
        } else {
            return;
        }

        if (rmsDb > this._peakDb) this._peakDb = rmsDb;

        this._noiseFloorSamples.push(rmsDb);
        if (this._noiseFloorSamples.length > 250) this._noiseFloorSamples.shift();

        this._renderLevelBar(rmsDb);
        this.dispatchEvent(new CustomEvent('levelupdate', {
            detail: { rmsDb, peakDb: this._peakDb },
        }));
    }

    // ─── Stream Manager (Server Session + WebSockets) ────────────────────

    async _startServerSession() {
        const source = this._getSource();

        if (source === 'browser') {
            const sr = this._audioCtx.sampleRate;
            const res = await apiFetch(`/api/live/start-browser?sr=${sr}`, {
                method: 'POST',
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);

            // Open audio WebSocket for sending PCM
            this._audioWs = new WebSocket(wsUrl(this._opts.wsBasePath + '/audio'));
            this._audioWs.binaryType = 'arraybuffer';

            await new Promise((resolve, reject) => {
                this._audioWs.onopen = resolve;
                this._audioWs.onerror = reject;
            });

            // Set up audio streaming (AudioWorklet or ScriptProcessor)
            this._setupAudioStreaming();
        } else {
            // Server mic
            const deviceId = this._els.deviceSelect?.value;
            const params = new URLSearchParams();
            if (deviceId) params.set('device', deviceId);
            const res = await apiFetch('/api/live/start?' + params, {
                method: 'POST',
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            if (data.status !== 'started' && data.status !== 'already_running') {
                throw new Error('Unexpected status: ' + data.status);
            }
        }
    }

    _connectMetricsWs() {
        if (this._metricsWs) return;
        this._metricsWs = new WebSocket(wsUrl(this._opts.wsBasePath));

        this._metricsWs.onmessage = (e) => {
            try {
                const frame = JSON.parse(e.data);
                if (frame.status === 'stopped') {
                    // Only honour "stopped" from server when we're actually recording.
                    // Guards against stale frames arriving during session startup.
                    if (this._state === 'recording') {
                        this.stop();
                    }
                    return;
                }
                this._currentFrame = frame;
                this.dispatchEvent(new CustomEvent('frame', { detail: frame }));
            } catch (err) {
                console.warn('[vc-rc] Bad WS message:', err);
            }
        };

        this._metricsWs.onclose = () => {
            this._metricsWs = null;
            if (this._state === 'recording') {
                // Reconnect after brief delay
                setTimeout(() => this._connectMetricsWs(), 1000);
            }
        };

        this._metricsWs.onerror = () => {
            this._metricsWs?.close();
        };
    }

    _disconnectMetricsWs() {
        if (this._metricsWs) {
            this._metricsWs.onclose = null;
            this._metricsWs.onmessage = null;
            this._metricsWs.close();
            this._metricsWs = null;
        }
    }

    // ─── Camera Manager ─────────────────────────────────────────────────

    async _initVideo() {
        try {
            const { VideoFeatureExtractor } = await import('./video_features.js');
            this._videoExtractor = new VideoFeatureExtractor();
            await this._videoExtractor.init();
            this._calibrated = this._videoExtractor.loadSavedCalibration();
            console.log('[vc-rc:video] Extractor initialized, calibrated:', this._calibrated);
        } catch (e) {
            console.warn('[vc-rc:video] Failed to init MediaPipe:', e);
            this._videoExtractor = null;
        }
    }

    async _startCamera() {
        if (!this._videoExtractor) return;

        // Open camera if not already previewing (e.g. from calibrate)
        if (!this._cameraActive) {
            await this._videoExtractor.startCamera();
            this._cameraActive = true;

            const videoEl = this._videoExtractor.getVideoElement();

            // Mount video thumbnail inline in the control bar
            if (this._els.cameraPreview) {
                this._els.cameraPreview.innerHTML = '';
                if (videoEl) {
                    videoEl.classList.add('vc-rc-camera-thumb');
                    this._els.cameraPreview.appendChild(videoEl);
                    this._els.cameraPreview.style.display = '';
                }
            }

            this.dispatchEvent(new CustomEvent('camerachange', {
                detail: { active: true, videoElement: videoEl },
            }));

            // Auto-calibrate if needed
            if (!this._calibrated) {
                this._runCalibration();
            }
        }

        // Connect video WebSocket for streaming features to server
        if (!this._videoWs) {
            this._videoWs = new WebSocket(wsUrl(this._opts.wsBasePath + '/video'));
            this._videoExtractor.setWebSocket(this._videoWs);
        }
    }

    async _stopCamera() {
        // Stop video recording first (before closing camera)
        await this._stopVideoRecording();

        // Disconnect video WS (stop streaming to server)
        if (this._videoWs) {
            this._videoWs.onclose = null;
            this._videoWs.close();
            this._videoWs = null;
        }
        if (this._videoExtractor) {
            this._videoExtractor.setWebSocket(null);
            this._videoExtractor.stopCamera();
        }
        if (this._cameraActive) {
            this._cameraActive = false;
            // Clear inline preview
            if (this._els.cameraPreview) {
                this._els.cameraPreview.innerHTML = '';
                this._els.cameraPreview.style.display = 'none';
            }
            this.dispatchEvent(new CustomEvent('camerachange', {
                detail: { active: false, videoElement: null },
            }));
        }
    }

    // ─── Video Recording (debug: raw WebM save) ────────────────────────

    _startVideoRecording() {
        if (!this._videoExtractor) return;
        const stream = this._videoExtractor.getMediaStream();
        if (!stream) return;

        this._videoChunks = [];
        try {
            const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
                ? 'video/webm;codecs=vp9'
                : 'video/webm;codecs=vp8';
            this._mediaRecorder = new MediaRecorder(stream, {
                mimeType,
                videoBitsPerSecond: 1_000_000,
            });
            this._mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) this._videoChunks.push(e.data);
            };
            this._mediaRecorder.start(1000);
            console.log('[vc-rc] Video recording started:', mimeType);
        } catch (e) {
            console.warn('[vc-rc] MediaRecorder failed:', e);
            this._mediaRecorder = null;
        }
    }

    _stopVideoRecording() {
        if (!this._mediaRecorder || this._mediaRecorder.state === 'inactive') return;
        return new Promise((resolve) => {
            this._mediaRecorder.onstop = () => {
                this._uploadVideo().then(resolve).catch(() => resolve());
            };
            this._mediaRecorder.stop();
        });
    }

    async _uploadVideo() {
        if (!this._videoChunks.length) return;
        const blob = new Blob(this._videoChunks, { type: 'video/webm' });
        this._videoChunks = [];
        console.log(`[vc-rc] Video blob: ${(blob.size / 1024).toFixed(0)} KB`);

        const filename = this._lastSavedFilename;
        if (!filename) {
            console.warn('[vc-rc] No audio filename to pair video with');
            return;
        }
        const videoName = filename.replace(/\.wav$/, '.webm');
        const form = new FormData();
        form.append('video', blob, videoName);
        try {
            await apiFetch('/api/live/save-video', { method: 'POST', body: form });
            console.log('[vc-rc] Video saved:', videoName);
        } catch (e) {
            console.warn('[vc-rc] Video upload failed:', e);
        }
    }

    _runCalibration() {
        this.dispatchEvent(new CustomEvent('calibration', {
            detail: {
                extractor: this._videoExtractor,
                onDone: () => { this._calibrated = true; },
            },
        }));
    }

    /** Expose the video extractor for pages that need calibration UI */
    get videoExtractor() { return this._videoExtractor; }
    get cameraActive() { return this._cameraActive; }

    // ─── Device Discovery ───────────────────────────────────────────────

    async _loadDevices() {
        // Server devices
        if (this._opts.showSourceSelect || this._opts.defaultSource === 'server') {
            try {
                const res = await apiFetch('/api/live/devices');
                const devices = await res.json();
                const sel = this._els.deviceSelect;
                if (sel) {
                    sel.innerHTML = '';
                    if (devices.length === 0) {
                        sel.innerHTML = '<option value="">No input devices</option>';
                    } else {
                        devices.forEach(d => {
                            const opt = document.createElement('option');
                            opt.value = d.id;
                            opt.textContent = `${d.name} (${d.channels}ch, ${Math.round(d.sr)}Hz)`;
                            sel.appendChild(opt);
                        });
                        try {
                            const saved = localStorage.getItem(LS_KEYS.serverDevice);
                            const opts = [...sel.options];
                            if (saved && opts.some(o => o.value === saved)) {
                                sel.value = saved;
                            }
                        } catch (e) { /* ignore */ }
                    }
                }
            } catch (e) { /* ignore */ }
        }

        // Browser devices
        this._loadBrowserDevices();

        // Check if already running
        try {
            const res = await apiFetch('/api/live/status');
            const data = await res.json();
            if (data.running) {
                this._connectMetricsWs();
                this._setState('recording');
            }
        } catch (e) { /* ignore */ }
    }

    async _loadBrowserDevices() {
        if (!navigator.mediaDevices) return;
        const sel = this._els.browserDeviceSelect;
        if (!sel) return;

        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const inputs = devices.filter(d => d.kind === 'audioinput');
            sel.innerHTML = '';
            if (inputs.length === 0 || (inputs.length === 1 && !inputs[0].label)) {
                sel.innerHTML = '<option value="">Default mic</option>';
                return;
            }
            for (const dev of inputs) {
                const opt = document.createElement('option');
                opt.value = dev.deviceId;
                opt.textContent = dev.label || `Mic ${dev.deviceId.slice(0, 8)}`;
                sel.appendChild(opt);
            }
            // Restore saved browser device
            try {
                const saved = localStorage.getItem(LS_KEYS.browserDevice);
                if (saved && [...sel.options].some(o => o.value === saved)) {
                    sel.value = saved;
                }
            } catch (e) { /* ignore */ }
        } catch (e) { /* ignore */ }
    }

    // ─── Preference Persistence ─────────────────────────────────────────

    _restorePrefs() {
        try {
            const savedSource = localStorage.getItem(LS_KEYS.source);
            if (savedSource) this._opts.defaultSource = savedSource;

            const savedChannel = localStorage.getItem(LS_KEYS.channel);
            if (savedChannel !== null) this._opts.defaultChannel = parseInt(savedChannel);

            // Camera checkbox: only default to checked if a valid calibration exists;
            // honour an explicit user override stored from last session.
            const hasCameraCalib = (() => {
                try {
                    const saved = localStorage.getItem('videoCalibration');
                    if (!saved) return false;
                    const calib = JSON.parse(saved);
                    return calib && (calib.jawOpenRef - calib.jawClosedRef) > 0.05;
                } catch (e) { return false; }
            })();

            const savedCameraState = localStorage.getItem(LS_KEYS.cameraCheckbox);
            if (savedCameraState !== null) {
                this._opts.cameraDefault = savedCameraState === 'true';
            } else {
                this._opts.cameraDefault = hasCameraCalib;
            }
        } catch (e) { /* localStorage unavailable */ }
    }

    _savePrefs() {
        try {
            if (this._els.sourceSelect) {
                localStorage.setItem(LS_KEYS.source, this._els.sourceSelect.value);
            }
            if (this._els.deviceSelect) {
                localStorage.setItem(LS_KEYS.serverDevice, this._els.deviceSelect.value);
            }
            if (this._els.browserDeviceSelect) {
                localStorage.setItem(LS_KEYS.browserDevice, this._els.browserDeviceSelect.value);
            }
            if (this._els.channelSelect) {
                localStorage.setItem(LS_KEYS.channel, this._els.channelSelect.value);
            }
        } catch (e) { /* localStorage unavailable */ }
    }

    // ─── Helpers ────────────────────────────────────────────────────────

    _getSource() {
        return this._els.sourceSelect?.value || this._opts.defaultSource;
    }

    _emitError(message, phase) {
        this.dispatchEvent(new CustomEvent('error', {
            detail: { message, phase },
        }));
    }

    // ─── UI Rendering ───────────────────────────────────────────────────

    _render() {
        const o = this._opts;
        const source = o.defaultSource;

        let html = `<div class="vc-rc-bar">`;

        // ── LEFT GROUP: Audio/mic setup ──────────────────────────────
        const hasSetup = o.showSourceSelect || o.showDeviceSelect || o.showChannelSelect;
        if (hasSetup) {
            html += `<button class="vc-rc-btn vc-rc-btn-check vc-rc-setup-toggle" data-rc="setupToggle" title="Mic setup">&#9881;</button>`;

            // Setup dropdowns inline (hidden by default, toggled by gear)
            html += `<span class="vc-rc-setup-inline" data-rc="setupPanel" style="display:none">`;
            if (o.showSourceSelect) {
                html += `
                    <select class="vc-rc-select" data-rc="sourceSelect">
                        <option value="server"${source === 'server' ? ' selected' : ''}>Server mic</option>
                        <option value="browser"${source === 'browser' ? ' selected' : ''}>Browser mic</option>
                    </select>`;
            }
            if (o.showDeviceSelect) {
                html += `
                    <select class="vc-rc-select" data-rc="deviceSelect"
                            style="display:${source === 'server' ? '' : 'none'}">
                        <option value="">Loading...</option>
                    </select>
                    <select class="vc-rc-select" data-rc="browserDeviceSelect"
                            style="display:${source === 'browser' ? '' : 'none'}">
                        <option value="">Default mic</option>
                    </select>`;
            }
            if (o.showChannelSelect) {
                html += `
                    <select class="vc-rc-select vc-rc-select-narrow" data-rc="channelSelect"
                            style="display:${source === 'browser' ? '' : 'none'}">
                        <option value="0">Left (ch0)</option>
                        <option value="1"${o.defaultChannel === 1 ? ' selected' : ''}>Right (ch1)</option>
                    </select>`;
            }
            html += `</span>`;
        }

        // ── CENTER: Action buttons ───────────────────────────────────
        html += `
            <span class="vc-rc-divider"></span>
            <button class="vc-rc-btn vc-rc-btn-check" data-rc="checkBtn">Check Level</button>
            <span class="vc-rc-start-stop-slot">
                <button class="vc-rc-btn vc-rc-btn-start" data-rc="startBtn">Start</button>
                <button class="vc-rc-btn vc-rc-btn-stop" data-rc="stopBtn" style="display:none">Stop</button>
            </span>`;

        // ── RIGHT GROUP: Camera + status ─────────────────────────────
        if (o.showCamera) {
            html += `
                <span class="vc-rc-divider"></span>
                <label class="vc-rc-camera-label">
                    <input type="checkbox" data-rc="cameraToggle"
                           ${o.cameraDefault ? 'checked' : ''}>
                    <span>Camera</span>
                </label>
                <button class="vc-rc-btn vc-rc-btn-check vc-rc-btn-cam" data-rc="calibrateBtn"
                        title="Preview camera &amp; calibrate">Calibrate</button>
                <div class="vc-rc-camera-preview" data-rc="cameraPreview" style="display:none"></div>`;
        }

        if (o.showDeleteButton) {
            html += `
                <button class="vc-rc-btn vc-rc-btn-delete" data-rc="deleteBtn" style="display:none"
                        title="Delete last saved recording">Delete</button>`;
        }

        // Status dot (far right)
        html += `
            <div class="vc-rc-status" data-rc="status">
                <div class="vc-rc-status-dot" data-rc="statusDot"></div>
                <span data-rc="statusText">Ready</span>
            </div>`;

        html += `</div>`; // .vc-rc-bar

        // Level meter (inline, hidden by default)
        html += `
            <div class="vc-rc-level" data-rc="levelPanel" style="display:none">
                <div class="vc-rc-level-track">
                    <div class="vc-rc-level-fill" data-rc="levelFill"></div>
                    <div class="vc-rc-level-zone vc-rc-zone-quiet"></div>
                    <div class="vc-rc-level-zone vc-rc-zone-good"></div>
                    <div class="vc-rc-level-zone vc-rc-zone-hot"></div>
                </div>
                <div class="vc-rc-level-labels">
                    <span>Too quiet</span><span>Good</span><span>Too hot</span>
                </div>
                <div class="vc-rc-level-info">
                    <span class="vc-rc-level-readout" data-rc="levelReadout">Waiting...</span>
                    <span class="vc-rc-level-peak" data-rc="levelPeak"></span>
                </div>
            </div>`;

        this._container.innerHTML = html;

        // Collect refs
        this._els = {};
        this._container.querySelectorAll('[data-rc]').forEach(el => {
            this._els[el.dataset.rc] = el;
        });

        // Calibrate button only visible when camera is checked
        if (this._els.calibrateBtn) {
            this._els.calibrateBtn.style.display = o.cameraDefault ? '' : 'none';
        }
    }

    _bindEvents() {
        // Source toggle
        this._els.sourceSelect?.addEventListener('change', () => {
            const isBrowser = this._getSource() === 'browser';
            if (this._els.deviceSelect)
                this._els.deviceSelect.style.display = isBrowser ? 'none' : '';
            if (this._els.browserDeviceSelect)
                this._els.browserDeviceSelect.style.display = isBrowser ? '' : 'none';
            if (this._els.channelSelect)
                this._els.channelSelect.style.display = isBrowser ? '' : 'none';
            if (isBrowser) this._loadBrowserDevices();
            this._updateCheckBtn();
            this._savePrefs();
        });

        // Save prefs on device/channel change
        this._els.deviceSelect?.addEventListener('change', () => this._savePrefs());
        this._els.browserDeviceSelect?.addEventListener('change', () => this._savePrefs());
        this._els.channelSelect?.addEventListener('change', () => this._savePrefs());

        // Device change re-enumerate
        if (navigator.mediaDevices) {
            navigator.mediaDevices.addEventListener('devicechange', () => {
                if (this._getSource() === 'browser') this._loadBrowserDevices();
            });
        }

        // Buttons
        this._els.checkBtn?.addEventListener('click', () => {
            if (this._state === 'idle') {
                this.checkLevel();
            } else if (this._state === 'mic_open') {
                this.stopCheck();
            }
        });

        this._els.startBtn?.addEventListener('click', () => this.start());
        this._els.stopBtn?.addEventListener('click', () => this.stop());
        this._els.deleteBtn?.addEventListener('click', () => this.deleteLastRecording());

        // Camera toggle: persist state, start/stop stream, show/hide Calibrate
        this._els.cameraToggle?.addEventListener('change', () => {
            const on = this._els.cameraToggle.checked;
            try { localStorage.setItem(LS_KEYS.cameraCheckbox, on); } catch (e) { /* ignore */ }
            if (this._els.calibrateBtn) {
                this._els.calibrateBtn.style.display = on ? '' : 'none';
            }
            if (on) {
                this._startCamera().catch(e => console.warn('[vc-rc] Camera start failed:', e));
            } else {
                this._stopCamera();
            }
        });

        // Camera calibrate
        this._els.calibrateBtn?.addEventListener('click', () => {
            this.calibrateCamera();
        });

        // Setup toggle — show/hide the mic setup panel
        this._els.setupToggle?.addEventListener('click', () => {
            const panel = this._els.setupPanel;
            if (!panel) return;
            const nowVisible = panel.style.display === 'none';
            panel.style.display = nowVisible ? '' : 'none';
            if (this._els.setupToggle) {
                this._els.setupToggle.classList.toggle('vc-rc-setup-toggle-active', nowVisible);
            }
        });

        // Spacebar: toggle recording (start or stop)
        document.addEventListener('keydown', (e) => {
            if (e.code !== 'Space') return;
            // Skip when typing in an input
            const tag = document.activeElement?.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
            if (document.activeElement?.isContentEditable) return;
            // Skip with modifier keys
            if (e.ctrlKey || e.altKey || e.metaKey) return;

            // Always prevent default (no page scroll), ignore key-repeat entirely
            e.preventDefault();
            if (e.repeat) return;

            if (this._state === 'idle' || this._state === 'mic_open') {
                this.start();
            } else if (this._state === 'recording') {
                this.stop();
            }
        });
    }

    _updateUI() {
        const s = this._state;
        const isIdle = s === 'idle';
        const isMicOpen = s === 'mic_open';
        const isRecording = s === 'recording';
        const isProcessing = s === 'processing';
        const isActive = isRecording || isProcessing;

        // Button visibility
        if (this._els.checkBtn) {
            this._els.checkBtn.style.display = isActive ? 'none' : '';
            this._els.checkBtn.textContent = isMicOpen ? 'Stop Check' : 'Check Level';
            this._updateCheckBtn();
        }
        if (this._els.startBtn) {
            this._els.startBtn.style.display = isActive ? 'none' : '';
            this._els.startBtn.disabled = false;
        }
        if (this._els.stopBtn) {
            this._els.stopBtn.style.display = isRecording ? '' : 'none';
            this._els.stopBtn.disabled = false;
        }
        if (this._els.deleteBtn) {
            this._els.deleteBtn.style.display =
                (isIdle && this._lastSavedFilename) ? '' : 'none';
        }

        // Status
        if (this._els.statusDot) {
            this._els.statusDot.classList.toggle('vc-rc-active', isActive);
        }
        if (this._els.statusText) {
            this._els.statusText.textContent =
                isRecording ? 'Recording' :
                isProcessing ? 'Processing...' :
                isMicOpen ? 'Checking' : 'Ready';
        }

        // Disable selects while active
        const locked = isMicOpen || isActive;
        for (const key of ['sourceSelect', 'deviceSelect', 'browserDeviceSelect', 'channelSelect']) {
            if (this._els[key]) this._els[key].disabled = locked;
        }

        // Level panel
        if (!isMicOpen) {
            this._showLevelMeter(false);
        }
    }

    _updateCheckBtn() {
        // Disable Check Level for server mic (no local AnalyserNode)
        if (this._els.checkBtn && this._state === 'idle') {
            const isServer = this._getSource() === 'server';
            this._els.checkBtn.disabled = isServer;
            this._els.checkBtn.title = isServer
                ? 'Switch to Browser mic for level check'
                : '';
        }
    }

    _showLevelMeter(show) {
        if (this._els.levelPanel) {
            this._els.levelPanel.style.display = show ? '' : 'none';
        }
    }

    _renderLevelBar(rmsDb) {
        const fill = this._els.levelFill;
        const readout = this._els.levelReadout;
        const peak = this._els.levelPeak;
        if (!fill) return;

        const pct = Math.max(0, Math.min(100, ((rmsDb + 60) / 60) * 100));
        fill.style.width = pct + '%';

        let color, status;
        if (rmsDb < LEVEL_ZONES.tooQuiet) {
            color = 'var(--blue)'; status = 'Too quiet';
        } else if (rmsDb < LEVEL_ZONES.quiet) {
            color = 'var(--yellow)'; status = 'A bit quiet';
        } else if (rmsDb <= LEVEL_ZONES.good) {
            color = 'var(--green)'; status = 'Good level';
        } else {
            color = 'var(--red)'; status = 'Too hot!';
        }
        fill.style.background = color;

        if (readout) {
            readout.textContent = `${status}  (${Math.round(rmsDb)} dBFS)`;
            readout.style.color = color;
        }
        if (peak) {
            peak.textContent = `Peak: ${Math.round(this._peakDb)} dBFS`;
        }
    }
}

// ─── Embedded Styles ────────────────────────────────────────────────────────
// Injected once on first import

const STYLE_ID = 'vc-rc-styles';
if (!document.getElementById(STYLE_ID)) {
    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = `
/* ── Recording Control ─────────────────────────────────────────────── */

.vc-rc-bar {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    flex-wrap: wrap;
    padding: 0.75rem 1.25rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
}

.vc-rc-select {
    background: var(--bg-input);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.45rem 0.7rem;
    font-family: inherit;
    font-size: 0.82rem;
    min-width: 160px;
    transition: border-color 0.15s;
}
.vc-rc-select:focus {
    outline: none;
    border-color: var(--accent);
}
.vc-rc-select:disabled {
    opacity: 0.45;
    cursor: not-allowed;
}
.vc-rc-select-narrow {
    min-width: 100px;
}

.vc-rc-btn-cam {
    padding: 0.3rem 0.7rem;
    font-size: 0.75rem;
}

.vc-rc-camera-label {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.82rem;
    color: var(--text-secondary);
    cursor: pointer;
    user-select: none;
}
.vc-rc-camera-label input[type="checkbox"] {
    accent-color: var(--accent);
}

.vc-rc-camera-preview {
    flex-shrink: 0;
}
.vc-rc-camera-thumb {
    width: 64px;
    height: 48px;
    object-fit: cover;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
}

.vc-rc-btn {
    padding: 0.45rem 1.2rem;
    border: none;
    border-radius: var(--radius-sm);
    font-family: inherit;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
}
.vc-rc-btn:disabled {
    opacity: 0.35;
    cursor: not-allowed;
}

.vc-rc-btn-check {
    background: var(--bg-input);
    color: var(--text-primary);
    border: 1px solid var(--border);
}
.vc-rc-btn-check:hover:not(:disabled) {
    border-color: var(--accent);
    color: var(--accent);
}

.vc-rc-btn-start {
    background: var(--green);
    color: #000;
}
.vc-rc-btn-start:hover:not(:disabled) {
    background: var(--green-dim);
}

.vc-rc-btn-stop {
    background: var(--red);
    color: #fff;
}
.vc-rc-btn-stop:hover:not(:disabled) {
    opacity: 0.85;
}

.vc-rc-btn-delete {
    background: var(--red);
    color: #fff;
}
.vc-rc-btn-delete:hover:not(:disabled) {
    opacity: 0.85;
}

.vc-rc-status {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-left: auto;
}

.vc-rc-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text-muted);
    transition: all 0.2s;
}
.vc-rc-status-dot.vc-rc-active {
    background: var(--green);
    box-shadow: 0 0 6px var(--green);
}

/* ── Setup toggle, divider, start/stop slot, setup panel ──────────── */

.vc-rc-setup-toggle {
    padding: 0.45rem 0.65rem;
    font-size: 1rem;
    line-height: 1;
}
.vc-rc-setup-toggle-active {
    border-color: var(--accent);
    color: var(--accent);
}

.vc-rc-divider {
    width: 1px;
    height: 1.5rem;
    background: var(--border);
    flex-shrink: 0;
}

.vc-rc-start-stop-slot {
    display: inline-flex;
    align-items: center;
}

.vc-rc-setup-inline {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Level Meter ───────────────────────────────────────────────────── */

.vc-rc-level {
    margin-top: 0.5rem;
    padding: 0.75rem 1.25rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
}

.vc-rc-level-track {
    position: relative;
    height: 28px;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    overflow: hidden;
}

.vc-rc-level-fill {
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 0%;
    background: var(--green);
    transition: width 0.08s, background 0.12s;
    z-index: 2;
    border-radius: 5px 0 0 5px;
}

.vc-rc-level-zone {
    position: absolute;
    top: 0; bottom: 0;
    opacity: 0.1;
    z-index: 1;
}
.vc-rc-zone-quiet {
    left: 0; width: 33.3%;
    background: var(--blue);
}
.vc-rc-zone-good {
    left: 33.3%; width: 41.7%;
    background: var(--green);
}
.vc-rc-zone-hot {
    left: 75%; width: 25%;
    background: var(--red);
}

.vc-rc-level-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.68rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
    padding: 0 2px;
}

.vc-rc-level-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 0.4rem;
}

.vc-rc-level-readout {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-secondary);
}

.vc-rc-level-peak {
    font-size: 0.78rem;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
}
`;
    document.head.appendChild(style);
}
