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
