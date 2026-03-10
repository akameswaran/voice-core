"""Real-time voice biofeedback — streaming capture + analysis engine.

Captures audio via sounddevice, runs pitch (torchcrepe), formant (Parselmouth),
and H1-H2 (FFT) analysis in parallel threads, and provides get_frame() for
WebSocket streaming to the live dashboard.
"""

import json
import logging
import threading
import time
from datetime import datetime
from math import gcd
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import soundfile as sf


class FrameLogger:
    """Append-only JSONL logger for combined audio+video telemetry.

    Rate-limited to 10 Hz (~36 KB/min). Skips ephemeral coaching/warning/exercise
    state to keep file size manageable.
    """

    # Keys to strip from frames before logging
    _SKIP_KEYS = {"coaching", "warnings", "exercise", "workshop"}

    def __init__(self, output_dir: Path, max_hz: float = 10.0):
        self._output_dir = output_dir
        self._min_interval = 1.0 / max_hz
        self._file = None
        self._last_write = 0.0

    def start(self):
        """Open a new JSONL file for this session."""
        self._output_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        path = self._output_dir / f"telemetry_{ts}.jsonl"
        self._file = open(path, "a")
        self._last_write = 0.0

    def log(self, frame: dict):
        """Write a frame if enough time has elapsed since the last write."""
        if self._file is None:
            return
        now = time.time()
        if now - self._last_write < self._min_interval:
            return
        self._last_write = now
        # Strip ephemeral state
        row = {k: v for k, v in frame.items() if k not in self._SKIP_KEYS}
        try:
            self._file.write(json.dumps(row, default=str) + "\n")
        except (OSError, ValueError):
            pass

    def stop(self):
        """Flush and close the JSONL file."""
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except OSError:
                pass
            self._file = None


class RingBuffer:
    """Thread-safe circular buffer for audio samples."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=np.float32)
        self.write_pos = 0
        self.total_written = 0
        self._lock = threading.Lock()

    def write(self, data: np.ndarray):
        """Append samples to the buffer."""
        n = len(data)
        with self._lock:
            if n >= self.capacity:
                self.buffer[:] = data[-self.capacity:]
                self.write_pos = 0
            else:
                end = self.write_pos + n
                if end <= self.capacity:
                    self.buffer[self.write_pos:end] = data
                else:
                    first = self.capacity - self.write_pos
                    self.buffer[self.write_pos:] = data[:first]
                    self.buffer[:n - first] = data[first:]
                self.write_pos = end % self.capacity
            self.total_written += n

    def read_last(self, n: int) -> np.ndarray:
        """Read the last n samples. Pads with zeros if not enough data yet."""
        with self._lock:
            available = min(n, self.total_written, self.capacity)
            if available == 0:
                return np.zeros(n, dtype=np.float32)

            start = (self.write_pos - available) % self.capacity
            if start + available <= self.capacity:
                out = self.buffer[start:start + available].copy()
            else:
                first = self.capacity - start
                out = np.concatenate([
                    self.buffer[start:],
                    self.buffer[:available - first],
                ])

            if available < n:
                result = np.zeros(n, dtype=np.float32)
                result[n - available:] = out
                return result
            return out


class LiveAnalyzer:
    """Real-time voice analysis engine with threaded workers.

    Workers:
    - Pitch (torchcrepe on GPU): ~25 Hz update rate
    - Formants (Parselmouth Burg): ~12 Hz update rate
    - H1-H2 (FFT peak picking): ~25 Hz update rate
    - RMS (in audio callback): per-block update
    """

    def __init__(self, device=None, sr: int = 48000, block_size: int = 1024,
                 crepe_device: str = "cuda:0", formant_ceiling: float = 5500.0,
                 realtime_coach=None, exercise_manager=None,
                 zone_classifier=None, recordings_dir=None):
        self.device = device
        self.sr = sr
        self.block_size = block_size
        self.crepe_device = crepe_device
        self.formant_ceiling = formant_ceiling
        self.ring = RingBuffer(sr * 2)  # 2 seconds of audio
        self.latest = {
            "ts": 0.0,
            "f0_hz": 0.0,
            "f0_confidence": 0.0,
            "f1_hz": 0.0,
            "f2_hz": 0.0,
            "f3_hz": 0.0,
            "f4_hz": 0.0,
            "delta_f_hz": 0.0,
            "delta_f_zone": "",
            "h1_h2_db": 0.0,
            "h1_h2_corrected_db": 0.0,
            "h1_a3_corrected_db": 0.0,
            "spectral_tilt_db": 0.0,
            "rms_db": -60.0,
            # Formant bandwidths
            "bw1_hz": 0.0,
            "bw2_hz": 0.0,
            "bw3_hz": 0.0,
            # F1/ΔF ratio (resonance balance indicator)
            "f1_delta_f_ratio": 0.0,
            # Safety monitor metrics
            "hnr_db": 0.0,
            "jitter_pct": 0.0,
            "shimmer_pct": 0.0,
        }
        self._lock = threading.Lock()
        self._running = False
        self._stream = None
        self._workers = []

        # Safety monitor — evaluates combined metrics for vocal health
        from voice_core.safety_monitor import SafetyMonitor
        self.safety = SafetyMonitor()
        self._latest_warnings: list = []

        # Video feature state — timestamped ring buffer for proper A/V sync
        self._latest_video: dict = {}
        self._video_ring: list[tuple[float, dict]] = []  # [(server_ts, features), ...]
        self._video_ring_max = 30  # ~2s at 15Hz
        from voice_core.video_monitor import VideoTensionMonitor
        self.video_monitor = VideoTensionMonitor()

        # Real-time coaching engine (injectable — None = disabled)
        self.coach = realtime_coach
        self._latest_coaching: dict | None = None
        self._coaching_log: list[dict] = []  # accumulated coaching messages with session_t

        # Exercise manager (injectable — None = disabled)
        self.exercises = exercise_manager

        # Zone classifier (injectable — None = disabled)
        self._classify_zone = zone_classifier

        # Session timing
        self._session_start_ts: float | None = None

        # Recording buffer — accumulates all incoming audio for playback
        self._recording_chunks: list[np.ndarray] = []
        self._total_samples_fed: int = 0  # running sample counter for WAV-aligned session_t
        if recordings_dir is not None:
            self._recordings_dir = Path(recordings_dir)
        else:
            # Fallback: project root / recordings (works when installed as package)
            self._recordings_dir = Path.cwd() / "recordings"
        self._recordings_dir.mkdir(exist_ok=True)

        # Telemetry logger
        self._frame_logger = FrameLogger(self._recordings_dir)

        # Clip marking for baseline wizard (record sub-clips without stopping)
        self._clip_start_idx: int | None = None
        self._last_autosave_len: int = 0  # track how much we've auto-saved

    @property
    def running(self) -> bool:
        return self._running

    def mark_clip_start(self):
        """Mark the start of a recording clip at the current buffer position."""
        self._clip_start_idx = sum(len(c) for c in self._recording_chunks)

    def save_clip(self, path: str) -> dict | None:
        """Save audio from clip mark to current position.

        Applies silence trimming and peak normalization to -3 dBFS.
        Returns {"path": str, "duration_s": float} or None if too short.
        """
        if not self._recording_chunks:
            return None
        all_audio = np.concatenate(self._recording_chunks)
        start = self._clip_start_idx if self._clip_start_idx is not None else 0
        clip = all_audio[start:]
        if len(clip) < self.sr * 0.5:
            return None

        # Trim silence from start and end (noise gate at -40 dBFS)
        threshold = 10 ** (-40 / 20)  # ~0.01
        abs_clip = np.abs(clip)
        # Smooth with short RMS window to avoid cutting on single-sample dips
        win = int(0.02 * self.sr)  # 20ms
        if len(clip) > win:
            rms_env = np.array([
                np.sqrt(np.mean(clip[i:i+win] ** 2))
                for i in range(0, len(clip) - win, win // 2)
            ])
            above = np.where(rms_env > threshold)[0]
            if len(above) > 0:
                trim_start = max(0, above[0] * (win // 2) - win)  # 20ms padding
                trim_end = min(len(clip), (above[-1] + 1) * (win // 2) + win)
                clip = clip[trim_start:trim_end]

        if len(clip) < self.sr * 0.3:
            return None

        clip = self._normalize_audio(clip)

        sf.write(path, clip, self.sr)
        self._clip_start_idx = None

        # Auto-save full recording buffer to prevent data loss
        self.autosave_recording()

        return {"path": path, "duration_s": round(len(clip) / self.sr, 1)}

    def start(self):
        """Start audio capture (sounddevice) and analysis workers."""
        import sounddevice as sd

        self._running = True
        self._recording_chunks.clear()
        self._total_samples_fed = 0
        self._coaching_log.clear()
        self._session_start_ts = time.time()
        self._frame_logger.start()

        self._stream = sd.InputStream(
            device=self.device,
            samplerate=self.sr,
            channels=1,
            blocksize=self.block_size,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        self._start_workers()

    def start_remote(self):
        """Start analysis workers only — audio fed externally via feed_audio()."""
        self._running = True
        self._recording_chunks.clear()
        self._total_samples_fed = 0
        self._coaching_log.clear()
        self._session_start_ts = None  # Set on first audio chunk
        self._frame_logger.start()
        self._start_workers()

    def feed_audio(self, data: np.ndarray):
        """Feed audio samples from an external source (e.g., browser WebSocket)."""
        self.ring.write(data)
        self._recording_chunks.append(data.copy())
        self._total_samples_fed += len(data)
        now = time.time()
        if self._session_start_ts is None:
            self._session_start_ts = now
        rms = np.sqrt(np.mean(data ** 2))
        rms_db = float(20 * np.log10(max(rms, 1e-10)))
        # session_t from sample count — directly matches WAV timeline
        # (immune to audio-clock vs system-clock drift)
        session_t = self._total_samples_fed / self.sr
        with self._lock:
            self.latest["rms_db"] = rms_db
            self.latest["ts"] = now
            self.latest["session_t"] = session_t

    def feed_video(self, features: dict):
        """Feed video feature vector from browser MediaPipe.

        Stores in a timestamped ring buffer for proper A/V sync.
        The client_ts field (performance.now() ms) is preserved if present.
        """
        now = time.time()
        with self._lock:
            self._latest_video = features
            self._video_ring.append((now, dict(features)))
            if len(self._video_ring) > self._video_ring_max:
                self._video_ring = self._video_ring[-self._video_ring_max:]

    def _start_workers(self):
        from voice_core.safety_monitor import hnr_worker_fn, jitter_shimmer_worker_fn

        for fn in [self._pitch_worker, self._formant_worker, self._h1h2_worker,
                    self._safety_worker]:
            t = threading.Thread(target=fn, daemon=True)
            t.start()
            self._workers.append(t)

        # HNR and jitter/shimmer workers from safety_monitor module
        for fn in [hnr_worker_fn, jitter_shimmer_worker_fn]:
            t = threading.Thread(target=fn, args=(self,), daemon=True)
            t.start()
            self._workers.append(t)

    def stop(self, save: bool = False) -> str | None:
        """Stop audio capture and analysis workers.

        Args:
            save: If True, save the recording buffer to disk. Default False.

        Returns saved recording filename if save=True and clip is long enough,
        else None.
        """
        self._running = False
        self._frame_logger.stop()

        # Diagnostic: log audio clock drift
        if self._session_start_ts and self._total_samples_fed > 0:
            wall_clock = time.time() - self._session_start_ts
            audio_clock = self._total_samples_fed / self.sr
            drift_pct = ((audio_clock / wall_clock) - 1) * 100 if wall_clock > 0 else 0
            logger.info(
                "Session timing: %.1fs wall-clock, %.1fs audio (%d samples @ %d Hz), "
                "drift: %+.2f%%",
                wall_clock, audio_clock, self._total_samples_fed, self.sr, drift_pct,
            )

        with self._lock:
            self._video_ring.clear()
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        # Wait briefly for workers to exit
        for w in self._workers:
            w.join(timeout=1.0)
        self._workers.clear()

        if save:
            return self.save_recording()
        # Keep _recording_chunks intact so save_recording() can be called after stop
        return None

    @staticmethod
    def _normalize_audio(audio: np.ndarray) -> np.ndarray:
        """Peak-normalize audio to -3 dBFS."""
        peak = np.max(np.abs(audio))
        if peak > 0:
            target = 10 ** (-3 / 20)  # ~0.708
            audio = audio * (target / peak)
        return audio

    def autosave_recording(self) -> str | None:
        """Save current recording buffer to disk without clearing it.

        Writes the full buffer each time (overwrites previous autosave).
        Called automatically after each clip save to prevent data loss.
        Returns path or None.
        """
        if not self._recording_chunks:
            return None
        total_samples = sum(len(c) for c in self._recording_chunks)
        if total_samples < self.sr * 1.0:
            return None
        # Only re-save if buffer has grown since last autosave
        if total_samples <= self._last_autosave_len:
            return None
        audio = self._normalize_audio(np.concatenate(self._recording_chunks))
        path = self._recordings_dir / "live_autosave.wav"
        sf.write(str(path), audio, self.sr)
        self._last_autosave_len = total_samples
        return str(path)

    def save_recording(self) -> str | None:
        """Concatenate recorded chunks, normalize, and save as WAV."""
        if not self._recording_chunks:
            return None
        audio = np.concatenate(self._recording_chunks)
        self._recording_chunks.clear()
        self._last_autosave_len = 0

        if len(audio) < self.sr * 0.5:  # skip clips under 0.5s
            return None

        # Trim leading silence from browser mic startup (zero-filled buffers)
        # and apply short fade-in to avoid click at signal onset
        threshold = 1e-4  # ~-80 dBFS — well below any real audio
        first_nonzero = 0
        for i, s in enumerate(audio):
            if abs(s) > threshold:
                first_nonzero = i
                break
        if first_nonzero > 0:
            audio = audio[first_nonzero:]
        # 5ms fade-in to smooth any remaining onset transient
        fade_len = min(int(self.sr * 0.005), len(audio))
        if fade_len > 0:
            audio[:fade_len] *= np.linspace(0, 1, fade_len)

        audio = self._normalize_audio(audio)

        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"live_{ts}.wav"
        path = self._recordings_dir / filename
        sf.write(str(path), audio, self.sr)

        # Clean up autosave file
        autosave = self._recordings_dir / "live_autosave.wav"
        if autosave.exists():
            try:
                autosave.unlink()
            except OSError:
                pass

        return str(path)

    def get_coaching_log(self) -> list[dict]:
        """Return accumulated coaching messages with session_t timestamps."""
        return list(self._coaching_log)

    def get_session_duration(self) -> float:
        """Return session duration in seconds from sample count."""
        return self._total_samples_fed / self.sr if self.sr > 0 else 0.0

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback — runs in audio thread, must be fast."""
        mono = indata[:, 0]
        self.ring.write(mono)
        self._recording_chunks.append(mono.copy())
        self._total_samples_fed += len(mono)

        # RMS is cheap enough for the audio callback
        rms = np.sqrt(np.mean(mono ** 2))
        rms_db = float(20 * np.log10(max(rms, 1e-10)))
        session_t = self._total_samples_fed / self.sr
        with self._lock:
            self.latest["rms_db"] = rms_db
            self.latest["ts"] = time.time()
            self.latest["session_t"] = session_t

    def _pitch_worker(self):
        """Thread: CREPE pitch tracking on GPU."""
        import torch
        import torchcrepe
        from scipy.signal import resample_poly

        # Compute resampling factors: sr -> 16000
        g = gcd(self.sr, 16000)
        up, down = 16000 // g, self.sr // g

        # Warmup: load the CREPE model with a dummy prediction
        try:
            dummy = torch.zeros(1, 1024, dtype=torch.float32, device=self.crepe_device)
            torchcrepe.predict(
                dummy, 16000, hop_length=1024, fmin=50.0, fmax=550.0,
                model="full", decoder=torchcrepe.decode.viterbi,
                return_periodicity=True, batch_size=64, device=self.crepe_device,
            )
        except Exception:
            pass

        while self._running:
            try:
                # 100ms window at native sample rate
                n_samples = int(self.sr * 0.1)
                chunk = self.ring.read_last(n_samples)

                if np.max(np.abs(chunk)) < 1e-5:
                    time.sleep(0.005)
                    continue

                # Resample to 16kHz
                chunk_16k = resample_poly(chunk, up, down).astype(np.float32)

                audio_tensor = torch.tensor(
                    chunk_16k, dtype=torch.float32
                ).unsqueeze(0).to(self.crepe_device)

                frequency, confidence = torchcrepe.predict(
                    audio_tensor, 16000,
                    hop_length=160,
                    fmin=50.0, fmax=550.0,
                    model="full",
                    decoder=torchcrepe.decode.viterbi,
                    return_periodicity=True,
                    batch_size=64,
                    device=self.crepe_device,
                )
                freq = frequency.squeeze(0).cpu().numpy()
                conf = confidence.squeeze(0).cpu().numpy()

                # Use last frame with good confidence
                valid = conf > 0.5
                if valid.any():
                    idx = int(np.where(valid)[0][-1])
                    f0 = float(freq[idx])
                    c = float(conf[idx])
                else:
                    f0 = 0.0
                    c = 0.0

                with self._lock:
                    self.latest["f0_hz"] = f0
                    self.latest["f0_confidence"] = c

            except Exception:
                pass

            time.sleep(0.005)

    def _formant_worker(self):
        """Thread: Parselmouth formant extraction + delta-F."""
        import parselmouth
        from parselmouth.praat import call
        from voice_core.analyze import _compute_delta_f

        while self._running:
            try:
                # 100ms window
                n_samples = int(self.sr * 0.1)
                chunk = self.ring.read_last(n_samples)

                if np.max(np.abs(chunk)) < 1e-5:
                    time.sleep(0.005)
                    continue

                snd = parselmouth.Sound(chunk, sampling_frequency=self.sr)
                # Single ceiling for speed (5500 = best for feminine-range analysis)
                formant = call(snd, "To Formant (burg)", 0.0, 5, self.formant_ceiling, 0.025, 50.0)
                n_frames = call(formant, "Get number of frames")

                f1s, f2s, f3s, f4s = [], [], [], []
                bw1s, bw2s, bw3s = [], [], []
                for i in range(1, n_frames + 1):
                    t = call(formant, "Get time from frame number", i)
                    for flist, n in [(f1s, 1), (f2s, 2), (f3s, 3), (f4s, 4)]:
                        f = call(formant, "Get value at time", n, t, "Hertz", "Linear")
                        if not np.isnan(f) and f > 0:
                            flist.append(f)
                    # Extract bandwidths BW1-BW3
                    for blist, n in [(bw1s, 1), (bw2s, 2), (bw3s, 3)]:
                        bw = call(formant, "Get bandwidth at time", n, t, "Hertz", "Linear")
                        if not np.isnan(bw) and bw > 0:
                            blist.append(bw)

                f1 = float(np.mean(f1s)) if f1s else 0.0
                f2 = float(np.mean(f2s)) if f2s else 0.0
                f3 = float(np.mean(f3s)) if f3s else 0.0
                f4 = float(np.mean(f4s)) if f4s else 0.0
                bw1 = float(np.mean(bw1s)) if bw1s else 0.0
                bw2 = float(np.mean(bw2s)) if bw2s else 0.0
                bw3 = float(np.mean(bw3s)) if bw3s else 0.0
                delta_f = _compute_delta_f(f1, f2, f3, f4)

                # F1/ΔF ratio — indicates resonance balance
                f1_delta_f_ratio = (f1 / delta_f) if f1 > 0 and delta_f > 0 else 0.0

                if self._classify_zone is not None and delta_f > 0:
                    zone = self._classify_zone(delta_f)
                else:
                    zone = ""

                with self._lock:
                    self.latest["f1_hz"] = f1
                    self.latest["f2_hz"] = f2
                    self.latest["f3_hz"] = f3
                    self.latest["f4_hz"] = f4
                    self.latest["bw1_hz"] = bw1
                    self.latest["bw2_hz"] = bw2
                    self.latest["bw3_hz"] = bw3
                    self.latest["delta_f_hz"] = delta_f
                    self.latest["delta_f_zone"] = zone
                    self.latest["f1_delta_f_ratio"] = f1_delta_f_ratio

            except Exception:
                pass

            time.sleep(0.005)

    @staticmethod
    def _iseli_correction(f_hz: float, formant_freqs: list,
                          formant_bws: list) -> float:
        """Iseli et al. (2007) vocal tract correction at frequency f_hz."""
        correction_db = 0.0
        for fi, bi in zip(formant_freqs, formant_bws):
            if fi <= 0 or bi <= 0:
                continue
            num = (fi**2 + bi**2)**2
            denom = (f_hz**2 - fi**2)**2 + (f_hz * bi)**2
            h_at_f = 10 * np.log10(num / denom) if denom > 0 else 0.0
            denom_center = bi**2 * fi**2
            h_at_center = 10 * np.log10(num / denom_center) if denom_center > 0 else 0.0
            correction_db += (h_at_f - h_at_center)
        return correction_db

    def _h1h2_worker(self):
        """Thread: H1-H2 and H1-A3 estimation with Iseli correction."""
        while self._running:
            try:
                # 50ms window
                n_samples = int(self.sr * 0.05)
                chunk = self.ring.read_last(n_samples)

                if np.max(np.abs(chunk)) < 1e-5:
                    time.sleep(0.005)
                    continue

                # Need current F0 and formants for correction
                with self._lock:
                    f0 = self.latest["f0_hz"]
                    f1 = self.latest["f1_hz"]
                    f2 = self.latest["f2_hz"]
                    f3 = self.latest["f3_hz"]
                    bw1 = self.latest["bw1_hz"]
                    bw2 = self.latest["bw2_hz"]
                    bw3 = self.latest["bw3_hz"]

                if f0 < 50:
                    time.sleep(0.005)
                    continue

                windowed = chunk * np.hamming(len(chunk))
                spectrum = np.abs(np.fft.rfft(windowed))
                freqs = np.fft.rfftfreq(len(chunk), 1.0 / self.sr)
                spectrum_db = 20 * np.log10(spectrum + 1e-10)

                h1_mask = (freqs >= f0 * 0.9) & (freqs <= f0 * 1.1)
                h2_mask = (freqs >= f0 * 1.8) & (freqs <= f0 * 2.2)

                if not (h1_mask.any() and h2_mask.any()):
                    time.sleep(0.005)
                    continue

                h1_db = float(np.max(spectrum_db[h1_mask]))
                h2_db = float(np.max(spectrum_db[h2_mask]))
                h1_h2_raw = h1_db - h2_db

                # Iseli correction when formants available
                has_formants = f1 > 0 and f2 > 0 and f3 > 0 and bw1 > 0
                formant_freqs = [f1, f2, f3]
                formant_bws = [bw1, bw2, bw3]

                h1_corr = h1_db
                if has_formants:
                    h1_corr = h1_db - self._iseli_correction(
                        f0, formant_freqs, formant_bws)
                    h2_corr = h2_db - self._iseli_correction(
                        2 * f0, formant_freqs, formant_bws)
                    h1_h2_corrected = h1_corr - h2_corr
                else:
                    h1_h2_corrected = h1_h2_raw

                # H1-A3: amplitude at nearest harmonic to F3
                h1_a3_corrected = 0.0
                if has_formants and f3 > 0:
                    # Find the harmonic closest to F3
                    a3_harmonic = round(f3 / f0) * f0
                    if a3_harmonic > 0:
                        a3_mask = (freqs >= a3_harmonic * 0.9) & (
                            freqs <= a3_harmonic * 1.1)
                        if a3_mask.any():
                            a3_db = float(np.max(spectrum_db[a3_mask]))
                            a3_corr = a3_db - self._iseli_correction(
                                a3_harmonic, formant_freqs, formant_bws)
                            h1_a3_corrected = h1_corr - a3_corr

                # Spectral tilt: fit slope across harmonics 1-10
                tilt = 0.0
                harm_freqs_log = []
                harm_amps = []
                for n in range(1, 11):
                    hf = n * f0
                    if hf > self.sr / 2:
                        break
                    hm = (freqs >= hf * 0.9) & (freqs <= hf * 1.1)
                    if hm.any():
                        harm_freqs_log.append(np.log2(hf))
                        harm_amps.append(float(np.max(spectrum_db[hm])))
                if len(harm_freqs_log) >= 3:
                    coeffs = np.polyfit(harm_freqs_log, harm_amps, 1)
                    tilt = float(coeffs[0])  # dB per octave

                with self._lock:
                    self.latest["h1_h2_db"] = h1_h2_raw
                    self.latest["h1_h2_corrected_db"] = h1_h2_corrected
                    self.latest["h1_a3_corrected_db"] = h1_a3_corrected
                    self.latest["spectral_tilt_db"] = tilt

            except Exception:
                pass

            time.sleep(0.005)

    def _safety_worker(self):
        """Thread: evaluate combined metrics through SafetyMonitor + CoachingEngine."""
        # Capture baseline after first few seconds of stable data
        baseline_captured = False
        baseline_delay = 3.0  # seconds before capturing baseline
        start_time = time.time()

        while self._running:
            try:
                with self._lock:
                    snapshot = dict(self.latest)

                # Capture baseline after initial delay
                if (not baseline_captured
                        and time.time() - start_time > baseline_delay
                        and snapshot.get("delta_f_hz", 0) > 0):
                    if self.coach is not None:
                        self.coach.set_baseline(snapshot)
                    # Set video tension baseline if video data exists
                    with self._lock:
                        vid = dict(self._latest_video) if self._latest_video else None
                    if vid and vid.get("pose_detected"):
                        self.video_monitor.set_baseline(vid)
                    baseline_captured = True

                # Safety checks
                warnings = self.safety.check(snapshot)
                if warnings:
                    self._latest_warnings = [w.to_dict() for w in warnings]

                # Video tension checks
                with self._lock:
                    video_snap = dict(self._latest_video) if self._latest_video else None
                if video_snap:
                    snapshot["video"] = video_snap
                    if not baseline_captured:
                        pass  # video baseline set below with audio baseline
                    video_alerts = self.video_monitor.check(video_snap)
                    if video_alerts:
                        self._latest_warnings.extend(
                            [a.to_dict() for a in video_alerts]
                        )

                # Coaching evaluation (general rules)
                if self.coach is not None:
                    msg = self.coach.evaluate(snapshot)
                    if msg:
                        self._latest_coaching = msg.to_dict()

                # Exercise-specific coaching (overrides general if active)
                if self.exercises is not None:
                    ex_msg = self.exercises.evaluate(snapshot)
                    if ex_msg:
                        self._latest_coaching = ex_msg  # already a dict

            except Exception:
                pass

            time.sleep(1.0)  # Check at 1 Hz

    def get_frame(self) -> dict:
        """Return a snapshot of all current metrics plus safety + coaching."""
        with self._lock:
            frame = dict(self.latest)

        # Attach video features — match by nearest timestamp
        with self._lock:
            audio_ts = frame.get("ts", 0)
            if self._video_ring and audio_ts > 0:
                # Find video frame closest to this audio timestamp
                best_idx = 0
                best_delta = abs(self._video_ring[0][0] - audio_ts)
                for i, (vts, _) in enumerate(self._video_ring):
                    delta = abs(vts - audio_ts)
                    if delta < best_delta:
                        best_delta = delta
                        best_idx = i
                vts, vfeatures = self._video_ring[best_idx]
                frame["video"] = dict(vfeatures)
                frame["video_sync"] = {
                    "audio_ts": audio_ts,
                    "video_ts": vts,
                    "delta_ms": (audio_ts - vts) * 1000,
                }
            elif self._latest_video:
                # Fallback: no ring data yet
                frame["video"] = dict(self._latest_video)

        # Attach any recent safety warnings
        if self._latest_warnings:
            frame["warnings"] = self._latest_warnings
            self._latest_warnings = []  # Clear after delivering
        else:
            frame["warnings"] = []

        # Attach coaching message if available
        if self._latest_coaching:
            frame["coaching"] = self._latest_coaching
            # Accumulate to coaching log with session_t before clearing
            log_entry = dict(self._latest_coaching)
            log_entry["session_t"] = frame.get("session_t", self._total_samples_fed / self.sr)
            self._coaching_log.append(log_entry)
            self._latest_coaching = None  # Clear after delivering
        else:
            frame["coaching"] = None

        # Attach exercise status if active
        if self.exercises is not None and self.exercises.active:
            ex = self.exercises.current_exercise
            exercise_data = {
                "name": ex.name,
                "display_name": ex.display_name,
                "display_mode": ex.display_mode,
                "targets": [
                    {"metric": t.metric, "min": t.min_val, "max": t.max_val,
                     "label": t.label}
                    for t in ex.targets
                ],
            }
            # Include mimicry similarity if mimicry exercise is active
            if ex.name == "mimicry" and self.exercises.mimicry_target:
                sim = self.exercises.mimicry_target.similarity(frame)
                exercise_data["mimicry"] = sim
            frame["exercise"] = exercise_data
        else:
            frame["exercise"] = None

        # Attach workshop data if a workshop session is active
        ws = getattr(self, "_workshop_session", None)
        if ws and ws.is_active:
            frame["workshop"] = ws.evaluate_frame(frame)
        else:
            frame["workshop"] = None

        # Log telemetry (rate-limited internally)
        self._frame_logger.log(frame)

        return frame
