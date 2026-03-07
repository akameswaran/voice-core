// ─── Video Feature Extraction ────────────────────────────────────────────────
// ES module: MediaPipe face/pose landmark detection, feature computation,
// calibration, and WebSocket streaming to server.
//
// Runs entirely in the browser via @mediapipe/tasks-vision WASM+GPU.
// Only small feature vectors (~1KB JSON at ~15Hz) cross the wire.

const MEDIAPIPE_WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm';
const FACE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const POSE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

// Landmark indices (MediaPipe canonical face mesh)
const LM = {
    NOSE_TIP: 1,
    FOREHEAD: 10,
    CHIN: 152,
    LEFT_EYE_OUTER: 33,
    RIGHT_EYE_OUTER: 263,
    LEFT_LIP_CORNER: 61,
    RIGHT_LIP_CORNER: 291,
    UPPER_LIP_INNER: 13,
    LOWER_LIP_INNER: 14,
    LEFT_INNER_BROW: 66,
    RIGHT_INNER_BROW: 107,
};

// Pose landmark indices
const POSE = {
    LEFT_EAR: 7,
    RIGHT_EAR: 8,
    LEFT_SHOULDER: 11,
    RIGHT_SHOULDER: 12,
    LEFT_HIP: 23,
    RIGHT_HIP: 24,
};

// ─── Utility ────────────────────────────────────────────────────────────────

function dist2d(a, b) {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
}

function mid(a, b) {
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

// ─── VideoFeatureExtractor ──────────────────────────────────────────────────

export class VideoFeatureExtractor {
    constructor() {
        this._vision = null;           // @mediapipe/tasks-vision module
        this._faceLandmarker = null;
        this._poseLandmarker = null;
        this._video = null;            // HTMLVideoElement
        this._stream = null;           // MediaStream
        this._running = false;
        this._frameCount = 0;
        this._animId = null;
        this._ws = null;               // WebSocket for streaming
        this._lastSendTime = 0;
        this._sendInterval = 66;       // ~15 Hz

        // Latest computed features
        this._features = {
            face_detected: false,
            pose_detected: false,
            lip_spread_ratio: 0,
            lip_width_norm: 0,
            lip_aperture_norm: 0,
            jaw_opening_norm: 0,
            head_pitch_deg: 0,
            brow_furrow_score: 0,
            shoulder_elevation_delta: 0,
            shoulder_asymmetry: 0,
            forward_head_offset: 0,
            tension_composite: 0,
        };

        // Calibration data
        this._calibration = null;
        this._calibrating = false;
    }

    // ── Init: load MediaPipe WASM models ──────────────────────────────────

    async init() {
        // Dynamic import of the tasks-vision module from CDN
        const { FaceLandmarker, PoseLandmarker, FilesetResolver } = await import(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18'
        );

        this._vision = { FaceLandmarker, PoseLandmarker, FilesetResolver };

        const filesetResolver = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_BASE);

        this._faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
                modelAssetPath: FACE_MODEL_URL,
                delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            numFaces: 1,
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: false,
        });

        this._poseLandmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
                modelAssetPath: POSE_MODEL_URL,
                delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            numPoses: 1,
        });

        console.log('[video] MediaPipe models loaded');
    }

    // ── Camera start/stop ─────────────────────────────────────────────────

    async startCamera(deviceId) {
        if (this._running) return;

        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 30 },
            },
        };
        if (deviceId) {
            constraints.video.deviceId = { exact: deviceId };
        }

        this._stream = await navigator.mediaDevices.getUserMedia(constraints);

        this._video = document.createElement('video');
        this._video.srcObject = this._stream;
        this._video.setAttribute('autoplay', '');
        this._video.setAttribute('muted', '');
        this._video.setAttribute('playsinline', '');
        await this._video.play();

        this._running = true;
        this._frameCount = 0;
        this._lastSendTime = 0;
        this._detectLoop();

        console.log('[video] Camera started');
    }

    stopCamera() {
        this._running = false;

        if (this._animId) {
            cancelAnimationFrame(this._animId);
            this._animId = null;
        }

        if (this._stream) {
            this._stream.getTracks().forEach(t => t.stop());
            this._stream = null;
        }

        if (this._video) {
            this._video.pause();
            this._video.srcObject = null;
        }

        this._features.face_detected = false;
        this._features.pose_detected = false;

        console.log('[video] Camera stopped');
    }

    // ── Detection loop ────────────────────────────────────────────────────

    _detectLoop() {
        if (!this._running || !this._video) return;

        this._animId = requestAnimationFrame(() => this._detectLoop());
        this._frameCount++;

        // Run detection every 2nd frame (~15 Hz on 30fps)
        if (this._frameCount % 2 !== 0) return;
        if (this._video.readyState < 2) return; // HAVE_CURRENT_DATA

        const now = performance.now();

        // Face landmarks
        let faceLandmarks = null;
        try {
            const faceResult = this._faceLandmarker.detectForVideo(this._video, now);
            if (faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0) {
                faceLandmarks = faceResult.faceLandmarks[0];
            }
        } catch (e) {
            // Detection can fail on some frames
        }

        // Pose landmarks
        let poseLandmarks = null;
        try {
            const poseResult = this._poseLandmarker.detectForVideo(this._video, now);
            if (poseResult.landmarks && poseResult.landmarks.length > 0) {
                poseLandmarks = poseResult.landmarks[0];
            }
        } catch (e) {
            // Detection can fail on some frames
        }

        this._computeFeatures(faceLandmarks, poseLandmarks);

        // Stream to server via WebSocket (with browser timestamp for A/V sync)
        if (this._ws && this._ws.readyState === WebSocket.OPEN) {
            if (now - this._lastSendTime >= this._sendInterval) {
                this._features.client_ts = now;  // performance.now() ms
                this._ws.send(JSON.stringify(this._features));
                this._lastSendTime = now;
            }
        }
    }

    // ── Feature computation ───────────────────────────────────────────────

    _computeFeatures(faceLandmarks, poseLandmarks) {
        this._features.face_detected = faceLandmarks !== null;
        this._features.pose_detected = poseLandmarks !== null;

        if (faceLandmarks) {
            this._computeFaceFeatures(faceLandmarks);
        }

        if (poseLandmarks) {
            this._computePoseFeatures(poseLandmarks);
        }
    }

    _computeFaceFeatures(lm) {
        // Inter-pupillary distance (IPD) as normalization reference
        const leftEye = lm[LM.LEFT_EYE_OUTER];
        const rightEye = lm[LM.RIGHT_EYE_OUTER];
        const ipd = dist2d(leftEye, rightEye);
        if (ipd < 0.001) return; // Too small / invalid

        // Lip spread ratio: width / height
        const leftCorner = lm[LM.LEFT_LIP_CORNER];
        const rightCorner = lm[LM.RIGHT_LIP_CORNER];
        const upperLip = lm[LM.UPPER_LIP_INNER];
        const lowerLip = lm[LM.LOWER_LIP_INNER];

        const lipWidth = dist2d(leftCorner, rightCorner);
        const lipAperture = dist2d(upperLip, lowerLip);
        this._features.lip_spread_ratio = lipWidth / Math.max(lipAperture, 0.001);
        this._features.lip_width_norm = lipWidth / ipd;
        this._features.lip_aperture_norm = lipAperture / ipd;

        // Jaw opening: chin-to-nose distance normalized by IPD
        const chin = lm[LM.CHIN];
        const nose = lm[LM.NOSE_TIP];
        const forehead = lm[LM.FOREHEAD];

        // Pose correction: project chin-nose distance along face vertical axis
        // to compensate for head tilt
        const faceVertX = nose.x - forehead.x;
        const faceVertY = nose.y - forehead.y;
        const faceVertLen = Math.sqrt(faceVertX * faceVertX + faceVertY * faceVertY);
        if (faceVertLen > 0.001) {
            // Unit face-vertical vector
            const uvx = faceVertX / faceVertLen;
            const uvy = faceVertY / faceVertLen;
            // Project chin-nose vector onto face vertical
            const cnx = chin.x - nose.x;
            const cny = chin.y - nose.y;
            const projectedDist = Math.abs(cnx * uvx + cny * uvy);
            const rawJaw = projectedDist / ipd;

            // Always log raw value for analysis (calibration-independent)
            this._features.jaw_opening_raw = rawJaw;

            if (this._calibration) {
                const range = this._calibration.jawOpenRef - this._calibration.jawClosedRef;
                this._features.jaw_opening_norm = range > 0.01
                    ? Math.max(0, Math.min(1, (rawJaw - this._calibration.jawClosedRef) / range))
                    : rawJaw;
            } else {
                this._features.jaw_opening_norm = rawJaw;
            }
        }

        // Head pitch: angle of nose-forehead line relative to vertical
        this._features.head_pitch_deg = Math.atan2(
            nose.x - forehead.x,
            nose.y - forehead.y
        ) * (180 / Math.PI);

        // Brow furrow: inner brow distance vs baseline
        const leftBrow = lm[LM.LEFT_INNER_BROW];
        const rightBrow = lm[LM.RIGHT_INNER_BROW];
        const browDist = dist2d(leftBrow, rightBrow) / ipd;
        if (this._calibration && this._calibration.browBaselineNorm > 0) {
            // Lower ratio = more furrowed (brows closer together)
            const ratio = browDist / this._calibration.browBaselineNorm;
            this._features.brow_furrow_score = Math.max(0, Math.min(1, 1 - ratio));
        } else {
            this._features.brow_furrow_score = 0;
        }
    }

    _computePoseFeatures(lm) {
        const leftShoulder = lm[POSE.LEFT_SHOULDER];
        const rightShoulder = lm[POSE.RIGHT_SHOULDER];
        const leftEar = lm[POSE.LEFT_EAR];
        const rightEar = lm[POSE.RIGHT_EAR];
        const leftHip = lm[POSE.LEFT_HIP];
        const rightHip = lm[POSE.RIGHT_HIP];

        // Torso length for normalization
        const shoulderMid = mid(leftShoulder, rightShoulder);
        const hipMid = mid(leftHip, rightHip);
        const torsoLen = dist2d(shoulderMid, hipMid);
        if (torsoLen < 0.01) return;

        // Shoulder elevation delta from baseline
        const shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
        if (this._calibration && this._calibration.shoulderBaselineY !== null) {
            // Negative y change = shoulders moved UP (in screen coords, y increases downward)
            const delta = this._calibration.shoulderBaselineY - shoulderY;
            this._features.shoulder_elevation_delta = delta / torsoLen;
        } else {
            this._features.shoulder_elevation_delta = 0;
        }

        // Shoulder asymmetry
        this._features.shoulder_asymmetry = Math.abs(leftShoulder.y - rightShoulder.y) / torsoLen;

        // Forward head offset: ear midpoint X vs shoulder midpoint X
        const earMid = mid(leftEar, rightEar);
        if (this._calibration && this._calibration.forwardHeadBaselineOffset !== null) {
            const currentOffset = earMid.x - shoulderMid.x;
            this._features.forward_head_offset = (currentOffset - this._calibration.forwardHeadBaselineOffset) / torsoLen;
        } else {
            this._features.forward_head_offset = 0;
        }

        // Tension composite
        this._features.tension_composite = Math.max(0, Math.min(1,
            Math.abs(this._features.shoulder_elevation_delta) * 0.5
            + this._features.shoulder_asymmetry * 0.2
            + Math.abs(this._features.forward_head_offset) * 0.3
        ));
    }

    // ── Calibration ───────────────────────────────────────────────────────

    /**
     * Run a 3-step calibration sequence.
     * @param {function} onStep - Called with {step, total, instruction, countdown}
     * @param {function} onDone - Called when calibration complete
     */
    startCalibration(onStep, onDone) {
        if (!this._running || this._calibrating) return;
        this._calibrating = true;

        const steps = [
            { instruction: 'Sit relaxed, look at camera', duration: 3000 },
            { instruction: 'Close your mouth naturally', duration: 2000 },
            { instruction: 'Open your mouth wide', duration: 2000 },
        ];

        const calibData = {
            shoulderBaselineY: null,
            browBaselineNorm: null,
            forwardHeadBaselineOffset: null,
            jawClosedRef: 0,
            jawOpenRef: 1,
        };

        let stepIdx = 0;

        const runStep = () => {
            if (stepIdx >= steps.length) {
                // Validate jaw range before saving calibration
                const jawRange = calibData.jawOpenRef - calibData.jawClosedRef;
                if (jawRange < 0.05) {
                    this._calibrating = false;
                    if (onStep) onStep({
                        step: 3, total: 3,
                        instruction: 'Not enough jaw movement detected — please try again',
                        countdown: 0,
                        error: true,
                    });
                    // Don't save calibration, don't call onDone
                    return;
                }

                // Valid — save and call onDone
                this._calibration = calibData;
                this._calibrating = false;
                // Save to localStorage
                try {
                    localStorage.setItem('videoCalibration', JSON.stringify(calibData));
                } catch (e) { /* ignore */ }
                if (onDone) onDone(calibData);
                return;
            }

            const step = steps[stepIdx];
            const samples = [];
            const sampleInterval = 100; // collect samples every 100ms
            const totalSamples = Math.floor(step.duration / sampleInterval);
            let collected = 0;

            if (onStep) onStep({
                step: stepIdx + 1,
                total: steps.length,
                instruction: step.instruction,
                countdown: step.duration / 1000,
            });

            // Countdown updates
            let remaining = step.duration;
            const countdownTimer = setInterval(() => {
                remaining -= 200;
                if (remaining > 0 && onStep) {
                    onStep({
                        step: stepIdx + 1,
                        total: steps.length,
                        instruction: step.instruction,
                        countdown: Math.max(0, remaining / 1000),
                    });
                }
            }, 200);

            const collectTimer = setInterval(() => {
                samples.push({ ...this._features });
                collected++;
                if (collected >= totalSamples) {
                    clearInterval(collectTimer);
                    clearInterval(countdownTimer);
                    this._processCalibStep(stepIdx, samples, calibData);
                    stepIdx++;
                    setTimeout(runStep, 300); // brief pause between steps
                }
            }, sampleInterval);
        };

        runStep();
    }

    _processCalibStep(stepIdx, samples, calibData) {
        // Filter to samples where we have detection
        const faceSamples = samples.filter(s => s.face_detected);
        const poseSamples = samples.filter(s => s.pose_detected);

        if (stepIdx === 0) {
            // "Sit relaxed" — capture baseline shoulders, brow, forward-head
            if (poseSamples.length > 0) {
                calibData.shoulderBaselineY = avg(poseSamples.map(s => s._raw_shoulder_y || 0));
                calibData.forwardHeadBaselineOffset = avg(poseSamples.map(s => s._raw_forward_offset || 0));

                // Get raw shoulder Y from the current feature computation
                // We'll use shoulder_elevation_delta=0 as baseline reference
                // Store the raw shoulder Y directly
                calibData.shoulderBaselineY = this._getRawShoulderY();
                calibData.forwardHeadBaselineOffset = this._getRawForwardOffset();
            }
            if (faceSamples.length > 0) {
                calibData.browBaselineNorm = avg(faceSamples.map(s => s._raw_brow_dist || 0));
                calibData.browBaselineNorm = this._getRawBrowDist();
            }
        } else if (stepIdx === 1) {
            // "Close mouth" — jaw closed reference (use raw to avoid clamping from old calibration)
            if (faceSamples.length > 0) {
                calibData.jawClosedRef = avg(faceSamples.map(s => s.jaw_opening_raw ?? s.jaw_opening_norm));
            }
        } else if (stepIdx === 2) {
            // "Open mouth wide" — jaw open reference (use raw to avoid clamping from old calibration)
            if (faceSamples.length > 0) {
                calibData.jawOpenRef = avg(faceSamples.map(s => s.jaw_opening_raw ?? s.jaw_opening_norm));
            }
        }
    }

    // Raw feature accessors for calibration baseline capture
    _getRawShoulderY() {
        if (!this._video || this._video.readyState < 2) return null;
        try {
            const now = performance.now();
            const poseResult = this._poseLandmarker.detectForVideo(this._video, now + 1);
            if (poseResult.landmarks && poseResult.landmarks.length > 0) {
                const lm = poseResult.landmarks[0];
                return (lm[POSE.LEFT_SHOULDER].y + lm[POSE.RIGHT_SHOULDER].y) / 2;
            }
        } catch (e) { /* ignore */ }
        return null;
    }

    _getRawForwardOffset() {
        if (!this._video || this._video.readyState < 2) return null;
        try {
            const now = performance.now();
            const poseResult = this._poseLandmarker.detectForVideo(this._video, now + 2);
            if (poseResult.landmarks && poseResult.landmarks.length > 0) {
                const lm = poseResult.landmarks[0];
                const earMid = mid(lm[POSE.LEFT_EAR], lm[POSE.RIGHT_EAR]);
                const shoulderMid = mid(lm[POSE.LEFT_SHOULDER], lm[POSE.RIGHT_SHOULDER]);
                return earMid.x - shoulderMid.x;
            }
        } catch (e) { /* ignore */ }
        return null;
    }

    _getRawBrowDist() {
        if (!this._video || this._video.readyState < 2) return null;
        try {
            const now = performance.now();
            const faceResult = this._faceLandmarker.detectForVideo(this._video, now + 3);
            if (faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0) {
                const lm = faceResult.faceLandmarks[0];
                const ipd = dist2d(lm[LM.LEFT_EYE_OUTER], lm[LM.RIGHT_EYE_OUTER]);
                if (ipd > 0.001) {
                    return dist2d(lm[LM.LEFT_INNER_BROW], lm[LM.RIGHT_INNER_BROW]) / ipd;
                }
            }
        } catch (e) { /* ignore */ }
        return null;
    }

    /**
     * Load saved calibration from localStorage.
     * @returns {boolean} true if calibration was loaded
     */
    loadSavedCalibration() {
        try {
            const saved = localStorage.getItem('videoCalibration');
            if (saved) {
                this._calibration = JSON.parse(saved);
                return true;
            }
        } catch (e) { /* ignore */ }
        return false;
    }

    // ── Public API ────────────────────────────────────────────────────────

    getLatestFeatures() {
        return { ...this._features };
    }

    isReady() {
        return (
            this._faceLandmarker !== null
            && this._poseLandmarker !== null
            && this._running
            && this._calibration !== null
        );
    }

    isInitialized() {
        return this._faceLandmarker !== null && this._poseLandmarker !== null;
    }

    isCameraActive() {
        return this._running;
    }

    isCalibrated() {
        return this._calibration !== null;
    }

    setWebSocket(ws) {
        this._ws = ws;
    }

    getVideoElement() {
        return this._video;
    }

    getMediaStream() {
        return this._stream;
    }
}

// ── Utility ─────────────────────────────────────────────────────────────────

function avg(arr) {
    if (!arr.length) return 0;
    const valid = arr.filter(v => v !== null && v !== undefined && !isNaN(v));
    if (!valid.length) return 0;
    return valid.reduce((a, b) => a + b, 0) / valid.length;
}
