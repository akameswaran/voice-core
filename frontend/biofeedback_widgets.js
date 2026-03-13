/**
 * Biofeedback Widgets — PitchStrip, ZoneGauge, GestureBars
 * ES module, reusable across Practice, Workshop, future coaches.
 *
 * All zone/threshold logic is injected by the consuming app — these
 * widgets contain ZERO domain-specific logic.
 */

// ── PitchStrip ──────────────────────────────────────────────
// Canvas sparkline with scrolling pitch history, Hz readout, zone badge.
export class PitchStrip {
  /**
   * @param {HTMLElement} containerEl
   * @param {Object} [opts]
   * @param {number} [opts.historyLength=60] - number of frames to display
   * @param {Function} [opts.zoneClassifier] - (hz) => { label, cssClass, color }
   *   Apps must supply this to get zone badges and line coloring.
   *   If omitted, no zone badge is shown and the line uses a default color.
   */
  constructor(containerEl, { historyLength = 60, zoneClassifier } = {}) {
    this.container = containerEl;
    this.maxLen = historyLength;
    this.history = [];
    this.zoneClassifier = zoneClassifier || null;

    // Build DOM
    containerEl.classList.add('vc-pitch-strip');
    containerEl.innerHTML = `
      <span class="vc-pitch-hz mono">-- Hz</span>
      <div class="vc-pitch-sparkline-wrap">
        <canvas></canvas>
        <span class="vc-pitch-limit-label max">--</span>
        <span class="vc-pitch-limit-label min">--</span>
      </div>
      <span class="vc-zone-badge">--</span>
    `;
    this.hzEl = containerEl.querySelector('.vc-pitch-hz');
    this.canvas = containerEl.querySelector('canvas');
    this.maxLabel = containerEl.querySelector('.vc-pitch-limit-label.max');
    this.minLabel = containerEl.querySelector('.vc-pitch-limit-label.min');
    this.zoneEl = containerEl.querySelector('.vc-zone-badge');
  }

  pushFrame({ f0_hz, f0_confidence }) {
    if (!f0_hz || f0_hz <= 0 || (f0_confidence !== undefined && f0_confidence < 0.3)) {
      // Unvoiced frame — skip updating value but keep sparkline gap
      return;
    }
    const hz = Math.round(f0_hz);
    this.history.push(hz);
    if (this.history.length > this.maxLen) this.history.shift();

    // Update Hz display
    this.hzEl.textContent = hz + ' Hz';

    // Zone (app-supplied classifier)
    if (this.zoneClassifier) {
      const zone = this.zoneClassifier(hz);
      this.zoneEl.className = `vc-zone-badge ${zone.cssClass || ''}`;
      this.zoneEl.textContent = zone.label;
    }

    this._draw();
  }

  setFrozen(avgHz, zone) {
    this.hzEl.textContent = Math.round(avgHz) + ' Hz';
    if (zone && typeof zone === 'object') {
      this.zoneEl.className = `vc-zone-badge ${zone.cssClass || ''}`;
      this.zoneEl.textContent = zone.label;
    } else if (zone && typeof zone === 'string') {
      this.zoneEl.className = 'vc-zone-badge';
      this.zoneEl.textContent = zone.toUpperCase();
    }
    this.maxLabel.textContent = '';
    this.minLabel.textContent = '';
  }

  clear() {
    this.history = [];
    this.hzEl.textContent = '-- Hz';
    this.zoneEl.className = 'vc-zone-badge';
    this.zoneEl.textContent = '--';
    this.maxLabel.textContent = '--';
    this.minLabel.textContent = '--';
    const ctx = this.canvas.getContext('2d');
    if (ctx) {
      this.canvas.width = this.canvas.offsetWidth;
      this.canvas.height = this.canvas.offsetHeight;
      ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  _draw() {
    const canvas = this.canvas;
    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;
    if (!w || !h) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const data = this.history;
    const minV = Math.min(...data);
    const maxV = Math.max(...data);
    const range = maxV - minV || 10;
    const pad = 4;

    // Midline
    ctx.strokeStyle = '#333336';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h * 0.5);
    ctx.lineTo(w, h * 0.5);
    ctx.stroke();

    // Pitch line — color from zone classifier if available, else default
    const latest = data[data.length - 1];
    let color = '#4ade80'; // default green
    if (this.zoneClassifier) {
      const zone = this.zoneClassifier(latest);
      color = zone.color || color;
    }
    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    data.forEach((val, i) => {
      const x = (i / (this.maxLen - 1)) * w;
      const y = pad + (1 - (val - minV) / range) * (h - pad * 2);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Y-axis labels
    if (data.length > 1) {
      this.maxLabel.textContent = maxV + ' Hz';
      this.minLabel.textContent = minV + ' Hz';
    }
  }
}


// ── ZoneGauge ───────────────────────────────────────────────
// HTML gauge bar with zone coloring, value display, zone badge.
export class ZoneGauge {
  /**
   * @param {HTMLElement} containerEl
   * @param {Object} opts
   * @param {string} opts.label - e.g. "Resonance"
   * @param {string} opts.unit - e.g. "Hz" or "dB"
   * @param {number} opts.min - gauge minimum (e.g. 900)
   * @param {number} opts.max - gauge maximum (e.g. 1400)
   * @param {Array} opts.zones - [{boundary, label, color, cssClass}]
   * @param {string} [opts.scaleMin] - left scale label
   * @param {string} [opts.scaleMax] - right scale label
   * @param {string} [opts.expanderLabel] - if set, adds an expander section
   */
  constructor(containerEl, opts) {
    this.container = containerEl;
    this.opts = opts;

    containerEl.innerHTML = `
      <div class="vc-card-title">${opts.label}</div>
      <div class="vc-gauge-label">
        <span class="vc-gauge-value mono">--</span>
        <span class="vc-zone-badge">--</span>
      </div>
      <div class="vc-gauge-wrap">
        <div class="vc-gauge-fill" style="width:0%"></div>
      </div>
      ${opts.scaleMin || opts.scaleMax ? `
      <div class="vc-gauge-scale">
        <span class="label">${opts.scaleMin || ''}</span>
        <span class="label">${opts.scaleMax || ''}</span>
      </div>` : ''}
      ${opts.expanderLabel ? `
      <div style="margin-top:10px">
        <div class="vc-expander-toggle" data-expander>
          <span class="vc-expander-arrow">&rsaquo;</span> ${opts.expanderLabel}
        </div>
        <div class="vc-expander-body" data-expander-body></div>
      </div>` : ''}
    `;

    this.valueEl = containerEl.querySelector('.vc-gauge-value');
    this.zoneEl = containerEl.querySelector('.vc-zone-badge');
    this.fillEl = containerEl.querySelector('.vc-gauge-fill');

    // Expander wiring
    const toggle = containerEl.querySelector('[data-expander]');
    if (toggle) {
      toggle.addEventListener('click', () => {
        const body = containerEl.querySelector('[data-expander-body]');
        toggle.classList.toggle('open');
        body.classList.toggle('open');
      });
    }
    this.expanderBody = containerEl.querySelector('[data-expander-body]');
  }

  update(value, zoneName) {
    const { min, max, unit, zones } = this.opts;
    const pct = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));

    this.valueEl.textContent = `${typeof value === 'number' ? (Number.isInteger(value) ? value : value.toFixed(1)) : value} ${unit}`;

    // Find zone
    const zone = zoneName || this._findZone(value);
    const zoneInfo = zones.find(z => z.label === zone) || zones[0];
    this.zoneEl.className = `vc-zone-badge ${zoneInfo.cssClass || ''}`;
    this.zoneEl.textContent = zone;

    this.fillEl.style.width = pct + '%';
    this.fillEl.style.background = zoneInfo.color || `var(--${zoneInfo.cssClass || zone})`;
  }

  setFrozen(value, zoneName) {
    this.update(value, zoneName);
  }

  _findZone(value) {
    const zones = this.opts.zones;
    for (let i = zones.length - 1; i >= 0; i--) {
      if (value >= zones[i].boundary) return zones[i].label;
    }
    return zones[0].label;
  }

  /** Set expander body HTML content */
  setExpanderContent(html) {
    if (this.expanderBody) this.expanderBody.innerHTML = html;
  }
}


// ── GestureBars ─────────────────────────────────────────────
// Collapsible gesture z-score bars (used inside gauge expanders or standalone).
export class GestureBars {
  /**
   * @param {HTMLElement} containerEl
   * @param {Object} opts
   * @param {Array} opts.bars - [{id, label}]
   * @param {Function} [opts.colorClassifier] - (pct) => cssColor string
   *   Apps supply this to map percentage values to colors.
   *   If omitted, a neutral gray is used.
   */
  constructor(containerEl, { bars, colorClassifier }) {
    this.container = containerEl;
    this.barDefs = bars;
    this.barEls = {};
    this.colorClassifier = colorClassifier || null;

    containerEl.innerHTML = `
      <div style="display:flex;flex-direction:column;gap:6px;">
        ${bars.map(b => `
          <div>
            <div class="label" style="margin-bottom:3px">${b.label}</div>
            <div class="vc-gauge-wrap">
              <div class="vc-gauge-fill" data-bar="${b.id}" style="width:50%;background:#888"></div>
            </div>
          </div>
        `).join('')}
      </div>
    `;

    bars.forEach(b => {
      this.barEls[b.id] = containerEl.querySelector(`[data-bar="${b.id}"]`);
    });
  }

  /**
   * @param {Object} values - e.g. { larynx: 45, opc: 70, tongue: 55 } (0-100)
   */
  update(values) {
    for (const [id, val] of Object.entries(values)) {
      const el = this.barEls[id];
      if (!el) continue;
      const pct = Math.max(0, Math.min(100, val));
      el.style.width = pct + '%';
      if (this.colorClassifier) {
        el.style.background = this.colorClassifier(pct);
      }
    }
  }
}


// ── StabilityRing ───────────────────────────────────────────
// SVG ring with stroke-dasharray fill driven by stability_pct.
// Confidence controls visibility: <0.2 hidden, 0.2-0.5 outline only, >=0.5 filled.
export class StabilityRing {
  /**
   * @param {HTMLElement} containerEl - Container for SVG ring
   */
  constructor(containerEl) {
    this.container = containerEl;

    // Zone colors from theme (same as ZoneGauge uses)
    this.zoneColors = {
      fem: '#4ade80',   // green
      andro: '#facc15', // yellow
      masc: '#f87171',  // red
    };

    // Build SVG ring
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 120 120');
    svg.setAttribute('class', 'vc-stability-ring-svg');
    svg.style.width = '100%';
    svg.style.height = '100%';
    svg.style.maxWidth = '120px';

    // Background track (full circle outline)
    const bgCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    bgCircle.setAttribute('cx', '60');
    bgCircle.setAttribute('cy', '60');
    bgCircle.setAttribute('r', '50');
    bgCircle.setAttribute('fill', 'none');
    bgCircle.setAttribute('stroke', '#e5e7eb');
    bgCircle.setAttribute('stroke-width', '8');
    svg.appendChild(bgCircle);

    // Fill arc (driven by stability_pct via stroke-dasharray/offset)
    const fillArc = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    fillArc.setAttribute('cx', '60');
    fillArc.setAttribute('cy', '60');
    fillArc.setAttribute('r', '50');
    fillArc.setAttribute('fill', 'none');
    fillArc.setAttribute('stroke', '#4ade80'); // default to fem
    fillArc.setAttribute('stroke-width', '8');
    fillArc.setAttribute('stroke-linecap', 'round');

    // circumference = 2 * π * r = 2 * π * 50 ≈ 314.16
    const circumference = 2 * Math.PI * 50;
    fillArc.setAttribute('stroke-dasharray', circumference);
    fillArc.setAttribute('stroke-dashoffset', circumference); // start at 0 fill

    // Rotate to start from top
    fillArc.setAttribute('transform', 'rotate(-90 60 60)');

    svg.appendChild(fillArc);

    containerEl.appendChild(svg);

    this.svg = svg;
    this.fillArc = fillArc;
    this.circumference = circumference;
    this.lastZone = ''; // track last zone to avoid unnecessary color updates
  }

  /**
   * @param {number|null} stability_pct - 0-100, or null for outline-only
   * @param {string} zone - "masc" | "andro" | "fem" | ""
   * @param {number} confidence - 0-1
   */
  update(stability_pct, zone, confidence) {
    // Hide entirely if confidence too low
    if (confidence < 0.2) {
      this.svg.style.opacity = '0';
      return;
    }

    this.svg.style.opacity = '1';

    // Update color if zone changed
    if (zone !== this.lastZone) {
      const color = this.zoneColors[zone] || '#4ade80';
      this.fillArc.setAttribute('stroke', color);
      this.lastZone = zone;
    }

    // Determine fill based on confidence and stability
    if (confidence < 0.5 || stability_pct === null) {
      // Warming up or no value — show outline only
      this.fillArc.setAttribute('stroke-dashoffset', this.circumference);
    } else {
      // confidence >= 0.5 — fill based on stability_pct
      const pct = Math.max(0, Math.min(100, stability_pct));
      const offset = this.circumference * (1 - pct / 100);
      this.fillArc.setAttribute('stroke-dashoffset', offset);
    }
  }
}
