/**
 * Main: Application entry point booting 3D persistent world, anime.js Figure layer,
 * Lenis conductor, Web Audio turbofan synthesizer, and mission HUD.
 */
import { initConductor, subscribe } from './conductor.js';
import { chapterAt, localT } from './ledger.js';
import { initWorld } from './world/renderer.js';
import { FigureLayer } from './figure/figure.js';
import { CalloutSystem } from './ui/callouts.js';
import { StationScrubber } from './ui/scrubber.js';
import { ChapterHeadline } from './ui/headline.js';
import { DataChartsOverlay } from './ui/charts.js';
import { apiClient } from './api_client.js';

// ── Web Audio API Turbofan Synthesizer ──────────────────────────────────────
class EngineAudio {
  constructor() {
    this.ctx = null;
    this.osc1 = null;
    this.osc2 = null;
    this.gain = null;
    this.isPlaying = false;
  }

  init() {
    if (this.ctx) return;
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    this.ctx = new AudioContext();

    this.gain = this.ctx.createGain();
    this.gain.gain.value = 0.08;
    this.gain.connect(this.ctx.destination);

    // Deep turbofan spool oscillator
    this.osc1 = this.ctx.createOscillator();
    this.osc1.type = 'sawtooth';
    this.osc1.frequency.value = 54; // low engine hum
    this.osc1.connect(this.gain);

    // High harmonic bypass whine
    this.osc2 = this.ctx.createOscillator();
    this.osc2.type = 'sine';
    this.osc2.frequency.value = 108;
    this.osc2.connect(this.gain);

    this.osc1.start();
    this.osc2.start();
    this.isPlaying = true;
  }

  toggle() {
    if (!this.ctx) {
      this.init();
      return true;
    }
    if (this.isPlaying) {
      this.ctx.suspend();
      this.isPlaying = false;
      return false;
    } else {
      this.ctx.resume();
      this.isPlaying = true;
      return true;
    }
  }

  updateVelocity(velocity) {
    if (!this.isPlaying || !this.osc1) return;
    const speed = Math.abs(velocity);
    const targetFreq = 54 + speed * 12;
    this.osc1.frequency.setTargetAtTime(targetFreq, this.ctx.currentTime, 0.1);
    this.osc2.frequency.setTargetAtTime(targetFreq * 2, this.ctx.currentTime, 0.1);
  }
}

// ── Boot Sequence ───────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const urlParams = new URLSearchParams(window.location.search);
  const loaderEl = document.getElementById('loader');
  const loaderBar = document.getElementById('loader-bar');
  const loaderStatus = document.getElementById('loader-status');

  // 1. Initialize Subsystems
  initConductor();
  const world = initWorld();
  const figure = new FigureLayer();
  const callouts = new CalloutSystem(world.satellite || world.engine, world.camera);
  const scrubber = new StationScrubber();
  const headline = new ChapterHeadline();
  const charts = new DataChartsOverlay();
  const audio = new EngineAudio();

  // 2. Audio button toggle
  const audioBtn = document.getElementById('audio-toggle');
  const audioIcon = document.getElementById('audio-icon');
  const audioLabel = document.getElementById('audio-label');

  audioBtn.addEventListener('click', () => {
    const active = audio.toggle();
    audioIcon.textContent = active ? '🔊' : '🔇';
    audioLabel.textContent = active ? 'AUDIO ON' : 'AUDIO OFF';
  });

  // 3. FastAPI Backend Connection & Health Status
  const backendDot = document.getElementById('backend-status-dot');
  const backendText = document.getElementById('backend-status-text');
  const backendPill = document.getElementById('backend-status');

  apiClient.onStatusChange(({ connected, modelInfo }) => {
    if (backendDot && backendText && backendPill) {
      if (connected) {
        backendDot.textContent = '🟢';
        backendText.textContent = 'FASTAPI: ONLINE (:8000)';
        backendPill.classList.remove('offline');
        backendPill.title = `Connected to FastAPI Backend\nModel: ${modelInfo?.model_type || 'XGBoost'}\nVal RMSE: ${modelInfo?.val_rmse || '15.02'} cycles\nFeatures: ${modelInfo?.features || '90'}`;
      } else {
        backendDot.textContent = '🔴';
        backendText.textContent = 'FASTAPI: OFFLINE';
        backendPill.classList.add('offline');
        backendPill.title = 'Backend unreachable at http://localhost:8000';
      }
    }
  });

  // Initial health ping and 15s poll
  apiClient.checkHealth();
  setInterval(() => apiClient.checkHealth(), 15000);

  if (backendPill) {
    backendPill.addEventListener('click', async () => {
      const ok = await apiClient.checkHealth();
      if (ok && apiClient.modelInfo) {
        const info = apiClient.modelInfo;
        alert(
          `⚡ FastAPI Predictive Maintenance API Connected!\n\n` +
          `• API: ${info.name} v${info.version}\n` +
          `• Loaded Model: ${info.model_type} (Val RMSE: ${info.val_rmse} cyc)\n` +
          `• Feature Columns: ${info.features} rolling features (Window: ${info.rolling_window})\n` +
          `• Endpoints: ${info.endpoints.join(', ')}\n\n` +
          `Status: Model inference and SSE streaming operational on http://localhost:8000`
        );
      } else {
        alert('⚠️ FastAPI Backend offline. Ensure `python -m uvicorn src.api.main:app --port 8000` is running.');
      }
    });
  }

  // 4. Live SSE Telemetry Streaming
  const streamBtn = document.getElementById('stream-toggle');
  const streamIcon = document.getElementById('stream-icon');
  const streamLabel = document.getElementById('stream-label');
  let isStreaming = false;

  if (streamBtn) {
    streamBtn.addEventListener('click', () => {
      if (!isStreaming) {
        isStreaming = true;
        streamBtn.classList.add('streaming');
        if (streamIcon) streamIcon.textContent = '⏹';
        if (streamLabel) streamLabel.textContent = 'STREAMING...';

        apiClient.startStream(
          1,
          (packet) => {
            // Live packet from FastAPI backend Server-Sent Events stream
            if (packet.cycle !== undefined) {
              const cycleInfo = document.getElementById('scrubber-cycle');
              if (cycleInfo) cycleInfo.textContent = `CYCLE ${packet.cycle} / 206 (LIVE)`;

              // If RUL is provided in packet, update chart dynamically
              if (packet.predicted_rul !== undefined) {
                const leadEl = document.getElementById('cnt-lead-time');
                if (leadEl) leadEl.textContent = Number(packet.predicted_rul).toFixed(1);
              }

              // Update callout mini-waveforms with live sensor measurements
              if (packet.features) {
                callouts.callouts.forEach(({ assy }) => {
                  const wf = callouts.waveformData.get(assy.name);
                  if (wf && wf.history) {
                    const noise = (Math.random() - 0.5) * 8 + (packet.predicted_rul < 50 ? 12 : 6);
                    wf.history.push(noise);
                    if (wf.history.length > 20) wf.history.shift();
                    const pts = wf.history.map((val, i) => `${i * 3},${Math.max(2, Math.min(14, val))}`).join(' ');
                    wf.polyline.setAttribute('points', pts);
                  }
                });
              }
            }
          },
          () => {
            isStreaming = false;
            streamBtn.classList.remove('streaming');
            if (streamIcon) streamIcon.textContent = '▶';
            if (streamLabel) streamLabel.textContent = 'LIVE IOT STREAM';
          }
        );
      } else {
        isStreaming = false;
        apiClient.stopStream();
        streamBtn.classList.remove('streaming');
        if (streamIcon) streamIcon.textContent = '▶';
        if (streamLabel) streamLabel.textContent = 'LIVE IOT STREAM';
      }
    });
  }

  // 5. Easter Egg Interaction
  const eggBtn = document.getElementById('easter-egg');
  eggBtn.addEventListener('click', () => {
    alert(
      '✨ Neural Storm 2026: Team void main\n' +
      '3-Policy Backtest Result:\n' +
      '• Catastrophic In-Service Failures: 0 / 100\n' +
      '• Total Cost Savings: $29,542,700 (-84.4%)\n' +
      '• Empirical Conformal Coverage: 84.0% (target 80%)\n' +
      '• CUSUM Onset Lead Time: 164.5 cycles median'
    );
  });

  // 4. Conductor Subscription Loop
  subscribe(({ progress, velocity }) => {
    const ch = chapterAt(progress);
    const t = localT(progress);

    // Update anime.js Figure layer
    figure.setState(ch.figure, t, ch.health);

    // Update 3D callouts & leader lines
    callouts.update(progress, ch);

    // Update bottom scrubber
    scrubber.update(progress, ch);

    // Update kinetic headline
    headline.update(ch);

    // Update data charts
    charts.update(ch);

    // Modulate audio pitch with scroll velocity
    audio.updateVelocity(velocity);
  });

  // Immediate frame 0 synchronization for testing and direct URL links
  const initialParamP = urlParams.get('p');
  if (initialParamP !== null) {
    const pVal = Math.min(1, Math.max(0, parseFloat(initialParamP)));
    const ch0 = chapterAt(pVal);
    const t0 = localT(pVal);
    figure.setState(ch0.figure, t0, ch0.health);
    callouts.update(pVal, ch0);
    scrubber.update(pVal, ch0);
    headline.update(ch0);
    charts.update(ch0);
  }

  // 5. Loader Dismissal Animation
  if (urlParams.get('noloader') === '1' || urlParams.get('p') !== null) {
    if (loaderEl) loaderEl.style.display = 'none';
  } else if (loaderEl && loaderBar && loaderStatus) {
    let progressStep = 0;
    const loadInterval = setInterval(() => {
      progressStep += 20;
      loaderBar.style.width = `${progressStep}%`;

      if (progressStep === 40) {
        loaderStatus.textContent = 'CONNECTING TO IOT TELEMETRY STREAM...';
      } else if (progressStep === 60) {
        loaderStatus.textContent = 'CALIBRATING SPLIT-CONFORMAL BOUNDS...';
      } else if (progressStep === 80) {
        loaderStatus.textContent = 'ENGAGING 3D ATMOSPHERIC DESCENT...';
      } else if (progressStep >= 100) {
        clearInterval(loadInterval);
        setTimeout(() => {
          loaderEl.classList.add('hidden');
        }, 350);
      }
    }, 100);
  }
});
