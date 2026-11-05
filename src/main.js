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

  // 3. Easter Egg Interaction
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
