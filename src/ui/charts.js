/**
 * Charts: Lightweight mathematical HUD overlays for the four data chapters.
 * Animates counters and SVG data curves cleanly when entering/exiting chapters.
 */
import { animate } from 'animejs';
import { METRICS } from '../ledger.js';

export class DataChartsOverlay {
  constructor() {
    this.container = document.getElementById('charts-overlay');
    this.activeChapter = null;
    this.hasAnimated = new Set();

    this.renderCards();
  }

  renderCards() {
    this.container.innerHTML = `
      <!-- 1. Detect Early Card -->
      <div id="chart-detect" class="chart-card">
        <div class="chart-card-header">
          <span class="chart-card-tag">// U1 ANOMALY DETECTION</span>
          <span style="font-family:var(--font-mono);font-size:0.65rem;color:var(--mut);">PCA + ISOLATION FOREST</span>
        </div>
        <div class="chart-metric-val" id="cnt-lead-time">0.0</div>
        <div class="chart-metric-label">Median cycles of lead time to failure (CUSUM onset)</div>
        <svg class="chart-svg-box" viewBox="0 0 380 90">
          <defs>
            <linearGradient id="grad-hi" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#3de0ff" stop-opacity="0.3"/>
              <stop offset="100%" stop-color="#3de0ff" stop-opacity="0.0"/>
            </linearGradient>
          </defs>
          <path d="M 10,20 Q 140,22 190,45 T 370,80 L 370,85 L 10,85 Z" fill="url(#grad-hi)"/>
          <path id="path-hi" d="M 10,20 Q 140,22 190,45 T 370,80" fill="none" stroke="#3de0ff" stroke-width="2.5"/>
          <line x1="190" y1="5" x2="190" y2="85" stroke="#ffb23d" stroke-width="1.5" stroke-dasharray="4 3"/>
          <text x="195" y="18" fill="#ffb23d" font-family="var(--font-mono)" font-size="9" font-weight="600">CUSUM ONSET (CYC 17)</text>
          <text x="370" y="88" fill="var(--mut)" font-family="var(--font-mono)" font-size="8" text-anchor="end">Cycle 200</text>
        </svg>
        <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--mut);">98.9% detected before RUL &le; 80 cycles</div>
      </div>

      <!-- 2. Calibrated Forecast Card -->
      <div id="chart-forecast" class="chart-card">
        <div class="chart-card-header">
          <span class="chart-card-tag">// U2 SPLIT-CONFORMAL UNCERTAINTY</span>
          <span style="font-family:var(--font-mono);font-size:0.65rem;color:var(--mut);">TEST RMSE 16.74 CYC</span>
        </div>
        <div class="chart-metric-val" id="cnt-coverage">0.0%</div>
        <div class="chart-metric-label">Empirical coverage on held-out units (Target 80%)</div>
        <svg class="chart-svg-box" viewBox="0 0 380 90">
          <!-- Fan Band [p10, p90] -->
          <polygon points="10,15 370,55 370,82 10,35" fill="rgba(61, 224, 255, 0.15)"/>
          <!-- Median Point Prediction -->
          <line x1="10" y1="25" x2="370" y2="68" stroke="#3de0ff" stroke-width="2.5"/>
          <!-- True RUL trajectory -->
          <line x1="10" y1="22" x2="370" y2="72" stroke="#ffffff" stroke-width="1.2" stroke-dasharray="3 3"/>
          <text x="15" y="16" fill="var(--cy)" font-family="var(--font-mono)" font-size="9">p90 upper</text>
          <text x="15" y="44" fill="var(--cy)" font-family="var(--font-mono)" font-size="9">p10 lower</text>
          <text x="370" y="65" fill="#ffffff" font-family="var(--font-mono)" font-size="9" text-anchor="end">RUL forecast</text>
        </svg>
        <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--mut);">NASA Asymmetric Score: S = 655.02 (Official FD001)</div>
      </div>

      <!-- 3. Decide with Dollars Card -->
      <div id="chart-decide" class="chart-card">
        <div class="chart-card-header">
          <span class="chart-card-tag" style="color:var(--gn);">// U3 COST-OPTIMAL POLICY</span>
          <span style="font-family:var(--font-mono);font-size:0.65rem;color:var(--gn);">-84.4% REDUCTION</span>
        </div>
        <div class="chart-metric-val" style="color:var(--gn);" id="cnt-savings">$0</div>
        <div class="chart-metric-label">Total fleet dollars saved vs reactive run-to-failure</div>
        <svg class="chart-svg-box" viewBox="0 0 380 90">
          <!-- Reactive Bar -->
          <rect x="30" y="10" width="130" height="45" rx="4" fill="#ff4d4d" opacity="0.8"/>
          <text x="40" y="36" fill="#ffffff" font-family="var(--font-mono)" font-size="12" font-weight="700">REACTIVE: $35.0M</text>
          <!-- Predictive Bar -->
          <rect x="190" y="35" width="160" height="20" rx="4" fill="#10b981" opacity="0.9"/>
          <text x="200" y="49" fill="#05070b" font-family="var(--font-mono)" font-size="10" font-weight="700">PREDICTIVE: $5.45M</text>
        </svg>
        <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--mut);">OR-Tools CP-SAT Hangar Scheduler: 0 Days Tardy</div>
      </div>

      <!-- 4. Agent & Ledger Card -->
      <div id="chart-agent" class="chart-card">
        <div class="chart-card-header">
          <span class="chart-card-tag">// U4 MULTI-AGENT & U9 LEDGER</span>
          <span style="font-family:var(--font-mono);font-size:0.65rem;color:var(--cy);">SHA-256 VERIFIED</span>
        </div>
        <div style="font-family:var(--font-mono);font-size:1.1rem;font-weight:700;color:var(--ink);margin-bottom:4px;">
          7-Node LangGraph + Ledger
        </div>
        <div class="chart-metric-label">Numeric Critic Verification + Human Approval Gate</div>
        <div class="pipeline-nodes">
          <div class="node-chip lit" id="node-0">TRIAGE</div>
          <div class="node-chip lit" id="node-1">DIAG</div>
          <div class="node-chip lit" id="node-2">PLAN</div>
          <div class="node-chip lit" id="node-3">CRITIC</div>
          <div class="node-chip lit" id="node-4">GATE</div>
          <div class="node-chip lit" id="node-5">DISPATCH</div>
          <div class="node-chip lit" id="node-6">REPORT</div>
        </div>
        <div style="font-family:var(--font-mono);font-size:0.62rem;color:var(--mut);margin-top:10px;">
          Chain: <span style="color:var(--gn);">0000..f92a</span> &rarr; <span style="color:var(--gn);">1a4e..61b8</span> &rarr; <span style="color:var(--gn);">d48c..0a72</span> [VALID]
        </div>
      </div>
    `;

    this.cardDetect = document.getElementById('chart-detect');
    this.cardForecast = document.getElementById('chart-forecast');
    this.cardDecide = document.getElementById('chart-decide');
    this.cardAgent = document.getElementById('chart-agent');

    this.cntLeadTime = document.getElementById('cnt-lead-time');
    this.cntCoverage = document.getElementById('cnt-coverage');
    this.cntSavings = document.getElementById('cnt-savings');
  }

  update(chapter) {
    if (this.activeChapter === chapter.id) return;
    this.activeChapter = chapter.id;

    // Toggle card visibility
    this.cardDetect.classList.toggle('visible', chapter.id === 'detect');
    this.cardForecast.classList.toggle('visible', chapter.id === 'forecast');
    this.cardDecide.classList.toggle('visible', chapter.id === 'decide');
    this.cardAgent.classList.toggle('visible', chapter.id === 'dawn');

    // Trigger metric counters on entry
    if (chapter.id === 'detect') {
      const obj = { val: 0 };
      animate(obj, {
        val: METRICS.medianLeadTimeCycles,
        duration: 1200,
        ease: 'outExpo',
        onUpdate: () => {
          this.cntLeadTime.textContent = obj.val.toFixed(1);
        },
      });
    } else if (chapter.id === 'forecast') {
      const obj = { val: 0 };
      animate(obj, {
        val: METRICS.empiricalCoveragePct,
        duration: 1200,
        ease: 'outExpo',
        onUpdate: () => {
          this.cntCoverage.textContent = `${obj.val.toFixed(1)}%`;
        },
      });
    } else if (chapter.id === 'decide') {
      const obj = { val: 0 };
      animate(obj, {
        val: METRICS.savingsUsd,
        duration: 1400,
        ease: 'outExpo',
        onUpdate: () => {
          this.cntSavings.textContent = `$${Math.round(obj.val).toLocaleString()}`;
        },
      });
    }
  }
}
