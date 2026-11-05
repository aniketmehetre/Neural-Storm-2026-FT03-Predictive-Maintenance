/**
 * Ledger: The authoritative chapter scroll map and mission ledger.
 * Maps normalized progress (0 to 1) to camera coords, sky health, headlines, and data.
 */

export const CHAPTERS = [
  {
    id: 'boarding',
    from: 0.00,
    to: 0.10,
    badge: 'CH 01 // BOARDING',
    headline: 'Hear the engine before it fails.',
    desc: 'High-altitude stratospheric cruise. Baseline acoustic and thermal stability. 100% health.',
    health: 1.0,
    camera: { pos: [0, 6, 26], look: [0, 0, 0] },
    explode: 0.0,
    heat: 0.0,
    figure: 'rest',
  },
  {
    id: 'dive',
    from: 0.10,
    to: 0.25,
    badge: 'CH 02 // THE DIVE',
    headline: 'Six assemblies. Fifteen sensors.',
    desc: 'Approaching the nacelle inlet. Subsystem telemetry ingested into 30-cycle causal buffers.',
    health: 0.95,
    camera: { pos: [0, 2.2, 13], look: [0, 0, 0] },
    explode: 0.05,
    heat: 0.05,
    figure: 'tighten',
  },
  {
    id: 'anatomy',
    from: 0.25,
    to: 0.50,
    badge: 'CH 03 // ANATOMY',
    headline: 'The anatomy of propulsion.',
    desc: 'Turbofan exploded along shaft axis: Fan, LPC, HPC, Combustor, HPT, and LPT. Physical telemetry active.',
    health: 0.85,
    camera: { pos: [5, 3.2, 11], look: [0, 0, 0] },
    explode: 1.0,
    heat: 0.25,
    figure: 'follow',
  },
  {
    id: 'detect',
    from: 0.50,
    to: 0.65,
    badge: 'CH 04 // DETECT EARLY',
    headline: 'The health index bends before the engine does.',
    desc: 'PCA + Isolation Forest fusion with CUSUM tracking. 164.5 median cycles lead time before failure.',
    health: 0.60,
    camera: { pos: [-2.5, 1.8, 9.5], look: [-0.6, 0.2, 0] },
    explode: 0.55,
    heat: 0.65,
    figure: 'curve',
  },
  {
    id: 'forecast',
    from: 0.65,
    to: 0.78,
    badge: 'CH 05 // CALIBRATED FORECAST',
    headline: 'Honest uncertainty, measured.',
    desc: 'Split-conformal calibrated bounds [p10, p90]. 84.0% empirical coverage verified on held-out units.',
    health: 0.40,
    camera: { pos: [0.5, 1.0, 7.5], look: [0, 0, 0] },
    explode: 0.35,
    heat: 0.85,
    figure: 'band',
  },
  {
    id: 'decide',
    from: 0.78,
    to: 0.90,
    badge: 'CH 06 // DECIDE WITH DOLLARS',
    headline: 'Zero in-service failures.',
    desc: 'Cost-optimal replacement policy + OR-Tools hangar scheduling. $29.54M fleet savings verified in backtest.',
    health: 0.20,
    camera: { pos: [4, 4.5, 16], look: [0, 0, 0] },
    explode: 0.15,
    heat: 1.0,
    figure: 'bars',
  },
  {
    id: 'dawn',
    from: 0.90,
    to: 1.00,
    badge: 'CH 07 // DAWN',
    headline: 'Predict. Prove. Prevent.',
    desc: 'Replaced unit airborne at dawn. Cryptographic audit ledger verified. Fleet operational readiness restored.',
    health: 1.0,
    camera: { pos: [0, 6, 24], look: [0, 0, 0] },
    explode: 0.0,
    heat: 0.0,
    figure: 'reform',
  },
];

export const METRICS = {
  medianLeadTimeCycles: 164.5,
  empiricalCoveragePct: 84.0,
  targetCoveragePct: 80.0,
  savingsUsd: 29542700,
  savingsPct: 84.4,
  reactiveCostUsd: 35000000,
  predictiveCostUsd: 5457300,
  rmseCycles: 16.74,
  r2Score: 0.8255,
  zeroFailures: true,
};

export function chapterAt(p) {
  const clamped = Math.min(0.9999, Math.max(0, p));
  for (const ch of CHAPTERS) {
    if (clamped >= ch.from && clamped < ch.to) {
      return ch;
    }
  }
  return CHAPTERS[CHAPTERS.length - 1];
}

export function localT(p) {
  const ch = chapterAt(p);
  const range = ch.to - ch.from;
  if (range <= 0) return 0;
  return Math.min(1, Math.max(0, (p - ch.from) / range));
}

