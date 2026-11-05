/**
 * Ledger: The authoritative chapter scroll map and mission ledger.
 * Maps normalized progress (0 to 1) to camera coords, sky health, headlines, and data.
 */

export const CHAPTERS = [
  {
    id: 'boarding',
    from: 0.00,
    to: 0.10,
    badge: 'CH 01 // ORBITAL INSERTION',
    headline: 'Deploy. Orient. Transmit.',
    desc: 'Geostationary orbital insertion. Fully deployed solar array wings. 100% nominal bus health.',
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
    badge: 'CH 02 // ROTATIONAL EXPLORATION',
    headline: 'Six subsystems. Real-time telemetry.',
    desc: 'Approaching the main satellite bus. Subsystem telemetry ingested into 30-cycle causal buffers.',
    health: 0.95,
    camera: { pos: [0, 2.2, 14], look: [0, 0, 0] },
    explode: 0.05,
    heat: 0.05,
    figure: 'tighten',
  },
  {
    id: 'anatomy',
    from: 0.25,
    to: 0.50,
    badge: 'CH 03 // COMPONENT REVEAL',
    headline: 'Aerospace digital twin architecture.',
    desc: 'Spacecraft bus, dual solar wings, parabolic high-gain dish, bipropellant propulsion, ADCS star trackers, and optical payload.',
    health: 0.85,
    camera: { pos: [5, 3.2, 12], look: [0, 0, 0] },
    explode: 0.6,
    heat: 0.25,
    figure: 'follow',
  },
  {
    id: 'detect',
    from: 0.50,
    to: 0.65,
    badge: 'CH 04 // EXPLODED ASSEMBLY',
    headline: 'Subsystems separate along flight axes.',
    desc: 'PCA + Isolation Forest anomaly detection. CUSUM flags degradation onset 164.5 cycles in advance.',
    health: 0.60,
    camera: { pos: [-2.5, 1.8, 10.5], look: [-0.6, 0.2, 0] },
    explode: 1.0,
    heat: 0.65,
    figure: 'curve',
  },
  {
    id: 'forecast',
    from: 0.65,
    to: 0.78,
    badge: 'CH 05 // DEEP INSPECTION',
    headline: 'Calibrated split-conformal telemetry.',
    desc: 'Split-conformal calibrated bounds [p10, p90]. 84.0% empirical coverage verified on held-out units.',
    health: 0.40,
    camera: { pos: [1.2, 1.4, 8.5], look: [0, 0, 0] },
    explode: 0.4,
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
    camera: { pos: [4, 4.5, 17], look: [0, 0, 0] },
    explode: 0.15,
    heat: 1.0,
    figure: 'bars',
  },
  {
    id: 'dawn',
    from: 0.90,
    to: 1.00,
    badge: 'CH 07 // REASSEMBLY & DAWN',
    headline: 'Predict. Prove. Prevent.',
    desc: 'Reassembled spacecraft operational at dawn. Cryptographic audit ledger verified. Readiness restored.',
    health: 1.0,
    camera: { pos: [0, 6, 25], look: [0, 0, 0] },
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

