/**
 * Scrubber: Interactive station navigation bar along the bottom viewport.
 * Provides instant camera rail seeking to any of the 6 turbofan assemblies.
 */
import { scrollToProgress } from '../conductor.js';

export class StationScrubber {
  constructor() {
    this.fill = document.getElementById('scrubber-fill');
    this.cycleEl = document.getElementById('scrubber-cycle');
    this.healthEl = document.getElementById('scrubber-health');
    this.ticks = Array.from(document.querySelectorAll('.station-tick'));

    this.initEvents();
  }

  initEvents() {
    this.ticks.forEach((tick) => {
      tick.addEventListener('click', (e) => {
        e.stopPropagation();
        const targetP = parseFloat(tick.getAttribute('data-p') || '0.25');
        scrollToProgress(targetP);
      });
    });
  }

  update(progress, chapter) {
    // 1. Fill percentage
    const pct = Math.min(100, Math.max(0, progress * 100));
    this.fill.style.width = `${pct}%`;

    // 2. Health & Cycle metrics
    const simulatedCycle = Math.round(progress * 206);
    this.cycleEl.textContent = `CYCLE ${simulatedCycle} / 206`;

    const healthPct = Math.round(chapter.health * 100);
    this.healthEl.textContent = `HEALTH ${healthPct}%`;

    if (chapter.health > 0.6) {
      this.fill.style.background = '#3de0ff';
      this.healthEl.style.color = '#3de0ff';
    } else if (chapter.health > 0.3) {
      this.fill.style.background = '#ffb23d';
      this.healthEl.style.color = '#ffb23d';
    } else {
      this.fill.style.background = '#ff4d4d';
      this.healthEl.style.color = '#ff4d4d';
    }

    // 3. Highlight current station tick
    this.ticks.forEach((tick) => {
      const p = parseFloat(tick.getAttribute('data-p') || '0');
      if (Math.abs(progress - p) < 0.035) {
        tick.classList.add('active');
      } else {
        tick.classList.remove('active');
      }
    });
  }
}

