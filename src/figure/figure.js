/**
 * Figure: Anime.js v4-powered kinetic geometric SVG background layer.
 * Morphs across 7 distinct operational states synced to the chapter ledger.
 */
import { animate, createTimeline, stagger } from 'animejs';

export class FigureLayer {
  constructor() {
    this.svg = document.getElementById('figure');
    this.currentState = 'rest';
    this.stateAmount = 0;
    this.elements = [];
    this.mouse = { x: 0, y: 0, targetX: 0, targetY: 0 };

    this.buildGeometry();
    this.initMouse();
  }

  buildGeometry() {
    const cx = 960;
    const cy = 540;

    // Center hub circle
    const centerG = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    centerG.setAttribute('id', 'fig-center');
    this.svg.appendChild(centerG);

    // 1. Concentric orbital circles (6 rings)
    this.rings = [];
    for (let r = 1; r <= 6; r++) {
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', cx);
      circle.setAttribute('cy', cy);
      circle.setAttribute('r', (r * 60).toString());
      circle.setAttribute('fill', 'none');
      circle.setAttribute('stroke', '#3de0ff');
      circle.setAttribute('stroke-width', (1.2 + (r % 2) * 0.8).toString());
      circle.setAttribute('stroke-dasharray', `${r * 15} ${r * 10}`);
      circle.setAttribute('opacity', (0.2 + (r / 6) * 0.4).toString());
      centerG.appendChild(circle);
      this.rings.push({ el: circle, baseR: r * 60, r });
    }

    // 2. Segmented radial spoke lines (24 rays)
    this.spokes = [];
    for (let i = 0; i < 24; i++) {
      const angle = (i / 24) * Math.PI * 2;
      const x1 = cx + Math.cos(angle) * 80;
      const y1 = cy + Math.sin(angle) * 80;
      const x2 = cx + Math.cos(angle) * 360;
      const y2 = cy + Math.sin(angle) * 360;

      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', x1);
      line.setAttribute('y1', y1);
      line.setAttribute('x2', x2);
      line.setAttribute('y2', y2);
      line.setAttribute('stroke', '#3de0ff');
      line.setAttribute('stroke-width', '1');
      line.setAttribute('stroke-dasharray', '8 12');
      line.setAttribute('opacity', '0.25');
      centerG.appendChild(line);
      this.spokes.push({ el: line, angle, x1, y1, x2, y2 });
    }

    // 3. Orbiting Diamond glyphs
    this.glyphs = [];
    for (let i = 0; i < 8; i++) {
      const angle = (i / 8) * Math.PI * 2;
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
      path.setAttribute('points', '-6,0 0,-6 6,0 0,6');
      path.setAttribute('fill', '#3de0ff');
      path.setAttribute('opacity', '0.6');
      centerG.appendChild(path);
      this.glyphs.push({ el: path, angle, radius: 240 });
    }

    // 4. Two vertical Comparative Bars (used in 'bars' state)
    this.bars = [];
    const barData = [
      { x: 780, h: 280, color: '#ff4d4d', label: 'Reactive $35M' },
      { x: 1140, h: 48, color: '#3de0ff', label: 'Predictive $5.5M' },
    ];
    barData.forEach((b) => {
      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('x', (b.x - 24).toString());
      rect.setAttribute('y', (cy + 140 - b.h).toString());
      rect.setAttribute('width', '48');
      rect.setAttribute('height', b.h.toString());
      rect.setAttribute('fill', b.color);
      rect.setAttribute('opacity', '0');
      rect.setAttribute('rx', '6');
      this.svg.appendChild(rect);
      this.bars.push({ el: rect, baseH: b.h });
    });
  }

  initMouse() {
    window.addEventListener('mousemove', (e) => {
      this.mouse.targetX = (e.clientX / window.innerWidth - 0.5) * 40;
      this.mouse.targetY = (e.clientY / window.innerHeight - 0.5) * 40;
    });
  }

  setState(name, localT, health = 1.0) {
    this.currentState = name;
    this.stateAmount = localT;

    // Morph colors based on health
    const strokeColor = health > 0.6 ? '#3de0ff' : health > 0.3 ? '#ffb23d' : '#ff4d4d';

    // Mouse spring
    this.mouse.x += (this.mouse.targetX - this.mouse.x) * 0.08;
    this.mouse.y += (this.mouse.targetY - this.mouse.y) * 0.08;

    const cx = 960 + this.mouse.x;
    const cy = 540 + this.mouse.y;

    // Apply state dynamics
    if (name === 'rest') {
      // Gentle breathing idle drift
      const t = Date.now() * 0.0015;
      this.rings.forEach((r, idx) => {
        const rad = r.baseR + Math.sin(t + idx * 0.4) * 12;
        r.el.setAttribute('cx', cx);
        r.el.setAttribute('cy', cy);
        r.el.setAttribute('r', rad);
        r.el.setAttribute('stroke', strokeColor);
        r.el.setAttribute('opacity', '0.35');
      });
      this.bars.forEach((b) => b.el.setAttribute('opacity', '0'));
    } else if (name === 'tighten') {
      // Contract inward toward core
      const contraction = 1.0 - localT * 0.45;
      this.rings.forEach((r) => {
        r.el.setAttribute('cx', cx);
        r.el.setAttribute('cy', cy);
        r.el.setAttribute('r', (r.baseR * contraction).toString());
        r.el.setAttribute('stroke', strokeColor);
        r.el.setAttribute('opacity', '0.45');
      });
      this.bars.forEach((b) => b.el.setAttribute('opacity', '0'));
    } else if (name === 'follow') {
      // Deconstruct laterally to follow exploded assemblies
      const spread = localT * 180;
      this.rings.forEach((r, idx) => {
        const xOffset = (idx - 2.5) * (spread / 3);
        r.el.setAttribute('cx', (cx + xOffset).toString());
        r.el.setAttribute('cy', cy);
        r.el.setAttribute('stroke', strokeColor);
      });
      this.bars.forEach((b) => b.el.setAttribute('opacity', '0'));
    } else if (name === 'curve' || name === 'band') {
      // Elongate into fan/band
      this.rings.forEach((r, idx) => {
        r.el.setAttribute('cx', cx);
        r.el.setAttribute('cy', cy);
        r.el.setAttribute('stroke', strokeColor);
        r.el.setAttribute('opacity', '0.25');
      });
      this.bars.forEach((b) => b.el.setAttribute('opacity', '0'));
    } else if (name === 'bars') {
      // Collapse into discrete cost bars
      this.rings.forEach((r) => r.el.setAttribute('opacity', '0.08'));
      this.bars.forEach((b) => b.el.setAttribute('opacity', '0.75'));
    } else if (name === 'reform') {
      // Reconstitute into pristine aerodynamic ring
      this.rings.forEach((r) => {
        r.el.setAttribute('cx', cx);
        r.el.setAttribute('cy', cy);
        r.el.setAttribute('r', r.baseR.toString());
        r.el.setAttribute('stroke', '#3de0ff');
        r.el.setAttribute('opacity', '0.4');
      });
      this.bars.forEach((b) => b.el.setAttribute('opacity', '0'));
    }

    // Rotate glyphs
    const timeSec = Date.now() * 0.001;
    this.glyphs.forEach((g, idx) => {
      const a = g.angle + timeSec * 0.2;
      const gx = cx + Math.cos(a) * g.radius;
      const gy = cy + Math.sin(a) * g.radius;
      g.el.setAttribute('transform', `translate(${gx}, ${gy})`);
      g.el.setAttribute('fill', strokeColor);
    });
  }
}

