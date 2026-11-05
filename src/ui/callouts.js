/**
 * Callouts: Cardless 3D-anchored labels tethered to physical turbofan assemblies.
 * Screen-projected in real time with dynamic mini-waveforms.
 */
import * as THREE from 'three';

export class CalloutSystem {
  constructor(engine, camera) {
    this.engine = engine;
    this.camera = camera;
    this.container = document.getElementById('callout-container');
    this.callouts = [];
    this.waveformData = new Map();

    this.initCallouts();
  }

  initCallouts() {
    this.container.innerHTML = '';

    this.engine.assemblies.forEach((assy) => {
      const anchorEl = document.createElement('div');
      anchorEl.className = 'callout-anchor';
      anchorEl.id = `callout-${assy.id || assy.name.toLowerCase().replace(/\s+/g, '-')}`;

      // Center anchor dot
      const dot = document.createElement('div');
      dot.className = 'callout-dot';
      anchorEl.appendChild(dot);

      // Info box
      const box = document.createElement('div');
      box.className = 'callout-box';
      box.style.width = '175px';

      const name = document.createElement('div');
      name.className = 'callout-name';
      name.textContent = assy.name;
      box.appendChild(name);

      if (assy.category) {
        const catBadge = document.createElement('span');
        catBadge.className = 'callout-fault-badge';
        catBadge.textContent = assy.category.toUpperCase();
        box.appendChild(catBadge);
      } else if (assy.isHPC) {
        const faultBadge = document.createElement('span');
        faultBadge.className = 'callout-fault-badge';
        faultBadge.textContent = 'TELEMETRY ACTIVE';
        box.appendChild(faultBadge);
      }

      // Mini-waveform SVG (60x16)
      const svgWave = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svgWave.setAttribute('class', 'callout-waveform');
      svgWave.setAttribute('viewBox', '0 0 60 16');

      const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
      polyline.setAttribute('fill', 'none');
      polyline.setAttribute('stroke', '#3de0ff');
      polyline.setAttribute('stroke-width', '1.5');
      svgWave.appendChild(polyline);
      box.appendChild(svgWave);

      anchorEl.appendChild(box);
      this.container.appendChild(anchorEl);

      // Seed waveform history
      const history = Array.from({ length: 15 }, () => 8);
      this.waveformData.set(assy.name, { history, polyline });

      this.callouts.push({
        assy,
        anchorEl,
        box,
        dot,
      });
    });
  }

  update(progress, chapter) {
    const isAnatomy = chapter.id === 'anatomy';
    const isDetect = chapter.id === 'detect';
    const showFull = isAnatomy || isDetect;

    const widthHalf = window.innerWidth / 2;
    const heightHalf = window.innerHeight / 2;
    const projVec = new THREE.Vector3();

    // Find assembly nearest screen center
    let closestAssy = null;
    let minCenterDist = Infinity;

    this.callouts.forEach(({ assy, anchorEl, box }) => {
      // 1. Project 3D anchor to 2D screen coordinates
      const worldPos = assy.worldAnchor();
      projVec.copy(worldPos).project(this.camera);

      // Check if in front of camera
      const isBehind = projVec.z > 1.0;
      if (isBehind) {
        anchorEl.style.opacity = '0';
        return;
      }

      const screenX = projVec.x * widthHalf + widthHalf;
      const screenY = -(projVec.y * heightHalf) + heightHalf;
      const distFromCenter = Math.hypot(screenX - widthHalf, screenY - heightHalf);

      if (distFromCenter < minCenterDist) {
        minCenterDist = distFromCenter;
        closestAssy = assy.name;
      }

      // Position anchor via translate3d for silky 60fps performance
      anchorEl.style.transform = `translate3d(${screenX}px, ${screenY}px, 0)`;

      // Visibility logic
      if (!showFull) {
        anchorEl.style.opacity = '0';
        anchorEl.classList.remove('active');
      } else {
        anchorEl.style.opacity = '1';
        // Active if closest or highlighted in detect mode
        const isActive = assy.name === closestAssy || (isDetect && (assy.id === 'hga' || assy.isHPC));
        if (isActive) {
          anchorEl.classList.add('active');
        } else {
          anchorEl.classList.remove('active');
        }
      }

      // 2. Animate mini-waveform
      const wave = this.waveformData.get(assy.name);
      if (wave) {
        wave.history.shift();
        const baseNoise = (Math.random() - 0.5) * ((assy.id === 'hga' || assy.isHPC) && isDetect ? 8 : 3.5);
        const nextVal = Math.max(2, Math.min(14, 8 + baseNoise));
        wave.history.push(nextVal);

        const points = wave.history
          .map((v, i) => `${(i / (wave.history.length - 1)) * 60},${v}`)
          .join(' ');
        wave.polyline.setAttribute('points', points);
      }
    });
  }
}

