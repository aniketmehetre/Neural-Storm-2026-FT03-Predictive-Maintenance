/**
 * Satellite Controls & Inspection System:
 * Handles 3D drag-to-rotate with inertia damping, zoom clamping, subsystem raycasting,
 * hover highlighting, camera focus transitions, and engineering information panels.
 */
import * as THREE from 'three';
import { animate } from 'animejs';

export class SatelliteInteractionController {
  constructor(satellite, camera, renderer) {
    this.satellite = satellite;
    this.camera = camera;
    this.renderer = renderer;
    this.domElement = renderer.domElement;

    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2(-999, -999);

    // Interaction states
    this.isDragging = false;
    this.isInspectionMode = false;
    this.selectedSubsystem = null;
    this.hoveredSubsystem = null;

    // Rotation damping
    this.rotationVelocity = { x: 0, y: 0 };
    this.previousMousePosition = { x: 0, y: 0 };

    // Camera targeting
    this.defaultCamPos = new THREE.Vector3(0, 6, 26);
    this.targetCamPos = this.defaultCamPos.clone();
    this.targetLookAt = new THREE.Vector3(0, 0, 0);
    this.currentLookAt = new THREE.Vector3(0, 0, 0);

    this.initHUD();
    this.initEventListeners();
  }

  initHUD() {
    // 1. Hover tooltip label
    let tip = document.getElementById('sat-hover-tooltip');
    if (!tip) {
      tip = document.createElement('div');
      tip.id = 'sat-hover-tooltip';
      tip.className = 'sat-tooltip';
      tip.innerHTML = `
        <span class="sat-tooltip-badge">SYSTEM</span>
        <span class="sat-tooltip-name">SUBSYSTEM</span>
      `;
      document.body.appendChild(tip);
    }
    this.tooltipEl = tip;

    // 2. Component Information Side Panel
    let panel = document.getElementById('sat-info-panel');
    if (!panel) {
      panel = document.createElement('div');
      panel.id = 'sat-info-panel';
      panel.className = 'sat-info-panel';
      panel.innerHTML = `
        <div class="sat-panel-header">
          <div>
            <div class="sat-panel-sub" id="sat-info-cat">SUBSYSTEM</div>
            <h2 class="sat-panel-title" id="sat-info-title">SOLAR ARRAY SYSTEM</h2>
          </div>
          <button class="sat-close-btn" id="sat-close-btn" aria-label="Close Inspection">✕</button>
        </div>
        <div class="sat-panel-body">
          <div class="sat-panel-section">
            <div class="sat-section-label">// PURPOSE & ARCHITECTURE</div>
            <p class="sat-section-text" id="sat-info-purpose">Description...</p>
          </div>
          <div class="sat-panel-section">
            <div class="sat-section-label">// ENGINEERING INSPECTION POINTS</div>
            <ul class="sat-checklist" id="sat-info-points">
              <li>Item 1</li>
            </ul>
          </div>
          <div class="sat-panel-actions">
            <button class="sat-action-btn primary" id="sat-reset-cam-btn">RESET CAMERA</button>
            <button class="sat-action-btn secondary" id="sat-toggle-inspect-btn">INSPECTION MODE: OFF</button>
          </div>
        </div>
      `;
      document.body.appendChild(panel);
    }
    this.panelEl = panel;

    // Close panel button
    document.getElementById('sat-close-btn').addEventListener('click', () => {
      this.deselectSubsystem();
    });

    // Reset camera button
    document.getElementById('sat-reset-cam-btn').addEventListener('click', () => {
      this.resetCamera();
    });

    // Toggle inspection mode button
    const inspectBtn = document.getElementById('sat-toggle-inspect-btn');
    inspectBtn.addEventListener('click', () => {
      this.isInspectionMode = !this.isInspectionMode;
      inspectBtn.textContent = `INSPECTION MODE: ${this.isInspectionMode ? 'ON' : 'OFF'}`;
      inspectBtn.classList.toggle('active', this.isInspectionMode);
    });
  }

  initEventListeners() {
    // Enable pointer events on canvas for drag & raycasting
    this.domElement.style.pointerEvents = 'auto';

    // Mouse move & Raycasting
    window.addEventListener('mousemove', (e) => {
      this.mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
      this.mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

      // Position tooltip near cursor
      if (this.tooltipEl.classList.contains('visible')) {
        this.tooltipEl.style.transform = `translate3d(${e.clientX + 16}px, ${e.clientY + 16}px, 0)`;
      }

      // Dragging satellite rotation
      if (this.isDragging) {
        const deltaX = e.clientX - this.previousMousePosition.x;
        const deltaY = e.clientY - this.previousMousePosition.y;

        this.rotationVelocity.y += deltaX * 0.003;
        this.rotationVelocity.x += deltaY * 0.003;

        this.previousMousePosition = { x: e.clientX, y: e.clientY };
      }
    });

    // Drag start
    this.domElement.addEventListener('mousedown', (e) => {
      if (e.button === 0) {
        this.isDragging = true;
        this.previousMousePosition = { x: e.clientX, y: e.clientY };
      }
    });

    // Drag end
    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    // Zoom clamping
    window.addEventListener('wheel', (e) => {
      const zoomFactor = e.deltaY * 0.015;
      const currentDist = this.camera.position.length();
      const newDist = THREE.MathUtils.clamp(currentDist + zoomFactor, 9, 45);
      this.camera.position.setLength(newDist);
    }, { passive: true });

    // Click selection
    this.domElement.addEventListener('click', (e) => {
      if (Math.hypot(e.clientX - this.previousMousePosition.x, e.clientY - this.previousMousePosition.y) > 5) {
        return; // Ignore if user was dragging
      }

      this.raycaster.setFromCamera(this.mouse, this.camera);
      const hits = this.raycaster.intersectObjects(this.satellite.interactiveMeshes, false);

      if (hits.length > 0) {
        const subDef = hits[0].object.userData.subsystemDef;
        if (subDef) {
          this.selectSubsystem(subDef);
        }
      }
    });
  }

  selectSubsystem(subDef) {
    this.selectedSubsystem = subDef;
    this.isInspectionMode = true;

    // 1. Move camera smoothly to subsystem focus point
    const worldAnchor = subDef.worldAnchor();
    this.targetLookAt.copy(worldAnchor);

    // Target position offset from subsystem anchor
    const offset = new THREE.Vector3(0, 1.2, 5.5);
    if (subDef.id.includes('solar')) {
      offset.set(subDef.explodeDir.x > 0 ? 3.0 : -3.0, 1.5, 6.0);
    } else if (subDef.id === 'hga') {
      offset.set(0, 3.2, 4.5);
    } else if (subDef.id === 'prop') {
      offset.set(0, -2.5, 4.5);
    }
    this.targetCamPos.copy(worldAnchor).add(offset);

    // 2. Dim other subsystems, highlight selected
    this.satellite.subsystems.forEach((s) => {
      const isTarget = s.id === subDef.id;
      s.group.traverse((c) => {
        if (c.isMesh && c.material) {
          if (isTarget) {
            c.material.emissive = new THREE.Color(0x3de0ff);
            c.material.emissiveIntensity = 0.25;
          } else {
            c.material.emissive = new THREE.Color(0x000000);
            c.material.emissiveIntensity = 0.0;
          }
        }
      });
    });

    // 3. Populate and show engineering info panel
    document.getElementById('sat-info-cat').textContent = subDef.category;
    document.getElementById('sat-info-title').textContent = subDef.name;
    document.getElementById('sat-info-purpose').textContent = subDef.purpose;

    const pointsList = document.getElementById('sat-info-points');
    pointsList.innerHTML = '';
    subDef.inspectionPoints.forEach((pt) => {
      const li = document.createElement('li');
      li.textContent = pt;
      pointsList.appendChild(li);
    });

    this.panelEl.classList.add('visible');

    // Anime.js entrance animation for panel
    animate(this.panelEl, {
      translateX: [320, 0],
      opacity: [0, 1],
      duration: 400,
      ease: 'outExpo',
    });
  }

  deselectSubsystem() {
    this.selectedSubsystem = null;
    this.panelEl.classList.remove('visible');

    // Restore all materials
    this.satellite.subsystems.forEach((s) => {
      s.group.traverse((c) => {
        if (c.isMesh && c.material) {
          c.material.emissive = new THREE.Color(0x000000);
          c.material.emissiveIntensity = 0.0;
        }
      });
    });

    this.resetCamera();
  }

  resetCamera() {
    this.targetCamPos.set(0, 6, 26);
    this.targetLookAt.set(0, 0, 0);
  }

  update() {
    // 1. Raycast hover check
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const hits = this.raycaster.intersectObjects(this.satellite.interactiveMeshes, false);

    if (hits.length > 0) {
      const subDef = hits[0].object.userData.subsystemDef;
      if (subDef && subDef !== this.hoveredSubsystem) {
        this.hoveredSubsystem = subDef;
        this.tooltipEl.querySelector('.sat-tooltip-name').textContent = subDef.name;
        this.tooltipEl.querySelector('.sat-tooltip-badge').textContent = subDef.category;
        this.tooltipEl.classList.add('visible');
      }
    } else {
      if (this.hoveredSubsystem) {
        this.hoveredSubsystem = null;
        this.tooltipEl.classList.remove('visible');
      }
    }

    // 2. Apply drag rotation with inertia damping
    this.satellite.root.rotation.y += this.rotationVelocity.y;
    this.satellite.root.rotation.x += this.rotationVelocity.x;

    this.rotationVelocity.x *= 0.92;
    this.rotationVelocity.y *= 0.92;

    // 3. Smooth camera position interpolation
    if (this.selectedSubsystem) {
      this.camera.position.lerp(this.targetCamPos, 0.08);
      this.currentLookAt.lerp(this.targetLookAt, 0.08);
      this.camera.lookAt(this.currentLookAt);
    }
  }
}

