/**
 * Rig: Smooth 3D camera rail powered by continuous Catmull-Rom splines.
 */
import * as THREE from 'three';
import { CHAPTERS } from '../ledger.js';

export class CameraRig {
  constructor(camera) {
    this.camera = camera;
    this.mouse = { x: 0, y: 0, targetX: 0, targetY: 0 };
    this.currentLookAt = new THREE.Vector3(0, 0, 0);

    // Build splines through chapter positions
    const posPoints = CHAPTERS.map((ch) => new THREE.Vector3(...ch.camera.pos));
    const lookPoints = CHAPTERS.map((ch) => new THREE.Vector3(...ch.camera.look));

    this.posCurve = new THREE.CatmullRomCurve3(posPoints, false, 'centripetal', 0.5);
    this.lookCurve = new THREE.CatmullRomCurve3(lookPoints, false, 'centripetal', 0.5);

    window.addEventListener('mousemove', (e) => {
      this.mouse.targetX = (e.clientX / window.innerWidth - 0.5) * 2;
      this.mouse.targetY = (e.clientY / window.innerHeight - 0.5) * 2;
    });
  }

  update(progress, time) {
    // Parallax spring
    this.mouse.x += (this.mouse.targetX - this.mouse.x) * 0.05;
    this.mouse.y += (this.mouse.targetY - this.mouse.y) * 0.05;

    // Sample rail
    const clampedP = Math.min(0.999, Math.max(0, progress));
    const rawPos = this.posCurve.getPointAt(clampedP);
    const rawLook = this.lookCurve.getPointAt(clampedP);

    // Subtle atmospheric sway & mouse parallax
    const swayX = Math.sin(time * 0.6) * 0.08 + this.mouse.x * 0.35;
    const swayY = Math.cos(time * 0.5) * 0.08 - this.mouse.y * 0.25;

    this.camera.position.set(rawPos.x + swayX, rawPos.y + swayY, rawPos.z);
    this.currentLookAt.lerp(rawLook, 0.08);
    this.camera.lookAt(this.currentLookAt);
  }
}
