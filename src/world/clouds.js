/**
 * Clouds: Procedural atmospheric cloud layers using canvas-generated noise textures.
 */
import * as THREE from 'three';

function createCloudTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  // Radial soft puff base
  const grad = ctx.createRadialGradient(256, 256, 40, 256, 256, 250);
  grad.addColorStop(0, 'rgba(255, 255, 255, 0.45)');
  grad.addColorStop(0.5, 'rgba(240, 245, 255, 0.18)');
  grad.addColorStop(1, 'rgba(255, 255, 255, 0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 512, 512);

  // Fractal noise blending
  for (let i = 0; i < 40; i++) {
    const rx = 100 + Math.random() * 312;
    const ry = 100 + Math.random() * 312;
    const r = 40 + Math.random() * 120;
    const g = ctx.createRadialGradient(rx, ry, 10, rx, ry, r);
    g.addColorStop(0, 'rgba(255, 255, 255, 0.15)');
    g.addColorStop(1, 'rgba(255, 255, 255, 0)');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(rx, ry, r, 0, Math.PI * 2);
    ctx.fill();
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  return texture;
}

export class CloudLayers {
  constructor(scene) {
    this.group = new THREE.Group();
    const texture = createCloudTexture();

    this.planes = [];
    const configs = [
      { y: -18, size: 280, opacity: 0.4, speed: 0.15 },
      { y: -26, size: 360, opacity: 0.3, speed: 0.08 },
      { y: -35, size: 480, opacity: 0.22, speed: 0.04 },
    ];

    for (const cfg of configs) {
      const geo = new THREE.PlaneGeometry(cfg.size, cfg.size);
      const mat = new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        opacity: cfg.opacity,
        depthWrite: false,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.y = cfg.y;
      this.group.add(mesh);
      this.planes.push({ mesh, mat, cfg });
    }

    scene.add(this.group);
  }

  update(health, time, progress) {
    for (const p of this.planes) {
      p.mesh.position.z = ((time * 2 * p.cfg.speed + progress * 20) % 100) - 50;
      // Tint with health: day (white) -> dusk (gold) -> night (dark amber/dim)
      const alpha = p.cfg.opacity * Math.max(0.05, Math.pow(health, 0.7));
      p.mat.opacity = alpha;
      if (health > 0.6) {
        p.mat.color.setHex(0xe8f0ff);
      } else if (health > 0.3) {
        p.mat.color.setHex(0xffba7a);
      } else {
        p.mat.color.setHex(0x5a1818);
      }
    }
  }
}
