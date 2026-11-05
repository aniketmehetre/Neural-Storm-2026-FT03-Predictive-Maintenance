/**
 * SpaceEnvironment: Renders the curved Earth planetary horizon viewed from orbit,
 * the luminous blue atmospheric Rayleigh scattering limb, and a 3D parallax starfield.
 */
import * as THREE from 'three';

/**
 * Creates procedural high-altitude Earth planetary texture with ocean bathymetry,
 * continental landmasses, and swirling weather cloud systems.
 */
function createEarthSurfaceTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  // Deep oceanic blue
  ctx.fillStyle = '#061328';
  ctx.fillRect(0, 0, 1024, 512);

  // Continental shelf shallows (lighter cyan-blue coastal fringes)
  for (let i = 0; i < 35; i++) {
    const cx = (i * 37) % 1024;
    const cy = 120 + ((i * 53) % 270);
    const rad = 40 + (i % 6) * 18;
    const grad = ctx.createRadialGradient(cx, cy, 10, cx, cy, rad);
    grad.addColorStop(0, 'rgba(14, 116, 144, 0.45)');
    grad.addColorStop(0.6, 'rgba(3, 105, 161, 0.25)');
    grad.addColorStop(1, 'rgba(6, 19, 40, 0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, rad, 0, Math.PI * 2);
    ctx.fill();
  }

  // Continental land masses (dark muted green/ochre terrain)
  for (let i = 0; i < 24; i++) {
    const lx = (i * 47 + 50) % 1024;
    const ly = 140 + ((i * 41) % 220);
    const w = 60 + (i % 5) * 35;
    const h = 40 + (i % 4) * 25;
    ctx.fillStyle = 'rgba(28, 48, 36, 0.65)';
    ctx.beginPath();
    ctx.ellipse(lx, ly, w, h, (i % 3) * 0.4, 0, Math.PI * 2);
    ctx.fill();
  }

  // Orbital atmospheric cloud bands (fine wispy cyclonic cloud spirals)
  ctx.fillStyle = 'rgba(255, 255, 255, 0.38)';
  for (let c = 0; c < 50; c++) {
    const sx = (c * 31) % 1024;
    const sy = 80 + ((c * 23) % 350);
    const r = 25 + (c % 8) * 15;
    const g = ctx.createRadialGradient(sx, sy, 5, sx, sy, r);
    g.addColorStop(0, 'rgba(255, 255, 255, 0.42)');
    g.addColorStop(0.5, 'rgba(235, 245, 255, 0.18)');
    g.addColorStop(1, 'rgba(255, 255, 255, 0)');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fill();
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  return texture;
}

export class CloudLayers {
  constructor(scene) {
    this.group = new THREE.Group();
    this.group.name = 'SpaceEnvironment';

    // 1. Earth Planetary Globe in Low Orbit
    // Large sphere placed below the satellite (radius 85, center at y = -92, z = 8)
    const earthGeo = new THREE.SphereGeometry(85, 48, 32);
    const earthTex = createEarthSurfaceTexture();
    const earthMat = new THREE.MeshStandardMaterial({
      map: earthTex,
      roughness: 0.65,
      metalness: 0.15,
      color: 0x93c5fd,
    });
    this.earthMesh = new THREE.Mesh(earthGeo, earthMat);
    this.earthMesh.position.set(0, -92, 6);
    this.group.add(this.earthMesh);

    // 2. Atmospheric Rayleigh Scattering Glow (The luminous "Thin Blue Line")
    const atmoGeo = new THREE.SphereGeometry(86.2, 48, 32);
    const atmoMat = new THREE.ShaderMaterial({
      side: THREE.BackSide,
      transparent: true,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uColor: { value: new THREE.Color(0x38bdf8) },
        uIntensity: { value: 1.2 },
      },
      vertexShader: `
        varying vec3 vNormal;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec3 uColor;
        uniform float uIntensity;
        varying vec3 vNormal;
        void main() {
          float rim = pow(1.0 - max(0.0, dot(vNormal, vec3(0.0, 0.0, 1.0))), 2.8);
          gl_FragColor = vec4(uColor, rim * uIntensity);
        }
      `,
    });
    this.atmoMesh = new THREE.Mesh(atmoGeo, atmoMat);
    this.atmoMesh.position.copy(this.earthMesh.position);
    this.group.add(this.atmoMesh);

    // 3. 3D Parallax Starfield Particle Points (2,400 multi-depth stars)
    const starCount = 2400;
    const starGeo = new THREE.BufferGeometry();
    const starPos = new Float32Array(starCount * 3);
    const starColor = new Float32Array(starCount * 3);

    for (let i = 0; i < starCount; i++) {
      // Distribute stars in spherical shell around spacecraft (radius 60 to 280)
      const u = Math.random();
      const v = Math.random();
      const theta = u * 2.0 * Math.PI;
      const phi = Math.acos(2.0 * v - 1.0);
      const r = 60 + Math.random() * 220;

      starPos[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
      starPos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      starPos[i * 3 + 2] = r * Math.cos(phi);

      // Star hues: blue-white, cool cyan, warm stellar gold
      const hueChoice = Math.random();
      if (hueChoice > 0.7) {
        starColor[i * 3]     = 0.75;
        starColor[i * 3 + 1] = 0.90;
        starColor[i * 3 + 2] = 1.0;
      } else if (hueChoice > 0.4) {
        starColor[i * 3]     = 1.0;
        starColor[i * 3 + 1] = 0.88;
        starColor[i * 3 + 2] = 0.72;
      } else {
        starColor[i * 3]     = 0.92;
        starColor[i * 3 + 1] = 0.96;
        starColor[i * 3 + 2] = 1.0;
      }
    }

    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
    starGeo.setAttribute('color', new THREE.BufferAttribute(starColor, 3));

    const starMat = new THREE.PointsMaterial({
      size: 1.25,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
    });

    this.starPoints = new THREE.Points(starGeo, starMat);
    this.group.add(this.starPoints);

    scene.add(this.group);
  }

  update(health, time, progress) {
    // 1. Slow orbital rotation of Earth beneath satellite
    this.earthMesh.rotation.y = time * 0.015 + progress * 0.4;
    this.earthMesh.rotation.x = -0.15;

    // 2. Gentle starfield parallax
    this.starPoints.rotation.y = time * 0.003;
    this.starPoints.rotation.x = time * 0.002;

    // 3. Shift atmospheric limb glow with health
    const atmoColor = health > 0.6 
      ? new THREE.Color(0x38bdf8) 
      : health > 0.3 
      ? new THREE.Color(0xf59e0b) 
      : new THREE.Color(0xef4444);
    this.atmoMesh.material.uniforms.uColor.value.lerp(atmoColor, 0.08);
  }
}
