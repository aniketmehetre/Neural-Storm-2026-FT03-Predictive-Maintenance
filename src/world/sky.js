/**
 * Sky: Deep space cosmic void shader with procedural twinkling starfields,
 * distant nebula dust bands, and celestial solar corona.
 */
import * as THREE from 'three';

const vertexShader = `
  varying vec3 vWorldPosition;
  varying vec3 vNormal;
  void main() {
    vNormal = normal;
    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
    vWorldPosition = worldPosition.xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const fragmentShader = `
  uniform float uHealth;
  uniform float uTime;
  varying vec3 vWorldPosition;

  // Pseudo-random hash
  float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
  }

  // 3D Value Noise for organic cosmic nebulae
  float noise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
      mix(mix(hash(i + vec3(0,0,0)), hash(i + vec3(1,0,0)), f.x),
          mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
      mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
          mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y), f.z
    );
  }

  void main() {
    vec3 dir = normalize(vWorldPosition);

    // Deep space void baseline (pitch cosmic black with subtle deep navy)
    vec3 spaceVoid = vec3(0.008, 0.012, 0.025);

    // Cosmic Nebula Cloud Bands
    float n1 = noise(dir * 3.5 + vec3(0.0, uTime * 0.01, 0.0));
    float n2 = noise(dir * 7.0 - vec3(uTime * 0.008, 0.0, 0.0));
    float nebulaDensity = smoothstep(0.42, 0.78, n1 * 0.65 + n2 * 0.35);

    // Nebula color shifting with health:
    // Health 1.0: Deep cyan / electric indigo celestial clouds
    // Health 0.6: Warm solar flare / auroral amber
    // Health 0.2: Deep crimson emergency alert void
    vec3 nebNominal = mix(vec3(0.02, 0.08, 0.18), vec3(0.12, 0.04, 0.22), dir.y * 0.5 + 0.5);
    vec3 nebOnset   = vec3(0.24, 0.11, 0.03);
    vec3 nebAlarm   = vec3(0.22, 0.02, 0.04);

    vec3 nebColor;
    if (uHealth > 0.6) {
      nebColor = mix(nebOnset, nebNominal, (uHealth - 0.6) / 0.4);
    } else {
      nebColor = mix(nebAlarm, nebOnset, clamp(uHealth / 0.6, 0.0, 1.0));
    }

    vec3 background = spaceVoid + nebColor * (nebulaDensity * 0.45);

    // Procedural Distant Stars (thousands of twinkling celestial pinpoints)
    vec3 starGrid = floor(dir * 280.0);
    float starRand = hash(starGrid);
    if (starRand > 0.985) {
      float starIntensity = pow((starRand - 0.985) / 0.015, 6.0);
      float twinkle = 0.7 + 0.3 * sin(uTime * 4.0 + starRand * 60.0);
      vec3 starColor = mix(vec3(0.85, 0.95, 1.0), vec3(1.0, 0.85, 0.7), fract(starRand * 13.0));
      background += starColor * (starIntensity * twinkle * 1.4);
    }

    // High-Magnitude Bright Guide Stars (sparse brilliant pinpricks)
    vec3 brightGrid = floor(dir * 75.0);
    float brightRand = hash(brightGrid);
    if (brightRand > 0.996) {
      float bIntensity = pow((brightRand - 0.996) / 0.004, 4.0);
      float bTwinkle = 0.8 + 0.2 * cos(uTime * 3.0 + brightRand * 40.0);
      background += vec3(0.9, 0.98, 1.0) * (bIntensity * bTwinkle * 2.5);
    }

    // Solar Corona & Flare (Deep space sun in upper quadrant)
    vec3 sunDir = normalize(vec3(0.4, 0.35, -0.85));
    float sunDot = max(0.0, dot(dir, sunDir));
    float sunDisc = smoothstep(0.9985, 0.9995, sunDot) * 4.0;
    float sunCorona = pow(sunDot, 128.0) * 0.9 + pow(sunDot, 16.0) * 0.25;
    vec3 sunColor = vec3(1.0, 0.97, 0.92);

    gl_FragColor = vec4(background + sunColor * (sunDisc + sunCorona), 1.0);
  }
`;

export class SkyDome {
  constructor(scene) {
    this.uniforms = {
      uHealth: { value: 1.0 },
      uTime: { value: 0.0 },
    };

    const geometry = new THREE.SphereGeometry(500, 48, 32);
    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: this.uniforms,
      side: THREE.BackSide,
      depthWrite: false,
    });

    this.mesh = new THREE.Mesh(geometry, material);
    scene.add(this.mesh);
  }

  update(health, time) {
    this.uniforms.uHealth.value += (health - this.uniforms.uHealth.value) * 0.08;
    this.uniforms.uTime.value = time;
  }
}
