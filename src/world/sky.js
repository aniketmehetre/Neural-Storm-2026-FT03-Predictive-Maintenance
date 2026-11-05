/**
 * Sky: Dynamic atmospheric hemispherical dome driven by scalar health metric uHealth.
 */
import * as THREE from 'three';

const vertexShader = `
  varying vec3 vWorldPosition;
  void main() {
    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
    vWorldPosition = worldPosition.xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const fragmentShader = `
  uniform float uHealth;
  uniform float uTime;
  varying vec3 vWorldPosition;

  void main() {
    vec3 dir = normalize(vWorldPosition);
    float h = clamp(dir.y, 0.0, 1.0);

    // Day gradient (uHealth ~ 1.0)
    vec3 dayTop = vec3(0.04, 0.12, 0.28);
    vec3 dayHorizon = vec3(0.45, 0.72, 0.95);
    vec3 daySky = mix(dayHorizon, dayTop, pow(h, 0.6));

    // Dusk gradient (uHealth ~ 0.6)
    vec3 duskTop = vec3(0.08, 0.05, 0.20);
    vec3 duskHorizon = vec3(0.85, 0.45, 0.15);
    vec3 duskSky = mix(duskHorizon, duskTop, pow(h, 0.65));

    // Night/Critical gradient (uHealth ~ 0.2)
    vec3 nightTop = vec3(0.02, 0.01, 0.05);
    vec3 nightHorizon = vec3(0.65, 0.08, 0.08);
    vec3 nightSky = mix(nightHorizon, nightTop, pow(h, 0.7));

    // Blend day -> dusk -> night based on uHealth
    vec3 skyColor;
    if (uHealth > 0.6) {
      float t = (uHealth - 0.6) / 0.4;
      skyColor = mix(duskSky, daySky, t);
    } else {
      float t = clamp(uHealth / 0.6, 0.0, 1.0);
      skyColor = mix(nightSky, duskSky, t);
    }

    // Solar flare / glow lowering into horizon
    vec3 sunDir = normalize(vec3(0.3, 0.15 + 0.4 * uHealth, -0.9));
    float sunDot = max(0.0, dot(dir, sunDir));
    float sunGlow = pow(sunDot, 64.0) * 0.8 + pow(sunDot, 8.0) * 0.25;
    vec3 sunColor = mix(vec3(1.0, 0.35, 0.1), vec3(1.0, 0.95, 0.8), uHealth);

    gl_FragColor = vec4(skyColor + sunColor * sunGlow, 1.0);
  }
`;

export class SkyDome {
  constructor(scene) {
    this.uniforms = {
      uHealth: { value: 1.0 },
      uTime: { value: 0.0 },
    };

    const geometry = new THREE.SphereGeometry(450, 32, 24);
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
