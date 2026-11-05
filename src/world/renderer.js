/**
 * Renderer: The single persistent WebGL world and requestAnimationFrame loop.
 */
import * as THREE from 'three';
import { SkyDome } from './sky.js';
import { CloudLayers } from './clouds.js';
import { SatelliteDigitalTwin } from './satellite.js';
import { SatelliteInteractionController } from './satellite_controls.js';
import { CameraRig } from './rig.js';
import { subscribe } from '../conductor.js';
import { chapterAt } from '../ledger.js';

export let scene = null;
export let camera = null;
export let renderer = null;
export let satellite = null;
export let satelliteControls = null;
export let cameraRig = null;
export let skyDome = null;
export let cloudLayers = null;

let isPaused = false;
let clock = null;

export function initWorld() {
  const canvas = document.getElementById('world');
  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x05070b, 0.005);

  camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 800);
  camera.position.set(0, 6, 26);

  const isMobile = window.innerWidth < 768;
  renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: !isMobile,
    powerPreference: 'high-performance',
    alpha: true,
  });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, isMobile ? 1.5 : 2.0));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;

  // Spacecraft illumination (Sun directional key, Earth albedo fill, Rim light)
  const ambLight = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambLight);

  const sunLight = new THREE.DirectionalLight(0xfffaed, 2.2);
  sunLight.position.set(16, 22, 18);
  scene.add(sunLight);

  const earthAlbedo = new THREE.DirectionalLight(0x38bdf8, 0.85);
  earthAlbedo.position.set(0, -15, 10);
  scene.add(earthAlbedo);

  const rimLight = new THREE.DirectionalLight(0x3de0ff, 0.9);
  rimLight.position.set(-18, -8, -12);
  scene.add(rimLight);

  // Entities
  skyDome = new SkyDome(scene);
  cloudLayers = new CloudLayers(scene);
  satellite = new SatelliteDigitalTwin(scene);
  satellite.assemblies = satellite.subsystems; // compatibility alias for callouts
  satelliteControls = new SatelliteInteractionController(satellite, camera, renderer);
  cameraRig = new CameraRig(camera);
  clock = new THREE.Clock();

  // Resize handling
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  // Tab visibility pause to save battery
  document.addEventListener('visibilitychange', () => {
    isPaused = document.hidden;
  });

  // Subscribe to conductor
  subscribe(({ progress, velocity }) => {
    if (isPaused) return;
    const time = clock.getElapsedTime();
    const ch = chapterAt(progress);

    // 1. Camera Rig (only if user is not actively inspecting an individual subsystem)
    if (!satelliteControls.selectedSubsystem) {
      cameraRig.update(progress, time);
    }

    // 2. Sky & Atmosphere
    skyDome.update(ch.health, time);
    cloudLayers.update(ch.health, time, progress);

    // 3. Central 3D Satellite Digital Twin
    satellite.update(time, ch.explode, ch.health, satelliteControls.isInspectionMode);

    // 4. Interactive Controller (damping, raycasting, camera transitions)
    satelliteControls.update();

    // 5. Render
    renderer.render(scene, camera);
  });

  return { scene, camera, renderer, satellite, satelliteControls, cameraRig };
}

