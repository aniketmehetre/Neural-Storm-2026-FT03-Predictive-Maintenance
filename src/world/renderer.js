/**
 * Renderer: The single persistent WebGL world and requestAnimationFrame loop.
 */
import * as THREE from 'three';
import { SkyDome } from './sky.js';
import { CloudLayers } from './clouds.js';
import { TurbofanEngine } from './engine.js';
import { CameraRig } from './rig.js';
import { subscribe } from '../conductor.js';
import { chapterAt } from '../ledger.js';

export let scene = null;
export let camera = null;
export let renderer = null;
export let engine = null;
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

  // Lighting
  const ambLight = new THREE.AmbientLight(0xffffff, 0.65);
  scene.add(ambLight);

  const sunLight = new THREE.DirectionalLight(0xfff5ea, 1.8);
  sunLight.position.set(12, 20, 15);
  scene.add(sunLight);

  const rimLight = new THREE.DirectionalLight(0x3de0ff, 0.8);
  rimLight.position.set(-15, -10, -10);
  scene.add(rimLight);

  // Entities
  skyDome = new SkyDome(scene);
  cloudLayers = new CloudLayers(scene);
  engine = new TurbofanEngine(scene);
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

    // 1. Rig
    cameraRig.update(progress, time);

    // 2. Sky
    skyDome.update(ch.health, time);

    // 3. Clouds
    cloudLayers.update(ch.health, time, progress);

    // 4. Engine
    engine.update(time, ch.explode, ch.heat, velocity);

    // 5. Render
    renderer.render(scene, camera);
  });

  return { scene, camera, renderer, engine, cameraRig };
}
