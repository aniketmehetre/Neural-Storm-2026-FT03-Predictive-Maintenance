/**
 * Conductor: Centralized smooth scroll engine powered by Lenis.
 * Exports smoothed normalized progress (0 to 1), raw progress, and frame subscription.
 */
import Lenis from 'lenis';

let lenis = null;
let rawProgress = 0;
let progress = 0;
let velocity = 0;
const subscribers = new Set();

export function initConductor() {
  lenis = new Lenis({
    duration: 1.2,
    easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
    orientation: 'vertical',
    gestureOrientation: 'vertical',
    smoothWheel: true,
    touchMultiplier: 1.5,
  });

  lenis.on('scroll', (e) => {
    velocity = e.velocity || 0;
  });

  // Support direct progress jumping via query param ?p=0.35
  const urlParams = new URLSearchParams(window.location.search);
  const initialP = urlParams.get('p');
  const forcedP = initialP !== null ? Math.min(1, Math.max(0, parseFloat(initialP))) : null;

  function raf(time) {
    lenis.raf(time);

    if (forcedP !== null) {
      rawProgress = forcedP;
      progress = forcedP;
    } else {
      // Calculate raw progress across the 1000vh document height
      const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
      rawProgress = maxScroll > 0 ? Math.min(1, Math.max(0, window.scrollY / maxScroll)) : 0;

      // Smooth progress toward rawProgress with lerp of 0.08
      progress += (rawProgress - progress) * 0.08;
      if (Math.abs(rawProgress - progress) < 0.0001) {
        progress = rawProgress;
      }
    }

    const state = { progress, rawProgress, velocity };
    for (const sub of subscribers) {
      sub(state);
    }

    requestAnimationFrame(raf);
  }

  requestAnimationFrame(raf);
}

export function subscribe(fn) {
  subscribers.add(fn);
  return () => subscribers.delete(fn);
}

export function scrollToProgress(p, immediate = false) {
  if (!lenis) return;
  const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
  lenis.scrollTo(p * maxScroll, { immediate });
}

export function getProgress() {
  return progress;
}

export function getRawProgress() {
  return rawProgress;
}
