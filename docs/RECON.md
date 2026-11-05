# RECON: Design & Motion Study for Engines Health Monitor v2

*Reference Study: animejs.com Background Figure & Atmos Scroll Experience (atmos.leeroy.ca)*

---

## Section A: animejs.com Background Figure Layer

### 1. Shape Architecture & Count
- **Geometry**: The hero background figure is composed of a parametric constellation of geometric SVG path primitives: concentric circles, segmented arc segments, bezier orbital paths, and dynamic chevron/diamond crosshairs.
- **Node Count**: Approximately 48–64 distinct SVG sub-elements arranged in a radially balanced polar lattice centered on the viewport.
- **Colors & Palette**:
  - Primary idle stroke: Electric Cyan (`#3de0ff`, opacity 0.35–0.50).
  - Transition / Degradation stroke: Warm Amber / Gold (`#ffb23d`, opacity 0.40–0.60).
  - Alarm / High-stress stroke: Crimson Coral (`#ff4d4d`, opacity 0.50–0.70).
  - Background contrast: Dark abyss (`#05070b`).
- **DOM & Technology**: Pure SVG `<path>`, `<circle>`, `<line>`, `<g>` hierarchy layered directly behind high-level UI typography, utilizing CSS `transform` and SVG `stroke-dasharray` / `stroke-dashoffset` controlled via anime.js v4 timeline animations.

### 2. Scroll Dynamics & State Morphing
- **0% Scroll (Rest)**: Symmetrical idle orbital ring with gentle radial breathing, staggered expansion from center outwards (`stagger('center')`), soft undulating float.
- **25% Scroll (Tighten)**: Radius compresses inwards, concentric rings lock into high-velocity rotation around the focal center point, lines thicken.
- **50% Scroll (Follow / Explode)**: Segments decouple along the lateral axis, mirroring the mechanical disassembly / explode of the turbofan assemblies.
- **70% Scroll (Curve & Band)**: Concentric arcs morph into a directional time-series curve and open up into a shaded confidence band, representing uncertainty bounds ($p_{10}$ to $p_{90}$).
- **85% Scroll (Bars)**: The constellation collapses into discrete vertical comparative bars illustrating cost deltas ($35M reactive vs $5.5M predictive).
- **100% Scroll (Reform)**: Nodes re-crystallize into a stable, pristine aerodynamic glyph at dawn.

### 3. Mouse Interaction
- Subtle parallax translation (15–25px viewport displacement) with spring easing (`createSpring` / damp factor ~0.15) tracking cursor coordinates relative to screen center.

---

## Section B: Atmos (atmos.leeroy.ca) Scroll Model

### 1. Camera & Coordinate Space
- **Persistent Unified World**: Zero page jumps or separate DOM viewports. Single continuous Three.js perspective camera riding a 3D rail modeled with a `CatmullRomCurve3` centripetal spline.
- **Coordinate progression**: Smooth camera translation from wide atmospheric approach $\rightarrow$ tight mechanical ingestion $\rightarrow$ orbit around core spool $\rightarrow$ high-altitude climb to dawn.

### 2. Atmosphere & Sky Shader
- Dynamic hemispherical shader driven by scalar health metric $uHealth \in [0.0, 1.0]$:
  - $uHealth = 1.0$: Crisp daytime stratosphere (Deep aero blue `#0b1a30` to pale cyan `#88c8f7`).
  - $uHealth = 0.6$: Twilight / Degradation onset (Violet dusk `#1a0f2e` to warm amber `#c25e1a` horizon).
  - $uHealth = 0.2$: Critical alarm night (Deep void `#05020a` to searing emergency red `#8b1010` horizon glow).
  - Dawn ($p \to 1.0$): Morning replenishment (Fresh cerulean dawn with gold horizon flare).

### 3. Typography & UI Integration
- Non-boxed, cardless HUD layout.
- Anchored callouts tethered by hairline leader lines connecting 3D world anchors directly to 2D screen positions via camera unproject/matrix math.
- Station scrubber on the bottom viewport providing instant jump anchors.
- Kinetic character-split typography driven by anime.js stagger transitions.

---

## What to Copy vs What to Avoid

### What to Copy
- Unified continuous WebGL canvas with a single requestAnimationFrame loop.
- One normalized scroll conductor driving both Three.js camera position and SVG figure states.
- Cardless anchored callouts with dynamic screen-space projection.
- Strict telemetry palette: Cyan (Healthy) $\rightarrow$ Amber (Onset / Warning) $\rightarrow$ Crimson (Critical).

### What to Avoid
- Heavy video textures or gigantic GLTF files (use procedural Three.js geometry for ultra-fast instant 60 FPS loading).
- Choppy window scroll listeners (use Lenis smooth scroll conductor with frame-rate independent interpolation).
- Cluttered card overlays that obscure the 3D turbofan asset.

