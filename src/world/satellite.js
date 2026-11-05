/**
 * Satellite: High-fidelity procedural Earth-observation & communications satellite digital twin.
 * Engineered with PBR materials, decoupled aerospace subsystems, realistic textures,
 * and reversible 6-axis exploded assembly.
 */
import * as THREE from 'three';

// ── Procedural Aerospace Textures ───────────────────────────────────────────

/**
 * Creates an authentic gold multilayer insulation (MLI) crinkled foil texture.
 */
function createGoldMLITexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  // Base gold foil hue
  ctx.fillStyle = '#b8860b';
  ctx.fillRect(0, 0, 512, 512);

  // Layered specular facet noise for crinkled thermal foil appearance
  for (let i = 0; i < 400; i++) {
    const x = Math.random() * 512;
    const y = Math.random() * 512;
    const w = 15 + Math.random() * 45;
    const h = 15 + Math.random() * 45;
    const brightness = Math.random();
    ctx.fillStyle = brightness > 0.5 
      ? `rgba(255, 215, 0, ${0.15 + brightness * 0.25})` 
      : `rgba(139, 90, 0, ${0.2 + (1 - brightness) * 0.3})`;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + w, y + (Math.random() - 0.5) * 10);
    ctx.lineTo(x + w + (Math.random() - 0.5) * 15, y + h);
    ctx.lineTo(x - (Math.random() - 0.5) * 10, y + h);
    ctx.closePath();
    ctx.fill();
  }

  // Quilted insulation seam lines
  ctx.strokeStyle = 'rgba(80, 50, 0, 0.45)';
  ctx.lineWidth = 1.5;
  const step = 64;
  for (let x = 0; x <= 512; x += step) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, 512);
    ctx.stroke();
  }
  for (let y = 0; y <= 512; y += step) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(512, y);
    ctx.stroke();
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  return texture;
}

/**
 * Creates high-efficiency multi-junction photovoltaic solar cell texture
 * with deep space indigo-black cells and fine silver collection grid lines.
 */
function createSolarCellTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  // Deep space photovoltaic semiconductor substrate
  ctx.fillStyle = '#060c18';
  ctx.fillRect(0, 0, 1024, 512);

  const cols = 8;
  const rows = 4;
  const cellW = 1024 / cols;
  const cellH = 512 / rows;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = c * cellW;
      const y = r * cellH;

      // Anti-reflective coating gradient (subtle deep blue iridescence)
      const grad = ctx.createLinearGradient(x, y, x + cellW, y + cellH);
      grad.addColorStop(0, '#0a1630');
      grad.addColorStop(0.5, '#071022');
      grad.addColorStop(1, '#0c1a38');
      ctx.fillStyle = grad;
      ctx.fillRect(x + 2, y + 2, cellW - 4, cellH - 4);

      // Fine silver collection fingers (sub-grid lines)
      ctx.strokeStyle = 'rgba(190, 215, 245, 0.25)';
      ctx.lineWidth = 1;
      const subLines = 10;
      for (let s = 1; s < subLines; s++) {
        const lx = x + (s * cellW) / subLines;
        ctx.beginPath();
        ctx.moveTo(lx, y + 3);
        ctx.lineTo(lx, y + cellH - 3);
        ctx.stroke();
      }

      // Main silver busbars (2 horizontal current collectors)
      ctx.strokeStyle = 'rgba(235, 245, 255, 0.75)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x + 2, y + cellH * 0.33);
      ctx.lineTo(x + cellW - 2, y + cellH * 0.33);
      ctx.moveTo(x + 2, y + cellH * 0.67);
      ctx.lineTo(x + cellW - 2, y + cellH * 0.67);
      ctx.stroke();
    }
  }

  // Inter-cell isolation channels
  ctx.strokeStyle = '#020408';
  ctx.lineWidth = 3;
  for (let c = 0; c <= cols; c++) {
    ctx.beginPath();
    ctx.moveTo(c * cellW, 0);
    ctx.lineTo(c * cellW, 512);
    ctx.stroke();
  }
  for (let r = 0; r <= rows; r++) {
    ctx.beginPath();
    ctx.moveTo(0, r * cellH);
    ctx.lineTo(1024, r * cellH);
    ctx.stroke();
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  return texture;
}

/**
 * Creates optical solar reflector (OSR) silver radiator tile texture.
 */
function createRadiatorTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#e2e8f0';
  ctx.fillRect(0, 0, 256, 256);

  // High-emissivity mirror tile grid
  ctx.strokeStyle = '#94a3b8';
  ctx.lineWidth = 1;
  const step = 32;
  for (let i = 0; i <= 256; i += step) {
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, 256);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i);
    ctx.lineTo(256, i);
    ctx.stroke();
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  return texture;
}

// ── Satellite Digital Twin Subsystems ───────────────────────────────────────

export class SatelliteDigitalTwin {
  constructor(scene) {
    this.root = new THREE.Group();
    this.root.name = 'SatelliteRoot';
    this.subsystems = [];
    this.interactiveMeshes = [];

    // Shared procedural textures
    this.goldTex = createGoldMLITexture();
    this.solarTex = createSolarCellTexture();
    this.radiatorTex = createRadiatorTexture();

    // Reusable aerospace PBR materials
    this.materials = {
      goldMLI: new THREE.MeshStandardMaterial({
        color: 0xffd700,
        map: this.goldTex,
        metalness: 0.88,
        roughness: 0.35,
        bumpMap: this.goldTex,
        bumpScale: 0.04,
      }),
      titanium: new THREE.MeshStandardMaterial({
        color: 0x64748b,
        metalness: 0.92,
        roughness: 0.28,
      }),
      aluminium: new THREE.MeshStandardMaterial({
        color: 0x94a3b8,
        metalness: 0.85,
        roughness: 0.32,
      }),
      carbonComposite: new THREE.MeshStandardMaterial({
        color: 0x1e293b,
        metalness: 0.6,
        roughness: 0.45,
      }),
      radiatorOSR: new THREE.MeshStandardMaterial({
        color: 0xf1f5f9,
        map: this.radiatorTex,
        metalness: 0.95,
        roughness: 0.12,
      }),
      solarCell: new THREE.MeshStandardMaterial({
        color: 0x1a2e56,
        map: this.solarTex,
        metalness: 0.82,
        roughness: 0.22,
      }),
      feedHornGold: new THREE.MeshStandardMaterial({
        color: 0xffaa00,
        metalness: 0.95,
        roughness: 0.18,
      }),
      thrusterNozzle: new THREE.MeshStandardMaterial({
        color: 0x334155,
        metalness: 0.95,
        roughness: 0.2,
      }),
      sensorOptics: new THREE.MeshStandardMaterial({
        color: 0x051024,
        metalness: 0.98,
        roughness: 0.05,
      }),
    };

    // Build all decoupled engineering subsystems
    this.buildSpacecraftBus();
    this.buildSolarArrays();
    this.buildCommunicationSystem();
    this.buildPropulsionSystem();
    this.buildAttitudeControlSystem();
    this.buildSensorPayload();

    scene.add(this.root);
  }

  /**
   * Registers a major subsystem with engineering metadata and interaction hooks.
   */
  registerSubsystem(def) {
    this.subsystems.push(def);
    // Tag recursive children for raycasting
    def.group.traverse((child) => {
      if (child.isMesh) {
        child.userData = {
          subsystemId: def.id,
          name: def.name,
          category: def.category,
          subsystemDef: def,
        };
        this.interactiveMeshes.push(child);
      }
    });
  }

  // 1. MAIN SPACECRAFT BUS
  buildSpacecraftBus() {
    const group = new THREE.Group();
    group.name = 'SpacecraftBus';

    // Primary central chassis (Dimensions: 2.4 x 3.6 x 2.2 m scaled)
    const busGeo = new THREE.BoxGeometry(2.4, 3.6, 2.2);
    const busMesh = new THREE.Mesh(busGeo, this.materials.goldMLI);
    group.add(busMesh);

    // Structural corner corner-posts (Titanium extrusion rails)
    const postGeo = new THREE.BoxGeometry(0.08, 3.65, 0.08);
    const postOffsets = [
      [-1.2, 0, -1.1],
      [1.2, 0, -1.1],
      [-1.2, 0, 1.1],
      [1.2, 0, 1.1],
    ];
    postOffsets.forEach(([px, py, pz]) => {
      const post = new THREE.Mesh(postGeo, this.materials.titanium);
      post.position.set(px, py, pz);
      group.add(post);
    });

    // Optical Solar Reflector (OSR) thermal radiator panels on -X and +X faces
    const radGeo = new THREE.PlaneGeometry(1.9, 3.0);
    const radLeft = new THREE.Mesh(radGeo, this.materials.radiatorOSR);
    radLeft.rotation.y = -Math.PI / 2;
    radLeft.position.set(-1.21, 0, 0);
    group.add(radLeft);

    const radRight = new THREE.Mesh(radGeo, this.materials.radiatorOSR);
    radRight.rotation.y = Math.PI / 2;
    radRight.position.set(1.21, 0, 0);
    group.add(radRight);

    // Electronics & Avionics bay access hatches
    const hatchGeo = new THREE.BoxGeometry(0.8, 1.0, 0.05);
    const hatch = new THREE.Mesh(hatchGeo, this.materials.aluminium);
    hatch.position.set(0, 0.6, 1.11);
    group.add(hatch);

    this.root.add(group);

    this.registerSubsystem({
      id: 'bus',
      name: 'SPACECRAFT BUS & THERMAL',
      category: 'Structure & Thermal Control',
      group,
      basePos: new THREE.Vector3(0, 0, 0),
      explodeDir: new THREE.Vector3(0, 0, 0), // core remains anchor
      worldAnchor: () => {
        const v = new THREE.Vector3(0, 1.8, 0);
        group.localToWorld(v);
        return v;
      },
      purpose: 'Houses core avionics, power conditioning, central computers, and structural load frames.',
      inspectionPoints: [
        'Gold Multi-Layer Insulation (MLI) thermal blankets',
        'Optical Solar Reflector (OSR) heat rejection radiators',
        'Titanium primary load-bearing corner trusses',
        'Hermetic avionics equipment bays',
      ],
    });
  }

  // 2. SOLAR ARRAY WINGS
  buildSolarArrays() {
    // Both Left and Right deployable wings
    const sides = [
      { id: 'solar-left', name: 'SOLAR ARRAY (PORT)', sign: -1, labelSide: 'Left' },
      { id: 'solar-right', name: 'SOLAR ARRAY (STARBOARD)', sign: 1, labelSide: 'Right' },
    ];

    sides.forEach(({ id, name, sign }) => {
      const wingGroup = new THREE.Group();
      wingGroup.name = id;

      // Solar Array Drive Mechanism (SADM) rotating yoke
      const yokeGeo = new THREE.CylinderGeometry(0.18, 0.18, 0.6, 16);
      yokeGeo.rotateZ(Math.PI / 2);
      const yoke = new THREE.Mesh(yokeGeo, this.materials.titanium);
      yoke.position.set(sign * 1.5, 0, 0);
      wingGroup.add(yoke);

      const boomGeo = new THREE.BoxGeometry(0.8, 0.1, 0.12);
      const boom = new THREE.Mesh(boomGeo, this.materials.carbonComposite);
      boom.position.set(sign * 1.9, 0, 0);
      wingGroup.add(boom);

      // 3 hinged panel segments per wing
      const panelCount = 3;
      const panelW = 2.4;
      const panelH = 1.6;
      const panelThick = 0.05;

      for (let p = 0; p < panelCount; p++) {
        const panelGroup = new THREE.Group();
        const panelCenterDist = 2.4 + p * (panelW + 0.12);
        panelGroup.position.set(sign * panelCenterDist, 0, 0);

        // Carbon composite structural support backing frame
        const frameGeo = new THREE.BoxGeometry(panelW, panelH, panelThick);
        const frameMesh = new THREE.Mesh(frameGeo, this.materials.carbonComposite);
        panelGroup.add(frameMesh);

        // Active Photovoltaic front cell face (+Z face)
        const cellGeo = new THREE.PlaneGeometry(panelW * 0.96, panelH * 0.94);
        const cellFront = new THREE.Mesh(cellGeo, this.materials.solarCell);
        cellFront.position.set(0, 0, panelThick / 2 + 0.002);
        panelGroup.add(cellFront);

        // Rear cell face for albedo collection
        const cellBack = new THREE.Mesh(cellGeo, this.materials.solarCell);
        cellBack.rotation.y = Math.PI;
        cellBack.position.set(0, 0, -panelThick / 2 - 0.002);
        panelGroup.add(cellBack);

        // Inter-panel deployment hinges
        if (p < panelCount - 1) {
          const hingeGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.35, 12);
          const hinge = new THREE.Mesh(hingeGeo, this.materials.titanium);
          hinge.position.set(sign * (panelW / 2 + 0.06), 0, 0);
          panelGroup.add(hinge);
        }

        wingGroup.add(panelGroup);
      }

      this.root.add(wingGroup);

      this.registerSubsystem({
        id,
        name,
        category: 'Electrical Power Subsystem (EPS)',
        group: wingGroup,
        basePos: new THREE.Vector3(0, 0, 0),
        explodeDir: new THREE.Vector3(sign * 3.5, 0, 0), // separates outward
        worldAnchor: () => {
          const v = new THREE.Vector3(sign * 4.5, 0.9, 0);
          wingGroup.localToWorld(v);
          return v;
        },
        purpose: 'Converts solar irradiance into regulated electrical power (8.5 kW BOL) for payload and bus operations.',
        inspectionPoints: [
          'High-efficiency triple-junction GaAs photovoltaic cells',
          'Solar Array Drive Mechanism (SADM) sun-tracking actuators',
          'Deployable spring-loaded torsion hinges and inter-panel wire harnesses',
          'Ultra-lightweight carbon composite backing frames',
        ],
      });
    });
  }

  // 3. HIGH-GAIN COMMUNICATION SYSTEM
  buildCommunicationSystem() {
    const group = new THREE.Group();
    group.name = 'CommunicationSystem';

    // 2-axis gimbal pointing mast mounted on +Y deck
    const mastGeo = new THREE.CylinderGeometry(0.12, 0.14, 0.8, 16);
    const mast = new THREE.Mesh(mastGeo, this.materials.titanium);
    mast.position.set(0, 2.2, 0.4);
    group.add(mast);

    const gimbalGeo = new THREE.SphereGeometry(0.2, 16, 12);
    const gimbal = new THREE.Mesh(gimbalGeo, this.materials.aluminium);
    gimbal.position.set(0, 2.65, 0.4);
    group.add(gimbal);

    // Parabolic High-Gain Antenna (HGA) Reflector Dish (Diameter 1.8m)
    // Modeled with lathe geometry for realistic parabolic curvature
    const points = [];
    for (let i = 0; i <= 20; i++) {
      const u = i / 20;
      const radius = u * 0.95;
      const depth = u * u * 0.32; // parabolic profile z = a * r^2
      points.push(new THREE.Vector2(radius, depth));
    }
    const dishGeo = new THREE.LatheGeometry(points, 32);
    dishGeo.rotateX(Math.PI / 2 + 0.35); // angled toward Earth ground stations
    const dish = new THREE.Mesh(dishGeo, this.materials.carbonComposite);
    dish.position.set(0, 3.2, 0.5);
    group.add(dish);

    // Feed horn support struts (Tripod assembly)
    const strutMat = this.materials.titanium;
    for (let a = 0; a < 3; a++) {
      const ang = (a / 3) * Math.PI * 2;
      const strutGeo = new THREE.CylinderGeometry(0.015, 0.015, 0.75);
      const strut = new THREE.Mesh(strutGeo, strutMat);
      strut.position.set(
        Math.cos(ang) * 0.4,
        3.2 + Math.sin(ang) * 0.25,
        0.5 + 0.35
      );
      strut.lookAt(0, 3.2, 0.5 + 0.65);
      group.add(strut);
    }

    // Central sub-reflector feed horn
    const feedGeo = new THREE.ConeGeometry(0.09, 0.18, 16);
    feedGeo.rotateX(-Math.PI / 2);
    const feed = new THREE.Mesh(feedGeo, this.materials.feedHornGold);
    feed.position.set(0, 3.2, 0.5 + 0.65);
    group.add(feed);

    // S-Band Low Gain Antennas (LGA) omni-directional masts
    const lgaGeo = new THREE.CylinderGeometry(0.02, 0.02, 0.6);
    const lga1 = new THREE.Mesh(lgaGeo, this.materials.feedHornGold);
    lga1.position.set(-1.1, 1.9, -0.9);
    group.add(lga1);

    const lga2 = new THREE.Mesh(lgaGeo, this.materials.feedHornGold);
    lga2.position.set(1.1, -1.9, 0.9);
    group.add(lga2);

    this.root.add(group);

    this.registerSubsystem({
      id: 'hga',
      name: 'HIGH-GAIN COMMUNICATION DISH',
      category: 'Telemetry, Tracking & Command (TT&C)',
      group,
      basePos: new THREE.Vector3(0, 0, 0),
      explodeDir: new THREE.Vector3(0, 2.6, 1.2), // moves upward and forward
      worldAnchor: () => {
        const v = new THREE.Vector3(0, 3.6, 0.5);
        group.localToWorld(v);
        return v;
      },
      purpose: 'Enables high-bandwidth X/Ka-band downlink (300 Mbps) of raw payload science data and command telemetry.',
      inspectionPoints: [
        'Parabolic carbon-fiber reinforced polymer (CFRP) reflector dish',
        'Precision sub-reflector feed horn with dual-polarization waveguide',
        'Dual-axis gimbal pointing mechanism for continuous ground-station tracking',
        'Omni-directional S-band emergency command antennas',
      ],
    });
  }

  // 4. PROPULSION SYSTEM
  buildPropulsionSystem() {
    const group = new THREE.Group();
    group.name = 'PropulsionSystem';

    // Main Liquid Apogee Engine (LAE) mounted on -Y aft deck
    const nozzlePoints = [
      new THREE.Vector2(0.12, 0),
      new THREE.Vector2(0.09, -0.15), // throat
      new THREE.Vector2(0.24, -0.55),
      new THREE.Vector2(0.42, -1.1),  // bell expansion skirt
    ];
    const nozzleGeo = new THREE.LatheGeometry(nozzlePoints, 24);
    const nozzle = new THREE.Mesh(nozzleGeo, this.materials.thrusterNozzle);
    nozzle.position.set(0, -1.8, 0);
    group.add(nozzle);

    // Heat shield ring around main nozzle
    const shieldGeo = new THREE.TorusGeometry(0.48, 0.08, 12, 24);
    shieldGeo.rotateX(Math.PI / 2);
    const shield = new THREE.Mesh(shieldGeo, this.materials.titanium);
    shield.position.set(0, -1.82, 0);
    group.add(shield);

    // Spherical Titanium Propellant Tanks (Hydrazine & Oxidizer)
    const tankGeo = new THREE.SphereGeometry(0.55, 20, 16);
    const tank1 = new THREE.Mesh(tankGeo, this.materials.titanium);
    tank1.position.set(0, -0.7, 0.45);
    group.add(tank1);

    const tank2 = new THREE.Mesh(tankGeo, this.materials.titanium);
    tank2.position.set(0, -0.7, -0.45);
    group.add(tank2);

    // 4 Corner Reaction Control System (RCS) Quad-Thruster Pods
    const rcsLocations = [
      [-1.18, 1.7, -1.05],
      [1.18, 1.7, -1.05],
      [-1.18, -1.7, -1.05],
      [1.18, -1.7, -1.05],
    ];

    rcsLocations.forEach(([rx, ry, rz]) => {
      const podGroup = new THREE.Group();
      podGroup.position.set(rx, ry, rz);

      const blockGeo = new THREE.BoxGeometry(0.16, 0.16, 0.16);
      const block = new THREE.Mesh(blockGeo, this.materials.aluminium);
      podGroup.add(block);

      // Micro-thruster nozzles (4 per pod pointing in orthogonal vectors)
      const dirs = [
        [0.1, 0, 0, 0, 0, Math.PI / 2],
        [-0.1, 0, 0, 0, 0, -Math.PI / 2],
        [0, 0.1, 0, 0, 0, 0],
        [0, 0, -0.1, Math.PI / 2, 0, 0],
      ];
      dirs.forEach(([dx, dy, dz, rotX, rotY, rotZ]) => {
        const microGeo = new THREE.ConeGeometry(0.03, 0.08, 8);
        const micro = new THREE.Mesh(microGeo, this.materials.thrusterNozzle);
        micro.position.set(dx, dy, dz);
        micro.rotation.set(rotX, rotY, rotZ);
        podGroup.add(micro);
      });

      group.add(podGroup);
    });

    this.root.add(group);

    this.registerSubsystem({
      id: 'prop',
      name: 'ORBITAL PROPULSION SYSTEM',
      category: 'Propulsion & Orbit Correction',
      group,
      basePos: new THREE.Vector3(0, 0, 0),
      explodeDir: new THREE.Vector3(0, -2.8, -0.6), // moves downward and aft
      worldAnchor: () => {
        const v = new THREE.Vector3(0, -2.6, 0);
        group.localToWorld(v);
        return v;
      },
      purpose: 'Performs orbit insertion, station-keeping manoeuvres, and debris-avoidance delta-V burns.',
      inspectionPoints: [
        '400N Bipropellant Liquid Apogee Engine (LAE) with columbium expansion nozzle',
        'Dual titanium propellant storage tanks (MON-3 & MMH)',
        '16-thruster cold-gas Reaction Control System (RCS) pods for 3-axis impulse',
        'High-pressure helium pressurant tanks and isolation manifolds',
      ],
    });
  }

  // 5. ATTITUDE DETERMINATION & CONTROL SYSTEM (ADCS)
  buildAttitudeControlSystem() {
    const group = new THREE.Group();
    group.name = 'AttitudeControlSystem';

    // Dual Star Tracker Baffles angled at 45 degrees away from Earth and Sun
    const baffleGeo = new THREE.CylinderGeometry(0.08, 0.14, 0.35, 16, 1, true);
    baffleGeo.rotateZ(Math.PI / 4);

    const star1 = new THREE.Mesh(baffleGeo, this.materials.aluminium);
    star1.position.set(-0.75, 1.4, -1.15);
    group.add(star1);

    const star2 = new THREE.Mesh(baffleGeo, this.materials.aluminium);
    star2.position.set(0.75, 1.4, -1.15);
    star2.rotation.y = Math.PI;
    group.add(star2);

    // 4 Skewed Reaction Wheels Assembly (pyramid configuration inside housing)
    const rwaHousingGeo = new THREE.BoxGeometry(0.9, 0.45, 0.9);
    const rwaHousing = new THREE.Mesh(rwaHousingGeo, this.materials.titanium);
    rwaHousing.position.set(0, 0.2, -0.65);
    group.add(rwaHousing);

    for (let w = 0; w < 4; w++) {
      const ang = (w / 4) * Math.PI * 2;
      const wheelGeo = new THREE.CylinderGeometry(0.14, 0.14, 0.08, 16);
      wheelGeo.rotateX(Math.PI / 4);
      wheelGeo.rotateY(ang);
      const wheel = new THREE.Mesh(wheelGeo, this.materials.aluminium);
      wheel.position.set(Math.cos(ang) * 0.28, 0.2, -0.65 + Math.sin(ang) * 0.28);
      group.add(wheel);
    }

    // Coarse Sun Sensors (CSS) on outer chassis faces
    const cssGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.04, 8);
    cssGeo.rotateX(Math.PI / 2);
    const css1 = new THREE.Mesh(cssGeo, this.materials.feedHornGold);
    css1.position.set(0, 1.82, 0.8);
    group.add(css1);

    this.root.add(group);

    this.registerSubsystem({
      id: 'adcs',
      name: 'ATTITUDE CONTROL & GUIDANCE (ADCS)',
      category: 'Guidance, Navigation & Control (GNC)',
      group,
      basePos: new THREE.Vector3(0, 0, 0),
      explodeDir: new THREE.Vector3(0, 0.5, -2.4), // separates backward
      worldAnchor: () => {
        const v = new THREE.Vector3(0, 1.4, -1.3);
        group.localToWorld(v);
        return v;
      },
      purpose: 'Measures and maintains spacecraft 3-axis pointing accuracy to within 0.005 degrees using celestial navigation.',
      inspectionPoints: [
        'Dual optical Star Tracker heads with sun-exclusion conical baffles',
        '4-wheel skewed reaction wheel array providing momentum exchange',
        'Redundant ring-laser gyroscopic Inertial Measurement Unit (IMU)',
        'Magnetic torque rods for momentum desaturation via Earth’s field',
      ],
    });
  }

  // 6. EARTH-OBSERVATION / SENSOR PAYLOAD
  buildSensorPayload() {
    const group = new THREE.Group();
    group.name = 'SensorPayload';

    // Primary High-Resolution Optical Telescope / SAR Instrument Barrel
    // Pointing in the Nadir direction (+Z face)
    const barrelGeo = new THREE.CylinderGeometry(0.42, 0.48, 1.4, 24);
    barrelGeo.rotateX(Math.PI / 2);
    const barrel = new THREE.Mesh(barrelGeo, this.materials.carbonComposite);
    barrel.position.set(0, -0.3, 1.6);
    group.add(barrel);

    // Primary optical entrance aperture lens
    const lensGeo = new THREE.CircleGeometry(0.38, 24);
    const lens = new THREE.Mesh(lensGeo, this.materials.sensorOptics);
    lens.position.set(0, -0.3, 2.31);
    group.add(lens);

    // Telescope sun-shade / baffle ring
    const ringGeo = new THREE.TorusGeometry(0.44, 0.04, 12, 24);
    const ring = new THREE.Mesh(ringGeo, this.materials.aluminium);
    ring.position.set(0, -0.3, 2.3);
    group.add(ring);

    // Multi-spectral spectrometer electronics bay
    const specGeo = new THREE.BoxGeometry(0.7, 0.7, 0.5);
    const spec = new THREE.Mesh(specGeo, this.materials.aluminium);
    spec.position.set(0, -0.3, 0.85);
    group.add(spec);

    this.root.add(group);

    this.registerSubsystem({
      id: 'payload',
      name: 'EARTH OBSERVATION SENSOR PAYLOAD',
      category: 'Scientific & Operational Payload',
      group,
      basePos: new THREE.Vector3(0, 0, 0),
      explodeDir: new THREE.Vector3(0, -0.4, 3.2), // moves forward along nadir axis
      worldAnchor: () => {
        const v = new THREE.Vector3(0, -0.3, 2.4);
        group.localToWorld(v);
        return v;
      },
      purpose: 'Acquires sub-meter multi-spectral earth observation telemetry and high-resolution environmental imagery.',
      inspectionPoints: [
        '0.8-meter Ritchey-Chrétien primary optical telescope aperture',
        'Cryo-cooled multi-spectral focal plane detector arrays',
        'Active carbon-composite zero-expansion structural barrel',
        'High-rate on-board image compression and solid-state data recorder',
      ],
    });
  }

  /**
   * Main per-frame update loop.
   * Handles idle floating, rotation, and scroll-driven exploded assembly separation.
   */
  update(time, explode = 0, health = 1.0, isInspectionMode = false) {
    if (!isInspectionMode) {
      // Gentle floating physics in low Earth orbit
      this.root.position.y = Math.sin(time * 0.4) * 0.12;
      // Controlled slow orbital yaw
      this.root.rotation.y = time * 0.04;
      this.root.rotation.x = Math.sin(time * 0.25) * 0.04;
    }

    // Apply exploded assembly displacement along dedicated physical vectors
    const easedExplode = Math.min(1.0, Math.max(0.0, explode));
    this.subsystems.forEach((sub) => {
      const targetPos = sub.basePos.clone().addScaledVector(sub.explodeDir, easedExplode);
      sub.group.position.lerp(targetPos, 0.09);
    });
  }
}

