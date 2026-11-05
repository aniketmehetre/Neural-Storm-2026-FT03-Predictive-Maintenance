/**
 * Engine: Procedural cutaway turbofan digital twin with 6 distinct assemblies.
 * Explodes on scroll and heats/tints based on subsystem degradation.
 */
import * as THREE from 'three';

export class TurbofanEngine {
  constructor(scene) {
    this.group = new THREE.Group();
    this.assemblies = [];

    // Central core shaft
    const shaftGeo = new THREE.CylinderGeometry(0.35, 0.35, 14, 32);
    shaftGeo.rotateZ(Math.PI / 2);
    const shaftMat = new THREE.MeshStandardMaterial({
      color: 0x242e3f,
      metalness: 0.9,
      roughness: 0.25,
    });
    this.shaft = new THREE.Mesh(shaftGeo, shaftMat);
    this.group.add(this.shaft);

    // Assembly definitions along X-axis
    const stationDefs = [
      { name: 'FAN',  x: -5.0, rIn: 0.8, rOut: 2.8, blades: 24, length: 1.2, speedMult: 0.6 },
      { name: 'LPC',  x: -2.8, rIn: 0.7, rOut: 1.8, blades: 20, length: 1.0, speedMult: 0.8 },
      { name: 'HPC',  x: -0.8, rIn: 0.6, rOut: 1.4, blades: 28, length: 1.6, speedMult: 1.4, isHPC: true },
      { name: 'COMB', x: 1.2,  rIn: 0.6, rOut: 1.3, blades: 0,  length: 1.1, isCombustor: true },
      { name: 'HPT',  x: 2.8,  rIn: 0.65, rOut: 1.35, blades: 22, length: 0.8, speedMult: 1.8 },
      { name: 'LPT',  x: 4.4,  rIn: 0.7,  rOut: 1.6, blades: 18, length: 1.2, speedMult: 1.2 },
    ];

    stationDefs.forEach((def, index) => {
      const assyGroup = new THREE.Group();
      assyGroup.position.x = def.x;

      // 1. Semi-cylindrical cutaway casing (half-cut LatheGeometry or cylinder)
      const casingMat = new THREE.MeshStandardMaterial({
        color: 0x1a2333,
        metalness: 0.85,
        roughness: 0.3,
        side: THREE.DoubleSide,
      });

      // Half-cylinder cutaway (PI radians)
      const casingGeo = new THREE.CylinderGeometry(
        def.rOut * 1.05,
        def.rOut * 1.05,
        def.length,
        32,
        1,
        true,
        0,
        Math.PI * 1.15
      );
      casingGeo.rotateZ(Math.PI / 2);
      const casingMesh = new THREE.Mesh(casingGeo, casingMat);
      assyGroup.add(casingMesh);

      // 2. Hub
      const hubGeo = new THREE.CylinderGeometry(def.rIn, def.rIn, def.length * 0.9, 24);
      hubGeo.rotateZ(Math.PI / 2);
      const hubMat = new THREE.MeshStandardMaterial({
        color: 0x334155,
        metalness: 0.8,
        roughness: 0.4,
      });
      const hubMesh = new THREE.Mesh(hubGeo, hubMat);
      assyGroup.add(hubMesh);

      // 3. Rotor Blades or Combustor Chamber
      let rotorGroup = null;
      if (def.isCombustor) {
        // Glowing combustion ring
        const torusGeo = new THREE.TorusGeometry(def.rIn + 0.35, 0.22, 16, 32);
        torusGeo.rotateY(Math.PI / 2);
        const combMat = new THREE.MeshStandardMaterial({
          color: 0xff7b22,
          emissive: 0xff4800,
          emissiveIntensity: 0.8,
          roughness: 0.2,
        });
        const combMesh = new THREE.Mesh(torusGeo, combMat);
        assyGroup.add(combMesh);
      } else if (def.blades > 0) {
        rotorGroup = new THREE.Group();
        const bladeHeight = def.rOut - def.rIn;
        const bladeGeo = new THREE.BoxGeometry(0.06, bladeHeight, 0.22);
        bladeGeo.translate(0, def.rIn + bladeHeight / 2, 0);

        const bladeMat = new THREE.MeshStandardMaterial({
          color: 0x64748b,
          metalness: 0.9,
          roughness: 0.2,
        });

        const instMesh = new THREE.InstancedMesh(bladeGeo, bladeMat, def.blades);
        const dummy = new THREE.Object3D();

        for (let b = 0; b < def.blades; b++) {
          const angle = (b / def.blades) * Math.PI * 2;
          dummy.rotation.x = angle;
          // Blade aerodynamic twist
          dummy.rotation.y = 0.35;
          dummy.updateMatrix();
          instMesh.setMatrixAt(b, dummy.matrix);
        }
        instMesh.instanceMatrix.needsUpdate = true;
        rotorGroup.add(instMesh);
        assyGroup.add(rotorGroup);
      }

      this.group.add(assyGroup);

      this.assemblies.push({
        index,
        name: def.name,
        baseX: def.x,
        group: assyGroup,
        rotor: rotorGroup,
        speedMult: def.speedMult || 1.0,
        isHPC: !!def.isHPC,
        casingMat,
        worldAnchor: () => {
          const v = new THREE.Vector3(0, def.rOut + 0.4, 0);
          assyGroup.localToWorld(v);
          return v;
        },
      });
    });

    scene.add(this.group);
  }

  update(time, explode, heat, velocity = 0) {
    // Spin shaft
    this.shaft.rotation.x += 0.05 + Math.abs(velocity) * 0.1;

    // Explode factor: displace assemblies along X axis
    const maxDisplacement = 2.4;

    this.assemblies.forEach((assy) => {
      // Displace outward from center (HPC at index 2 is near origin)
      const offsetDirection = assy.index - 2.5;
      assy.group.position.x = assy.baseX + offsetDirection * explode * maxDisplacement;

      // Rotate blades
      if (assy.rotor) {
        assy.rotor.rotation.x += (0.04 * assy.speedMult) + Math.abs(velocity) * 0.15;
      }

      // Heat coloring: HPC gets full heat; others get half
      const assyHeat = assy.isHPC ? heat : heat * 0.45;
      if (assyHeat > 0.6) {
        // Red / thermal warning
        assy.casingMat.color.setHex(assy.isHPC ? 0x9b1b1b : 0x4a1f28);
        assy.casingMat.emissive.setHex(assy.isHPC ? 0x660808 : 0x1a0505);
        assy.casingMat.emissiveIntensity = assyHeat * 0.8;
      } else if (assyHeat > 0.25) {
        // Amber onset
        assy.casingMat.color.setHex(assy.isHPC ? 0x8b5413 : 0x2b2433);
        assy.casingMat.emissive.setHex(0x381e05);
        assy.casingMat.emissiveIntensity = assyHeat * 0.5;
      } else {
        // Healthy dark alloy
        assy.casingMat.color.setHex(0x1a2333);
        assy.casingMat.emissive.setHex(0x000000);
        assy.casingMat.emissiveIntensity = 0.0;
      }
    });
  }
}

