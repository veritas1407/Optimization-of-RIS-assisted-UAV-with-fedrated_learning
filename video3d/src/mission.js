// One recorded mission as a 3-D scene. Everything that moves is read from the trace JSON.
import * as THREE from 'three';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { P, PV, clamp, lerp, catmull, mulberry32, facadeTex, groundTex, gridTex, glowTex, skyTex, humanTex, outerTex, setSegment, firstHit } from './util.js';

export const DRONE_SCALE = 12, RIS_SCALE = 8, LAMBDA = 0.125;
export const UAV_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'];
const RED = '#d62728', BLUE = '#1f77b4', GREEN = '#35d66b', GOLD = '#ffc933';
const PHASE_SHADES = ['#5a4208', '#a37a12', '#e0ae2c', '#fff0a8'].map((c) => new THREE.Color(c));
const IDLE_GOLD = new THREE.Color('#9a7c1e');

let SHARED = null;
function shared() {
  if (!SHARED) SHARED = { outer: outerTex(), glow: glowTex(), sky: skyTex(), grid: gridTex(), human: humanTex(),
    facades: [facadeTex(11, '#d6cfbf', '#4a6a86'), facadeTex(23, '#c6cbd1', '#44688a'), facadeTex(37, '#d9cdb6', '#527082')] };
  return SHARED;
}

export class Mission {
  constructor(trace, opts = {}) {
    this.tr = trace; this.dt = trace.slot_s; this.M = trace.n_slots; this.T = this.M * this.dt;
    this.nA = trace.n_agents; this.K = trace.n_collectors;
    this.aerial = !!(trace.ris && trace.ris.kind === 'aerial');
    this.carrier = this.aerial ? trace.ris.carrier_uav_index : -1;
    // path knots (three coords): k = 0 is the start, k = 1..M the slot ends
    this.pts = [];
    for (let a = 0; a < this.nA; a++) this.pts.push([PV(trace.start.pos[a]), ...trace.slots.map((s) => PV(s.pos[a]))]);
    this.cumBits = [0]; this.cumEnergy = [0];
    for (const s of trace.slots) { this.cumBits.push(this.cumBits.at(-1) + s.bits_this_slot); this.cumEnergy.push(this.cumEnergy.at(-1) + s.energy_this_slot_J); }
    this.materialsLine = [];
    this.scene = new THREE.Scene();
    this._world(opts); this._sensors(); this._drones(); this._ris(); this._links(); this._extras();
  }

  // ------------------------------------------------------------------ trace queries (also used by check.mjs)
  posAt(t) {
    const tt = clamp(t, 0, this.T), k = Math.min(Math.floor(tt / this.dt), this.M - 1), u = (tt - k * this.dt) / this.dt;
    return this.pts.map((p) => {
      const p1 = p[k], p2 = p[k + 1];
      const p0 = k > 0 ? p[k - 1] : p1.clone().multiplyScalar(2).sub(p2);
      const p3 = k + 2 <= this.M ? p[k + 2] : p2.clone().multiplyScalar(2).sub(p1);
      return catmull(p0, p1, p2, p3, u);
    });
  }
  simPosAt(t) { return this.posAt(t).map((v) => [v.x, -v.z, v.y]); }
  dataLeftAt(t) {
    const tt = clamp(t, 0, this.T), k = Math.min(Math.floor(tt / this.dt), this.M - 1), u = (tt - k * this.dt) / this.dt;
    return this.tr.devices.map((d, i) => lerp(k === 0 ? d.data_bits : this.tr.slots[k - 1].data_left_bits[i], this.tr.slots[k].data_left_bits[i], u));
  }
  countersAt(t) {
    const tt = clamp(t, 0, this.T), k = Math.min(Math.floor(tt / this.dt), this.M - 1), u = (tt - k * this.dt) / this.dt;
    const bits = lerp(this.cumBits[k], this.cumBits[k + 1], u), energy = lerp(this.cumEnergy[k], this.cumEnergy[k + 1], u);
    return { t: tt, bits, energy, ee: energy > 0 ? bits / energy : 0 };
  }
  // links drawn at sim time t; a link exists only inside the slots where the trace lists it
  linksAt(t, fadeSim = 0.5) {
    if (t < 0 || t >= this.T) return [];
    const k = Math.floor(t / this.dt), f = Math.min(fadeSim, 0.3 * this.dt);
    const has = (kk, l) => kk >= 0 && kk < this.M && this.tr.slots[kk].uploads.some((o) => o.device === l.device && o.uav === l.uav && o.via_ris === l.via_ris);
    return this.tr.slots[k].uploads.map((l) => {
      let a = 1;
      if (!has(k - 1, l)) a *= clamp((t - k * this.dt) / f);
      if (!has(k + 1, l)) a *= clamp(((k + 1) * this.dt - t) / f);
      return { ...l, alpha: a };
    });
  }

  // ------------------------------------------------------------------ static world
  _world(opts) {
    const S = shared(), tr = this.tr, A = tr.area_m, sc = this.scene;
    sc.background = S.sky; sc.fog = new THREE.Fog(0xd8d2c4, 900, 3600);
    sc.add(new THREE.HemisphereLight(0xcfe0ff, 0x5a5040, 1.15));
    const sun = new THREE.DirectionalLight(0xffdcb0, 3.1), dir = P(0.85, 0.12, 0.5).normalize();
    sun.position.copy(P(A / 2, A / 2, 0)).addScaledVector(dir, 800); sun.target.position.copy(P(A / 2, A / 2, 0));
    sun.castShadow = true; sun.shadow.mapSize.set(opts.shadow || 4096, opts.shadow || 4096);
    Object.assign(sun.shadow.camera, { left: -430, right: 430, top: 430, bottom: -430, near: 100, far: 1700 });
    sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.6;
    sc.add(sun, sun.target);

    const outer = new THREE.Mesh(new THREE.PlaneGeometry(6000, 6000), new THREE.MeshStandardMaterial({ map: S.outer, roughness: 1 }));
    outer.rotation.x = -Math.PI / 2; outer.position.copy(P(A / 2, A / 2, -0.3)); outer.receiveShadow = true; sc.add(outer);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(A, A), new THREE.MeshStandardMaterial({ map: groundTex(tr), roughness: 0.95 }));
    ground.rotation.x = -Math.PI / 2; ground.position.copy(P(A / 2, A / 2, 0)); ground.receiveShadow = true; sc.add(ground);
    this.grid = new THREE.Mesh(new THREE.PlaneGeometry(A, A), new THREE.MeshBasicMaterial({ map: S.grid, transparent: true, opacity: 0, depthWrite: false }));
    this.grid.rotation.x = -Math.PI / 2; this.grid.position.copy(P(A / 2, A / 2, 0.15)); sc.add(this.grid);

    // context blocks outside the simulated area (decoration only; desaturated, low)
    const r = mulberry32(2026), boxes = [];
    for (let i = 0; i < 900 && boxes.length < 300; i++) {
      const x = -900 + r() * 2300, y = -900 + r() * 2300, w = 22 + r() * 46, d = 22 + r() * 46, h = 5 + r() * 10, m = 45;
      if (x + w / 2 > -m && x - w / 2 < A + m && y + d / 2 > -m && y - d / 2 < A + m) continue;
      boxes.push([x, y, w, d, h]);
    }
    const ctx = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: 0xb9bbba, roughness: 1 }), boxes.length);
    const mtx = new THREE.Matrix4();
    boxes.forEach(([x, y, w, d, h], i) => { mtx.compose(P(x, y, h / 2), new THREE.Quaternion(), new THREE.Vector3(w, h, d)); ctx.setMatrixAt(i, mtx); });
    ctx.receiveShadow = true; sc.add(ctx);

    // the simulated buildings (exact footprints and heights from the trace)
    tr.buildings.forEach((b, i) => {
      const w = b.x1 - b.x0, d = b.y1 - b.y0, h = b.height_m, base = S.facades[i % S.facades.length];
      const side = (len) => { const t = base.clone(); t.needsUpdate = true; t.repeat.set(len / 16, h / 13.6); return new THREE.MeshStandardMaterial({ map: t, roughness: 0.85 }); };
      const roof = new THREE.MeshStandardMaterial({ color: 0x9a9b9c, roughness: 0.95 });
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), [side(d), side(d), roof, roof, side(w), side(w)]);
      mesh.position.copy(P((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2, h / 2)); mesh.castShadow = mesh.receiveShadow = true; sc.add(mesh);
      const par = new THREE.Mesh(new THREE.BoxGeometry(w + 0.6, 0.9, d + 0.6), new THREE.MeshStandardMaterial({ color: 0x9c9d9e, roughness: 1 }));
      par.position.copy(P((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2, h + 0.3)); par.castShadow = true; sc.add(par);
      // rooftop units (decoration), kept away from the RIS mount
      const rr = mulberry32(100 + i), unitM = new THREE.MeshStandardMaterial({ color: 0xc9cbcc, roughness: 0.8 }), rc = tr.ris && tr.ris.center;
      for (let u = 0; u < Math.round((w * d) / 700); u++) {
        const ux = b.x0 + 5 + rr() * (w - 10), uy = b.y0 + 5 + rr() * (d - 10), uw = 3 + rr() * 4, ud = 3 + rr() * 4, uh = 1.4 + rr() * 1.6;
        if (rc && Math.hypot(ux - rc[0], uy - rc[1]) < 14) continue;
        const um = new THREE.Mesh(new THREE.BoxGeometry(uw, uh, ud), unitM); um.position.copy(P(ux, uy, h + uh / 2)); um.castShadow = um.receiveShadow = true; sc.add(um);
      }
    });

    // area boundary, start pads, landing zone
    this.boundary = this._fatLine([[0, 0, 0.4], [A, 0, 0.4], [A, A, 0.4], [0, A, 0.4], [0, 0, 0.4]], '#ffffff', 2.2, 0.6); sc.add(this.boundary);
    const fz = tr.final_zone, ring = new THREE.Mesh(new THREE.RingGeometry(fz.radius_m - 1.6, fz.radius_m, 96), new THREE.MeshBasicMaterial({ color: 0xff5fd2, transparent: true, opacity: 0.9, side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2; ring.position.copy(P(fz.center[0], fz.center[1], 0.5)); sc.add(ring);
    const disc = new THREE.Mesh(new THREE.CircleGeometry(fz.radius_m, 64), new THREE.MeshBasicMaterial({ color: 0xff5fd2, transparent: true, opacity: 0.1, depthWrite: false }));
    disc.rotation.x = -Math.PI / 2; disc.position.copy(P(fz.center[0], fz.center[1], 0.35)); sc.add(disc);
    tr.start.pos.slice(0, this.K).forEach((p) => {
      const pad = new THREE.Mesh(new THREE.RingGeometry(4.5, 6, 40), new THREE.MeshBasicMaterial({ color: 0x7dffa8, transparent: true, opacity: 0.85, side: THREE.DoubleSide }));
      pad.rotation.x = -Math.PI / 2; pad.position.copy(P(p[0], p[1], 0.5)); sc.add(pad);
    });
  }

  _fatLine(simPts, color, width, opacity = 1, vertexColors = false) {
    const g = new LineGeometry(); g.setPositions(simPts.flatMap((p) => [p[0], p[2], -p[1]]));
    const m = new LineMaterial({ color, linewidth: width, transparent: opacity < 1 || vertexColors, opacity, vertexColors, depthWrite: false });
    this.materialsLine.push(m);
    return new Line2(g, m);
  }
  // update a Line2 in place (positions/colours as flat xyz arrays of the polyline's points)
  _setLine(line, pts, cols) {
    const g = line.geometry;
    if (!g.attributes.instanceColorStart) { g.setPositions(pts); g.setColors(cols); return; }
    const ps = g.attributes.instanceStart.data, cs = g.attributes.instanceColorStart.data, n = pts.length / 3 - 1;
    for (let i = 0; i < n; i++) for (let j = 0; j < 6; j++) { ps.array[i * 6 + j] = pts[i * 3 + j]; cs.array[i * 6 + j] = cols[i * 3 + j]; }
    ps.needsUpdate = true; cs.needsUpdate = true;
  }
  setResolution(w, h) { this.materialsLine.forEach((m) => m.resolution.set(w, h)); }

  // ------------------------------------------------------------------ sensors (markers enlarged; the data bar shows data still waiting)
  _sensors() {
    const S = shared(); this.sensors = [];
    for (const d of this.tr.devices) {
      const col = new THREE.Color(d.in_courtyard ? RED : BLUE), g = new THREE.Group(); g.position.copy(P(d.xyz[0], d.xyz[1], 0));
      const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.45, 6, 10), new THREE.MeshStandardMaterial({ color: 0x30343a, roughness: 0.7 })); pole.position.y = 3; pole.castShadow = true;
      const boxM = new THREE.MeshStandardMaterial({ color: col.clone(), emissive: col.clone(), emissiveIntensity: 0.5, roughness: 0.5 });
      const box = new THREE.Mesh(new THREE.BoxGeometry(2.8, 2.8, 2.8), boxM); box.position.y = 7.4; box.castShadow = true;
      const H0 = 0.55 * d.data_bits / 1e6;                                       // 0.55 m of bar per Mbit
      const barM = new THREE.MeshBasicMaterial({ color: col.clone().lerp(new THREE.Color('#ffffff'), 0.25), transparent: true, opacity: 0.92 });
      const bar = new THREE.Mesh(new THREE.BoxGeometry(2, 1, 2), barM); bar.position.set(3.4, H0 / 2, 0);
      const frame = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(2.1, H0, 2.1)), new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 })); frame.position.set(3.4, H0 / 2, 0);
      const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: S.glow, color: col.clone().lerp(new THREE.Color('#ffffff'), 0.2), transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }));
      glow.position.y = 7.4; glow.scale.setScalar(26);
      const ringM = new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false });
      const ring = new THREE.Mesh(new THREE.RingGeometry(6.2, 7.6, 40), ringM); ring.rotation.x = -Math.PI / 2; ring.position.y = 0.45;
      g.add(pole, box, bar, frame, glow, ring); this.scene.add(g);
      this.sensors.push({ d, col, g, boxM, bar, barM, frame, glow, ringM, ring, H0 });
    }
  }

  // ------------------------------------------------------------------ drones (drawn x12)
  _drones() {
    const S = shared(); this.drones = []; this.trails = [];
    for (let a = 0; a < this.nA; a++) {
      const col = new THREE.Color(UAV_COLORS[a === this.carrier ? 3 : a]), g = new THREE.Group();
      const bodyM = new THREE.MeshStandardMaterial({ color: col, emissive: col, emissiveIntensity: 0.55, roughness: 0.4 });
      const body = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.09, 0.26), bodyM); body.castShadow = true; g.add(body);
      const nose = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.05, 0.06), new THREE.MeshBasicMaterial({ color: 0xffffff })); nose.position.set(0, 0.01, 0.15); g.add(nose);
      const dark = new THREE.MeshStandardMaterial({ color: 0x1c1f24, roughness: 0.6 });
      for (const [sx, sz] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
        const arm = new THREE.Mesh(new THREE.BoxGeometry(0.035, 0.03, 0.34), dark); arm.position.set(sx * 0.115, 0, sz * 0.115); arm.rotation.y = sx * sz > 0 ? Math.PI / 4 : -Math.PI / 4; arm.castShadow = true; g.add(arm);
        const rotor = new THREE.Mesh(new THREE.CylinderGeometry(0.135, 0.135, 0.012, 24), new THREE.MeshStandardMaterial({ color: 0xe8eef6, transparent: true, opacity: 0.45, roughness: 0.3 }));
        rotor.position.set(sx * 0.23, 0.05, sz * 0.23); rotor.castShadow = true; g.add(rotor);
        const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.07, 10), dark); hub.position.set(sx * 0.23, 0.02, sz * 0.23); g.add(hub);
      }
      g.scale.setScalar(DRONE_SCALE);
      const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: S.glow, color: col, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false })); halo.scale.setScalar(20);
      const holder = new THREE.Group(); holder.add(g, halo); this.scene.add(holder);
      const drop = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 6), new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.22, depthWrite: false })); this.scene.add(drop);
      this.drones.push({ holder, g, drop, col });
      const trail = this._fatLine(Array.from({ length: 81 }, () => [0, 0, 0]), '#ffffff', 3.2, 0.95, true); trail.frustumCulled = false; this.scene.add(trail);
      this.trails.push(trail);
    }
  }

  // ------------------------------------------------------------------ RIS panel (16 x 16 tiles; drawn x8 except in the true-scale close-up)
  _ris() {
    const r = this.tr.ris; this.ris = null; if (!r) return;
    const n3 = this.aerial ? new THREE.Vector3(0, -1, 0) : new THREE.Vector3(r.normal[0], r.normal[2], -r.normal[1]).normalize();
    let e1 = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), n3); if (e1.lengthSq() < 1e-6) e1.set(1, 0, 0); e1.normalize();
    const e2 = new THREE.Vector3().crossVectors(n3, e1).normalize(), s = r.element_spacing_m, N = r.elements[0];
    const g = new THREE.Group(); g.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(e1, e2, n3));
    const back = new THREE.Mesh(new THREE.BoxGeometry(N * s + 0.08, N * s + 0.08, 0.04), new THREE.MeshStandardMaterial({ color: 0x8a6f24, roughness: 0.55, emissive: 0x3a2c08, emissiveIntensity: 0.6 })); back.position.z = -0.045; back.castShadow = true; g.add(back);
    const tiles = new THREE.InstancedMesh(new THREE.PlaneGeometry(s * 0.88, s * 0.88), new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, toneMapped: false }), N * N);
    const m = new THREE.Matrix4(), local = [];
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { const x = (i - (N - 1) / 2) * s, y = (j - (N - 1) / 2) * s; local.push([x, y]); m.makeTranslation(x, y, 0.004); tiles.setMatrixAt(j * N + i, m); tiles.setColorAt(j * N + i, IDLE_GOLD); }
    g.add(tiles); this.scene.add(g);
    const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: shared().glow, color: 0xffd66b, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false })); this.scene.add(glow);
    this.ris = { g, tiles, local, e1, e2, n3, glow, center: this.aerial ? null : PV(r.center), target: null };
    if (!this.aerial) {
      g.position.copy(this.ris.center);
      // mast on the roof edge behind the panel (true scale)
      const c = r.center, hx = -r.normal[0], hy = -r.normal[1], hn = Math.hypot(hx, hy) || 1, roof = Math.max(...this.tr.buildings.map((b) => b.height_m));
      const bx = c[0] + (hx / hn) * 0.9, by = c[1] + (hy / hn) * 0.9, steel = new THREE.MeshStandardMaterial({ color: 0x3a3d42, roughness: 0.5 });
      const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.09, c[2] - roof + 0.7, 10), steel); pole.position.copy(P(bx, by, (c[2] + roof + 0.7) / 2)); pole.castShadow = true;
      const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 1, 8), steel); setSegment(arm, P(bx, by, c[2]), P(c[0], c[1], c[2]), 1); arm.scale.x = arm.scale.z = 1;
      this.scene.add(pole, arm);
    }
  }
  // Display position of the panel centre. The building panel keeps its bottom edge in place when drawn larger,
  // so the enlarged panel does not sink into the roof (a display-only shift of at most 3.5 m).
  risCenter(pos, scale = RIS_SCALE) {
    if (!this.ris) return null;
    if (this.aerial) return pos[this.carrier].clone().add(new THREE.Vector3(0, -4.5, 0));
    return this.ris.center.clone().addScaledVector(this.ris.e2, (scale - 1) * 0.5 * this.tr.ris.side_m);
  }

  // ------------------------------------------------------------------ radio links, pulses, extras
  _links() {
    const cyl = new THREE.CylinderGeometry(1, 1, 1, 10, 1, true), sph = new THREE.SphereGeometry(1, 12, 10);
    this.linkPool = []; this.pulsePool = [];
    for (let i = 0; i < 8; i++) {
      const core = new THREE.Mesh(cyl, new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, toneMapped: false }));
      const halo = new THREE.Mesh(cyl, new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false }));
      core.visible = halo.visible = false; this.scene.add(core, halo); this.linkPool.push({ core, halo });
    }
    for (let i = 0; i < 16; i++) { const p = new THREE.Mesh(sph, new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, toneMapped: false })); p.visible = false; this.scene.add(p); this.pulsePool.push(p); }
  }
  _extras() {
    const S = shared();
    this.human = new THREE.Sprite(new THREE.SpriteMaterial({ map: S.human, transparent: true })); this.human.scale.set(0.6, 1.8, 1); this.human.center.set(0.5, 0); this.human.visible = false; this.scene.add(this.human);
    // "blocked direct path" illustration (geometry computed from the trace's buildings)
    const red = new THREE.MeshBasicMaterial({ color: 0xff3b30, transparent: true, opacity: 0.95, depthWrite: false, toneMapped: false });
    this.blk = { beam: new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 10), red), halo: new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 10), new THREE.MeshBasicMaterial({ color: 0xff3b30, transparent: true, opacity: 0.25, blending: THREE.AdditiveBlending, depthWrite: false })),
      x1: new THREE.Mesh(new THREE.BoxGeometry(9, 1.3, 1.3), red), x2: new THREE.Mesh(new THREE.BoxGeometry(9, 1.3, 1.3), red), dots: [] };
    for (let i = 0; i < 40; i++) { const dmesh = new THREE.Mesh(new THREE.SphereGeometry(0.8, 8, 6), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55, depthWrite: false })); this.blk.dots.push(dmesh); this.scene.add(dmesh); }
    this.scene.add(this.blk.beam, this.blk.halo, this.blk.x1, this.blk.x2); this.hideBlocked();
  }
  // run fn with the radio links hidden (used for the picture-in-picture of the panel)
  withoutLinks(fn) {
    const all = [...this.linkPool.flatMap((l) => [l.core, l.halo]), ...this.pulsePool, this.ris ? this.ris.glow : null].filter(Boolean), was = all.map((m) => m.visible);
    all.forEach((m) => (m.visible = false)); fn(); all.forEach((m, i) => (m.visible = was[i]));
  }
  hideBlocked() { const b = this.blk; b.beam.visible = b.halo.visible = b.x1.visible = b.x2.visible = false; b.dots.forEach((d) => (d.visible = false)); }
  // Direct sensor -> drone path at sim time t; returns the wall hit (sim coords) or null if the path is clear.
  showBlocked(device, uav, t, progress) {
    const a = this.tr.devices[device].xyz.slice(); a[2] = 7.4; const b = this.simPosAt(t)[uav];
    const hit = firstHit([a[0], a[1], 0.5], b, this.tr.buildings); this.hideBlocked(); if (!hit) return null;
    const bl = this.blk, A3 = P(...a), H3 = P(...hit.point), u = clamp(progress), tip = A3.clone().lerp(H3, u);
    setSegment(bl.beam, A3, tip, 0.55); setSegment(bl.halo, A3, tip, 1.6); bl.beam.visible = bl.halo.visible = u > 0.01;
    if (u >= 1) {
      const n = H3.clone().sub(A3).normalize(); bl.x1.position.copy(H3).addScaledVector(n, -0.8); bl.x2.position.copy(bl.x1.position);
      bl.x1.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), n); bl.x2.quaternion.copy(bl.x1.quaternion);
      bl.x1.rotateZ(Math.PI / 4); bl.x2.rotateZ(-Math.PI / 4); bl.x1.visible = bl.x2.visible = true;
      const O3 = P(...hit.out), B3 = P(...b), len = O3.distanceTo(B3), nd = Math.min(bl.dots.length, Math.floor(len / 7));
      for (let i = 0; i < nd; i++) { bl.dots[i].position.copy(O3).lerp(B3, (i + 0.5) / nd); bl.dots[i].visible = true; }
    }
    return { hit: hit.point, sensor: a, drone: b };
  }

  // ------------------------------------------------------------------ per-frame update
  // o: { links, fadeSim, vt (video time, s), risScale, trails, human, risGlow }
  update(t, o = {}) {
    const pos = this.posAt(t), left = this.dataLeftAt(t), vt = o.vt || 0, tr = this.tr;
    // drones + trails
    const ahead = this.posAt(t + 0.4), behind = this.posAt(t - 0.4);
    this.drones.forEach((dr, a) => {
      dr.holder.position.copy(pos[a]);
      const dx = ahead[a].x - behind[a].x, dz = ahead[a].z - behind[a].z; if (Math.hypot(dx, dz) > 0.2) dr.g.rotation.y = Math.atan2(dx, dz);
      setSegment(dr.drop, new THREE.Vector3(pos[a].x, 0.2, pos[a].z), pos[a], 0.25);
      const arr = [], cols = [], base = dr.col, dim = new THREE.Color('#5d6658');
      for (let i = 0; i <= 80; i++) { const tp = Math.max(t - i * 0.25, 0), p = this.posAt(tp)[a], f = Math.pow(i / 80, 0.7); arr.push(p.x, p.y, p.z); const c = base.clone().lerp(dim, f); cols.push(c.r, c.g, c.b); }
      this._setLine(this.trails[a], arr, cols); this.trails[a].visible = o.trails !== false && t > 0.05;
    });
    // sensors
    this.sensors.forEach((s, i) => {
      const d = s.d, awake = t >= d.awake_from_s && t < d.awake_until_s && left[i] > 1, done = left[i] <= 1, closed = t >= d.awake_until_s;
      const h = Math.max(s.H0 * left[i] / d.data_bits, 0.02); s.bar.scale.y = h; s.bar.position.y = h / 2; s.bar.visible = !done;
      const grey = new THREE.Color('#7b8087'), c = s.col.clone().lerp(grey, awake ? 0 : done || closed ? 0.8 : 0.45);
      s.boxM.color.copy(c); s.boxM.emissive.copy(c); s.boxM.emissiveIntensity = awake ? 1.1 : 0.25;
      s.barM.opacity = awake ? 0.95 : 0.5; s.frame.material.opacity = done || closed ? 0.2 : 0.5;
      s.glow.material.opacity = awake ? 0.75 + 0.15 * Math.sin(vt * 5 + i) : 0;
      s.ringM.opacity = awake ? 0.75 : 0; const rs = 1 + 0.1 * Math.sin(vt * 5 + i); s.ring.scale.set(rs, rs, 1);
    });
    // RIS panel: position, display scale and illustrative 2-bit phase pattern for the link it currently serves
    const links = o.links === false ? [] : this.linksAt(t, o.fadeSim ?? 0.5), rc = this.risCenter(pos, o.risScale ?? RIS_SCALE);
    if (this.ris) {
      const R = this.ris, scale = o.risScale ?? RIS_SCALE; R.g.scale.setScalar(scale); R.g.position.copy(rc);
      const k = Math.min(Math.floor(clamp(t, 0, this.T - 1e-6) / this.dt), this.M - 1), via = links.filter((l) => l.via_ris);
      R.target = null;
      if (via.length) {
        const tgt = via[Math.min(Math.floor(((t - k * this.dt) / this.dt) * via.length), via.length - 1)]; R.target = tgt;
        const sp = this.simPosAt(t), cT = this.aerial ? [sp[this.carrier][0], sp[this.carrier][1], sp[this.carrier][2] - 0.3] : tr.ris.center;
        const cen = P(...cT), sen = P(...tr.devices[tgt.device].xyz), drn = pos[tgt.uav], e = new THREE.Vector3();
        R.local.forEach(([x, y], idx) => {
          e.copy(cen).addScaledVector(R.e1, x).addScaledVector(R.e2, y);
          const th = (((-2 * Math.PI / LAMBDA) * (e.distanceTo(sen) + e.distanceTo(drn))) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI);
          R.tiles.setColorAt(idx, PHASE_SHADES[Math.round(th / (Math.PI / 2)) % 4]);
        });
      } else R.local.forEach((_, idx) => R.tiles.setColorAt(idx, IDLE_GOLD));
      R.tiles.instanceColor.needsUpdate = true;
      R.glow.position.copy(rc); R.glow.scale.setScalar(5 * scale); R.glow.material.opacity = (via.length ? 0.55 : 0.12) * (o.risGlow ?? 1);
    }
    // links (only those listed by the trace for the current slot) with travelling pulses
    let li = 0, pi = 0;
    for (const l of links) {
      const s3 = P(tr.devices[l.device].xyz[0], tr.devices[l.device].xyz[1], 7.4), d3 = pos[l.uav], col = new THREE.Color(l.via_ris ? GOLD : GREEN);
      const path = l.via_ris && rc ? [s3, rc, d3] : [s3, d3];
      for (let j = 0; j + 1 < path.length && li < this.linkPool.length; j++, li++) {
        const { core, halo } = this.linkPool[li]; setSegment(core, path[j], path[j + 1], 0.55); setSegment(halo, path[j], path[j + 1], 1.7);
        core.material.color.copy(col); halo.material.color.copy(col); core.material.opacity = 0.9 * l.alpha; halo.material.opacity = 0.22 * l.alpha; core.visible = halo.visible = l.alpha > 0.01;
      }
      const lens = path.slice(1).map((p, j) => p.distanceTo(path[j])), total = lens.reduce((a, b) => a + b, 0);
      for (let q = 0; q < 3 && pi < this.pulsePool.length; q++, pi++) {
        let s = ((vt * 0.75 + q / 3 + l.device * 0.137) % 1) * total, j = 0; while (j < lens.length - 1 && s > lens[j]) { s -= lens[j]; j++; }
        const p = this.pulsePool[pi]; p.position.copy(path[j]).lerp(path[j + 1], clamp(s / lens[j])); p.scale.setScalar(1.9);
        p.material.color.copy(col).lerp(new THREE.Color('#ffffff'), 0.55); p.material.opacity = l.alpha; p.visible = l.alpha > 0.01;
      }
    }
    for (; li < this.linkPool.length; li++) this.linkPool[li].core.visible = this.linkPool[li].halo.visible = false;
    for (; pi < this.pulsePool.length; pi++) this.pulsePool[pi].visible = false;
    // human silhouette for the true-scale close-up (on the roof next to the mast)
    this.human.visible = !!o.human && !this.aerial && !!this.ris;
    if (this.human.visible) { const c = tr.ris.center, roof = Math.max(...tr.buildings.map((b) => b.height_m)); this.human.position.copy(P(c[0] + 1.9, c[1] - 1.2, roof + 0.75)); }
    return { pos, links, rc };
  }
}
