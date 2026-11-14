// Stylised federated-learning overview. Area layouts come from fl_areas.json; the packets,
// the cloud node and the padlocks are illustrative (labelled as such in the video).
import * as THREE from 'three';
import { clamp, lerp, smooth, ramp, glowTex, padlockTex } from './util.js';
import { UAV_COLORS } from './mission.js';

// Areas sampled in the first three rounds of the heterogeneous FL run with seed 0:
// numpy.random.default_rng(1000).choice(12, 6, replace=False), sorted, as in v9/fedhet.py.
export const ROUNDS = [[1, 4, 5, 6, 7, 9], [1, 2, 3, 8, 9, 11], [1, 4, 6, 7, 9, 11]];
export const FL_T = { reveal: 3, round: 4, test: 15, end: 21 };

const TILE = 100, GAP = 44, CLOUD = new THREE.Vector3(0, 200, -40);

export class FLScene {
  constructor(data) {
    this.scene = new THREE.Scene(); this.scene.background = new THREE.Color('#0b1220');
    this.scene.add(new THREE.HemisphereLight(0xcfe0ff, 0x20283a, 1.5));
    const sun = new THREE.DirectionalLight(0xffffff, 1.8); sun.position.set(-300, 600, 400); this.scene.add(sun);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(4000, 4000), new THREE.MeshStandardMaterial({ color: 0x0f1828, roughness: 1 })); floor.rotation.x = -Math.PI / 2; floor.position.y = -4; this.scene.add(floor);
    this.glow = glowTex(); this.lock = padlockTex();
    const train = data.areas.filter((a) => a.split === 'train'), test = data.areas.filter((a) => a.split === 'test_unseen');
    this.train = train.map((a, i) => this._tile(a, ((i % 4) - 1.5) * (TILE + GAP), (Math.floor(i / 4) - 1) * (TILE + GAP) - 40, TILE, i));
    const W = 4 * TILE + 3 * GAP, ts = 56, tg = (W - 8 * ts) / 7;
    this.test = test.map((a, i) => this._tile(a, -W / 2 + ts / 2 + i * (ts + tg), 1 * (TILE + GAP) - 40 + TILE / 2 + 66, ts, i, true));
    // cloud aggregator (illustrative)
    this.cloud = new THREE.Group(); this.cloud.position.copy(CLOUD);
    this.cloudCore = new THREE.Mesh(new THREE.IcosahedronGeometry(15, 1), new THREE.MeshStandardMaterial({ color: 0x7fd3ff, emissive: 0x3aa6ff, emissiveIntensity: 1.2, roughness: 0.3, flatShading: true }));
    const wire = new THREE.Mesh(new THREE.IcosahedronGeometry(24, 1), new THREE.MeshBasicMaterial({ color: 0xbfe6ff, wireframe: true, transparent: true, opacity: 0.55 }));
    this.cloudGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.glow, color: 0x6fc6ff, transparent: true, opacity: 0.7, blending: THREE.AdditiveBlending, depthWrite: false })); this.cloudGlow.scale.setScalar(120);
    this.cloud.add(this.cloudCore, wire, this.cloudGlow); this.wire = wire; this.scene.add(this.cloud);
    // packets
    this.packets = [];
    for (let i = 0; i < 14; i++) {
      const m = new THREE.Mesh(new THREE.SphereGeometry(4.2, 14, 10), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true }));
      const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.glow, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })); s.scale.setScalar(34); m.add(s);
      m.visible = false; this.scene.add(m); this.packets.push({ m, s });
    }
  }

  _tile(a, x, z, size, idx, isTest = false) {
    const g = new THREE.Group(); g.position.set(x, 0, z); const k = size / 500, half = size / 2;
    const plateM = new THREE.MeshStandardMaterial({ color: 0x25354d, roughness: 0.9, emissive: 0x2a84c8, emissiveIntensity: 0, transparent: true });
    const plate = new THREE.Mesh(new THREE.BoxGeometry(size, 4, size), plateM); plate.position.y = -2; g.add(plate);
    const mats = [plateM];
    const bM = new THREE.MeshStandardMaterial({ color: 0xc3cad4, roughness: 0.8, transparent: true }); mats.push(bM);
    for (const b of a.buildings) { const w = (b.x1 - b.x0) * k, d = (b.y1 - b.y0) * k, h = b.height_m * k * 1.6;
      const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), bM); m.position.set(-half + (b.x0 + b.x1) / 2 * k, h / 2, half - (b.y0 + b.y1) / 2 * k); g.add(m); }
    const c = a.courtyard_region, cM = new THREE.MeshBasicMaterial({ color: 0xd62728, transparent: true, opacity: 0.75 }); mats.push(cM);
    const court = new THREE.Mesh(new THREE.PlaneGeometry((c[2] - c[0]) * k, (c[3] - c[1]) * k), cM); court.rotation.x = -Math.PI / 2; court.position.set(-half + (c[0] + c[2]) / 2 * k, 0.3, half - (c[1] + c[3]) / 2 * k); g.add(court);
    const rM = new THREE.MeshBasicMaterial({ color: 0xffc933, transparent: true }); mats.push(rM);
    const ris = new THREE.Mesh(new THREE.OctahedronGeometry(size * 0.035), rM); ris.position.set(-half + a.ris_center[0] * k, a.ris_center[2] * k * 1.6 + size * 0.03, half - a.ris_center[1] * k); g.add(ris);
    // ground station at the front-left corner
    const sM = new THREE.MeshStandardMaterial({ color: 0xe8eef6, roughness: 0.5, transparent: true }); mats.push(sM);
    const st = new THREE.Group(); st.position.set(-half + size * 0.1, 0, half - size * 0.1);
    const hut = new THREE.Mesh(new THREE.BoxGeometry(size * 0.09, size * 0.06, size * 0.09), sM); hut.position.y = size * 0.03;
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(size * 0.006, size * 0.006, size * 0.16, 6), sM); mast.position.y = size * 0.13;
    const dish = new THREE.Mesh(new THREE.SphereGeometry(size * 0.035, 12, 8, 0, Math.PI * 2, 0, Math.PI / 2), sM); dish.position.y = size * 0.21; dish.rotation.x = Math.PI;
    st.add(hut, mast, dish); g.add(st);
    // border (4 bars) whose colour shows the tile state
    const brM = new THREE.MeshBasicMaterial({ color: 0x5f7896, transparent: true }); mats.push(brM);
    for (const [bx, bz, bw, bd] of [[0, -half, size + 3, 3], [0, half, size + 3, 3], [-half, 0, 3, size + 3], [half, 0, 3, size + 3]]) { const m = new THREE.Mesh(new THREE.BoxGeometry(bw, 2.2, bd), brM); m.position.set(bx, 0.6, bz); g.add(m); }
    // padlock: raw data stays in the area
    const lockM = new THREE.SpriteMaterial({ map: this.lock, transparent: true, depthTest: false }); const lock = new THREE.Sprite(lockM); lock.scale.setScalar(size * 0.2); lock.position.set(half - size * 0.13, size * 0.16, half - size * 0.13); g.add(lock);
    // three training drones (illustrative dots)
    const dots = UAV_COLORS.slice(0, 3).map((col) => { const m = new THREE.Mesh(new THREE.SphereGeometry(size * 0.026, 10, 8), new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0 })); g.add(m); return m; });
    this.scene.add(g);
    const station = new THREE.Vector3(x - half + size * 0.1, size * 0.22, z + half - size * 0.1);
    return { a, g, size, plateM, brM, mats, lockM, dots, station, isTest, idx, label: new THREE.Vector3(x, 0, z + half + 2) };
  }

  _packet(i, from, to, u, color, lift = 60) {
    const p = this.packets[i]; if (!p) return; if (u <= 0 || u >= 1) { p.m.visible = false; return; }
    const e = smooth(u), mid = from.clone().lerp(to, 0.5); mid.y += lift;
    const a = from.clone().lerp(mid, e), b = mid.clone().lerp(to, e); p.m.position.copy(a.lerp(b, e));
    p.m.material.color.set(color); p.s.material.color.set(color); p.m.material.opacity = 1; p.s.material.opacity = 0.85; p.m.visible = true;
  }

  // tl: seconds since the start of the FL shot. Returns info for the DOM overlay.
  update(tl) {
    this.packets.forEach((p) => (p.m.visible = false));
    const reveal = (i) => ramp(tl, 0.2 + i * 0.12, 1.2 + i * 0.12);
    let round = -1, phase = '', chosen = [], cloudPulse = 0;
    if (tl >= FL_T.reveal && tl < FL_T.test) { round = Math.min(Math.floor((tl - FL_T.reveal) / FL_T.round), ROUNDS.length - 1); chosen = ROUNDS[round]; }
    const rt = round >= 0 ? tl - FL_T.reveal - round * FL_T.round : -1;
    this.train.forEach((T, i) => {
      const vis = reveal(i), on = chosen.includes(i) ? Math.min(ramp(rt, 0, 0.35), 1 - ramp(rt, 3.4, 3.9)) : 0;
      T.mats.forEach((m) => (m.opacity = vis)); T.lockM.opacity = vis; T.g.position.y = lerp(-30, 0, vis);
      T.brM.color.set(0x5f7896).lerp(new THREE.Color(0x4fe3ff), on); T.plateM.emissiveIntensity = 0.5 * on;
      const train = on > 0 ? Math.min(ramp(rt, 0.8, 1.1), 1 - ramp(rt, 2.0, 2.3)) : 0; if (train > 0) phase = 'local training';
      T.plateM.emissiveIntensity += 0.9 * train * (0.6 + 0.4 * Math.sin(rt * 14));
      T.dots.forEach((d, j) => { const ang = rt * 3.2 + j * 2.1 + i; d.position.set(Math.cos(ang) * T.size * 0.26, T.size * 0.2 + 3 * Math.sin(ang * 2), Math.sin(ang) * T.size * 0.26); d.material.opacity = train; });
    });
    if (round >= 0) {
      chosen.forEach((i, j) => {
        if (rt < 0.9) { this._packet(j, CLOUD, this.train[i].station, rt / 0.9, 0x6fd0ff); phase = 'global model down'; }
        else if (rt >= 2.2 && rt < 3.2) { this._packet(j, this.train[i].station, CLOUD, (rt - 2.2) / 1.0, 0xffd66b); phase = 'model updates up'; }
      });
      if (rt >= 3.2) { cloudPulse = Math.sin(clamp((rt - 3.2) / 0.6) * Math.PI); phase = 'averaging (FedAvg)'; }
    }
    // unseen test areas: appear grey, receive the global model, turn green
    const tt = tl - FL_T.test; let deployed = 0;
    this.test.forEach((T, i) => {
      const vis = ramp(tt, 0.1 + i * 0.06, 1.0 + i * 0.06), got = ramp(tt, 2.6 + i * 0.12, 3.2 + i * 0.12); deployed += got;
      T.mats.forEach((m) => (m.opacity = vis)); T.lockM.opacity = 0; T.g.position.y = lerp(-20, 0, vis);
      T.brM.color.set(0x77808c).lerp(new THREE.Color(0x4dff8f), got); T.plateM.emissive.set(0x35c46a); T.plateM.emissiveIntensity = 0.55 * got;
      if (tt >= 1.4) this._packet(i, CLOUD, T.station, (tt - 1.4 - i * 0.12) / 1.2, 0x6fd0ff, 30);
    });
    const s = 1 + 0.25 * cloudPulse; this.cloudCore.scale.setScalar(s); this.wire.rotation.y = tl * 0.4; this.wire.rotation.x = tl * 0.23;
    this.cloudGlow.material.opacity = 0.55 + 0.4 * cloudPulse; this.cloud.visible = tl > 0.8;
    return { round, phase, deployed: deployed / this.test.length, testVisible: ramp(tt, 0.1, 1.0) };
  }
}
