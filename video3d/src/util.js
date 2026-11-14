// Shared helpers: coordinate mapping, easing, splines, procedural textures.
import * as THREE from 'three';

// Simulation coordinates (x east, y north, z up) -> three.js (X east, Y up, Z south).
export const P = (x, y, z = 0) => new THREE.Vector3(x, z, -y);
export const PV = (a) => new THREE.Vector3(a[0], a[2] ?? 0, -a[1]);

export const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, u) => a + (b - a) * u;
export const smooth = (u) => { u = clamp(u); return u * u * (3 - 2 * u); };
export const smoother = (u) => { u = clamp(u); return u * u * u * (u * (u * 6 - 15) + 10); };
// 0 -> 1 over [a, b]
export const ramp = (t, a, b) => smooth((t - a) / (b - a));
// 1 inside [a, b] with soft edges of width e
export const win = (t, a, b, e = 0.4) => Math.min(ramp(t, a, a + e), 1 - ramp(t, b - e, b));

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// Centripetal Catmull-Rom (Barry-Goldman): point between p1 and p2 at u in [0, 1].
export function catmull(p0, p1, p2, p3, u) {
  const d = (a, b) => Math.max(Math.sqrt(a.distanceTo(b)), 1e-4);
  const t0 = 0, t1 = t0 + d(p0, p1), t2 = t1 + d(p1, p2), t3 = t2 + d(p2, p3);
  const t = t1 + u * (t2 - t1);
  const L = (a, b, ta, tb) => a.clone().multiplyScalar((tb - t) / (tb - ta)).add(b.clone().multiplyScalar((t - ta) / (tb - ta)));
  const A1 = L(p0, p1, t0, t1), A2 = L(p1, p2, t1, t2), A3 = L(p2, p3, t2, t3);
  const B1 = L(A1, A2, t0, t2), B2 = L(A2, A3, t1, t3);
  return L(B1, B2, t1, t2);
}

// Camera path through keyframes [{t, pos:[x,y,z], look:[x,y,z], fov}] in SIM coordinates.
// Monotone cubic (Fritsch-Carlson tangents, zero at both ends): no overshoot between unevenly spaced keys.
export function camPath(keys, t) {
  const n = keys.length;
  if (t <= keys[0].t) return keys[0];
  if (t >= keys[n - 1].t) return keys[n - 1];
  let i = 0; while (i < n - 2 && t > keys[i + 1].t) i++;
  const h = keys[i + 1].t - keys[i].t, u = (t - keys[i].t) / h, u2 = u * u, u3 = u2 * u;
  const comp = (get) => {
    const y = keys.map(get), d = []; for (let k = 0; k < n - 1; k++) d.push((y[k + 1] - y[k]) / (keys[k + 1].t - keys[k].t));
    const m = y.map((_, k) => (k === 0 || k === n - 1 || d[k - 1] * d[k] <= 0 ? 0 : (2 * d[k - 1] * d[k]) / (d[k - 1] + d[k])));
    return (2 * u3 - 3 * u2 + 1) * y[i] + (u3 - 2 * u2 + u) * h * m[i] + (-2 * u3 + 3 * u2) * y[i + 1] + (u3 - u2) * h * m[i + 1];
  };
  return { pos: [0, 1, 2].map((j) => comp((k) => k.pos[j])), look: [0, 1, 2].map((j) => comp((k) => k.look[j])), fov: comp((k) => k.fov) };
}
export const mixCam = (a, b, u) => ({ pos: a.pos.map((v, i) => lerp(v, b.pos[i], u)), look: a.look.map((v, i) => lerp(v, b.look[i], u)), fov: lerp(a.fov, b.fov, u) });

export function canvasTex(w, h, draw, repeat = false) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8;
  if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; }
  return t;
}

// One facade tile = 1 bay (4 m) x 1 floor (3.4 m).
export function facadeTex(seed, base = '#b9b4aa', glass = '#3d5266') {
  const r = mulberry32(seed);
  return canvasTex(256, 256, (g, w, h) => {
    g.fillStyle = base; g.fillRect(0, 0, w, h);
    for (let i = 0; i < 500; i++) { g.fillStyle = `rgba(0,0,0,${0.02 + 0.04 * r()})`; g.fillRect(r() * w, r() * h, 2 + r() * 5, 1 + r() * 3); }
    for (let by = 0; by < 4; by++) for (let bx = 0; bx < 4; bx++) {
      const x = bx * 64 + 12, y = by * 64 + 14;
      g.fillStyle = r() < 0.16 ? '#e7d9a4' : glass; g.fillRect(x, y, 40, 34);
      g.fillStyle = 'rgba(255,255,255,0.14)'; g.fillRect(x, y, 40, 9);
      g.strokeStyle = 'rgba(20,24,30,0.65)'; g.lineWidth = 2; g.strokeRect(x, y, 40, 34);
      g.beginPath(); g.moveTo(x + 20, y); g.lineTo(x + 20, y + 34); g.stroke();
    }
    g.fillStyle = 'rgba(0,0,0,0.13)'; for (let by = 0; by < 4; by++) g.fillRect(0, by * 64 + 58, w, 4);
  }, true);
}

export function groundTex(trace) {
  const r = mulberry32(7), S = 2048, k = S / trace.area_m;
  return canvasTex(S, S, (g) => {
    g.fillStyle = '#566b4c'; g.fillRect(0, 0, S, S);
    for (let i = 0; i < 9000; i++) {
      const v = r(); g.fillStyle = v < 0.5 ? `rgba(40,62,36,${0.05 + 0.1 * r()})` : `rgba(150,170,110,${0.04 + 0.07 * r()})`;
      const s = 4 + r() * 26; g.beginPath(); g.ellipse(r() * S, r() * S, s, s * (0.4 + r() * 0.6), r() * 3, 0, 7); g.fill();
    }
    // paved plaza around the urban block and the courtyard floor (canvas y grows downward = south)
    const bs = trace.buildings, x0 = Math.min(...bs.map((b) => b.x0)), x1 = Math.max(...bs.map((b) => b.x1));
    const y0 = Math.min(...bs.map((b) => b.y0)), y1 = Math.max(...bs.map((b) => b.y1)), m = 14;
    g.fillStyle = '#7d7c78'; g.fillRect((x0 - m) * k, S - (y1 + m) * k, (x1 - x0 + 2 * m) * k, (y1 - y0 + 2 * m) * k);
    for (let i = 0; i < 1500; i++) { g.fillStyle = `rgba(0,0,0,${0.03 + 0.05 * r()})`;
      g.fillRect((x0 - m + r() * (x1 - x0 + 2 * m)) * k, S - (y1 + m - r() * (y1 - y0 + 2 * m)) * k, 3 + r() * 10, 2 + r() * 6); }
    const c = trace.courtyard_region; g.fillStyle = '#9b958a';
    g.fillRect((c[0] - 10) * k, S - (c[3] + 5) * k, (c[2] - c[0] + 20) * k, (c[3] - c[1] + 10) * k);
  });
}

export function gridTex() {
  return canvasTex(1024, 1024, (g, w) => {
    g.clearRect(0, 0, w, w); g.strokeStyle = 'rgba(255,255,255,0.55)'; g.lineWidth = 2;
    for (let i = 0; i <= 10; i++) { const p = (i / 10) * w; g.beginPath(); g.moveTo(p, 0); g.lineTo(p, w); g.moveTo(0, p); g.lineTo(w, p); g.stroke(); }
  });
}

export function glowTex() {
  return canvasTex(128, 128, (g, w) => {
    const gr = g.createRadialGradient(w / 2, w / 2, 0, w / 2, w / 2, w / 2);
    gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.25, 'rgba(255,255,255,0.55)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = gr; g.fillRect(0, 0, w, w);
  });
}

export function skyTex() {
  return canvasTex(16, 512, (g, w, h) => {
    const gr = g.createLinearGradient(0, 0, 0, h);
    gr.addColorStop(0, '#5d8fd0'); gr.addColorStop(0.45, '#a9c7e8'); gr.addColorStop(0.62, '#f3d9b4'); gr.addColorStop(0.72, '#e9c9a0'); gr.addColorStop(1, '#cdb894');
    g.fillStyle = gr; g.fillRect(0, 0, w, h);
  });
}

export function humanTex() {
  return canvasTex(128, 384, (g, w, h) => {
    g.clearRect(0, 0, w, h); g.fillStyle = '#1b2a44'; g.strokeStyle = '#ffffff'; g.lineWidth = 5;
    const body = new Path2D();
    body.ellipse(64, 36, 26, 30, 0, 0, 7);                                      // head
    body.roundRect(26, 74, 76, 150, 26);                                        // torso
    body.roundRect(8, 84, 22, 120, 11); body.roundRect(98, 84, 22, 120, 11);    // arms
    body.roundRect(32, 214, 28, 164, 12); body.roundRect(68, 214, 28, 164, 12); // legs
    g.stroke(body); g.fill(body);
  });
}

export function padlockTex() {
  return canvasTex(128, 128, (g) => {
    g.clearRect(0, 0, 128, 128); g.strokeStyle = '#ffffff'; g.lineWidth = 12; g.lineCap = 'round';
    g.beginPath(); g.arc(64, 54, 24, Math.PI, 0); g.lineTo(88, 66); g.moveTo(40, 54); g.lineTo(40, 66); g.stroke();
    g.fillStyle = '#ffd66b'; g.beginPath(); g.roundRect(26, 62, 76, 54, 10); g.fill();
    g.fillStyle = '#1b2a44'; g.beginPath(); g.arc(64, 84, 8, 0, 7); g.fill(); g.fillRect(60, 86, 8, 16);
  });
}

// Orient a unit-height Y cylinder between two points.
const _up = new THREE.Vector3(0, 1, 0), _d = new THREE.Vector3();
export function setSegment(mesh, a, b, radius) {
  _d.subVectors(b, a); const len = Math.max(_d.length(), 1e-6);
  mesh.position.copy(a).addScaledVector(_d, 0.5);
  mesh.quaternion.setFromUnitVectors(_up, _d.clone().divideScalar(len));
  mesh.scale.set(radius, len, radius);
}

// First intersection of segment a->b (sim coords, arrays) with axis-aligned building boxes; returns {u, point} or null.
export function firstHit(a, b, buildings) {
  let best = null;
  for (const bd of buildings) {
    const lo = [bd.x0, bd.y0, 0], hi = [bd.x1, bd.y1, bd.height_m];
    let u0 = 0, u1 = 1, ok = true;
    for (let i = 0; i < 3; i++) {
      const d = b[i] - a[i];
      if (Math.abs(d) < 1e-9) { if (a[i] < lo[i] || a[i] > hi[i]) { ok = false; break; } continue; }
      let ta = (lo[i] - a[i]) / d, tb = (hi[i] - a[i]) / d; if (ta > tb) [ta, tb] = [tb, ta];
      u0 = Math.max(u0, ta); u1 = Math.min(u1, tb); if (u0 > u1) { ok = false; break; }
    }
    if (ok && (best === null || u0 < best.u)) best = { u: u0, uOut: u1, point: a.map((v, i) => v + (b[i] - a[i]) * u0), out: a.map((v, i) => v + (b[i] - a[i]) * u1) };
  }
  return best;
}
