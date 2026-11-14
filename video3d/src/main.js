// Timeline: window.renderFrame(i) draws frame i (time i / 30 s) deterministically. No real-time clock is used.
import * as THREE from 'three';
import { Mission, RIS_SCALE } from './mission.js';
import { FLScene, ROUNDS, FL_T } from './fl.js';
import { P, clamp, lerp, smooth, smoother, ramp, win, camPath, mixCam } from './util.js';

const q = new URLSearchParams(location.search);
const W = +(q.get('w') || 1920), H = +(q.get('h') || 1080), K = W / 1920, FPS = 30;
const DATA = '../results_v9/animation/';
const $ = (id) => document.getElementById(id);
window.__err = null; window.addEventListener('error', (e) => { window.__err = String(e.message || e); });

const canvas = $('gl'); canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1); renderer.setSize(W, H, false);
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.0; renderer.outputColorSpace = THREE.SRGBColorSpace;
$('ov').style.transform = `scale(${K})`;
const cam = new THREE.PerspectiveCamera(40, 16 / 9, 1, 6000), pipCam = new THREE.PerspectiveCamera(28, 1, 0.5, 4000);
let curVP = [0, 0, 1920, 1080];

// ------------------------------------------------------------------ shot list (seconds)
export const SHOTS = [
  { id: 1, name: 'Title', a: 0, b: 6 }, { id: 2, name: 'The scenario', a: 6, b: 17 }, { id: 3, name: 'The problem', a: 17, b: 31 },
  { id: 4, name: 'The RIS', a: 31, b: 47 }, { id: 5, name: 'One mission with the RIS', a: 47, b: 80 },
  { id: 6, name: 'Same mission, with and without the RIS', a: 80, b: 98.5 }, { id: 7, name: 'Aerial RIS', a: 98.5, b: 110.5 },
  { id: 8, name: 'Federated learning across city areas', a: 110.5, b: 131.5 }, { id: 9, name: 'Results', a: 131.5, b: 141.5 }, { id: 10, name: 'End', a: 141.5, b: 146 },
];
export const DURATION = 146, PLAY5 = 30, PLAY6 = 15, PLAY7 = 9.5, T_PROBLEM = 10, DEV_PROBLEM = 20, UAV_PROBLEM = 2;

// Every static on-screen string lives here (check.mjs verifies their numbers against NUMBERS.md).
export const TEXT = {
  title: 'Optimization of RIS-assisted UAVs<br>with Federated Learning',
  subtitle: 'Energy-efficient IoT data collection in a shadowed urban courtyard',
  tag: 'B.Tech project · 3-D visualisation of simulation results',
  scale: 'Simulation · drones ×12, RIS ×8 and sensor markers drawn larger for visibility',
  scaleTrue: 'Simulation · RIS panel and person shown at true scale',
  c2: '3 drones · 24 IoT sensors · 100 s mission', c2s: '500 m × 500 m city area · drones fly at 60 m · maximise bits collected per joule of flight energy',
  c3a: 'Courtyard sensors are shadowed by the buildings around them', c3as: 'The direct radio path to a drone in the open field is blocked',
  c3b: 'Without help, a drone must fly almost overhead', c3bs: 'A courtyard sensor can be served in time from only 3.4 % of the area',
  c4a: 'The RIS: a 1 m × 1 m panel of 16 × 16 = 256 reflecting elements', c4as: 'Mounted 3 m above the roof edge, facing the courtyard and the flight zone',
  c4b: 'Each element applies one of 4 phase shifts (2-bit), set in closed form', c4bs: 'The reflections add up at the drone · area that can serve a courtyard sensor in time: 3.4 % → 90 %',
  c5: 'Learned flight (multi-agent RL) + closed-form RIS: one recorded mission', c5s: 'green = direct upload · gold = upload via the RIS · bars = data still waiting at each sensor',
  c6: 'Same mission, same sensors, same radio fading: only the RIS differs', c6s: 'Energy efficiency (EE) = bits collected ÷ flight energy',
  c7: 'Alternative: the same panel carried by a 4th drone', c7s: 'Its flight energy counts against energy efficiency',
  c7end: "the 4th drone's flight energy costs more than the RIS gains",
  fl1: '<b>12 city areas</b>, each with its own ground station and 3 drones. They differ in layout, number of sensors, data, transmit power and RIS size.',
  fl2: 'Each round, a random <b>50 %</b> of the areas train locally and send only <b>model updates</b> to the cloud, which averages them (FedAvg).',
  flRounds: 'showing 3 of 60 rounds',
  fl3: 'The averaged model is deployed to <b>8 city areas it has never seen</b>:',
  flA: '<span class="big good">+9.7 %</span><br>energy efficiency vs each area learning alone',
  flB: '<b>As good as pooling all raw data</b>, with <span class="good">13× less</span> cross-area traffic',
  flC: '8-bit model updates: <span class="good">38 % less</span> traffic, no measurable loss',
  legDown: 'global model (down)', legUp: 'model update (up)', legLock: 'raw flight data never leaves the area', legNote: 'packets and cloud node are illustrative',
  resHead: 'What the simulations show',
  res: [['Building RIS', '<span class="good">+13–26 %</span> energy efficiency'], ['Aerial RIS', '<span class="bad">−38 %</span> vs the building RIS: the carrier burns too much energy'],
    ['Learned flight', 'within 1 % of a tuned same-information heuristic, 6 % below a full-information planner'],
    ['Federated learning', 'generalises to unseen city areas without sharing raw data']],
  resFoot: 'All results from simulation · 5 training seeds · 96–100 test missions per result · 95 % confidence intervals in the report',
  repo: 'github.com/veritas1407/Optimization-of-RIS-assisted-UAV-with-fedrated_learning', endTag: 'B.Tech project · simulation study',
  pip: 'RIS phase pattern<br>(illustrative of the 2-bit configuration)',
};

// ------------------------------------------------------------------ overlay helpers
const OV = ['lower', 'scale', 'hudL', 'hudR', 'sideL', 'sideR', 'split', 'pip', 'pipl', 'cardA', 'cardB', 'fltext', 'flleg', 'results', 'caption', 'title', 'dim'];
function resetOverlay() { OV.forEach((id) => ($(id).style.opacity = 0)); $('labels').innerHTML = ''; $('fade').style.opacity = 0; }
const show = (id, alpha, html) => { const e = $(id); e.style.opacity = clamp(alpha); if (html !== undefined && e.__h !== html) { e.innerHTML = html; e.__h = html; } return e; };
const caption = (main, sub, alpha) => show('caption', alpha, main + (sub ? `<span class="sub">${sub}</span>` : ''));
const lower = (shot, alpha) => show('lower', alpha, `<span class="n">${shot.id - 1}</span>${shot.name}`);
const fmt = (x) => Math.round(x).toLocaleString('en-US');
function hud(id, c, title, alpha, hl = false) {
  const e = $(id); e.style.opacity = clamp(alpha); e.classList.add('live');
  e.innerHTML = `<div class="ttl">${title}</div><div class="row"><span class="k">time</span><span class="v">${c.t.toFixed(1)} s</span></div>`
    + `<div class="row"><span class="k">data collected</span><span class="v">${(c.bits / 8e6).toFixed(1)} MB</span></div>`
    + `<div class="row${hl ? ' hl' : ''}"><span class="k">flight energy</span><span class="v">${(c.energy / 1e3).toFixed(1)} kJ</span></div>`
    + `<div class="row ee"><span class="k">EE</span><span class="v">${fmt(c.ee)} bit/J</span></div>`;
}
function label3(text, v3, cls = '', alpha = 1, dy = 0, live = false) {
  const p = v3.clone().project(cam); if (p.z > 1 || alpha <= 0.01) return;
  const [x, y, w, h] = curVP, e = document.createElement('div');
  e.className = `abs lbl ${cls}${live ? ' live' : ''}`; e.innerHTML = text; e.style.opacity = clamp(alpha);
  e.style.left = (x + (p.x * 0.5 + 0.5) * w) + 'px'; e.style.top = (y + (-p.y * 0.5 + 0.5) * h + dy) + 'px'; $('labels').appendChild(e);
}
const label = (text, sim, cls, alpha, dy, live) => label3(text, P(...sim), cls, alpha, dy, live);
const edge = (tl, dur, e = 0.5) => Math.max(1 - ramp(tl, 0, e), ramp(tl, dur - e, dur));

function view3(scene, pos, look, fov, vp = [0, 0, 1920, 1080], camera = cam, up = null) {
  const [x, y, w, h] = vp; camera.fov = fov; camera.aspect = w / h; camera.position.copy(pos); camera.up.copy(up || new THREE.Vector3(0, 1, 0)); camera.lookAt(look);
  camera.updateProjectionMatrix(); camera.updateMatrixWorld();
  renderer.setViewport(x * K, (1080 - y - h) * K, w * K, h * K); renderer.setScissor(x * K, (1080 - y - h) * K, w * K, h * K); renderer.setScissorTest(true);
  renderer.render(scene, camera); if (camera === cam) curVP = vp;
}
const view = (m, spec, vp) => { m.setResolution((vp ? vp[2] : 1920) * K, (vp ? vp[3] : 1080) * K); view3(m.scene, P(...spec.pos), P(...spec.look), spec.fov, vp); };

// picture-in-picture of the RIS tiles, seen along the panel normal
const PIP = [1512, 560, 300, 300];
function pip(m, alpha) {
  if (alpha <= 0.01 || !m.ris) return;
  const R = m.ris, c = R.g.position, s = R.g.scale.x;
  m.setResolution(PIP[2] * K, PIP[3] * K); m.withoutLinks(() => view3(m.scene, c.clone().addScaledVector(R.n3, 2.35 * s), c, 28, PIP, pipCam, R.e2));
  const e = show('pip', alpha); Object.assign(e.style, { left: PIP[0] - 2 + 'px', top: PIP[1] - 2 + 'px', width: PIP[2] + 4 + 'px', height: PIP[3] + 4 + 'px' });
  const l = show('pipl', alpha, TEXT.pip); Object.assign(l.style, { left: PIP[0] - 30 + 'px', top: PIP[1] + PIP[3] + 10 + 'px', width: PIP[2] + 60 + 'px' });
}

// ------------------------------------------------------------------ cameras (simulation coordinates)
function centroid(m, t) {
  let cx = 0, cy = 0; const n = 9;
  for (let i = 0; i < n; i++) { const sp = m.simPosAt(clamp(t + (i - 4) * 2.2, 0, m.T)); for (let a = 0; a < m.K; a++) { cx += sp[a][0]; cy += sp[a][1]; } }
  return [cx / (n * m.K), cy / (n * m.K)];
}
const trackCam = (m, t, off = [-235, -175, 250], fov = 40) => { const c = centroid(m, t); return { pos: [c[0] + off[0], c[1] + off[1], off[2]], look: [c[0] + 35, c[1] + 28, 5], fov }; };
const C_CLOSE_A = { pos: [362, 106, 48.5], look: [370.3, 95.2, 47.3], fov: 20 }, C_CLOSE_B = { pos: [360.6, 107.2, 49.3], look: [370.3, 95.2, 47.3], fov: 20 };
const C_WIDE = { pos: [620, 118, 265], look: [250, 85, 35], fov: 38 }, C_WIDE2 = { pos: [602, 126, 256], look: [246, 84, 35], fov: 38 };

const M = {}; let FL = null; const LAST = {};
function upd(key, t, o) { const r = M[key].update(t, o); LAST[key] = { t, links: r.links.filter((l) => l.alpha > 0.01).map((l) => ({ device: l.device, uav: l.uav, via_ris: l.via_ris })) }; return r; }
const res = (key) => M[key].tr.mission_result;

// ------------------------------------------------------------------ shots
const S = {
  1(tl, dur) {
    upd('noris', 0, { links: false, vt: tl }); const u = tl / dur;
    view(M.noris, { pos: [lerp(-300, -245, u), lerp(-345, -295, u), lerp(360, 322, u)], look: [250, 250, 0], fov: 42 });
    show('dim', 0.35 * ramp(tl, 0.3, 2));
    show('title', win(tl, 0.8, dur - 0.2, 0.7), `<h1>${TEXT.title}</h1><h2>${TEXT.subtitle}</h2><div class="tag">${TEXT.tag}</div>`);
    $('fade').style.opacity = Math.max(1 - ramp(tl, 0, 3.0), ramp(tl, dur - 0.5, dur));
  },
  2(tl, dur, shot) {
    upd('noris', 0, { links: false, vt: tl });
    view(M.noris, camPath([{ t: 0, pos: [-235, -265, 305], look: [250, 250, 0], fov: 42 }, { t: 5.5, pos: [60, -305, 235], look: [290, 215, 0], fov: 42 }, { t: 11, pos: [335, -255, 195], look: [335, 195, 10], fov: 42 }], tl));
    const a = win(tl, 2.0, dur - 0.4, 0.5), tr = M.noris.tr;
    label('start', [45, 20, 78], '', a); label('landing zone', [tr.final_zone.center[0], tr.final_zone.center[1], 12], '', a);
    label('courtyard: 6 shadowed sensors', [370, 110, 62], 'red', a); label('18 open-field sensors', [150, 330, 18], '', win(tl, 3.2, dur - 0.4, 0.5));
    lower(shot, win(tl, 0.5, dur - 0.3)); show('scale', win(tl, 0.5, dur - 0.3), TEXT.scale); caption(TEXT.c2, TEXT.c2s, win(tl, 1.0, dur - 0.3));
    $('fade').style.opacity = edge(tl, dur);
  },
  3(tl, dur, shot) {
    upd('noris', T_PROBLEM, { links: false, vt: tl }); M.noris.grid.material.opacity = 0.22;
    const info = M.noris.showBlocked(DEV_PROBLEM, UAV_PROBLEM, T_PROBLEM, (tl - 2.6) / 2.0);
    view(M.noris, camPath([{ t: 0, pos: [590, 30, 240], look: [325, 112, 15], fov: 40 }, { t: 6, pos: [548, 98, 158], look: [300, 112, 18], fov: 39 }, { t: 14, pos: [524, 108, 128], look: [292, 112, 20], fov: 37 }], tl));
    if (info) {
      const d = Math.hypot(...info.sensor.map((v, i) => info.drone[i] - v)), a = win(tl, 4.7, dur - 0.4, 0.5);
      label('direct path blocked by building', [info.hit[0], info.hit[1], info.hit[2] + 9], 'red', a);
      label(`drone, ${d.toFixed(0)} m away`, [info.drone[0], info.drone[1], info.drone[2] + 12], '', a, 0, true);
      label('courtyard sensor', [info.sensor[0], info.sensor[1], 20], 'red', win(tl, 1.2, dur - 0.4, 0.5));
    }
    lower(shot, win(tl, 0.5, dur - 0.3)); show('scale', win(tl, 0.5, dur - 0.3), TEXT.scale);
    caption(TEXT.c3a, TEXT.c3as, win(tl, 0.8, 7.4)); if (tl > 7.4) caption(TEXT.c3b, TEXT.c3bs, win(tl, 7.6, dur - 0.3));
    $('fade').style.opacity = edge(tl, dur); M.noris.grid.material.opacity = 0;
  },
  4(tl, dur, shot) {
    const play = tl >= 9.8, ts = clamp((tl - 9.8) * 0.75, 0, 4.9), sc = lerp(1, RIS_SCALE, ramp(tl, 6.3, 8.6));
    upd('ris', ts, { links: play, fadeSim: 0.15, vt: tl, risScale: sc, human: tl < 6.9, risGlow: ramp(tl, 6.3, 8.6) });
    view(M.ris, camPath([{ t: 0, ...C_CLOSE_A }, { t: 5.5, ...C_CLOSE_B }, { t: 9.6, ...C_WIDE }, { t: 16, ...C_WIDE2 }], tl));
    label('1.8 m person', [371.9, 94.3, 47.95], 'sm', win(tl, 1.0, 5.6, 0.4)); label('RIS panel (1 m × 1 m)', [370, 95.5, 48.75], 'gold sm', win(tl, 1.0, 5.6, 0.4));
    label('RIS, drawn ×8 from here on', [370, 95.5, 56], 'gold', win(tl, 7.4, 10.6, 0.4));
    if (play) { const r = M.ris.tr; label('drones, about 300 m from the RIS', [78, 30, 76], '', win(tl, 10.6, dur - 0.4, 0.5)); label('RIS', [r.ris.center[0], r.ris.center[1], 60], 'gold', win(tl, 10.6, dur - 0.4, 0.5)); }
    pip(M.ris, ramp(tl, 10.4, 11.2) * (1 - ramp(tl, dur - 0.5, dur)));
    lower(shot, win(tl, 0.5, dur - 0.3)); show('scale', win(tl, 0.5, dur - 0.3), tl < 6.6 ? TEXT.scaleTrue : TEXT.scale);
    caption(TEXT.c4a, TEXT.c4as, win(tl, 0.8, 9.2)); if (tl > 9.2) caption(TEXT.c4b, TEXT.c4bs, win(tl, 9.4, dur - 0.3));
    $('fade').style.opacity = edge(tl, dur);
  },
  5(tl, dur, shot) {
    const ts = clamp(tl / PLAY5) * 100, r = upd('ris', ts, { fadeSim: 0.2 * 100 / PLAY5, vt: tl });
    view(M.ris, mixCam(C_WIDE2, trackCam(M.ris, ts), smoother((tl - 3.6) / 6.5)));
    pip(M.ris, (M.ris.ris.target ? 1 : 0.0) * win(tl, 0.6, PLAY5, 0.4));
    hud('hudL', M.ris.countersAt(ts), 'building RIS', win(tl, 0.5, dur - 0.3));
    lower(shot, win(tl, 0.5, dur - 0.3)); show('scale', win(tl, 0.5, dur - 0.3), TEXT.scale); caption(TEXT.c5, TEXT.c5s, win(tl, 0.8, PLAY5 - 0.2));
    const m = res('ris'), e = show('cardA', ramp(tl, PLAY5 + 0.2, PLAY5 + 0.8) * (1 - ramp(tl, dur - 0.5, dur)),
      `Mission complete<br><span class="big good">EE ${fmt(m.ee)} bit/J</span><br>courtyard data delivered: <b>${(100 * m.shadow_delivered_frac).toFixed(0)} %</b>`);
    Object.assign(e.style, { left: '1170px', top: '150px', width: '654px' });
    $('fade').style.opacity = edge(tl, dur); void r;
  },
  6(tl, dur) {
    const ts = clamp(tl / PLAY6) * 100, o = { fadeSim: 0.2 * 100 / PLAY6, vt: tl }; upd('ris', ts, o); upd('noris', ts, o);
    const c = trackCam(M.ris, ts, [-275, -205, 315], 50);
    view(M.ris, c, [0, 0, 958, 1080]); view(M.noris, c, [962, 0, 958, 1080]);
    const a = win(tl, 0.5, dur - 0.3); show('split', a); show('sideL', a, 'with RIS'); show('sideR', a, 'no RIS');
    hud('hudL', M.ris.countersAt(ts), 'building RIS', a); hud('hudR', M.noris.countersAt(ts), 'no RIS', a);
    caption(TEXT.c6, TEXT.c6s, win(tl, 0.8, dur - 0.3));
    const ea = ramp(tl, PLAY6 + 0.2, PLAY6 + 0.8) * (1 - ramp(tl, dur - 0.5, dur));
    const card = (id, key, left, cls) => { const m = res(key), e = show(id, ea, `<span class="big ${cls}">EE ${fmt(m.ee)} bit/J</span><br>courtyard data delivered: <b>${(100 * m.shadow_delivered_frac).toFixed(0)} %</b>`); Object.assign(e.style, { left, top: '420px', width: '560px' }); };
    card('cardA', 'ris', '96px', 'good'); card('cardB', 'noris', '1056px', 'bad');
    $('fade').style.opacity = edge(tl, dur);
  },
  7(tl, dur, shot) {
    const ts = clamp(tl / PLAY7) * 100, r = upd('aerial', ts, { fadeSim: 0.2 * 100 / PLAY7, vt: tl });
    view(M.aerial, trackCam(M.aerial, ts, [-255, -190, 275], 42));
    const cp = r.pos[M.aerial.carrier]; label3('RIS carrier (4th drone)', cp.clone().add(new THREE.Vector3(0, 16, 0)), 'gold', win(tl, 0.6, PLAY7, 0.4));
    hud('hudL', M.aerial.countersAt(ts), 'aerial RIS', win(tl, 0.5, dur - 0.3), true);
    lower(shot, win(tl, 0.5, dur - 0.3)); show('scale', win(tl, 0.5, dur - 0.3), TEXT.scale); caption(TEXT.c7, TEXT.c7s, win(tl, 0.8, PLAY7 - 0.2));
    const m = res('aerial'), b = res('ris'), e = show('cardA', ramp(tl, PLAY7 + 0.15, PLAY7 + 0.6) * (1 - ramp(tl, dur - 0.5, dur)),
      `<span class="big bad">EE ${fmt(m.ee)} bit/J</span><br>flight energy <b>${(m.energy / 1e3).toFixed(1)} kJ</b> vs ${(b.energy / 1e3).toFixed(1)} kJ with the building RIS<br><small>${TEXT.c7end}</small>`);
    Object.assign(e.style, { left: '1080px', top: '150px', width: '744px' });
    $('fade').style.opacity = edge(tl, dur);
  },
  8(tl, dur, shot) {
    const st = FL.update(tl), u = smooth(tl / dur);
    view3(FL.scene, new THREE.Vector3(-185, lerp(610, 585, u), lerp(790, 755, u)), new THREE.Vector3(-185, 30, 62), 40);
    FL.train.forEach((T, i) => label3(`${T.a.n_devices} sensors · RIS ${T.a.ris_elements[0]}×${T.a.ris_elements[1]}`, T.label, 'sm', ramp(tl, 1.4 + i * 0.1, 2.0 + i * 0.1) * (1 - ramp(tl, dur - 0.5, dur)), 34, true));
    if (st.testVisible > 0) label3('8 unseen test areas', FL.test[0].label.clone().add(new THREE.Vector3(-70, 0, -30)), 'sm', st.testVisible * (1 - ramp(tl, dur - 0.5, dur)), 0);
    label3('cloud aggregator (FedAvg)', FL.cloud.position.clone().add(new THREE.Vector3(0, 34, 0)), 'sm', win(tl, 1.2, dur - 0.5, 0.4));
    lower(shot, win(tl, 0.4, dur - 0.3));
    let h = '';
    if (tl < FL_T.test) {
      h += `<div class="card" style="opacity:${ramp(tl, 0.6, 1.2)}">${TEXT.fl1}</div>`;
      h += `<div class="card" style="opacity:${ramp(tl, FL_T.reveal - 0.2, FL_T.reveal + 0.4)}">${TEXT.fl2}</div>`;
      if (st.round >= 0) h += `<div class="card" style="font-size:25px"><b>Round ${st.round + 1}</b> <small>(${TEXT.flRounds})</small><br>${st.phase || '&nbsp;'}</div>`;
    } else {
      const tt = tl - FL_T.test;
      h += `<div class="card" style="opacity:${ramp(tt, 0.2, 0.8)}">${TEXT.fl3}</div>`;
      [TEXT.flA, TEXT.flB, TEXT.flC].forEach((s, i) => (h += `<div class="card" style="opacity:${ramp(tt, 1.6 + i * 1.0, 2.2 + i * 1.0)}">${s}</div>`));
    }
    const ft = $('fltext'); ft.style.opacity = 1 - ramp(tl, dur - 0.5, dur); ft.innerHTML = h;
    const lock = '<svg width="20" height="22" viewBox="0 0 20 22" style="vertical-align:middle;margin-right:8px"><path d="M5 10V6a5 5 0 0 1 10 0v4" fill="none" stroke="#fff" stroke-width="2.4"/><rect x="2" y="10" width="16" height="11" rx="2" fill="#ffd66b"/></svg>';
    show('flleg', win(tl, 2.6, dur - 0.5, 0.5), `<span class="dot" style="background:#6fd0ff"></span>${TEXT.legDown}<br><span class="dot" style="background:#ffd66b"></span>${TEXT.legUp}<br>${lock}${TEXT.legLock}<br><small>${TEXT.legNote}</small>`);
    $('fade').style.opacity = edge(tl, dur);
  },
  9(tl, dur) {
    upd('ris', 100, { vt: tl }); const ang = 3.9 + 0.012 * tl;
    view(M.ris, { pos: [250 + 520 * Math.cos(ang), 250 + 520 * Math.sin(ang), 300], look: [270, 230, 0], fov: 42 });
    show('dim', 0.74 * ramp(tl, 0, 0.6));
    let h = `<h3>${TEXT.resHead}</h3>`; TEXT.res.forEach(([a, b], i) => (h += `<div class="line" style="opacity:${ramp(tl, 0.7 + i * 0.8, 1.3 + i * 0.8)}"><span class="h">${a}</span><span class="t">${b}</span></div>`));
    h += `<div class="foot" style="opacity:${ramp(tl, 4.2, 4.8)}">${TEXT.resFoot}</div>`;
    const e = $('results'); e.style.opacity = win(tl, 0.4, dur - 0.2, 0.5); e.innerHTML = h;
    $('fade').style.opacity = edge(tl, dur);
  },
  10(tl, dur) {
    upd('ris', 100, { vt: tl }); const ang = 4.05 + 0.012 * tl;
    view(M.ris, { pos: [250 + 520 * Math.cos(ang), 250 + 520 * Math.sin(ang), 300], look: [270, 230, 0], fov: 42 });
    show('dim', 0.6);
    show('title', win(tl, 0.3, dur - 0.9, 0.5), `<h1>${TEXT.title}</h1><h2>${TEXT.endTag}</h2><div class="tag" style="text-transform:none;letter-spacing:0.04em">${TEXT.repo}</div>`);
    $('fade').style.opacity = Math.max(1 - ramp(tl, 0, 0.5), ramp(tl, dur - 1.0, dur));
  },
};

function renderFrame(i) {
  const t = Math.min(i / FPS, DURATION - 1e-6), shot = SHOTS.find((s) => t >= s.a && t < s.b) || SHOTS.at(-1);
  resetOverlay(); document.querySelectorAll('.live').forEach((e) => e.classList.remove('live')); M.noris.hideBlocked();
  for (const k in LAST) delete LAST[k];
  renderer.setScissorTest(false); renderer.clear();
  S[shot.id](t - shot.a, shot.b - shot.a, shot);
  renderer.getContext().finish();
  return shot.id;
}

async function init() {
  const get = async (f) => (await fetch(DATA + f)).json();
  const [ris, noris, aerial, areas] = await Promise.all([get('trace_building_ctde_5000002.json'), get('trace_building_noris_ctde_5000002.json'), get('trace_aerial_ctde_5000002.json'), get('fl_areas.json')]);
  const shadow = +(q.get('shadow') || (W >= 1920 ? 4096 : 2048));
  M.ris = new Mission(ris, { shadow }); M.noris = new Mission(noris, { shadow }); M.aerial = new Mission(aerial, { shadow }); FL = new FLScene(areas);
  window.renderFrame = renderFrame;
  // read-only hooks for check.mjs
  window.debug = {
    fps: FPS, duration: DURATION, frames: Math.round(DURATION * FPS), shots: SHOTS, play: { 5: PLAY5, 6: PLAY6, 7: PLAY7 }, TEXT, ROUNDS,
    simPos: (key, t) => M[key].simPosAt(t), links: (key, t, f) => M[key].linksAt(t, f), counters: (key, t) => M[key].countersAt(t), last: () => JSON.parse(JSON.stringify(LAST)),
    texts: () => [...$('ov').querySelectorAll('*')].filter((e) => e.children.length === 0 || e.classList.contains('lbl')).filter((e) => !e.closest('.live') && e.textContent.trim() && +getComputedStyle(e.closest('#ov > *') || e).opacity > 0.02).map((e) => e.textContent.trim()),
    hud: (id) => $(id).innerText,
  };
  renderFrame(0); window.ready = true;
}
init().catch((e) => { window.__err = String(e && e.stack || e); });
