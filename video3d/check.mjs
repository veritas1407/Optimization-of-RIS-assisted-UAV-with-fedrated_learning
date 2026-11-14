// Automated truthfulness checks. Run: node check.mjs
import { createRequire } from 'module'; import fs from 'fs'; import path from 'path'; import { fileURLToPath } from 'url';
import { serve, openPage } from './render.mjs';
const require = createRequire(import.meta.url);
let chromium; try { ({ chromium } = require('playwright')); } catch { ({ chromium } = require('/opt/node-tools/node_modules/playwright')); }
const here = path.dirname(fileURLToPath(import.meta.url)), data = path.resolve(here, '../results_v9/animation');
const TR = { ris: 'trace_building_ctde_5000002.json', noris: 'trace_building_noris_ctde_5000002.json', aerial: 'trace_aerial_ctde_5000002.json' };
const traces = Object.fromEntries(Object.entries(TR).map(([k, f]) => [k, JSON.parse(fs.readFileSync(path.join(data, f)))]));
let fails = 0; const ok = (c, msg) => { console.log(`${c ? 'PASS' : 'FAIL'}  ${msg}`); if (!c) fails++; };

const srv = await serve(), browser = await chromium.launch(), page = await openPage(browser, srv.address().port, 960, 540);
const dbg = await page.evaluate('({fps: debug.fps, frames: debug.frames, shots: debug.shots, play: debug.play})');

// 1. drone positions at every slot end match the trace
for (const [k, tr] of Object.entries(traces)) {
  let worst = 0;
  for (let s = 0; s <= tr.n_slots; s++) {
    const got = await page.evaluate(`debug.simPos('${k}', ${s * tr.slot_s})`), want = s === 0 ? tr.start.pos : tr.slots[s - 1].pos;
    got.forEach((p, a) => { worst = Math.max(worst, Math.hypot(p[0] - want[a][0], p[1] - want[a][1], p[2] - want[a][2])); });
  }
  ok(worst <= 0.01, `${k}: drone positions at the ${tr.n_slots + 1} slot boundaries match the trace (max error ${worst.toExponential(2)} m)`);
}

// 2. links exist only in slots where the trace lists them (link function sampled every 0.05 s, then the frames actually drawn)
const listed = (tr, t, l) => { const s = tr.slots[Math.floor(t / tr.slot_s + 1e-9)]; return !!s && s.uploads.some((u) => u.device === l.device && u.uav === l.uav && u.via_ris === l.via_ris); };
for (const [k, tr] of Object.entries(traces)) {
  let bad = 0, n = 0, missing = 0;
  for (let t = 0.01; t < tr.n_slots * tr.slot_s; t += 0.05) {
    const ls = await page.evaluate(`debug.links('${k}', ${t}, 0.67)`); n += ls.length; ls.forEach((l) => { if (!listed(tr, t, l)) bad++; });
    if (ls.length !== tr.slots[Math.floor(t / tr.slot_s + 1e-9)].uploads.length) missing++;
  }
  ok(bad === 0 && missing === 0, `${k}: link function returns exactly the trace's uploads in every slot (${n} link samples, ${bad} not listed, ${missing} mismatched counts)`);
}
const shot = (id) => dbg.shots.find((s) => s.id === id);
let drawn = 0, badDrawn = 0, framesChecked = 0;
for (const id of [4, 5, 6, 7]) {
  const s = shot(id);
  for (let i = Math.round(s.a * dbg.fps); i < Math.round(s.b * dbg.fps); i += 3) {
    await page.evaluate(`renderFrame(${i})`); const last = await page.evaluate('debug.last()'); framesChecked++;
    for (const [k, v] of Object.entries(last)) v.links.forEach((l) => { drawn++; if (!listed(traces[k], v.t, l)) badDrawn++; });
  }
}
ok(badDrawn === 0 && drawn > 0, `rendered frames of shots 4-7: every drawn link is listed by the trace for that slot (${drawn} drawn links in ${framesChecked} frames, ${badDrawn} not listed)`);

// 3. HUD and end cards
const s5 = shot(5); await page.evaluate(`renderFrame(${Math.round(s5.b * dbg.fps) - 16})`);
const hud = await page.evaluate("debug.hud('hudL')"), ee = +(/EE\s*([\d,]+)/.exec(hud) || [0, '0'])[1].replace(/,/g, '');
ok(Math.abs(ee - traces.ris.mission_result.ee) <= 1, `shot 5 final HUD EE = ${ee} bit/J (trace mission_result.ee = ${traces.ris.mission_result.ee.toFixed(1)})`);
for (const [k, tr] of Object.entries(traces)) {
  const c = await page.evaluate(`debug.counters('${k}', 100)`);
  ok(Math.abs(c.bits - tr.mission_result.bits) < 1 && Math.abs(c.energy - tr.mission_result.energy) < 1e-3, `${k}: cumulative bits and energy at t = 100 s equal mission_result`);
}

// 4. every number in the on-screen text is in NUMBERS.md
const allowed = new Set(/```tokens\n([\s\S]*?)```/.exec(fs.readFileSync(path.join(here, 'NUMBERS.md'), 'utf8'))[1].split(/\s+/).filter(Boolean));
const seen = new Map();
for (let i = 0; i < dbg.frames; i += 10) {
  await page.evaluate(`renderFrame(${i})`);
  for (const txt of await page.evaluate('debug.texts()')) for (const m of txt.match(/\d[\d,.]*\d|\d/g) || []) if (!seen.has(m)) seen.set(m, `${(i / dbg.fps).toFixed(1)} s: "${txt.slice(0, 70)}"`);
}
const unknown = [...seen].filter(([t]) => !allowed.has(t));
ok(unknown.length === 0, `all ${seen.size} distinct numbers in the on-screen text are listed in NUMBERS.md` + (unknown.length ? '\n      not listed: ' + unknown.map(([t, w]) => `${t} (${w})`).join('; ') : ''));
ok([...allowed].every((t) => seen.has(t)), `no stale tokens in NUMBERS.md` + ([...allowed].filter((t) => !seen.has(t)).length ? ' (unused: ' + [...allowed].filter((t) => !seen.has(t)).join(' ') + ')' : ''));

await browser.close(); srv.close();
console.log(fails ? `\n${fails} check(s) FAILED` : '\nall checks passed'); process.exit(fails ? 1 : 0);
