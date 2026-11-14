// Deterministic offline renderer: Playwright (preinstalled Chromium) -> PNG frames -> ffmpeg (H.264).
//   node render.mjs                       full video, 1920x1080, 30 fps -> ../results_v9/animation/ris_uav_3d.mp4
//   node render.mjs --preview             every 10th frame at 960x540 -> out/preview.mp4
//   node render.mjs --stills 5,15,25      PNG stills at the given times (s) -> out/still_<t>.png   (add --preview for 960x540)
//   node render.mjs --start 31 --end 98.5 --out out/part.mp4     a time range
import { createRequire } from 'module';
import http from 'http'; import fs from 'fs'; import path from 'path'; import { spawn } from 'child_process'; import { fileURLToPath } from 'url';
const require = createRequire(import.meta.url);
let chromium; try { ({ chromium } = require('playwright')); } catch { ({ chromium } = require('/opt/node-tools/node_modules/playwright')); }

const here = path.dirname(fileURLToPath(import.meta.url)), root = path.resolve(here, '..');
const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i < 0 ? d : (process.argv[i + 1] && !process.argv[i + 1].startsWith('--') ? process.argv[i + 1] : true); };
const preview = !!arg('preview', false), W = preview ? 960 : 1920, H = preview ? 540 : 1080, step = preview ? 10 : 1, FPS = 30;
const workers = +arg('workers', 4), stills = arg('stills', null);
const out = path.resolve(here, arg('out', preview ? 'out/preview.mp4' : '../results_v9/animation/ris_uav_3d.mp4'));
fs.mkdirSync(path.join(here, 'out'), { recursive: true });

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.css': 'text/css' };
export function serve(port = 0) {
  return new Promise((res) => {
    const srv = http.createServer((q, s) => {
      const f = path.join(root, decodeURIComponent(q.url.split('?')[0]));
      if (!f.startsWith(root)) { s.writeHead(403); s.end(); return; }
      fs.readFile(f, (e, d) => { if (e) { s.writeHead(404); s.end(); return; } s.writeHead(200, { 'Content-Type': MIME[path.extname(f)] || 'application/octet-stream' }); s.end(d); });
    }).listen(port, '127.0.0.1', () => res(srv));
  });
}
export async function openPage(browser, port, w = W, h = H) {
  const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('page error:', e.message));
  await page.goto(`http://127.0.0.1:${port}/video3d/index.html?w=${w}&h=${h}`);
  await page.waitForFunction('window.ready === true || window.__err', null, { timeout: 120000 });
  const err = await page.evaluate('window.__err'); if (err) throw new Error(err);
  return page;
}

async function main() {
  const srv = await serve(), port = srv.address().port, t0 = Date.now();
  if (stills) {
    const browser = await chromium.launch(), page = await openPage(browser, port);
    for (const t of String(stills).split(',').map(Number)) {
      await page.evaluate(`renderFrame(${Math.round(t * FPS)})`);
      const f = path.join(here, 'out', `still_${String(t).replace('.', '_')}${preview ? '_p' : ''}.png`); await page.screenshot({ path: f, type: 'png' }); console.log('wrote', f);
    }
    await browser.close(); srv.close(); return;
  }
  const b0 = await chromium.launch(), p0 = await openPage(b0, port), total = await p0.evaluate('debug.frames'); await b0.close();
  const f0 = Math.round(+arg('start', 0) * FPS), f1 = Math.min(total, Math.round(+arg('end', total / FPS) * FPS));
  const frames = []; for (let i = f0; i < f1; i += step) frames.push(i);
  const n = Math.min(workers, frames.length), chunk = Math.ceil(frames.length / n), segs = [];
  let done = 0; const tick = setInterval(() => process.stdout.write(`\r  ${done}/${frames.length} frames  ${((Date.now() - t0) / 1000).toFixed(0)} s   `), 5000);
  await Promise.all(Array.from({ length: n }, async (_, w) => {
    const mine = frames.slice(w * chunk, (w + 1) * chunk); if (!mine.length) return;
    const seg = path.join(here, 'out', `seg_${w}.mp4`); segs[w] = seg;
    const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS / step), '-c:v', 'png', '-i', '-',
      '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'slow', '-r', String(FPS / step), seg], { stdio: ['pipe', 'inherit', 'inherit'] });
    const closed = new Promise((res, rej) => ff.on('close', (c) => (c === 0 ? res() : rej(new Error('ffmpeg exit ' + c)))));
    const browser = await chromium.launch(), page = await openPage(browser, port);
    for (const i of mine) {
      await page.evaluate(`renderFrame(${i})`);
      const png = await page.screenshot({ type: 'png' });
      if (!ff.stdin.write(png)) await new Promise((r) => ff.stdin.once('drain', r));
      done++;
    }
    ff.stdin.end(); await closed; await browser.close();
  }));
  clearInterval(tick);
  const list = path.join(here, 'out', 'segs.txt'); fs.writeFileSync(list, segs.filter(Boolean).map((s) => `file '${s}'`).join('\n'));
  await new Promise((res, rej) => spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', list, '-c', 'copy', '-movflags', '+faststart', out], { stdio: 'inherit' }).on('close', (c) => (c === 0 ? res() : rej(new Error('concat ' + c)))));
  segs.filter(Boolean).forEach((s) => fs.unlinkSync(s)); fs.unlinkSync(list);
  console.log(`\nwrote ${out}  (${frames.length} frames, ${W}x${H}, ${((Date.now() - t0) / 1000).toFixed(0)} s)`);
  srv.close();
}
if (process.argv[1] === fileURLToPath(import.meta.url)) main().catch((e) => { console.error(e); process.exit(1); });
