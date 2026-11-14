# video3d: 3-D simulation video of the project

Renders `results_v9/animation/ris_uav_3d.mp4` (146 s, 1920×1080, 30 fps, H.264): a 3-D
replay of the recorded simulation missions and an overview of the federated-learning study.
Everything that moves is read from the trace files in `results_v9/animation/`; see
[`PLAN.md`](PLAN.md) for the storyboard and for what is data and what is illustration, and
[`NUMBERS.md`](NUMBERS.md) for the source of every number on screen.

## Re-render

```bash
python -m v9.export_trace          # (repo root) re-export the traces if the simulation changed
cd video3d
npm install                        # three@0.160.0 (pinned in package.json)
node render.mjs --preview          # every 10th frame at 960×540 -> out/preview.mp4   (≈ 2 min)
node render.mjs --stills 15,60,90  # PNG stills at those times -> out/still_*.png
node check.mjs                     # truthfulness checks (≈ 10 min)
./render_all.sh                    # full video in restartable parts -> ../results_v9/animation/ris_uav_3d.mp4
```

Open `index.html` through any static file server rooted at the repository (the page loads
`../results_v9/animation/*.json`) and call `renderFrame(i)` in the console to look at frame `i`.

## How it works

- `index.html`, `src/main.js`: the timeline. `window.renderFrame(i)` draws frame `i` (time
  `i / 30` s) synchronously; nothing depends on a real-time clock, so renders are reproducible.
- `src/mission.js`: one recorded mission as a 3-D scene (city, sensors, drones, RIS, links).
- `src/fl.js`: the federated-learning overview (area layouts from `fl_areas.json`).
- `src/util.js`: coordinate mapping, splines, procedural textures (no external assets).
- `render.mjs`: Playwright (headless Chromium, software WebGL) renders each frame, PNG
  screenshots are piped to ffmpeg (`libx264`, `yuv420p`, `crf 18`, `preset slow`); 4 browser
  workers render frame ranges in parallel and the segments are concatenated without re-encoding.
- `check.mjs`: verifies that drone positions at slot ends equal the trace, that every drawn
  link is listed by the trace for that slot, that HUD totals equal `mission_result`, and that
  every number in the on-screen text is listed in `NUMBERS.md`.

## Dependencies (exact versions used)

| Tool | Version |
|---|---|
| three.js | 0.160.0 (npm, pinned) |
| Node.js | 22.23.4 |
| Playwright + Chromium | the preinstalled build (chromium-1243); any recent version works |
| ffmpeg | 6.1.1 |

Render time on a 4-core CPU without GPU (software WebGL): about 50 minutes for the full
video (≈ 1.5 frames/s), about 2 minutes for the preview.
