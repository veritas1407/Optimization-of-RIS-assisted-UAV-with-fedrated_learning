# Prompt: 3-D simulation video of the RIS-assisted UAV + federated-learning project

*Copy everything below the line into a new Claude session (Fable) that has this repository
checked out on branch `claude/btp-project-analysis-425r7n`.*

---

You are a technical 3-D visualisation engineer and science-communication director. Your job
is to produce a **2–2.5 minute, 1920×1080, 30 fps MP4 video** that shows, in a realistic 3-D
city, what this B.Tech project does and what it found. The audience is a B.Tech project
panel (electrical/communications engineering professors) watching a 15-minute presentation.
The video must be **beautiful, clear, and strictly truthful**: every motion, link and number
on screen must come from the project's recorded simulation data or its results files. This
is a visualisation of a simulation, not a fictional product film.

## 1. What the project is (read this first)

Project title: **"Optimization of RIS-assisted UAV with federated learning."**

- A 500 m × 500 m city area. Three collector UAVs (drones) fly at 60 m for a 100 s mission
  (40 slots × 2.5 s) from the south-west corner to a landing zone in the north-east corner.
- 24 ground IoT sensors hold 10–30 Mbit each and are awake for one 30 s window each. Six of
  them sit in a **courtyard enclosed by 40–45 m buildings**, so a drone's direct radio link to
  them is blocked unless it flies almost overhead.
- A **Reconfigurable Intelligent Surface (RIS)**: a 1 m × 1 m panel of 16 × 16 = 256 tiny
  reflectors mounted high on the courtyard wall. Each element applies one of four phase
  shifts (0°, 90°, 180°, 270°). Aligned correctly, they redirect the courtyard sensor's
  signal to a drone anywhere in the flight zone. It is set by a closed-form formula.
- **Goal (the optimisation):** maximise energy efficiency, EE = bits collected ÷ flight
  energy (bit/J). The drones' trajectories are learned with multi-agent reinforcement
  learning (MAPPO). The RIS phases are computed in closed form.
- **Federated learning (FL):** many city areas each have a ground station that trains its own
  drones. Instead of shipping raw flight data to a cloud server, the stations send only model
  updates, which are averaged (FedAvg). Each round, a random 50 % of areas take part. The
  averaged model generalises to city areas it has never seen.
- Alternative compared in the project: an **aerial RIS**, the same panel carried by a 4th
  drone, whose flight energy counts against EE.

Read `docs/v9_explained.md` (Parts 0, 2, 5, 6, 9, 13) and `docs/v9_results.md` before you
start. They are the authoritative description and the only allowed source of result numbers.

## 2. Data you must drive the video from

All files are under `results_v9/animation/`. Units are metres, seconds, bits and joules.
Coordinates: x = east, y = north, z = up. The ground is z = 0 and the area is [0, 500]².

| File | Content |
|---|---|
| `trace_building_ctde_5000002.json` | One real mission: learned drones + building RIS (EE 8,148 bit/J) |
| `trace_building_noris_ctde_5000002.json` | **Same** mission (same sensors, data, wake windows, fading), no RIS (EE 6,941) |
| `trace_aerial_ctde_5000002.json` | Same mission with the RIS carried by a 4th drone (EE 5,238) |
| `fl_areas.json` | 24 procedurally generated city areas of the FL study (12 train, 4 validation, 8 unseen test): buildings, RIS, device counts, power, RIS size |
| `*_final.png`, `*.mp4` | The existing 2-D top-down animations, as a visual reference for colours and semantics |

Trace JSON fields:
- `buildings[]`: `x0, y0, x1, y1, height_m`, axis-aligned boxes.
- `ris`: `center` (x, y, z), `normal` (unit vector the panel faces), `elements` [16, 16], `side_m` ≈ 1.0, `phase_bits` 2. In the aerial trace there is `carrier_uav_index` instead: the panel hangs under that drone.
- `devices[]`: `xyz`, `in_courtyard`, `data_bits`, `awake_from_s`, `awake_until_s`.
- `final_zone` (center, radius), `landing_slots[]`, `start.pos` (drone positions at t = 0).
- `slots[k]` (k = 0..39, ending at `t_end_s` = 2.5·(k+1)):
  - `pos[uav] = [x, y, z]` at the end of the slot, and `mean_vel_xy`.
  - `uploads[]`: `{device, uav, via_ris}`. These are the links active during this slot; `via_ris` true means the RIS is bouncing that link.
  - `data_left_bits[device]` after the slot, plus `bits_this_slot` and `energy_this_slot_J`.
- `mission_result`: final `ee`, `bits`, `energy`, `delivered_frac`, `shadow_delivered_frac`, etc.

Interpolation inside a slot: the velocity changes linearly, so use a centripetal
Catmull-Rom spline through `start.pos` and the slot-end positions. Positions at every slot end
must match the trace exactly (tolerance 0.01 m). Live counters are cumulative: data =
Σ bits / 8e6 MB, energy = Σ energy / 1e3 kJ, and EE = Σ bits / Σ energy. Interpolate the
counters linearly within each slot.

## 3. Technical approach (use this unless it is impossible; then explain why)

- **Three.js** (pin an exact version, e.g. `three@0.160.0`, installed with npm into a new
  `video3d/` folder) in a single page `video3d/index.html` plus ES modules.
- **Deterministic offline rendering.** Expose `window.renderFrame(frameIndex)`, which sets the
  timeline time to frameIndex/30 and renders synchronously with no real-time clock and no
  `requestAnimationFrame` timing. A Node script (`video3d/render.mjs`) uses **Playwright with
  the preinstalled Chromium** to render each frame at 1920×1080 and stream PNGs to
  **ffmpeg**:
  - Chromium: `require('playwright')` works and `chromium.launch()` needs no options. If a pinned Playwright version cannot find the browser, use `executablePath: '/opt/pw-browsers/chromium'`. Do not run `playwright install`.
  - Encoding: `-c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow -r 30`.
- **Preview mode.** A `--preview` flag renders every 10th frame at 960×540, so you can iterate
  quickly. Render the full video only when the storyboard is right.
- Enable shadows and PCF soft shadow maps. Use a hemisphere light plus a sun at a low-morning
  angle (warm key, cool fill), ACES filmic tone mapping, light fog for depth, and 4× MSAA or FXAA.
- **No external assets that need a licence or network at render time.** Generate all geometry
  procedurally. Use system fonts or CSS for overlays.

## 4. Visual design rules

- **Scale honesty.** The world is at true scale. Drones (~0.5 m) and the 1 m RIS panel would
  be invisible at 500 m, so draw drones about 12× larger and the panel about 8× larger. Show a
  small permanent caption: "Drones ×12, RIS ×8 drawn larger for visibility". In the RIS
  close-up shot, show the panel at true size next to a 1.8 m human silhouette.
- **City.** Inside the 500 m area, render only the buildings in the trace, extruded with
  concrete or glass façade shaders and window grids. Outside the area you may add low-detail
  "context" blocks (desaturated, clearly secondary) for realism. They must not occlude the
  action, and nothing outside the area affects the simulation. Ground: asphalt and grass
  patches, with a subtle 50 m grid in the analysis shots only.
- **Colour code.** Keep it identical to the 2-D animations so the panel recognises it:
  - Collector drones: #1f77b4, #ff7f0e, #2ca02c. The aerial RIS carrier: #9467bd.
  - Courtyard sensors: red #d62728. Open-field sensors: blue #1f77b4.
  - RIS panel: gold. Direct radio link: green. RIS-bounced link: gold, drawn as two segments, sensor → panel → drone.
- **Sensors.** A small box on a pole with a vertical "data left" bar that shrinks as data is
  collected (height ∝ `data_left_bits`). Glow while awake (`awake_from_s` ≤ t < `awake_until_s`).
  Dim when empty or asleep.
- **Links.** Animated travelling pulses along the link, from sensor to drone. The pulse speed is
  only a visual cue. The link exists only during slots where it appears in `uploads`. Fade
  links in and out over 0.2 s at slot boundaries.
- **RIS panel.** A 16 × 16 tile grid. While a `via_ris` link is active, colour the tiles by a
  2-bit phase pattern: four shades of gold. Compute the pattern from the geometry with the
  formula θ_l = −(2π/λ)(d_sensor→element + d_element→drone) mod 2π, quantised to 4 levels,
  λ = 0.125 m. Recompute it when the served drone or sensor changes, so the pattern visibly
  re-steers. Label it "phase pattern (illustrative of the 2-bit configuration)".
- **Blocked link.** In the problem shot, draw the direct sensor → drone line in red, stopping at
  the wall where it is blocked, with a small "blocked by building" tag.
- **Trails.** Fading trails behind each drone, showing the last 20 s.
- **Typography and HUD.** Clean sans-serif. Top-left: t (s), data collected (MB), flight energy
  (kJ), EE (bit/J). Bottom: a one-line shot caption. Section titles are lower-thirds, never
  covering the action. Respect safe margins of 5 %.
- **Camera.** Smooth, eased spline moves, with no cuts faster than 2 s. Stay at least 10 m away
  from building faces. No shaky-cam.

## 5. Storyboard (target ≈ 140 s; adjust ±15 s if pacing needs it)

| # | Time | Shot | Content |
|---|---|---|---|
| 1 | 0–6 s | Title | Dark-to-dawn fade over the city. Title: "Optimization of RIS-assisted UAVs with Federated Learning". Subtitle: "Energy-efficient IoT data collection in a shadowed urban courtyard (simulation)". |
| 2 | 6–18 s | Establishing fly-over | Orbit from high south-west over the 500 m area. Reveal the start point (SW), the landing zone (NE, ring of radius 40 m), the urban block with the courtyard, and the 24 sensors. Caption: "3 drones · 24 IoT sensors · 100 s mission". |
| 3 | 18–33 s | The problem | Descend toward the courtyard. Show a red courtyard sensor and a drone in the open field. The direct link is drawn red and stops at the wall ("blocked by building"). Caption: "Courtyard sensors are shadowed: without help, a drone must fly almost overhead". Optionally show "3.4 % of the area can serve a courtyard sensor in time". |
| 4 | 33–50 s | The RIS | Close-up on the wall-mounted panel at true scale next to the human silhouette, then pull out to the 8× display scale. A courtyard sensor's signal hits the panel, the tile phase pattern lights up, and a gold beam re-steers to a drone 300 m away. Caption: "256 reflecting elements, 2-bit phases, set in closed form. Usable area: 3.4 % → 90 %". |
| 5 | 50–80 s | Mission replay with RIS | Play `trace_building_ctde_5000002.json` at 100 s → 30 s (3.33×). Camera: slow high three-quarter view tracking the swarm centroid, with a 4 s dip toward the courtyard when the first `via_ris` link appears. HUD counters live. End on the landing zone and freeze the HUD at EE 8,148 bit/J and courtyard data delivered 90 %. |
| 6 | 80–98 s | Same mission without the RIS | Split screen, left "with RIS" and right "no RIS", with both traces in sync at 6.7× (100 s → 15 s). The no-RIS drones dive toward the courtyard. End cards: "EE 8,148 vs 6,941 bit/J" and "courtyard data 90 % vs 47 %" (from `mission_result`). |
| 7 | 98–110 s | Aerial RIS | `trace_aerial_ctde_5000002.json`: the purple carrier drone flies with the panel hanging below. Gold links go sensor → carrier → collector. Energy counter emphasised. End card: "EE 5,238 bit/J: the 4th drone's flight energy costs more than the RIS gains". |
| 8 | 110–130 s | Federated learning | Pull far out to a stylised "city map" grid of the 12 training areas from `fl_areas.json`. Show each area's building block and RIS, small at map scale, and a ground station. Show 3 rounds: each lights a different random 6 of 12 areas (50 % participation), their drones train (brief pulse), and glowing model packets fly up to a central cloud node and back. A padlock icon on each area reads "raw flight data never leaves the area". Then the 8 unseen test areas appear greyed out and turn green as the global model is deployed. On-screen numbers, only these, from `docs/v9_results.md` §4b: "+9.7 % EE vs each area learning alone", "as good as pooling all raw data with 13× less traffic", "8-bit updates: 38 % less traffic, no loss". |
| 9 | 130–140 s | Results and honesty | Clean card with 4 lines: "Building RIS: +13–26 % EE"; "Aerial RIS: −38 %"; "Learned flight: within 1 % of a tuned same-information heuristic, 6 % below a full-information planner"; "FL: generalises to unseen city areas without sharing raw data". Footer: "All results from simulation (5 seeds, 100 test missions, 95 % CIs)". |
| 10 | 140–145 s | End card | Title, "B.Tech project", and a link line to the repository. |

## 6. Truthfulness rules (non-negotiable)

1. Every number shown must come from a trace file's `mission_result` or from
   `docs/v9_results.md` or `docs/v9_explained.md`. Do not round in a way that changes meaning.
   Keep a `video3d/NUMBERS.md` mapping each on-screen number to its source file and line.
2. Drone paths, link timing, `via_ris` flags, sensor wake times and data bars come from the
   traces. Never invent extra links, beams, signal heat-maps or manoeuvres. Illustrative-only
   elements (the phase-pattern colours, the FL "packets", the cloud node, context buildings)
   must be visually distinct and, where needed, labelled "illustrative".
3. Label the video as a simulation. Do not depict a real city, real brands or logos, or claim
   real-world deployment or hardware results.
4. If something in the storyboard conflicts with the data (e.g. no `via_ris` link exists when a
   shot needs one), change the shot timing or camera to fit the data, never the data, and note it.

## 7. Workflow and verification

1. Read the docs and inspect the JSON files. Write `video3d/PLAN.md` with your scene list,
   timings, camera keyframes and which trace slots each shot uses.
2. Build the scene and the timeline. Render a preview. Extract frames at 5, 15, 25, 40, 60, 75,
   90, 105, 120 and 135 s and **look at them**. Check readability at 1080p and at half size,
   label overlap, scale captions, and colour code.
3. Automated checks in `video3d/check.mjs`:
   - Drone positions at slot ends match the trace (≤ 0.01 m).
   - Links are drawn only in slots where they exist.
   - HUD EE at the final frame of shot 5 equals 8,148 ± 1.
   - Every number on screen appears in NUMBERS.md.
4. Render the final MP4 to `results_v9/animation/ris_uav_3d.mp4`. Also save 6 key stills
   (PNG, 1920×1080) and a 20 s highlight cut (`ris_uav_3d_highlight.mp4`, shots 4–6).
5. Write `video3d/README.md` covering how to re-render, the dependencies with exact versions,
   and the render time.
6. Commit everything except `node_modules` and frame dumps to the branch
   `claude/btp-project-analysis-425r7n`, and push. Do not open a pull request.

## 8. Definition of done

- The MP4 plays in a standard player (H.264, yuv420p) and is 2–2.5 minutes long.
- A viewer with no context understands four things: (1) why courtyard sensors are hard to reach, (2) how the RIS fixes it, (3) that learned drones collect more data per joule with the RIS than without, and (4) what federated learning shares and what it does not.
- All automated checks pass, NUMBERS.md is complete, and the stills look professional.
- Your final message lists what was built, the render time, any storyboard deviations and why,
  and the exact file paths.
