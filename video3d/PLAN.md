# Plan: 3-D simulation video (146 s, 1920×1080, 30 fps)

Everything that moves is replayed from the recorded traces in `results_v9/animation/`
(R = building RIS, N = same mission without RIS, A = aerial RIS; mission seed 5000002).

| # | Time (s) | Shot | Scene / trace time | Camera (simulation coordinates x, y, z) |
|---|---|---|---|---|
| 1 | 0–6 | Title | N frozen at t = 0 | slow drift from (−300, −345, 360) toward (−245, −295, 322), looking at the area centre |
| 2 | 6–17 | The scenario | N frozen at t = 0 | orbit (−235, −265, 305) → (60, −305, 235) → (335, −255, 195) |
| 3 | 17–31 | The problem | N frozen at t = 10 s; direct path sensor 20 → drone 2, cut at the wall it hits (slab test on the trace's buildings) | descend from the east along the courtyard axis: (590, 30, 240) → (548, 98, 158) → (524, 108, 128) |
| 4 | 31–47 | The RIS | R: t = 0 (panel at true scale, 0–6.3 s), panel grows ×1 → ×8 (6.3–8.6 s), then slots 0–1 replayed at 0.75× (from 9.8 s) | close-up (362, 106, 48.5) → wide east view (620, 118, 265) |
| 5 | 47–80 | Mission with the RIS | R: 0–100 s in 30 s (3.33×), then 3 s hold | starts on the courtyard (first RIS links are in slot 0), blends to a tracking camera: swarm centroid + (−235, −175, 250) |
| 6 | 80–98.5 | With vs without the RIS | R (left) and N (right), 0–100 s in 15 s (6.7×), 3.5 s hold | same tracking camera in both halves: R centroid + (−275, −205, 315) |
| 7 | 98.5–110.5 | Aerial RIS | A: 0–100 s in 9.5 s, 2.5 s hold | tracking camera: collectors' centroid + (−255, −190, 275) |
| 8 | 110.5–131.5 | Federated learning | `fl_areas.json`: 12 training tiles, 3 rounds (4 s each) with the areas sampled by the real run (seed 0), then the 8 unseen test areas | fixed three-quarter view of the tile map, slow push-in |
| 9 | 131.5–141.5 | Results | R at t = 100 s, dimmed | slow orbit |
| 10 | 141.5–146 | End card | R at t = 100 s, dimmed | slow orbit |

Shots are separated by 0.5 s dips to dark, so time jumps of the replay are never shown as motion.

## What is data and what is illustration

| Element | Source |
|---|---|
| Buildings, RIS position and facing, sensors, start pads, landing zone | trace JSON (exact coordinates) |
| Drone paths | centripetal Catmull-Rom through the slot-end positions (exact at slot ends) |
| Links, green/gold colour, which links use the RIS | `slots[k].uploads` |
| Sensor data bars, awake glow | `data_left_bits`, `awake_from_s`, `awake_until_s` |
| HUD counters | cumulative `bits_this_slot`, `energy_this_slot_J` |
| Blocked direct path (shot 3) | geometry: first building hit of the sensor → drone segment at t = 10 s of N |
| RIS tile colours | illustrative: θ = −(2π/λ)(d₁ + d₂) mod 2π quantised to 4 levels for the link currently served, λ = 0.125 m (labelled on screen) |
| Pulses along links, drone/sensor glow, context buildings outside the 500 m area, FL packets, cloud node, padlocks | illustrative (labelled or outside the simulated area) |
| Display scale | drones ×12, RIS ×8 (bottom edge kept in place), sensor markers enlarged; noted on screen; shot 4 opens at true scale |

## Deviations from the storyboard

1. Total length 146 s (storyboard ≈ 145 s): holds after shots 5–7 so the end cards can be read.
2. Shot 6 caption does not say "the no-RIS drones dive toward the courtyard": in the recorded
   mission drones of both runs pass near the block, so the claim would not be supported.
   The caption states only what is true: same mission, only the RIS differs.
3. Shot 5: the first RIS link is in slot 0, so the "dip toward the courtyard" is the opening
   of the shot (it continues the wide view of shot 4) instead of a mid-shot move.
4. Shot 4 wide view shows the panel from behind (it faces north-west and down, into the
   courtyard): from the front the courtyard floor is hidden by the 45 m rows. The picture-in-
   picture shows the panel from the front.
5. FL statement uses "no measurable loss" (not "no loss") for 8-bit updates, as in the results.
