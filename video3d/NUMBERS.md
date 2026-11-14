# Every number shown in the video and where it comes from

Paths are relative to the repository root. `trace:` means a field of the trace JSON in
`results_v9/animation/` (R = `trace_building_ctde_5000002.json`, N =
`trace_building_noris_ctde_5000002.json`, A = `trace_aerial_ctde_5000002.json`).
`check.mjs` verifies that every number in the on-screen text appears in the token list at the
bottom of this file, and re-computes the live values from the traces.

## Live values (computed every frame from the traces)

| On screen | Computed from |
|---|---|
| HUD time | playback time of the trace, 0–100 s |
| HUD data collected (MB) | Σ `slots[k].bits_this_slot` / 8e6, linear within a slot |
| HUD flight energy (kJ) | Σ `slots[k].energy_this_slot_J` / 1e3, linear within a slot |
| HUD EE (bit/J) | Σ bits / Σ energy |
| "drone, 204 m away" (shot 3) | distance between courtyard sensor 20 and drone 2 at t = 10 s in N |
| "… sensors · RIS n×n" under each FL tile | `n_devices`, `ris_elements` in `fl_areas.json` |
| "Round 1/2/3" | the three rounds shown (illustrative) |

## Static numbers

| Number on screen | Shot | Source |
|---|---|---|
| 3 drones, 24 IoT sensors, 100 s mission | 2 | trace: `n_collectors`, `devices.length`, `n_slots × slot_s`; `docs/v9_explained.md` Part 2 |
| 500 m × 500 m, 60 m | 2 | trace: `area_m`, `uav_altitude_m`; `docs/v9_explained.md` lines 66, 76 |
| 6 shadowed sensors, 18 open-field sensors | 2 | trace: count of `devices[].in_courtyard` true / false |
| 3.4 % of the area | 3, 4 | `docs/v9_results.md` line 46 (`results_v9/gate/summary.json` `area_frac_shadow_feasible_noris`) |
| 3.4 % → 90 % | 4 | `docs/v9_results.md` line 46 (90.0 %; `area_frac_shadow_feasible_ris`) |
| 1 m × 1 m, 16 × 16 = 256 elements | 4 | trace: `ris.side_m` (0.999), `ris.elements`; `docs/v9_explained.md` line 179 |
| mounted 3 m above the roof edge | 4 | trace: `ris.center[2]` = 48 m, south-row `height_m` = 45 m |
| 4 phase shifts, 2-bit | 4 | trace: `ris.phase_bits` = 2 |
| 1.8 m person | 4 | scale reference only (not a result) |
| drones, about 300 m from the RIS | 4 | R, slot 0: drone-to-RIS distances 296 m and 320 m for the two RIS links |
| ×12, ×8 | all mission shots | display scale factors of this video (`DRONE_SCALE`, `RIS_SCALE` in `src/mission.js`) |
| EE 8,148 bit/J, courtyard data 90 % | 5, 6 | R `mission_result.ee` = 8147.6, `shadow_delivered_frac` = 0.898 |
| EE 6,941 bit/J, courtyard data 47 % | 6 | N `mission_result.ee` = 6941.4, `shadow_delivered_frac` = 0.470 |
| EE 5,238 bit/J | 7 | A `mission_result.ee` = 5237.9 |
| flight energy 59.5 kJ vs 43.4 kJ | 7 | A `mission_result.energy` = 59,516 J; R `mission_result.energy` = 43,378 J |
| 4th drone | 7 | A: `n_agents` = 4, `ris.carrier_uav_index` = 3 |
| 12 city areas, 3 drones each | 8 | `fl_areas.json` (12 with `split` = train); `docs/v9_results.md` line 214 |
| a random 50 % of the areas | 8 | `docs/v9_results.md` §4b ("Partial participation … 50 %"); `v9/fedhet.py` `participation = 0.5` |
| showing 3 of 60 rounds | 8 | `docs/v9_results.md` lines 249–253 (60 rounds at 50 % participation, E = 4) |
| which 6 areas light up in rounds 1–3 | 8 | `numpy.random.default_rng(1000).choice(12, 6, replace=False)` ×3, the sampler of `v9/fedhet.py` for seed 0: [1,4,5,6,7,9], [1,2,3,8,9,11], [1,4,6,7,9,11] |
| 8 city areas it has never seen / 8 unseen test areas | 8 | `fl_areas.json` (8 with `split` = test_unseen); `docs/v9_results.md` line 216 |
| +9.7 % vs each area learning alone | 8 | `docs/v9_results.md` line 240 (+695 ± 108 bit/J, 5 of 5 seeds) |
| 13× less cross-area traffic, as good as pooling | 8 | `docs/v9_results.md` line 243 (85 MB vs 1,081 MB; +1.7 %, not significant) |
| 8-bit updates: 38 % less traffic, no measurable loss | 8 | `docs/v9_results.md` line 245 (38–40 % less; −0.3 %, not significant); 53.2 vs 85.2 MB |
| Building RIS +13–26 % | 9 | `docs/v9_results.md`: +12.8 % learned flight (line 286), +13.8 % (line 74), +14.7 % (line 75), +25.8 % (line 72) |
| Aerial RIS −38 % | 9 | `docs/v9_results.md` line 288 (−37.6 %) |
| within 1 %, 6 % below | 9 | `docs/v9_results.md` lines 109–110 (−0.98 ± 0.74 %; 6.4 % below) |
| 5 training seeds, 96–100 test missions, 95 % confidence intervals | 9 | `docs/v9_results.md` "How to read the numbers" and line 225 |
| section numbers 1–7 in the lower-third titles | 2–8 | section counter of the video |

## Token list used by check.mjs

```tokens
1 2 3 4 5 6 7 8 12 13 16 18 24 26 38 47 50 60 90 95 96 100 256 300 500
1.8 3.4 9.7 43.4 59.5 5,238 6,941 8,148
```
