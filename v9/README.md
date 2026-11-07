# v9 — RIS-assisted multi-UAV IoT data collection with federated MARL

This package replaces the v4–v8 scripts (kept untouched as the archive). It fixes the
defects found in the v8 audit (`docs/analysis/01-technical-audit.md`) and moves the study
from the 20 m toy scene to a 500 m urban scene where the RIS has a real job.

## What changed vs v8

| | v8 | v9 |
|---|---|---|
| World | 20 × 20 m, UAVs at 5 m, no obstacles | 500 × 500 m, UAVs at 60 m, a courtyard of 40–45 m buildings that shadows 6 of 24 devices |
| RIS | ground level, 50 elements, each UAV applies its own private phases | one 16 × 16 (1 m²) panel on a building face, 2-bit phases, **one configuration at a time**, closed-form alignment |
| RL action | 52 values (2 velocity + 50 phases) | 2 values (velocity command), acceleration-limited |
| Devices | 12 fixed, unlimited data, always on, counted at all 3 UAVs | 24 placed randomly each episode, 10–30 Mbit each, 30 s active window, each device's bits counted once |
| Channel | Rayleigh/Rician with tuned ρ₀ | Al-Hourani urban air-to-ground LoS probability, explicit building blockage, FSPL anchor, real noise and bandwidth (bit/J is real) |
| Reward | `100·EE + 0.4·progress` | Dinkelbach form `(bits − λ·energy)/(λ·E_ref)` + reachability shaping + collision penalty |
| Learning | MAPPO only, PPO math issues | MAPPO-CTDE, IPPO, FedAvg, FedProx with corrected PPO and metered communication |
| Evaluation | 1 seed, monotonicity post-processing | held-out validation/test seeds, mean ± std over seeds, no post-processing |

## Layout

```
v9/config.py      scenario presets: building (main), aerial (RIS carried by a UAV), legacy_small (old scene)
v9/physics.py     geometry, blockage, channels, RIS alignment/quantization, propulsion
v9/env.py         RISUAVEnv: kinematics, energy, scheduling, RIS arbitration, reward, observations
v9/baselines.py   random / straight-line / greedy policies + evaluation loop
v9/analysis.py    static link-budget analysis (coverage maps, upload-time table)
v9/ppo.py         squashed-Gaussian actor, critic, GAE, PPO update (+ FedProx term)
v9/trainers.py    ctde / ippo / fedavg / fedprox training loops, comm metering, checkpoints
v9/run.py         CLI entry point
v9/plots.py       figures from saved CSV/NPZ/JSON files only
v9/tests/         unit tests (pytest)
```

## Running

```bash
pip install -r requirements_v9.txt
python -m pytest v9/tests -q

# gate: coverage maps + baselines (≈ 30 s)
python -m v9.run --exp gate && python -m v9.plots gate

# training (≈ 7 min per 2000-episode run on one CPU core)
python -m v9.run --exp train --algo ctde    --seed 0
python -m v9.run --exp train --algo ippo    --seed 0
python -m v9.run --exp train --algo fedavg  --E 4 --seed 0
python -m v9.run --exp train --algo fedprox --E 4 --mu 0.01 --seed 0
python -m v9.plots train building
```

Results go to `results_v9/<exp>/…`. Seeds: training episodes use `10_000_000·(seed+1)+ep`,
checkpoint selection uses validation seeds `2_000_000+j`, and every reported test number
(learned and baseline) uses the same held-out test seeds `1_000_000+j`.

## Gate results (building scene)

* A shadowed device can upload its 20 Mbit batch within its 30 s window from **3.4 %** of
  the area without the RIS and from **33 %** with the RIS (`results_v9/gate/fig_coverage.png`).
* The aligned RIS raises EE by 11–16 % for every baseline flight policy; random RIS phases
  give nothing. In the old 20 m scene all three RIS settings give identical EE
  (`fig_old_vs_new.png`) — the old scene could not show the RIS's value.
* Aerial RIS (panel carried by a 4th UAV at 100 m) currently lowers EE: its propulsion
  energy (+37 % of fleet energy) outweighs its gain for the baseline policies.

## Known limitations

Perfect CSI at the RIS controller; buildings are boxes with a fixed +20 dB blockage loss;
fixed altitude; Shannon rates; only propulsion energy counted (RIS-carrier scaled by a
payload factor); data size, bandwidth and receivers per UAV are design choices within
realistic ranges; FL is simulated on one machine (it fixes what is exchanged, not real
hardware); 3 seeds per method.
