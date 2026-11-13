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
| Evaluation | 1 seed, monotonicity post-processing | held-out validation/test seeds, mean ± 95 % CI over seeds, no post-processing |

## Layout

```
v9/config.py      scenario presets: building (main), building_noris, aerial (RIS carried by a UAV),
                  legacy_small (old 20 m scene), random_area(seed) for multi-area FL
v9/physics.py     geometry, blockage, channels, RIS element pattern + alignment/quantization, propulsion
v9/env.py         RISUAVEnv: kinematics, energy, scheduling, RIS arbitration, reward, observations
v9/baselines.py   random / straight-line / Greedy (centralized or decentralized) + evaluation loop
v9/tune.py        tunes the Greedy heuristics on validation episodes
v9/analysis.py    static link-budget analysis (coverage maps, upload-time table)
v9/ppo.py         squashed-Gaussian actor, critic, GAE, PPO update (+ FedProx term)
v9/trainers.py    ctde / ippo / fedavg / fedprox / fedper training loops, comm metering, checkpoints
v9/multiarea.py   multi-area hierarchical FL (local / fedavg / pooled)
v9/stats.py       95 % confidence intervals, paired differences
v9/run.py         CLI entry point for one experiment
v9/launch.py      runs groups of experiments in parallel (pilots, campaign, post)
v9/plots.py       figures from saved CSV/NPZ/JSON files only
v9/tests/         unit and regression tests (pytest)
```

## Running

```bash
pip install -r requirements_v9.txt
python -m pytest v9/tests -q

python -m v9.tune --target building          # also building_noris, aerial, multiarea (≈ 1–2 min each)
python -m v9.run --exp gate && python -m v9.plots gate

# one run (≈ 9 min for 4000 episodes on one CPU core)
python -m v9.run --exp train --algo ctde --episodes 4000 --ppo '{"lr_anneal": true}' --seed 0

# the whole campaign, 4 runs in parallel (≈ 3 h on 4 cores), then sweeps and figures
python -m v9.launch campaign --episodes 4000
python -m v9.launch post
python -m v9.plots train building; python -m v9.plots multiarea; python -m v9.plots ris_value
python -m v9.plots sweeps building
```

Results go to `results_v9/<exp>/…` (v9.0 results are kept in `results_v9/archive_v9.0/`).
Seeds: training episodes use `10_000_000·(seed+1)+ep`; checkpoint selection and heuristic
tuning use validation seeds `2_000_000+j`; every reported test number (learned and
heuristic) uses the same fresh test seeds `5_000_000+j` (100 episodes), which were not used
during development.

## Results (v9.1, 5 seeds, 100 fresh test missions, 95 % CIs)

Full tables, figures and discussion: [`docs/v9_results.md`](../docs/v9_results.md);
ground-up explanation: [`docs/v9_explained.md`](../docs/v9_explained.md). Headlines:

* **The RIS matters in the new scene, not in the old one.** Building RIS: +25.8 % EE for
  straight flight, +13.8–14.7 % for tuned heuristics, +12.8 % under learned flight; the
  old 20 m scene: +0.5 %. A courtyard device can finish its upload from 3.4 % of the area
  without the RIS and 90 % with it.
* **Aerial (UAV-carried) RIS: −37.6 %** vs the building RIS, because the carrier adds
  ~41 % flight energy.
* **Learned vs heuristics (single area):** MAPPO-CTDE 8,119 ± 61 bit/J — +14.9 % over the
  straight line, within 1 % of (slightly below) a tuned decentralised heuristic with the
  same information (−0.98 ± 0.74 %), 6.4 % below a centralised full-information heuristic.
* **FL inside one area** (3 UAVs) does not help: FedAvg/FedProx 7,647–7,829, FedPer 7,989,
  IPPO 8,027.
* **FL across city areas** (generalisation to 8 unseen areas): FedAvg 7,953 ± 62 vs local
  7,581 ± 115 (+4.9 %, all 5 seeds) vs pooled raw data 8,162 ± 59; FedAvg recovers 64 % of
  the pooling gain with 8.8× less cross-area traffic (1.6× less in total) and no raw data
  leaving an area.

## Known limitations

Perfect CSI at the RIS controller; buildings are boxes with a fixed +20 dB blockage loss;
fixed altitude; Shannon rates; only propulsion energy counted (RIS-carrier scaled by a
payload factor); data size, bandwidth and receivers per UAV are design choices within
realistic ranges; FL is simulated on one machine (it fixes what is exchanged, not real
hardware); the heuristic references were tuned per scenario; 5 seeds per method.
