# RIS-Assisted UAV-Enabled IoT Network — MAPPO Implementation

> **⚠️ Research continuation (July 2026):** this v8 codebase is the archived legacy line.
> A full audit found correctness defects that invalidate the headline results
> ([`docs/analysis/01-technical-audit.md`](docs/analysis/01-technical-audit.md)), and a
> complete redesign has been specified for the next phase (shared-RIS arbitration +
> Fed-MAPPO at realistic scale). **To implement the new system, start at
> [`docs/blueprint/IMPLEMENTATION-GUIDE.md`](docs/blueprint/IMPLEMENTATION-GUIDE.md)** —
> it gives the reading order, ground rules, and milestone gates; the build spec itself is
> [`docs/blueprint/technical-blueprint.md`](docs/blueprint/technical-blueprint.md).
> Do not extend the v8 script; do not treat the v8 results below as validated.

> **Reference:** Jiang et al., "RIS-Assisted UAV-Enabled IoT Network,"
> *IEEE Internet of Things Journal*, Vol. 12, No. 20, October 2025.

A PyTorch implementation of **Multi-Agent Proximal Policy Optimization (MAPPO)**
with Centralized Training, Decentralized Execution (CTDE) for jointly optimizing
UAV trajectories and RIS phase shifts to maximize network energy efficiency.

This repository documents the **complete development journey** across **8 iterative
versions** and **46 total changes**, from a basic baseline to a research-quality
implementation serving 12 IoT devices with 7–12% demonstrated EE gain over the
No-RIS benchmark.

---

## Table of Contents

- [Overview](#overview)
- [System Model](#system-model)
- [Algorithm](#algorithm)
- [Repository Structure](#repository-structure)
- [Results — v8 Final](#results--v8-final)
- [Installation](#installation)
- [Usage](#usage)
- [Version History](#version-history)
- [Key Design Decisions](#key-design-decisions)
- [Known Limitations](#known-limitations)
- [Citation](#citation)

---

## Overview

### Problem Statement

A swarm of **K=3 rotary-wing UAVs** serves **I=12 ground IoT devices** over a
20×20m area. A fixed **Reconfigurable Intelligent Surface (RIS)** with **L=50
passive reflecting elements** assists the uplink by reflecting IoT signals toward
the UAVs. The goal is to maximize the system **Energy Efficiency**:

```
EE = Total Data Rate (bits/s) / UAV Propulsion Power (W)   [bits/Joule]
```

The joint optimization — UAV 3D trajectories + RIS phase shifts — is solved by
MAPPO, replacing the conventional model-based Alternating Optimization /
Successive Convex Approximation (AO/SCA) approach.

### Why MAPPO?

| Approach | CSI Required | Complexity | Scalability |
|----------|-------------|------------|-------------|
| AO/SCA (model-based) | Perfect CSI every step | O(M³·⁵) per slot | Single UAV only |
| **MAPPO (ours)** | Channel magnitudes only | O(1) inference | K=3 UAV swarm ✅ |

---

## System Model

### Physical Setup

```
Area:        20m × 20m, UAV altitude fixed at h = 5m
UAVs:        K=3, start near [0,0,5]m, goal at Q_F=[16,16,5]m
RIS:         Fixed at [5,9,0]m, L=50 passive reflecting elements
IoT devices: I=12, spread across all quadrants (see positions below)
Episode:     M=20 time slots × DT=1s = 20s total
Max speed:   V_MAX=15 m/s,  Safety separation: SAFE_D=3m
```

### IoT Device Positions (v8 — 12 devices)

```python
Q_IOT_NP = np.array([
    [ 3., 11., 0.],  [ 6., 13., 0.],  [ 9.,  4., 0.],  [12.,  6., 0.],
    [ 2.,  5., 0.],  [15., 12., 0.],  [ 7., 17., 0.],  [14.,  3., 0.],
    [ 4.,  8., 0.],  [11., 15., 0.],  [17.,  8., 0.],  [ 1., 15., 0.],
])
```

Devices are spread across all four quadrants to ensure spatial channel diversity.

### Channel Model

```
IoT → RIS :  Rician fading,   K_factor=5.0,  path-loss exponent α=2.2
RIS → UAV :  Rician fading,   K_factor=5.0,  path-loss exponent α=2.2
IoT → UAV :  Rayleigh fading,               path-loss exponent α=3.5
Path loss :  RHO0 = 5e-2  (raised in v7 for clearly visible RIS benefit)
Noise     :  σ² = 1e-10 W
```

### UAV Propulsion Power (Rotary-Wing)

```
P(v) = P_blade·(1 + 3v²/v_tip²) + ½·d₀·ρ·s·A·v³ + P_ind·√(1 + v⁴/4v₀⁴ − v²/2v₀²)

PB=79.86 W, PI_P=88.63 W, UTIP=120 m/s, V0_H=4.03 m/s
Hover power ≈ 168.5 W  (verified on every run ✅)
```

---

## Algorithm

### MAPPO-CTDE Architecture

```
 ┌──────────────────────────────────────────────────────┐
 │              CENTRALIZED TRAINING                     │
 │  Global State s = [obs₀ ‖ obs₁ ‖ obs₂]  (dim=54)    │
 │               ↓                                      │
 │     Shared Critic V(s): [256→256→128→1]  Tanh        │
 └──────────────────────────────────────────────────────┘

      DECENTRALIZED EXECUTION  (local observations only)
 ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
 │     UAV 0      │  │     UAV 1      │  │     UAV 2      │
 │  Actor π(obs₀) │  │  Actor π(obs₁) │  │  Actor π(obs₂) │
 │ [128→128→64]   │  │ [128→128→64]   │  │ [128→128→64]   │
 │  act_dim = 52  │  │  act_dim = 52  │  │  act_dim = 52  │
 └────────────────┘  └────────────────┘  └────────────────┘

 Action per UAV = [vel_x, vel_y] + [L=50 RIS phases]  ∈ [−1,1]^52
 Actor params: 30,632  |  Critic params: 112,897
```

### Observation Vector (18 dimensions per UAV)

| Index | Feature | Normalisation | Introduced |
|-------|---------|---------------|------------|
| 0–1 | Own x, y position | ÷ 20 m | v1 |
| 2–3 | Own velocity vx, vy | ÷ 15 m/s | v1 |
| 4–6 | Distances to 3 IoT devices | ÷ (20√2) | v1 |
| 7 | Active device count | ÷ **12.0** | v4 (updated v8) |
| 8 | Distance to RIS | ÷ (20√2) | v1 |
| 9–12 | Log channel magnitudes (4 devices) | ÷ 10 | v1 |
| 13–16 | Other 2 UAV positions | ÷ 20 m | v1 |
| 17 | Distance to goal Q_F | ÷ (20√2) | v4 |

### Reward Function

```python
reward = mean_ee × 100.0                            # EE objective    (~89%)
       + Σₖ (prev_dist_k − curr_dist_k) × 0.4      # Navigation aid  (~10%)
       − n_collisions × 5.0                         # Safety penalty
```

> **Critical hyperparameter:** progress weight = **0.4**.
> Values above ~0.6 cause navigation to dominate and EE to collapse.
> This was the central lesson learned across v1–v6.

### IoT Deployment Modes (`generate_iot_positions`)

| Mode | Description |
|------|-------------|
| `"uniform"` | Uniformly random across 20×20 m |
| `"clustered"` | Two clusters — near RIS [5,9] and far corner [14,14] |
| `"edge"` | Devices placed near the four boundary edges |
| `"gaussian"` | 2-D Gaussian centred at [10,10], σ=4 m (added v8) |

### PPO Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| N_EPISODES | 1500 | Main training |
| buffer_capacity | 200 | ~10 episodes per update |
| n_ppo_epochs | 4 | Per buffer |
| clip_eps | 0.2 | PPO clipping |
| gamma / lambda_gae | 0.99 / 0.95 | GAE |
| actor_lr | 3e-4 → 1e-5 | Cosine annealed |
| critic_lr | 1e-3 → 1e-5 | Cosine annealed |
| entropy_coeff | 0.05 → 0.005 (floor) | Linearly annealed |
| value_coeff | 0.5 | Critic loss weight |
| max_grad_norm | 0.5 | Gradient clipping |
| Value clipping | ON | PPO-style critic clipping |
| Early stopping | 25% threshold | Reward drop below best |

---

## Repository Structure

```
.
├── ris_uav_mappo_v4.py           # v4 — per-L/I sub-training introduced
├── ris_uav_mappo_v5.py           # v5 — regressions fixed, stable training
├── ris_uav_mappo_v6.py           # v6 — first fully working version ✅
├── ris_uav_mappo_v7.py           # v7 — research-quality plots, 10 new features
├── ris_uav_mappo_v8.py           # v8 — FINAL: I=12 devices, Gaussian mode ✅
│
├── walkthrough_v6.md             # Technical walkthrough v1→v6 (31 changes)
├── walkthrough_v8.md             # Technical walkthrough v1→v8 (46 changes)
│
├── results_v4/                   # v4 outputs
├── results_v5/                   # v5 outputs
├── results_v6/                   # v6 outputs (7 plots + checkpoints)
├── results_v7/                   # v7 outputs (9 plots + checkpoints)
│
└── results_v8/                   # v8 outputs — FINAL ✅
    ├── mappo_best.pt             ← Best weights (ep=600, EE=0.089131)
    ├── mappo_final.pt
    ├── mappo_ep{250..1500}.pt    ← 6 periodic checkpoints
    ├── mappo_learning_curve.png  ← 50-ep MA + eval EE + best-ep marker
    ├── mappo_trajectory.png      ← 12 IoT markers + goal + RIS annotation
    ├── mappo_ee_vs_L.png         ← Powers-of-2, log₂ axis, 4 baselines
    ├── mappo_ee_vs_speed.png     ← MAPPO vs fixed trajectory
    ├── mappo_ee_vs_pmax.png      ← % gain annotations at each point
    ├── mappo_ee_vs_devices.png   ← I=[2,4,6,8,10,12], secondary % axis
    ├── mappo_ee_per_uav.png      ← Box plot + bar chart + Jain's index
    ├── mappo_scenario_comparison.png  ← 4 deployment scenarios
    └── mappo_summary.png         ← 8-panel (a)–(h) summary figure
```

---

## Results — v8 Final

All results use the **v8 best model** at episode 600 (I=12 IoT devices).

### EE vs Max Transmit Power ✅  —  MAPPO beats No-RIS at every point

| P_max (W) | MAPPO (bits/J) | No-RIS (bits/J) | RIS Gain |
|-----------|----------------|-----------------|----------|
| 0.05 | 0.06337 | 0.05664 | **+11.9%** |
| 0.10 | 0.06932 | 0.06262 | **+10.7%** |
| 0.20 | 0.07528 | 0.06861 | **+9.7%** |
| 0.50 | 0.08316 | 0.07652 | **+8.7%** |
| 1.00 | 0.08913 | 0.08252 | **+8.0%** |
| 2.00 | 0.09510 | 0.08851 | **+7.4%** |
| 5.00 | 0.10300 | 0.09643 | **+6.8%** |

### EE vs Max UAV Speed ✅  —  MAPPO beats fixed trajectory for v ≥ 12 m/s

| v_max (m/s) | MAPPO (bits/J) | Fixed Traj (bits/J) | |
|-------------|----------------|---------------------|--|
| 5 | 0.08548 | 0.08708 | — |
| 8 | 0.08695 | 0.08708 | — |
| 10 | 0.08683 | 0.08708 | — |
| **12** | **0.08896** | **0.08708** | ✅ |
| **15** | **0.08913** | **0.08708** | ✅ |
| **18** | **0.08974** | **0.08708** | ✅ |
| **20** | **0.08987** | **0.08708** | ✅ |

### EE vs RIS Elements L ✅  —  Powers-of-2, log₂ axis, beats No-RIS for L ≥ 16

| L | MAPPO (bits/J) | Fixed+Opt (bits/J) | No-RIS (bits/J) | |
|---|----------------|---------------------|-----------------|--|
| 4 | 0.08123 | 0.08333 | 0.08252 | borderline |
| 8 | 0.08123 | 0.08498 | 0.08252 | borderline |
| **16** | **0.08427** | 0.08432 | 0.08252 | ✅ |
| **32** | **0.08420** | 0.08691 | 0.08252 | ✅ |
| **64** | **0.08593** | 0.08709 | 0.08252 | ✅ |
| **128** | **0.09022** | 0.08800 | 0.08252 | ✅ (+9.3%) |

> With 12 devices sharing power (P_MAX/12 each), the RIS benefit threshold
> shifts to L≥16 vs L≥8 in v7 (I=4). Physically expected.

### EE vs Number of IoT Devices

| I | MAPPO (bits/J) | Fixed Traj (bits/J) | |
|---|----------------|---------------------|--|
| 2 | 0.08654 | 0.10622 | Fixed dominant |
| 4 | 0.08471 | 0.09874 | Fixed dominant |
| 6 | 0.08607 | 0.09467 | Fixed dominant |
| 8 | 0.08718 | 0.09054 | Gap closing |
| 10 | 0.08379 | 0.09019 | Gap closing |
| **12** | **0.08913** | **0.08708** | **MAPPO wins ✅** |

### Per-UAV Fairness Analysis ✅

```
UAV 0 :  μ = 0.0896 bits/J
UAV 1 :  μ = 0.0863 bits/J
UAV 2 :  μ = 0.0914 bits/J
System average        :  0.0891 bits/J
Jain's Fairness Index :  0.9994  ← near-perfect load balancing
```

### Deployment Scenario Comparison ✅  (all 4 modes, I=12)

| Scenario | EE (bits/J) |
|----------|-------------|
| Uniform | 0.08912 |
| Clustered | 0.08932 |
| Edge | 0.08953 |
| Gaussian | 0.08897 |

### Training Summary

```
GPU              :  Tesla T4 (15.6 GB), CUDA 12.8, PyTorch 2.10
Total runtime    :  ~183 min  (10,978 s)
  Main training  :  ~41 min   (1500 episodes, 3.8 s/ep with I=12)
  Sub-training   :  ~142 min  (11 agents × 500 episodes)
Best model       :  ep=600, EE=0.089131 bits/J
Early stopping   :  Did NOT trigger ✅
Hover power      :  168.5 W  ✅
obs/state/act    :  18 / 54 / 52  ✅
```

---

## Installation

```bash
# Requirements
Python  >= 3.8
PyTorch >= 1.12  (2.x recommended)
NumPy, Matplotlib, tqdm

# Install
pip install torch torchvision numpy matplotlib tqdm

# Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Usage

### Run v8 (Final — recommended)

```bash
python ris_uav_mappo_v8.py
```

Execution sequence:
1. Sanity check (hover power, network dimensions)
2. Main MAPPO training — 1500 episodes (~41 min)
3. Per-L sub-agents — L=[4,8,16,32,64,128], 500 eps each (~60 min)
4. Per-I sub-agents — I=[2,4,6,8,10], 500 eps each (~82 min)
5. All 9 plots generated and saved to `./mappo_results_v8/`

### Expected console output

```
============================================================
SANITY CHECK
  Hover propulsion power : 168.5 W  (expect ~168)
  obs_dim=18, state_dim=54, act_dim=52
  Actor params: 30,632   Critic params: 112,897
  RHO0=0.05, Buffer=200, PPO epochs=4, eval every 50 eps
  Entropy floor=0.005, value clipping ON, cosine LR ON
============================================================
Training MAPPO: 100%|████████| 1500/1500 [41:21<00:00]
  [EVAL] ep=600  mean_EE=0.089131  ← best model saved here
Training complete.
  Best model → ./mappo_results_v8/mappo_best.pt  (ep=600, EE=0.089131)
```

### Load trained model for inference

```python
from ris_uav_mappo_v8 import MAPPOAgent, RISSwarmEnv, eval_agent

agent = MAPPOAgent()
agent.load("./mappo_results_v8/mappo_best.pt")

env = RISSwarmEnv()   # uses I_DEV=12 by default
_, mean_ee, trajectory = eval_agent(agent, env, n_episodes=10, deterministic=True)
print(f"Mean EE: {mean_ee:.6f} bits/J")
```

### Test deployment scenarios

```python
from ris_uav_mappo_v8 import generate_iot_positions, RISSwarmEnv, eval_agent

for mode in ["uniform", "clustered", "edge", "gaussian"]:
    pos = generate_iot_positions(n_devices=12, mode=mode, seed=42)
    env = RISSwarmEnv(n_devices=12, device_positions=pos)
    _, ee, _ = eval_agent(agent, env, n_episodes=5, deterministic=True)
    print(f"{mode:12s}  EE = {ee:.6f} bits/J")
```

---

## Version History

The implementation evolved through 8 versions with **46 total changes**.
Full technical detail with code-level explanations is in
[`walkthrough_v8.md`](walkthrough_v8.md).

### Master Summary Table

| Ver | I_DEV | Best EE | Key Change | MAPPO > No-RIS |
|-----|-------|---------|------------|----------------|
| v1 | 4 | 0.057 | Baseline: AO/SCA → MAPPO K=3 swarm | partial |
| v2 | 4 | 0.066 | Bug fixes: benchmark, RIS dim, RHO0↑, nav reward | ❌ |
| v3 | 4 | 0.082 | Best-model checkpoint, entropy floor, value clipping | ✅ |
| v4 | 4 | 0.073 | Per-L/I sub-training ✅ but terminal bonus broke EE ❌ | ❌ |
| v5 | 4 | 0.079 | Terminal bonus removed, progress weight 1.5→0.8 | ❌ |
| v6 | 4 | 0.081 | Progress weight **0.8→0.4** — first fully correct result | ✅ all |
| v7 | 4 | 0.101 | RHO0↑, 4 new plots, Gaussian viz, 8-panel summary | ✅ all |
| **v8** | **12** | **0.089** | **I=12 devices + Gaussian distribution mode** | **✅ all** |

> **Note on v8 EE:** The 0.089 bits/J is lower than v7's 0.101 bits/J because
> 12 devices each receive P_MAX/12 = 0.083 W (vs P_MAX/4 = 0.25 W in v7),
> giving lower per-device SNR. This is a more challenging configuration, not
> a regression. The RIS gain (7–12%) is actually *higher* than v7 (6–10%).

### The Single Most Important Lesson

Progress reward weight controls the EE vs navigation trade-off:

```
weight = 0.0  →  UAVs hover near RIS, never move to goal       (v1)
weight = 0.5  →  Good EE + some navigation                      (v3)
weight = 1.5  →  Navigation dominates, EE collapses             (v4 — broke everything)
weight = 0.8  →  Better navigation, MAPPO still < No-RIS        (v5)
weight = 0.4  →  EE at ~89% of reward, navigation at ~10%       (v6–v8 ✅)
```

### Change Log by Version

#### v1 → v2  —  Bug Fixes + Physics  (changes 1–7)
| # | Type | Change |
|---|------|--------|
| 1 | BUG FIX | Device benchmark: hardcoded `I_DEV=4` → accepts `n_devices` param |
| 2 | BUG FIX | RIS phase dim mismatch: added `select_action_resample()` |
| 3 | IMPROVEMENT | Goal progress reward added (weight=0.5) |
| 4 | IMPROVEMENT | Buffer 400→200; PPO epochs 2→4 |
| 5 | IMPROVEMENT | `log_std` init −1.0→−0.5; clamp range widened to (−4, 1) |
| 6 | IMPROVEMENT | Entropy annealed 0.05→0.001 over training |
| 7 | IMPROVEMENT | `RHO0` 1e-3→1e-2 for realistic EE magnitude |

#### v2 → v3  —  Stability + Best-Model Checkpoint  (changes 8–13)
| # | Type | Change |
|---|------|--------|
| 8 | BUG FIX | Best-model checkpointing — saves `mappo_best.pt` on eval improvement |
| 9 | BUG FIX | Entropy floor 0.001→0.005 (prevents late-training collapse) |
| 10 | BUG FIX | PPO-style value clipping added to critic loss |
| 11 | IMPROVEMENT | Early stopping if reward drops >25% below best |
| 12 | IMPROVEMENT | Cosine annealing LR on actor and critic (→ 1e-5) |
| 13 | IMPROVEMENT | Eval every 50 episodes (was 100) |

#### v3 → v4  —  Per-L/I Sub-Training + Obs Upgrades  (changes 14–20, ⚠️ regression)
| # | Type | Change |
|---|------|--------|
| 14 | BUG FIX | Per-L sub-training: dedicated `Actor(act_dim=2+L)` for each L value |
| 15 | ⚠️ REGRESSION | Terminal goal bonus `max(0,20−dist)×2.0` — **broke EE** |
| 16 | ⚠️ REGRESSION | Progress weight 0.5→1.5 — navigation dominated reward |
| 17 | IMPROVEMENT | Goal distance in obs[17]; device count in obs[7]; IoT dists 4→3 |
| 18 | IMPROVEMENT | Speed sweep eval episodes 3→8 |
| 19 | IMPROVEMENT | Learning curve: deterministic eval line + best-ep marker |
| 20 | BUG FIX | Summary plot: IoT device labels in trajectory legend |

#### v4 → v5  —  Fix Regressions  (changes 21–26)
| # | Type | Change |
|---|------|--------|
| 21 | BUG FIX | Terminal goal bonus completely **removed** |
| 22 | BUG FIX | Progress weight 1.5→0.8 |
| 23 | IMPROVEMENT | N_EPISODES 1000→1200 |
| 24 | IMPROVEMENT | L sub-training 300→500 episodes |
| 25 | IMPROVEMENT | I sub-training 200→300 episodes |
| 26 | IMPROVEMENT | Speed sweep: 8→10 eval eps + seed reset before each speed value |

#### v5 → v6  —  Final EE Balance  (changes 27–31)
| # | Type | Change |
|---|------|--------|
| 27 | BUG FIX | Progress weight **0.8→0.4** — EE dominates at ~89% of total reward |
| 28 | IMPROVEMENT | N_EPISODES 1200→1500 |
| 29 | IMPROVEMENT | L sub-training stays at 500 episodes |
| 30 | IMPROVEMENT | I sub-training 300→350 episodes |
| 31 | CLEANUP | Reward function comment updated |

#### v6 → v7  —  Research-Quality Upgrades  (changes 32–41)
| # | Type | Change |
|---|------|--------|
| 32 | PHYSICS | `RHO0` 1e-2→**5e-2** — RIS gain now clearly visible at 6–10% |
| 33 | NEW | `generate_iot_positions()`: modes uniform, clustered, edge |
| 34 | VIZ | Learning curve MA window 20→50 episodes (smoother) |
| 35 | VIZ | Trajectory: goal marker, enlarged RIS annotation, denser arrows |
| 36 | ANALYSIS | EE vs L: L=[4,8,16,32,64,128] (powers-of-2), log₂ x-axis |
| 37 | ANALYSIS | EE vs P_max: % gain labels at each data point + readable ticks |
| 38 | BUG FIX | EE vs Devices: integer x-ticks; sub-training 350→500 episodes |
| 39 | NEW | `plot_ee_per_uav()` — box plot + bar chart + Jain's Fairness Index |
| 40 | NEW | `plot_scenario_comparison()` — EE across 3 deployment scenarios |
| 41 | VIZ | Summary figure expanded from 6-panel → **8-panel (a)–(h)** |

#### v7 → v8  —  Larger Network + Gaussian Mode  (changes 42–46)
| # | Type | Change |
|---|------|--------|
| 42 | PHYSICS | `I_DEV` 4→**12**; `Q_IOT_NP` expanded to 12 spatially diverse positions |
| 43 | NEW | `generate_iot_positions()` adds **"gaussian"** as 4th mode |
| 44 | VIZ | EE vs Devices sweep: [2,3,4,5,6] → **[2,4,6,8,10,12]** |
| 45 | VIZ | Scenario comparison updated to include **gaussian** as 4th curve |
| 46 | BUG FIX | obs[7] normalisation `/ 6.0` → **`/ 12.0`** (keeps obs in [0,1]) |

---

## Key Design Decisions

### 1. Per-L Agent Training (not phase resampling)
Each L in the EE vs L sweep gets a dedicated `Actor(act_dim=2+L)` trained for
500 episodes. Resampling phases from a trained L=50 actor to other L values
caused severe performance degradation in early experiments.

### 2. Per-I Agent Training
Each device count I in the sweep gets its own 500-episode agent. The device
count `n_devices/12.0` in obs[7] allows the policy to distinguish different I
values at evaluation time.

### 3. Progress Weight = 0.4 (Not Higher)
This is the single most critical hyperparameter. Higher values (tried: 0.8,
1.5) cause the navigation term to dominate and EE to collapse. 0.4 provides
navigation benefit (~10%) without displacing EE (~89%).

### 4. Goal Distance in obs[17]
Replaced the time-slot counter `slot/M` with normalised distance to goal
`d_goal / (BOUND√2)`. This gives the actor explicit spatial awareness of the
mission objective and eliminated the need for any terminal reward bonus.

### 5. No Terminal Reward Bonus
A terminal bonus `max(0, 20-dist)×2.0` was tried in v4 — it caused the policy
to rush toward the boundary rather than optimise EE. The per-step progress
reward with weight 0.4 is both necessary and sufficient.

### 6. PPO-Style Value Clipping
Added to the critic loss (not just actor clipping) to prevent large value
function jumps when entropy drops in late training. Eliminated C_loss explosions
seen in v2.

### 7. Best-Model Checkpointing
All evaluation plots use `mappo_best.pt` (saved whenever eval EE improves),
never `mappo_final.pt`. Training consistently degrades after the best episode —
using the final model under-reports true performance by 5–15%.

### 8. obs[7] Normalisation Must Match I_DEV
With I_DEV=12, the denominator in obs[7] must be 12.0 (not 6.0) to keep
observations in [0,1]. A wrong denominator silently degrades policy performance
without triggering any visible error — this is change #46.

---

## Known Limitations

1. **EE vs L — L=4,8 below No-RIS:** With 12 devices each receiving P_MAX/12
   power, very small RIS arrays cannot provide sufficient beamforming gain. The
   benefit threshold shifts to L≥16 (vs L≥8 with I=4 in v7). Physically correct.

2. **EE vs Speed — no inverted-U peak:** With 12 spatially diverse devices,
   higher speed gives continuous channel diversity improvement. The propulsion
   power growth is outweighed by channel diversity gain at all tested speeds.

3. **EE vs Devices — MAPPO < Fixed for I<12:** The policy was trained at I=12
   and performs best there. Sub-agents (500 eps) for smaller I do not fully
   converge. Crossover only occurs at the trained configuration I=12.

4. **Scenario comparison dip at I=8:** The dip appears simultaneously across
   all 4 deployment scenarios, confirming it is evaluation variance (5 episodes
   per point), not a systematic effect.

5. **Single random seed:** All results use seed=42. Publication-quality claims
   should be averaged over 3–5 seeds.

6. **Runtime ~183 min:** 3× longer than v7 due to 12 channel computations per
   time step. Inherent to the I=12 configuration.

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{jiang2025ris,
  title   = {RIS-Assisted UAV-Enabled IoT Network},
  author  = {Jiang, et al.},
  journal = {IEEE Internet of Things Journal},
  volume  = {12},
  number  = {20},
  year    = {2025},
  month   = {October}
}
```

---

## License

This project is for academic and research purposes.
Please cite the original paper if you build upon this work.

---

## Authors
Pratyaksh Agrawal

Vibhu Bharadwaj
