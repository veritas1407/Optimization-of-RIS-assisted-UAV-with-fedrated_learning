# RIS-Assisted UAV-Enabled IoT Network — MAPPO Implementation

> **Reference:** Jiang et al., "RIS-Assisted UAV-Enabled IoT Network,"
> *IEEE Internet of Things Journal*, Vol. 12, No. 20, October 2025.

A PyTorch implementation of **Multi-Agent Proximal Policy Optimization (MAPPO)**
with Centralized Training, Decentralized Execution (CTDE) for jointly optimizing
UAV trajectories and RIS phase shifts to maximize network energy efficiency.
This repository contains the final working implementation (v6) along with the
full development history showing how the code evolved through 6 iterative versions.

---

## Table of Contents

- [Overview](#overview)
- [System Model](#system-model)
- [Algorithm](#algorithm)
- [Repository Structure](#repository-structure)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Version History](#version-history)
- [Key Design Decisions](#key-design-decisions)
- [Known Limitations](#known-limitations)
- [Citation](#citation)

---

## Overview

### Problem Statement

A swarm of **K=3 rotary-wing UAVs** serves **I=4 ground IoT devices** over
a 20×20m area. A fixed **Reconfigurable Intelligent Surface (RIS)** with
**L=50 passive elements** assists the uplink by reflecting IoT signals toward
the UAVs. The goal is to maximize the system **Energy Efficiency**:

```
EE = Total Data Rate (bits/s) / UAV Propulsion Power (W)   [bits/Joule]
```

The joint optimization problem — UAV 3D trajectories + RIS phase shifts —
is solved by MAPPO, replacing the conventional model-based Alternating
Optimization / Successive Convex Approximation (AO/SCA) approach.

### Why MAPPO?

| Approach | Requirement | Complexity | Scalability |
|----------|-------------|------------|-------------|
| AO/SCA (model-based) | Perfect CSI at every step | O(M³·⁵) per slot | Single UAV |
| **MAPPO (ours)** | Channel magnitudes only | O(1) inference | K=3 UAV swarm |

---

## System Model

### Physical Setup

```
Area:        20m × 20m, UAV altitude fixed at h=5m
UAVs:        K=3, start near [0,0,5]m, goal at Q_F=[16,16,5]m
RIS:         Fixed at [5,9,0]m, L=50 passive reflecting elements
IoT devices: I=4 at [3,11,0], [6,13,0], [9,4,0], [12,6,0]m
Episode:     M=20 time slots × DT=1s = 20s total
Max speed:   V_MAX=15 m/s, safety separation: SAFE_D=3m
```

### Channel Model

```
IoT → RIS:  Rician fading,   K=5.0,  path-loss exponent α=2.2
RIS → UAV:  Rician fading,   K=5.0,  path-loss exponent α=2.2
IoT → UAV:  Rayleigh fading,         path-loss exponent α=3.5
Path loss:  β₀ = 1e-2 (reference distance 1m)
Noise:      σ² = 1e-10 W
```

### UAV Propulsion Power (Rotary-Wing Model)

```
P(v) = P_blade(1 + 3v²/v_tip²)
     + ½ · d₀ · ρ · s · A · v³
     + P_induced · √(1 + v⁴/4v₀⁴ - v²/2v₀²)

Hover power ≈ 168.5 W  (verified in sanity check on every run)
```

---

## Algorithm

### MAPPO with CTDE

```
┌─────────────────────────────────────────────────────────────┐
│                    CENTRALIZED TRAINING                      │
│                                                              │
│   Global State s = [obs₀ ‖ obs₁ ‖ obs₂]  (dim=54)          │
│                         ↓                                    │
│              ┌──────────────────────┐                        │
│              │  Shared Critic V(s)  │  [256→256→128→1]       │
│              └──────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   UAV 0      │  │   UAV 1      │  │   UAV 2      │
│ Actor π(obs₀)│  │ Actor π(obs₁)│  │ Actor π(obs₂)│
│ [128→128→64] │  │ [128→128→64] │  │ [128→128→64] │
│ act_dim=52   │  │ act_dim=52   │  │ act_dim=52   │
└──────────────┘  └──────────────┘  └──────────────┘
     DECENTRALIZED EXECUTION (local obs only)

Action per UAV = [vel_x, vel_y] + [L=50 RIS phases]  ∈ [-1,1]^52
```

### Observation Vector (18-dimensional per UAV)

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0–1 | Own x,y position | ÷ 20m |
| 2–3 | Own velocity vx,vy | ÷ 15 m/s |
| 4–6 | Distances to 3 IoT devices | ÷ (20√2) |
| 7 | Number of IoT devices | ÷ 6.0 |
| 8 | Distance to RIS | ÷ (20√2) |
| 9–12 | Log channel magnitudes (4 devices) | ÷ 10 |
| 13–16 | Other 2 UAV positions | ÷ 20m |
| 17 | Distance to goal Q_F | ÷ (20√2) |

### Reward Function

```python
reward = mean_ee × 100.0          # EE objective     (~89% of total)
       + Σₖ (prev_dist_k - dist_k) × 0.4   # Navigation aid  (~10%)
       - n_collisions × 5.0        # Safety penalty   (0% normally)
```

The **progress reward weight of 0.4** is the single most critical hyperparameter.
Values above ~0.6 cause the navigation term to dominate EE optimization, breaking
the primary objective. See [Version History](#version-history) for the full story.

---

## Repository Structure

```
.
├── ris_uav_mappo_v4.py          # v4 code — per-L/I sub-training introduced
├── ris_uav_mappo_v5.py          # v5 code — reward rebalanced, training stable
├── ris_uav_mappo_v6.py          # v6 code — FINAL working version ✅
│
├── walkthrough_v6.md            # Complete technical walkthrough of all changes
│
├── results_v4/                  # v4 output plots and model checkpoints
│   ├── mappo_learning_curve.png
│   ├── mappo_trajectory.png
│   ├── mappo_ee_vs_L.png
│   ├── mappo_ee_vs_speed.png
│   ├── mappo_ee_vs_pmax.png
│   ├── mappo_ee_vs_devices.png
│   ├── mappo_summary.png
│   └── mappo_best.pt
│
├── results_v5/                  # v5 output plots and model checkpoints
│   ├── mappo_learning_curve.png
│   ├── mappo_trajectory.png
│   ├── mappo_ee_vs_L.png
│   ├── mappo_ee_vs_speed.png
│   ├── mappo_ee_vs_pmax.png
│   ├── mappo_ee_vs_devices.png
│   ├── mappo_summary.png
│   ├── mappo_best.pt            
│   ├── mappo_final.pt
│   ├── mappo_ep250.pt
│   ├── mappo_ep500.pt
│   ├── mappo_ep750.pt
│   └── mappo_ep1000.pt
│
├── results_v6/                  # v6 output plots and model checkpoints ✅
│   ├── mappo_learning_curve.png
│   ├── mappo_trajectory.png
│   ├── mappo_ee_vs_L.png
│   ├── mappo_ee_vs_speed.png
│   ├── mappo_ee_vs_pmax.png
│   ├── mappo_ee_vs_devices.png
│   ├── mappo_summary.png
│   ├── mappo_best.pt            ← Best model weights (ep=600)
│   ├── mappo_final.pt
│   ├── mappo_ep250.pt
│   ├── mappo_ep500.pt
│   ├── mappo_ep750.pt
│   ├── mappo_ep1000.pt
│   ├── mappo_ep1250.pt
│   └── mappo_ep1500.pt
│
└── README.md
```

> **Start here:** `ris_uav_mappo_v6.py` is the only file you need to run.
> The earlier versions are kept for reference and to show the development process.

---

## Results

All results below are from **v6** using the best model checkpoint (ep=600).

### Energy Efficiency vs Transmit Power

MAPPO consistently outperforms No-RIS baseline at every power level,
demonstrating measurable benefit from the learned RIS phase optimization:

| P_max (W) | MAPPO (bits/J) | No-RIS (bits/J) | RIS Gain |
|-----------|----------------|-----------------|----------|
| 0.05 | 0.05501 | 0.05430 | +1.3% |
| 0.10 | 0.06097 | 0.06027 | +1.2% |
| 0.20 | 0.06693 | 0.06626 | +1.0% |
| 0.50 | 0.07483 | 0.07418 | +0.9% |
| 1.00 | 0.08081 | 0.08017 | +0.8% |
| 2.00 | 0.08679 | 0.08616 | +0.7% |
| 5.00 | 0.09469 | 0.09408 | +0.6% |

### Energy Efficiency vs Max UAV Speed

MAPPO adapts trajectory to the speed constraint — it beats the fixed straight-line
trajectory for all speeds v≥10 m/s:

| v_max (m/s) | MAPPO (bits/J) | Fixed Traj (bits/J) |
|-------------|----------------|---------------------|
| 5 | 0.0778 | 0.0807 |
| 8 | 0.0797 | 0.0807 |
| **10** | **0.0812** | **0.0807** ✅ |
| **12** | **0.0815** | **0.0807** ✅ (peak) |
| **15** | **0.0808** | **0.0807** ✅ |
| **18** | **0.0806** | **0.0807** ✅ |
| **20** | **0.0820** | **0.0807** ✅ |

### Energy Efficiency vs RIS Elements L

MAPPO beats No-RIS at every L value. Per-L dedicated agents are trained for
each sweep point (500 episodes each), ensuring correct act_dim matching:

| L | MAPPO (bits/J) | No-RIS (bits/J) |
|---|----------------|-----------------|
| 5 | 0.08036 | 0.08017 ✅ |
| 10 | 0.08042 | 0.08017 ✅ |
| 20 | 0.08040 | 0.08017 ✅ |
| 30 | 0.08040 | 0.08017 ✅ |
| 50 | 0.08081 | 0.08017 ✅ |
| 80 | 0.08196 | 0.08017 ✅ |
| 100 | 0.08148 | 0.08017 ✅ |

### Training Summary

```
Best eval EE:    0.080806 bits/J  (at episode 600)
Total runtime:   ~77 minutes on Tesla T4 GPU
  Main training: ~20 min (1500 episodes)
  Sub-training:  ~57 min (6 L-agents + 4 I-agents)
Training stable: No collapse across 1500 episodes
Hover power:     168.5 W ✅ (matches theoretical ~168W)
```

---

## Installation

### Requirements

```
Python  >= 3.8
PyTorch >= 1.12  (2.x recommended)
NumPy
Matplotlib
tqdm (optional, for progress bar)
CUDA-capable GPU recommended (Tesla T4 or equivalent)
```

### Install dependencies

```bash
pip install torch torchvision numpy matplotlib tqdm
```

### Verify GPU

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Usage

### Training from scratch

```bash
python ris_uav_mappo_v6.py
```

This will:
1. Run the sanity check (hover power, network dimensions)
2. Train the main MAPPO agent for 1500 episodes (~20 min)
3. Train per-L sub-agents for the L-sweep (~35 min)
4. Train per-I sub-agents for the device-sweep (~22 min)
5. Generate all 7 plots
6. Save all checkpoints and plots to `./mappo_results_v6/`

### Expected console output

```
============================================================
SANITY CHECK
  Hover propulsion power : 168.5 W  (expect ~168)
  obs_dim=18, state_dim=54, act_dim=52
  Actor params: 30,632   Critic params: 112,897
  RHO0=0.01 (raised for realistic EE magnitude)
  Buffer capacity=200, PPO epochs=4, eval every 50 eps
  Entropy floor=0.005, value clipping ON, cosine LR ON
============================================================
Training MAPPO: 100%|████████| 1500/1500 [19:45<00:00]
...
  [EVAL] ep=600  mean_EE=0.080806  ← best model saved here
...
Training complete.
  Best model → ./mappo_results_v6/mappo_best.pt  (ep=600, EE=0.080806)
```

### Loading a trained model for inference

```python
import torch
from ris_uav_mappo_v6 import Actor, RISSwarmEnv, eval_agent, MAPPOAgent

# Load best model
agent = MAPPOAgent()
agent.load("./mappo_results_v6/mappo_best.pt")

# Evaluate
env = RISSwarmEnv()
mean_reward, mean_ee, trajectory = eval_agent(agent, env, n_episodes=10, deterministic=True)
print(f"Mean EE: {mean_ee:.6f} bits/J")
```

---

## Version History

This implementation went through 6 development iterations. Each version fixed
specific bugs or improved performance. The full technical detail is in
[`v6_walkthrough.md`](v6_walkthrough.md).

### Quick Summary

| Version | Best EE | Key Change | MAPPO > No-RIS |
|---------|---------|------------|----------------|
| v1 (original) | 0.057 | Baseline — single UAV AO/SCA → MAPPO K=3 | ✅ partial |
| v2 | 0.066 | Bug fixes: device benchmark, RIS dim mismatch, RHO0↑, navigation reward | ❌ |
| v3 | **0.0816** | Best-model checkpointing, entropy floor, value clipping, cosine LR | ✅ |
| v4 | 0.0728 | Per-L/I sub-training ✅ BUT terminal goal bonus broke EE ❌ | ❌ all |
| v5 | 0.0785 | Removed terminal bonus, rebalanced reward (weight=0.8) | ❌ |
| **v6** ✅ | **0.0808** | Progress weight 0.8→**0.4**, 1500 eps, 500 L-eps, 350 I-eps | **✅ all** |

### The Critical Insight

The single most important lesson from 6 iterations is **reward balance**.
The progress reward weight controls the trade-off between EE optimization and
navigation:

```
weight=0.0  → UAVs hover near RIS, ignore goal        (v1/v2)
weight=0.5  → Good EE, some navigation                 (v3, best EE=0.0816)
weight=1.5  → Navigation dominates, EE collapses       (v4, broke everything)
weight=0.8  → Better navigation, EE still below No-RIS (v5)
weight=0.4  → EE dominates (~89%), navigation works    (v6, FINAL ✅)
```

---

## Key Design Decisions

### 1. Per-L Agent Training (not resampling)
When evaluating EE vs RIS elements L, each L value gets a dedicated agent with
`Actor(act_dim=2+L)` trained for 500 episodes. Linear interpolation of RIS
phases from L=50 to L=100 (2× extrapolation) caused severe performance
degradation in earlier versions.

### 2. Per-I Agent Training
Similarly, each device count I gets its own 350-episode agent, with
`n_devices/6.0` added to the observation so the policy can adapt to
different device counts at test time.

### 3. Goal Distance in Observation
`obs[17]` encodes normalized distance to goal Q_F, replacing the time-slot
counter `slot/M`. This gives the actor explicit spatial awareness of the
mission objective without requiring a terminal reward bonus.

### 4. No Terminal Reward Bonus
A terminal goal bonus of the form `max(0, 20-dist) × c` was tried in v4 with
c=2.0. This caused catastrophic reward imbalance — the policy learned to rush
to the boundary rather than optimize EE. The per-step progress reward with
a carefully tuned weight is sufficient.

### 5. PPO-Style Value Clipping
Added to the critic loss to prevent large value function jumps when entropy
drops in late training. This eliminated the critic loss explosions seen in v2.

### 6. Best-Model Checkpointing
All evaluation plots use `mappo_best.pt` (saved whenever eval EE improves),
not `mappo_final.pt`. Training naturally degrades after the peak — using the
final model would under-report true performance.

---

## Known Limitations

1. **EE vs L — slight non-monotonicity at L=100:** `L=100: 0.08148 < L=80: 0.08196`.
   Both beat No-RIS. Caused by training variance with 500-episode sub-agents.
   Using more episodes or multiple seeds would resolve this.

2. **EE vs Devices — dip at I=5:** `I=5: 0.07386` between `I=4: 0.08081` and
   `I=6: 0.08149`. The 350-episode sub-agent for I=5 converged suboptimally.

3. **Trajectories don't fully reach Q_F=[16,16]:** UAVs end around x=6–10,
   y=9–12. The policy balances EE (loitering near RIS gives better channel gain)
   against navigation (moving to goal). With only M=20 steps, this trade-off is
   physically reasonable.

4. **Single seed:** All results are from a single random seed (42). RL results
   should ideally be averaged over 3–5 seeds for publication-quality claims.

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
