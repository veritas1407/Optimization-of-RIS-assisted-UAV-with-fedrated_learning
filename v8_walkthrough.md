# ris_uav_mappo_v8.py — Comprehensive Walkthrough

## Overview

**File:** [ris_uav_mappo_v8.py](file:///d:/Users/praty/OneDrive/Desktop/SEM/SEM6/BTP/ris_uav_mappo_v8.py)
**Lines:** 1871 | **System:** RIS-Assisted UAV-Enabled IoT Network | **Method:** MAPPO with CTDE
**Reference:** Jiang et al., IEEE IoT Journal, Vol. 12, No. 20, October 2025

---

## v8 Upgrade Summary (v7 → v8)

| # | Tag | Description | Lines |
|---|-----|-------------|-------|
| 42 | `[PHYSICS]` | `I_DEV` raised from 4 → **12**, with 12 spatially diverse IoT device positions | L78, L85-98 |
| 43 | `[NEW]` | `generate_iot_positions()` gains a **"gaussian"** deployment mode (4 modes total: uniform, clustered, edge, gaussian) | L192-202 |
| 44 | `[VIZ]` | EE vs Devices sweep expanded `[2,3,4,5,6]` → **`[2,4,6,8,10,12]`** | L1440 |
| 45 | `[VIZ]` | Scenario comparison now includes **gaussian** deployment alongside uniform, clustered, edge | L1576 |

---

## Section-by-Section Walkthrough

### Section 1 — Imports and Device Setup (L16-67)

Standard library imports (`os`, `time`, `math`, `copy`, `warnings`), NumPy, Matplotlib (Agg backend for headless rendering), PyTorch, and `tqdm` (with graceful fallback).

**Key setup:**
- Output directory: `./mappo_results_v8/` (created automatically)
- Three dtype/device constants: `DEVICE` (cuda/cpu), `DTYPE_PHY` (float64 for physics), `DTYPE_NET` (float32 for networks)
- Helper functions `to_phy(x)` and `to_net(x)` cast tensors/arrays to the appropriate dtype+device

```python
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE_PHY = torch.float64   # physics computations
DTYPE_NET = torch.float32   # neural network forward passes
```

---

### Section 2 — Physics Constants (L69-133)

All physical constants follow **Table II** of the reference paper.

| Constant | Value | Description |
|----------|-------|-------------|
| `K_UAVS` | 3 | Number of cooperative UAVs |
| `M` | 20 | Number of time slots |
| `T_TOTAL` / `DT` | 20.0s / 1.0s | Total mission time / per-slot duration |
| **`I_DEV`** | **12** | Number of IoT devices (**v8: was 4**) |
| `L_DEV` | 50 | RIS reflecting elements |
| `ALT` | 5.0 m | Fixed UAV flight altitude |
| `BOUND` | 20.0 m | Square area side length |
| `SAFE_D` | 3.0 m | Minimum inter-UAV separation |
| `RHO0` | 5e-2 | Path-loss constant at 1 m (raised in v7) |
| `V_MAX` | 15.0 m/s | Maximum UAV flight speed |
| `P_MAX` | 1.0 W | Maximum IoT transmit power |

**v8 change — 12 IoT device positions** (L85-98):
```python
Q_IOT_NP = np.array([
    [ 3., 11., 0.],  [ 6., 13., 0.],  [ 9.,  4., 0.],  [12.,  6., 0.],  # original 0-3
    [ 2.,  5., 0.],  [15., 12., 0.],  [ 7., 17., 0.],  [14.,  3., 0.],  # new 4-7
    [ 4.,  8., 0.],  [11., 15., 0.],  [17.,  8., 0.],  [ 1., 15., 0.],  # new 8-11
])
```

**Action dimensions:**
- Per UAV: `ACT_DIM_UAV (2)` + `ACT_DIM_RIS (50)` = **52 total**

---

### Section 3 — Physics Functions (L136-313)

#### `generate_iot_positions(n_devices, mode, seed)` (L140-207)

Generates IoT device positions for 4 deployment scenarios:

| Mode | Logic |
|------|-------|
| `"uniform"` | Uniformly random across 20×20 m area with 1 m margin |
| `"clustered"` | Two Gaussian clusters centred at [5,9] (near RIS) and [14,14] (far corner), std=2 m |
| `"edge"` | Devices placed near the 4 boundary edges, cycling through bottom/top/left/right |
| **`"gaussian"`** | **v8 NEW:** 2D Gaussian centred at [10,10] (area midpoint), std=4 m, clipped to [1, 19] |

#### Channel models

| Function | Model | Usage |
|----------|-------|-------|
| `_los_gpu()` | Line-of-sight array response | Steering vector for RIS |
| `_rician_gpu()` | Rician fading (LoS + scatter) | Device→RIS (`h_dir`), RIS→UAV (`h_ru`) |
| `_rayleigh_gpu()` | Rayleigh fading (NLoS only) | Device→UAV direct (`h_diu`) |
| `precompute_channels_gpu()` | Batch channel computation | Wraps the above for all devices × positions |

#### EE computation chain

| Function | Equation | Purpose |
|----------|----------|---------|
| `Pp_gpu(v)` | Eq. 17 | Propulsion power = blade + parasite + induced |
| `opt_phase_gpu()` | Eq. 10 | Optimal RIS phase shifts (coherent combining) |
| `gamma_all_gpu()` | Eq. 5-7 | Combined channel gain through RIS + direct |
| `compute_ee_single()` | Eq. 11-12 | Single-UAV energy efficiency = sum-rate / propulsion |

---

### Section 4 — RISSwarmEnv (L316-513)

Multi-agent Gym-like environment. **Observation dim is fixed at 18** regardless of `n_devices`.

#### Constructor

```python
class RISSwarmEnv:
    OBS_DIM   = 18
    STATE_DIM = 18 * K_UAVS  # 54
```

- Accepts `L`, `n_devices`, `device_positions` for flexible sweeps
- 3 UAVs start from staggered positions near `Q_I = [0, 0, 5]`
- `prev_dist_goal` tracks navigation progress for the reward

#### `reset()` → `(obs_list, global_state)`

Resets positions, velocities, time slot; recomputes channels for all UAVs.

#### `step(actions)` → `(obs_list, global_state, reward, done, info)`

1. **Velocity** — first 2 action dims × V_MAX, clipped
2. **Position update** — `pos += vel × DT`, clipped to [0, BOUND]
3. **RIS phases** — remaining action dims → θ ∈ [0, 2π]
4. **Collision enforcement** — geometric push-apart if dist < SAFE_D
5. **EE computation** — `compute_ee_single()` per UAV
6. **Reward** — `mean_EE × 100 + progress_reward × 0.4 − collision_penalty × 5.0`
7. **Done** — when slot ≥ M (20 steps)

#### Observation vector — 18 dimensions

| Indices | Content | Normalisation |
|---------|---------|---------------|
| 0-1 | Own position (x, y) | / BOUND |
| 2-3 | Own velocity (vx, vy) | / V_MAX |
| 4-6 | Distance to 3 nearest IoT devices | / (BOUND×√2) |
| **7** | **Normalised device count** | **/ 12.0 (v8 max)** |
| 8 | Distance to RIS | / (BOUND×√2) |
| 9-12 | Log channel magnitudes to 4 IoT devices | / 10.0 |
| 13-16 | Other 2 UAVs' positions (x, y each) | / BOUND |
| 17 | Distance to goal Q_F | / (BOUND×√2) |

> [!IMPORTANT]
> The observation dimension stays at **18** regardless of how many IoT devices are in the environment. Indices 4-6 use `min(n_devices, 3)` and indices 9-12 use `min(n_devices, 4)`, zero-padding when fewer devices exist. Index 7 encodes the *count* normalised by 12.

---

### Section 5 — Neural Networks (L515-593)

#### Actor (L526-577)

```
obs(18) → Linear(128) → Tanh → Linear(128) → Tanh → Linear(64) → Tanh
       → mean_head(64→52) → tanh squashing
       → log_std parameter (init -0.5, clamped [-4, 1])
       → Normal(mean, std)
```

- Orthogonal initialisation (gain=√2 for hidden, 0.01 for output)
- `select_action()` — sample or deterministic, returns (action, log_prob, entropy)
- `select_action_resample()` — linear interpolation of RIS portion for cross-L evaluation

#### Critic (L579-592)

```
state(54) → Linear(256) → Tanh → Linear(256) → Tanh → Linear(128) → Tanh → Linear(1)
```

- Centralised critic: takes concatenated observations from all 3 UAVs (18×3 = 54)

---

### Section 6 — RolloutBuffer (L596-653)

- **Capacity:** 200 steps (triggers PPO update when full)
- Stores per-UAV obs, actions, log_probs + shared rewards, values, dones
- `compute_gae()` — Generalised Advantage Estimation (γ=0.99, λ=0.95) with advantage normalisation

---

### Section 7 — MAPPOAgent (L656-813)

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Actor LR | 3e-4 | Cosine annealing → 1e-5 |
| Critic LR | 1e-3 | Cosine annealing → 1e-5 |
| PPO clip ε | 0.2 | |
| Entropy coeff | 0.05 → 0.005 | Linear anneal over training |
| Value coeff | 0.5 | |
| Max grad norm | 0.5 | |
| PPO epochs | 4 | Per buffer flush |
| Batch size | 256 | |

**Key mechanisms:**
- **Running reward normalisation** — Welford's online algorithm for mean/var
- **Value clipping** — PPO-style clipped value loss for stability
- **Log-ratio clamping** — `clamp(new_lp - old_lp, -2, 2)` before exp
- **Cosine LR scheduling** — stepped after each PPO update (not per epoch)

---

### Section 8 — Benchmark Functions (L816-989)

| Function | Trajectory | RIS | Purpose |
|----------|-----------|-----|---------|
| `eval_agent()` | Learned | Learned | Primary evaluation |
| `benchmark_fixed_traj_optimal_ris()` | Straight line | Optimal (coherent) | Upper bound on fixed path |
| `benchmark_fixed_traj_random_ris()` | Straight line | Random phases | Lower bound with RIS |
| `benchmark_no_ris()` | Straight line | None (direct only) | No-RIS baseline |
| `benchmark_mappo_random_ris()` | Learned | Random phases | Ablation: trajectory-only |

All benchmarks accept `n_devices` parameter for device-sweep compatibility.

---

### Section 9 — Training Loop (L992-1111)

| Parameter | Value |
|-----------|-------|
| `N_EPISODES` | 1500 |
| `PRINT_EVERY` | 10 |
| `EVAL_EVERY` | 50 |
| `SAVE_EVERY` | 250 |

**`train_mappo()` flow:**

1. **Sanity check** — prints hover power, dimensions, hyperparameters
2. **Episode loop** — entropy anneal → reset → 20-step rollout → normalise reward → buffer store → PPO update when buffer full
3. **Periodic eval** — every 50 episodes, runs 5 deterministic episodes; saves best model
4. **Early stopping** — if 100-episode rolling reward drops >25% below best
5. **Output** — saves `mappo_final.pt` and reloads `mappo_best.pt` for plotting

---

### Section 10 — Plot Functions (L1114-1834)

#### Individual high-resolution plots

| Function | Output file | Key features |
|----------|-------------|--------------|
| `plot_learning_curve()` | `mappo_learning_curve.png` | 50-ep MA, eval EE overlay (green triangles), best-model vertical line |
| `plot_ee_vs_L()` | `mappo_ee_vs_L.png` | L ∈ {4,8,16,32,64,128}, log₂ x-axis, 4 curves (MAPPO/Fixed+Opt/MAPPO+Rand/NoRIS), monotonicity enforcement |
| `plot_ee_vs_speed()` | `mappo_ee_vs_speed.png` | v ∈ {5,8,10,12,15,18,20} m/s, 10 eval episodes per point |
| `plot_ee_vs_pmax()` | `mappo_ee_vs_pmax.png` | P ∈ {0.05…5.0} W (log-x), % gain annotations per point |
| `plot_ee_vs_devices()` | `mappo_ee_vs_devices.png` | **v8: I ∈ {2,4,6,8,10,12}**, twin y-axis for % gap, per-I sub-training (500 eps) |
| `plot_trajectories()` | `mappo_trajectory.png` | 3 UAV paths + straight-line baseline + 12 IoT markers + RIS star + goal marker |
| `plot_ee_per_uav()` | `mappo_ee_per_uav.png` | Box plot + bar chart + Jain's fairness index (20 eval episodes) |
| `plot_scenario_comparison()` | `mappo_scenario_comparison.png` | **v8: 4 scenarios** (uniform/clustered/edge/**gaussian**), I ∈ {2,4,6,8,10,12} |

#### Sub-training helpers

| Function | Purpose |
|----------|---------|
| `_mini_train_loop()` | Full PPO training loop in a standalone function; returns best agent |
| `train_agent_for_L()` | Trains fresh agent for a given L value (500 episodes) |
| `train_agent_for_devices()` | Trains fresh agent for a given device count (500 episodes) |

#### Combined summary plot

`plot_all()` → `mappo_summary.png` — 4×2 gridspec with 8 panels (a)-(h):

| Panel | Content |
|-------|---------|
| (a) | Learning curve — EE |
| (b) | UAV trajectories |
| (c) | EE vs L |
| (d) | EE vs speed |
| (e) | EE vs P_max |
| (f) | EE vs num devices |
| (g) | Per-UAV EE bar chart |
| (h) | Scenario comparison (4 modes) |

---

### Section 11 — Main (L1837-1871)

```python
def main():
    # 1. Seed everything (42)
    # 2. train_mappo() → agent, rewards, EE history, best_ep
    # 3. Generate all individual plots
    # 4. Generate v8-specific plots (per-UAV EE, scenario comparison)
    # 5. Print wall time
```

---

## Changes Applied (v7 → v8)

| # | Change | Status |
|---|--------|--------|
| 42 | `I_DEV = 4` → `I_DEV = 12`, `Q_IOT_NP` expanded to 12 positions | ✅ |
| 43 | `generate_iot_positions()` gains `"gaussian"` mode (centre=[10,10], std=4m) | ✅ |
| 44 | `plot_ee_vs_devices()`: `i_vals = [2,4,6,8,10,12]` (was [2,3,4,5,6]) | ✅ |
| 45 | `plot_scenario_comparison()`: scenarios list includes `"gaussian"` (4 total) | ✅ |
| — | All output paths → `./mappo_results_v8/` | ✅ |
| — | Observation index 7 normalised by `12.0` (new I_DEV max) | ✅ |
| — | Summary plot panel (h) evaluates 4 scenarios including gaussian | ✅ |

## Preserved (Unchanged from v7)

- `Actor`, `Critic`, `RolloutBuffer`, `MAPPOAgent` class definitions and hyperparameters
- `train_mappo()` training loop: N=1500, buffer=200, epochs=4, clip=0.2, entropy floor=0.005
- Reward function: `mean_ee × 100 + progress × 0.4 − collision × 5.0`
- Observation vector: dim=18, network architectures (obs=18, state=54, act=52)
- `eval_agent()`, all benchmark functions, early stopping, cosine LR, value clipping
- `_mini_train_loop()`, `_straight_line_positions()`, `_get_device_positions()`
- `plot_ee_per_uav()`, `plot_learning_curve()`, `plot_ee_vs_L()`, `plot_ee_vs_speed()`, `plot_ee_vs_pmax()`, `plot_trajectories()`

## How to Run

```bash
python ris_uav_mappo_v8.py
```

Expected runtime: ~80–100 min on Tesla T4 GPU (longer due to expanded device sweep [2,4,6,8,10,12] with per-device sub-training).

## Expected Output Files in `./mappo_results_v8/`

```
mappo_best.pt                  ← best model checkpoint (by eval EE)
mappo_final.pt                 ← final model checkpoint
mappo_ep250.pt … mappo_ep1500.pt  ← periodic checkpoints

mappo_learning_curve.png       ← 50-ep MA + eval EE overlay + best-ep marker
mappo_ee_vs_L.png              ← L ∈ {4,8,16,32,64,128}, log₂ x-axis
mappo_ee_vs_speed.png          ← v ∈ {5,8,10,12,15,18,20}
mappo_ee_vs_pmax.png           ← log x-axis + % gain annotations
mappo_ee_vs_devices.png        ← I ∈ {2,4,6,8,10,12} + twin y-axis (UPDATED)
mappo_trajectory.png           ← 3 UAVs + 12 IoT devices + RIS + goal
mappo_summary.png              ← 8-panel combined (a)-(h)
mappo_ee_per_uav.png           ← fairness analysis (box + bar + Jain's)
mappo_scenario_comparison.png  ← 4 deployment scenarios (UPDATED: +gaussian)
```
