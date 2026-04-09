# ris_uav_mappo_v7.py — Upgrade Walkthrough

## Summary

All 13 upgrade items from the v6→v7 specification were applied as targeted, additive modifications. No training logic, reward function, observation vector, or network architecture was changed.

## Changes Applied

| # | File | Change | Status |
|---|------|--------|--------|
| 1 | All paths | `mappo_results_v6` → `mappo_results_v7` (14 occurrences) | ✅ |
| 2 | Section 2 | `RHO0 = 1e-2` → `RHO0 = 5e-2` | ✅ |
| 3 | Section 3 | Added `generate_iot_positions()` (uniform/clustered/edge) | ✅ |
| 4 | `plot_learning_curve()` | MA window 20→50, offsets 19→49, labels updated | ✅ |
| 5 | `plot_trajectories()` | Bigger RIS/start/end markers, goal marker, denser arrows (//7) | ✅ |
| 6 | `plot_ee_vs_L()` | L values → [4,8,16,32,64,128], log-2 x-axis | ✅ |
| 7 | `plot_ee_vs_pmax()` | % gain annotations, fixed tick labels, NullLocator | ✅ |
| 8 | `plot_ee_vs_devices()` | Integer ticks, twin y-axis for % gap vs Fixed | ✅ |
| 9 | `train_agent_for_devices()` | Default 300→500 episodes | ✅ |
| 10 | New function | `plot_ee_per_uav()` — box plot + bar chart + Jain's fairness | ✅ |
| 11 | New function | `plot_scenario_comparison()` — 3 IoT deployment scenarios | ✅ |
| 12 | `plot_all()` | gridspec 3×2→4×2, figsize 20×13→20×18, panels (g)+(h) | ✅ |
| 13 | `main()` | Added calls to new plots, updated output print | ✅ |

## Preserved (Unchanged)

- `Actor`, `Critic`, `RolloutBuffer`, `MAPPOAgent` class definitions
- `train_mappo()` training loop, all hyperparameters (N=1500, buffer=200, epochs=4, clip=0.2, entropy floor=0.005)
- Reward function: `mean_ee*100 + progress*0.4 - collision*5.0`
- Observation vector: dim=18, network architectures (obs=18, state=54, act=52)
- `eval_agent()`, all benchmark functions, early stopping, cosine LR, value clipping
- `_mini_train_loop()`, `_straight_line_positions()`, `_get_device_positions()`

## Validation

- **Syntax check:** `py_compile.compile()` → passed
- **20/20 automated content checks:** all passed (output dir, RHO0, L values, MA window, new functions, gridspec, reward weights, obs dim, etc.)

## How to Run

```bash
python ris_uav_mappo_v7.py
```


## Expected Output Files in `./mappo_results_v7/`

```
mappo_best.pt, mappo_final.pt, mappo_ep250.pt ... mappo_ep1500.pt
mappo_learning_curve.png       ← 50-ep MA
mappo_ee_vs_L.png              ← powers-of-2, log-2 axis
mappo_ee_vs_speed.png          ← unchanged logic
mappo_ee_vs_pmax.png           ← % gain annotations
mappo_ee_vs_devices.png        ← integer ticks, twin y-axis
mappo_trajectory.png           ← enhanced markers + goal
mappo_summary.png              ← 8-panel (a)-(h)
mappo_ee_per_uav.png           ← NEW: fairness analysis
mappo_scenario_comparison.png  ← NEW: 3 deployment scenarios
```
