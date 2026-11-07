# v9 Results — RIS-Assisted Multi-UAV IoT Data Collection with Federated MARL

All numbers come from `results_v9/` and are reproducible with the commands in
`v9/README.md`. Learned methods: mean ± std over **3 training seeds**, each tested on the
same **20 held-out test episodes** as the baselines (checkpoints chosen on separate
validation episodes). EE = delivered bits ÷ total UAV propulsion energy (bit/J).

## 1. Why the environment had to change (old scene vs new scene)

| | Old 20 m scene (v8 / Jiang et al.) | New 500 m urban scene (v9) |
|---|---|---|
| Greedy flight, no RIS → aligned RIS | 126,930 → 126,954 bit/J (**+0.0 %**) | 6,795 → 7,724 bit/J (**+13.7 %**) |
| Straight flight, no RIS → aligned RIS | 102,894 → 103,021 (**+0.1 %**) | 5,620 → 6,535 (**+16.3 %**) |

In the old scene every link is strong, so the RIS cannot matter; trajectories barely
matter either. Figure: `results_v9/gate/fig_old_vs_new.png`.

**Coverage (rule check).** A device inside the shadowing courtyard can upload a 20 Mbit
batch within its 30 s window from **3.4 %** of the area without the RIS and from **33 %**
with it (`results_v9/gate/fig_coverage.png`). Upload time from 200 m away: 194 s direct vs
27 s via the RIS (`fig_upload_time.png`). Across 14 randomly generated city layouts the
same check gives 2–5 % vs 29–39 %.

## 2. Learning beats hand-made flight (single area, building RIS)

| Method | Test EE (bit/J) | Delivered | Mission completed | Training traffic |
|---|---|---|---|---|
| Straight line + RIS | 6,535 | 56 % | 100 % | — |
| Greedy heuristic + RIS | 7,724 | 72 % | 100 % | — |
| **IPPO** (each UAV learns alone) | **8,142 ± 93** | 74 % | 100 % | 0 MB |
| **MAPPO-CTDE** (centralized critic) | **8,049 ± 36** | 73 % | 100 % | 201 MB |
| FedAvg, E = 16 | 7,973 ± 154 | 74 % | 100 % | 27 MB |
| FedAvg, E = 4 | 7,848 ± 124 | 72 % | 100 % | 109 MB |
| FedAvg, E = 1 | 7,818 ± 256 | 72 % | 100 % | 439 MB |
| FedProx, E = 4, μ = 0.01 | 7,784 ± 187 | 71 % | 100 % | 109 MB |
| FedPer, E = 4 (personal heads) | 7,743 ± 226 | 71 % | 100 % | 108 MB |

* Every learned method beats the best heuristic, finishes every mission, ~0 collisions.
* Centralized training learns fastest (clear lead from episode 500 to ~2000).
* FedAvg with infrequent averaging (E = 16) reaches **99 % of CTDE's EE with 13 % of its
  communication**; averaging after every update (E = 1) costs 16× more traffic for nothing.
* Honest finding: **in a single area, independent learning (IPPO) is as good as anything** at
  zero communication — every UAV has plenty of its own data. FedProx and FedPer do not beat
  plain FedAvg here. This motivated the multi-area study (§3).

Figures: `results_v9/train/building/fig_learning_curves.png`, `fig_ee_vs_comm.png`.

## 3. Federated learning across city areas (the FL contribution)

Setting: 4 training areas with different building layouts and RIS positions (one of each
layout type), each with its own 3-UAV swarm trained by MAPPO at its own ground station.
Judged on **8 never-seen areas** (`results_v9/multiarea/fig_areas.png`).

| Method | EE on unseen areas | Cross-area traffic | Raw data leaves area? |
|---|---|---|---|
| Greedy heuristic + RIS | 7,245 | — | — |
| Local only (each area alone) | 7,301 ± 15 | 0 MB | no |
| **FedAvg across areas, actor only (E = 4)** | **7,542 ± 111** | **61 MB** | **no** |
| FedAvg across areas, actor + critic (E = 4) | 7,596 ± 182 | 340 MB | no |
| Pooled raw data in the cloud (upper bound) | 7,538 ± 30 | 790 MB | yes |

* Local models **overfit their own area**: 7,865 on their own area on average, but only
  7,301 on new areas (one area's model drops from 8,203 to 7,065, −14 %).
* Federation recovers the generalization of pooling all raw data — **+3.3 % over local,
  equal to pooled, with 13× less traffic and no raw data leaving any area**.
* Averaging the critic too adds a little EE (+0.7 %) for 5.6× more traffic; the actor-only
  design (critics stay at each ground station) is the better trade.

Figures: `results_v9/multiarea/fig_generalization.png`, `fig_multiarea_curves.png`.

## 4. Where should the RIS be? (placement study)

| Setup | Greedy heuristic | Learned MAPPO | Flight energy |
|---|---|---|---|
| No RIS | 6,795 | 7,441 ± 221 | ~42 kJ |
| **Building RIS** (1 m², rooftop) | **7,724** | **8,049 ± 36** | ~43 kJ |
| Aerial RIS (carried by a 4th UAV at 70 m) | 5,368 | 5,354 ± 170 | ~60 kJ |

* The building RIS adds **+8.2 % EE even under learned (optimized) flight**, and +13.7 % for
  the heuristic. Learning helps most without a RIS (+9.5 %): it partly compensates by
  detouring over the shadowed courtyard.
* The aerial RIS delivers about as much data as the building RIS but its carrier adds
  **+40 % flight energy**, so EE falls below even no RIS. A UAV-carried RIS only pays off
  where no building can host a panel or the shadowed zones move. Static analysis agrees:
  even hovering overhead, the aerial RIS's cascaded link is weaker than the building
  RIS's at every distance.

Figure: `results_v9/fig_ris_value.png`.

## 5. Sensitivity (trained policies re-used, no retraining)

Because the RIS phases are computed in closed form and the observation has a fixed size,
the trained policies (MAPPO-CTDE and FedAvg E = 16, 3 seeds each) are evaluated at every
setting **without retraining** (v8 needed 11 separately trained sub-agents for its sweeps).
Figure: `results_v9/sweeps/building/fig_sweeps.png`.

| Parameter | Finding (learned MAPPO EE unless noted) |
|---|---|
| RIS size L = 64 / 144 / 256 / 400 | 7,681 / 7,900 / 8,049 / 8,285 — steady gain with diminishing returns |
| Phase bits 1 / 2 / 3 / continuous | 8,002 / 8,049 / 8,141 / 8,120 — **2-bit is within ~1 % of continuous** |
| Phase error 0 / 10 / 20 / 40° (std) | 8,049 / 8,097 / 8,101 / 7,971 — **robust to imperfect CSI** (−1 % at 40°) |
| Device power 0 → 20 dBm | 5,261 → 10,431; learned stays 3–6 % above greedy at every power |
| Data per device ×0.5 / ×1 / ×1.5 / ×2 | learned − greedy gap grows with scarcity: +1.8 % / +4.2 % / +6.1 % / +3.9 % |
| RIS mode: time-share vs common | 8,049 vs 8,110 — equal at this load (one device per UAV per slot) |

The "no RIS" and "random phases" entries of that last panel use policies trained *with* the
RIS (out of distribution); the properly trained no-RIS policy is in §4 (7,441).

## 6. What was fixed relative to v8

Phase-alignment sign error (the old "optimal RIS" baseline was worse than random); EE
computed on the wrong device set in sweeps; one physical RIS simulated as three private
copies; frozen fading; triple-counted throughput; PPO log-prob averaging and clamping
mismatch; collisions counted after the safety push; result-rewriting monotonicity code.
Details: `docs/analysis/01-technical-audit.md`.

## 7. Limitations (state these in the report)

1. Perfect channel knowledge at the RIS controller for the closed-form phases (as in Jiang
   et al.); quantified by the phase-error sweep.
2. Buildings are boxes with a fixed +20 dB blockage loss (no diffraction/reflection).
3. Fixed altitude (2-D flight); Shannon rates; ideal synchronization.
4. Only propulsion energy counted; the aerial carrier is charged ×1.2 for the panel.
5. Data volume, bandwidth and one receiver per UAV are design choices within realistic
   ranges (NB-IoT 180 kHz, 20 dBm-class devices at 10 dBm, 1.25–3.75 MB per device).
6. A return-to-home autopilot guarantees mission completion; it overrides the policy in
   about a quarter of the slots near the end of the mission and is reported, not hidden.
7. FL is simulated on one machine — it fixes what is exchanged, not real hardware.
8. 3 training seeds per method (BTP scope).
