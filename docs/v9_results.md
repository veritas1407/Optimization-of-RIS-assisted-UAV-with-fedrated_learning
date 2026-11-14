# v9.1 Results: RIS-assisted multi-UAV IoT data collection with federated MARL

*Final results of the v9.1 campaign (2026-11-09). These supersede every v9.0 number (kept
only in `results_v9/archive_v9.0/`). For the ground-up explanation of the system see
[`docs/v9_explained.md`](v9_explained.md); for the audit that triggered v9.1 see
[`docs/analysis/06-v9-audit.md`](analysis/06-v9-audit.md).*

## The problem being solved

Maximise mission energy efficiency (bits ÷ flight energy) over UAV trajectories, device
scheduling, RIS phases and RIS time shares, under speed, acceleration, separation,
landing, receiver, duty-cycle, data and 2-bit-phase constraints — problem **(P1)**, written
out with its solution architecture in [`v9_explained.md`, Part 7½](v9_explained.md). Solver:
Dinkelbach outer loop; closed-form RIS phases; optimised or equal RIS time shares;
deadline-aware scheduling; MAPPO trajectory policy trained with federated learning across
city areas.

## How to read the numbers

- **EE** = bits delivered ÷ total UAV propulsion energy (bit/J), over a 100 s mission.
- **Test set:** 100 fresh test missions (seeds 5,000,000 + j), never used during development.
  Every policy, learned or heuristic, flies the *same* 100 missions.
- **Learned methods:** 5 training seeds each; checkpoint chosen on 20 validation missions;
  reported as mean ± **95 % confidence interval over seeds** (Student t, df = 4).
- **Heuristics:** tuned on validation missions only; reported as mean ± 95 % CI over the 100
  test missions.
- **"vs X"** = paired comparison on the same test missions: for each seed, the mean over the
  100 missions of (learned − heuristic), then a 95 % CI over seeds.
- **Violations** = separation violations (two UAVs closer than 20 m) per mission.
- Reproduce: `python -m v9.run --exp gate`, `python -m v9.launch campaign --episodes 4000`,
  `python -m v9.launch post`, then `python -m v9.plots …` (see `v9/README.md`).

---

## 1. The environment shows what the old one could not

| | Old 20 m scene (v8 / Jiang et al.) | New 500 m urban scene (v9) |
|---|---|---|
| Straight-line flight: no RIS → aligned RIS | 103,252 → 103,815 bit/J (**+0.5 %**) | 5,616 → 7,065 bit/J (**+25.8 %**) |
| Random flight: no RIS → aligned RIS | 119,161 → 119,873 (**+0.6 %**) | 5,101 → 6,238 (**+22.3 %**) |

In the old scene every link is already strong, so the RIS cannot matter.
Figure: `results_v9/gate/fig_old_vs_new.png`.

**Shadowing and the RIS (link budget).** A courtyard device can upload a 20 Mbit batch within
its 30 s window from **3.4 %** of the area without the RIS and from **90.0 %** with it
(`results_v9/gate/fig_coverage.png`). Time to upload 20 Mbit from a courtyard device:

| UAV distance | 50 m | 200 m | 400 m |
|---|---|---|---|
| no RIS | 55 s | 194 s | 513 s |
| with RIS | 11 s | 17 s | 23 s |
| open-field device, no RIS (reference) | 7.5 s | 16 s | 31 s |

Hand check (`docs/v9_explained.md` §5.4): device ~25–50 m from the 1 m² panel, UAV 300 m away
→ cascaded loss ≈ −105 dB → SNR ≈ 20 dB → ≈ 1.2 Mbit/s. The RIS link to the UAV is mostly
line-of-sight because the panel is elevated; a ground device 400 m from the UAV is mostly
non-line-of-sight in the urban model. That is why, at long range, the RIS path even beats an
open-field direct link.

**All 16 generated city layouts** (`results_v9/gate/area_feasibility.csv`): courtyard-feasible
area 2–5 % → 75–91 % with the RIS; RIS EE gain +23 to +33 % for the straight line and +8 to
+19 % for the tuned greedy.

---

## 2. Heuristic references (building scene, 100 test missions)

| Policy | No RIS | Random RIS phases | Aligned RIS (time-share) | Aligned RIS (common) | RIS gain (paired) |
|---|---|---|---|---|---|
| random | 5,101 ± 140 | 5,138 | 6,238 ± 140 | 6,243 | +22.3 % |
| straight line | 5,616 ± 129 | 5,645 | 7,065 ± 126 | 7,061 | **+25.8 %** (CI ±68 bit/J) |
| greedy, untuned (v9.0 rule) | 6,688 ± 174 | 6,760 | 7,694 ± 179 | 7,711 | |
| **greedy, decentralised** (same information as the learned UAVs) | 7,208 ± 194 | 6,875 | **8,199 ± 160** | 8,234 | **+13.8 %** (±175) |
| **greedy, centralised** (full information, joint assignment) | 7,564 ± 143 | 7,701 | **8,675 ± 162** | 8,689 | **+14.7 %** (±126) |

- Each configuration uses the heuristic tuned for it on validation missions (no-RIS runs use
  the no-RIS tuning). All policies reach the final zone in 100 % of missions; the tuned
  heuristics have ≤ 0.15 violations per mission (untuned greedy 0.5–0.7, random 3.2).
- Heuristic tuning has its own noise: the centralised greedy tuned on the RIS scene scores
  7,698 when flown without the RIS, more than the one tuned for the no-RIS scene (7,564).
  Against that stronger no-RIS reference the centralised RIS gain is +12.7 % instead of
  +14.7 %; the RIS gain is robustly **+13–15 %** for tuned heuristics.
- Random RIS phases ≈ no RIS; time-share ≈ common arbitration (only one courtyard device is
  usually active at a time).

---

## 3. Single area: learning methods and federated learning

(`results_v9/train/building/summary.csv`, `fig_learning_curves.png`, `fig_ee_vs_comm.png`)

| Method | Test EE (bit/J) | vs decentralised greedy | vs centralised greedy | vs straight line | Courtyard data delivered | Violations | Training traffic |
|---|---|---|---|---|---|---|---|
| **MAPPO-CTDE** | **8,119 ± 61** | −80 ± 61 (−1.0 %) | −555 ± 61 (−6.4 %) | +1,055 ± 61 (+14.9 %) | 70.0 % | 0.25 | 375 MB |
| IPPO (no exchange) | 8,027 ± 262 | −172 ± 262 (−2.1 %) | −647 ± 262 (−7.5 %) | +963 (+13.6 %) | 69.8 % | 0.13 | 0 MB |
| FedPer (E = 4, personal heads) | 7,989 ± 113 | −211 ± 113 (−2.6 %) | −686 ± 113 (−7.9 %) | +924 (+13.1 %) | 70.6 % | 0.06 | 176 MB |
| FedAvg (E = 1) | 7,829 ± 150 | −370 ± 150 (−4.5 %) | −845 ± 150 (−9.7 %) | +765 (+10.8 %) | 66.1 % | 1.03 | 708 MB |
| FedProx (E = 4, μ = 0.01) | 7,827 ± 115 | −373 ± 115 (−4.5 %) | −848 ± 115 (−9.8 %) | +762 (+10.8 %) | 66.8 % | 0.88 | 177 MB |
| FedAvg (E = 4) | 7,742 ± 184 | −458 ± 184 (−5.6 %) | −933 ± 184 (−10.8 %) | +677 (+9.6 %) | 65.3 % | 1.12 | 177 MB |
| FedAvg (E = 16) | 7,647 ± 166 | −552 ± 166 (−6.7 %) | −1,027 ± 166 (−11.8 %) | +583 (+8.2 %) | 63.7 % | 3.88 | 44 MB |

All methods reach the final zone in 100 % of test missions; flight energy ≈ 42.6–43.1 kJ for
all of them, so the differences come from bits delivered, not energy.

**What this shows**
1. **Learned vs heuristics.** Every learned method clearly beats the straight line
   (+8 to +15 %). The best (CTDE) is within 1 % of, but slightly below, the tuned
   decentralised heuristic that has the same information (−0.98 ± 0.74 %, the CI excludes
   zero), and 6.4 % below the centralised heuristic that sees all devices and assigns UAVs
   jointly. These CIs count training-seed variation only: the heuristic is treated as fixed,
   although re-tuning it on other validation missions moves it by ~100–150 bit/J (e.g. the
   centralised greedy tuned for the no-RIS scene scores 7,564 there, the RIS-tuned one 7,698),
   so differences of ~1–2 % between learned and heuristic policies are not meaningful. **The learned policies do not
   beat a well-tuned heuristic.** Pilots showed that more training, a wider network or more
   devices in view do not close the gap (§6).
2. **FL inside one area does not help.** With only three UAVs flying in the same area,
   averaging their models costs EE relative to independent learning (IPPO); there is a trend
   toward worse results the less often they average (E = 1 / 4 / 16: 7,829 / 7,742 / 7,647),
   but the intervals overlap, so the ordering is not established — the averaged model is a compromise between three
   drifted local models and also collides more (FedAvg E = 16: 3.9 violations per mission).
   **FedPer**, which keeps each UAV's output layer personal, is the best federated variant
   (level with IPPO, −2.6 % vs the decentralised heuristic, and the fewest violations).
3. **Communication.** CTDE uploads every transition to the central learner (375 MB per run);
   FedAvg E = 4 uses about half of that, E = 16 an eighth; IPPO exchanges no learning data.
   Note that all methods share one team reward and one centralised scheduler that knows every
   channel (the same for every method), so "0 MB" counts learning traffic only, not the
   coordination the environment itself provides.

---

## 4. Multi-area federated learning: generalising to unseen city areas

(`results_v9/multiarea/`, `fig_generalization.png`, `fig_multiarea_curves.png`)

4 training areas (each with its own ground station training its 3 UAVs with CTDE), 4
validation areas (checkpoint selection), **8 test areas never seen in training**; 3000
missions per training area; 12 test missions per test area (96 per number), 5 seeds.

| Mode | EE on 8 unseen areas | Cross-area traffic per run | Raw data leaves the area? |
|---|---|---|---|
| Local only (each area alone; mean over the 4 area models) | 7,581 ± 115 | 0 MB | no |
| **FedAvg across areas (actors only, E = 4)** | **7,953 ± 62** | **88 MB** | **no** |
| FedAvg across areas (actors + critics, E = 4) | 7,972 ± 43 | 512 MB | no |
| Pooled raw data (one cloud learner) | 8,162 ± 59 | 771 MB | yes |
| *heuristic: decentralised greedy + RIS* | *8,052 ± 167* | — | — |
| *heuristic: centralised greedy + RIS* | *8,470 ± 151* | — | — |
| *heuristic: straight line + RIS* | *6,884 ± 138* | — | — |

Paired by seed (same initialisation and training missions):
- **FedAvg − local: +373 ± 130 bit/J (+4.9 %), better in 5 of 5 seeds.**
- Pooled − FedAvg: +208 ± 88 bit/J (+2.6 %), better in 5 of 5 seeds.
- Actors + critics − actors only: +18 ± 84 bit/J (+0.2 %) — no measurable gain for 5.8× the
  traffic.
- Intra-area traffic (UAVs ↔ their ground station) is the same for all modes: 1,126 MB.

**Equal-data control (v9.2).** The FedAvg model is built from 4 areas × 3,000 missions, each
local model from 3,000. To check that FedAvg does not win simply by seeing more data, each
area was also trained alone on **12,000** of its own missions (`multiarea/local_x4`, 5 seeds):

| Comparison (paired by seed, unseen areas) | Difference |
|---|---|
| local ×4 data − local | +83 ± 171 bit/J (+1.1 %, not significant; 7,663 ± 120) |
| **FedAvg − local ×4 data** | **+290 ± 94 bit/J (+3.8 %), 5 of 5 seeds** |
| pooled − local ×4 data | +498 ± 131 bit/J (+6.5 %), 5 of 5 seeds |

Four times more missions from the *same* area barely helps; what generalises is experience
from *different* areas — which FedAvg obtains without moving raw data.

**What this shows**
1. **Local models overfit their own layout.** On their own area they score 8,167 bit/J; on
   unseen areas 7,581 (−7 %). Part of that drop is that the training areas happen to be
   easier, so the cross-area matrix (`results_v9/multiarea/crossarea.json`: each area's model
   flown in every training area, 5 seeds × 24 missions) compares models **on the same area**:
   the area's own model beats the other areas' models there by +3.4 / +3.8 / +9.7 / +9.6 %
   (mean **+6.6 %**). The overfitting is real, not an artefact of area difficulty.
2. **FedAvg recovers most of what pooling gains** (+373 of +581 bit/J over local, i.e. 64 %)
   while sending 8.8× less cross-area traffic than pooling and **no raw data**. Counting the
   within-area traffic (UAVs ↔ ground station, 1,126 MB in every mode) as well, the total
   saving is 1.6× (1,214 vs 1,896 MB per run).
3. **Traffic needed to reach a given quality** (`fig_traffic_to_target.png`,
   `traffic_to_target.json`; total traffic = cross-area + UAVs ↔ ground station; validation
   EE on unseen areas, mean over seeds):

   | Mode | MB to reach 7,600 | MB to reach 7,800 | MB to reach 8,000 | total MB per run |
   |---|---|---|---|---|
   | Local only | 863 (4 of 5 seeds) | never | never | 1,126 |
   | **FedAvg (actors)** | **291** | **405** | **647** | 1,214 |
   | FedAvg (actors + critics) | 392 | 612 | 846 (4 of 5) | 1,637 |
   | Pooled raw data | 253 | 354 | 632 | 1,896 |

   FedAvg reaches the same validation quality as pooling for about the same total traffic
   (647 vs 632 MB to reach 8,000) without moving raw data, while local training never gets
   there.
   **Energy of this traffic is negligible:** even if every within-area byte is sent by a UAV
   radio (0.5 W at 10 Mbit/s ≈ 0.4 J/MB), 1,126 MB over 12,000 training missions is ≈ 0.04 J
   per mission, about 10⁻⁶ of the ≈ 42.6 kJ flight energy; cross-area traffic runs on the
   ground stations' backhaul and costs the UAVs nothing.
4. **Averaging only the actors is enough**; averaging the critics too costs 5.8× the traffic
   for nothing.
5. Against heuristics on unseen areas: pooled learning (8,162) is level with the tuned
   decentralised heuristic (8,052 ± 167; overlapping intervals), FedAvg slightly below, and
   the centralised full-information heuristic remains best (8,470). Note that the heuristics
   were tuned on the 4 validation areas — the same areas the learned models use for
   checkpoint selection — so both sides had the same access to non-training layouts.

---

## 4b. Heterogeneous (non-IID) multi-area FL with partial participation (v9.2)

(`results_v9/fedhet/`, `fig_fedhet.png`, `summary.json`; code `v9/fedhet.py`.) The harder,
more realistic FL setting:

- **12 training areas (clients)** that differ in layout *and* deployment — 15–31 devices,
  data per device × 0.6–1.5, device power 6–14 dBm, RIS 10 × 10 to 20 × 20
  (`config.hetero_area`); 4 validation and **8 unseen test areas** drawn the same way.
  The decentralised heuristic's EE varies from 7,030 to 8,790 bit/J across the test areas
  (straight line: 5,896–7,814).
- **Partial participation:** each round the cloud samples 50 % of the areas; they start from
  the global actor, run E = 4 local MAPPO-CTDE updates and upload their update.
- **Equal budget:** every mode uses the same 11,520 training missions in total.
- **Compressed uploads:** 8-bit quantised updates, or top-10 % sparsified updates with
  client-side error feedback.

| Mode (5 seeds, 96 test missions on 8 unseen areas) | EE (bit/J) | Cross-area traffic |
|---|---|---|
| Local only (mean of the 12 area models) | 7,200 ± 111 | 0 |
| Pooled raw data | 7,761 ± 142 | 1,081 MB |
| **FedAvg, 50 % of areas per round** | **7,895 ± 73** | **85 MB** |
| FedAvg 50 %, 8-bit updates | 7,869 ± 90 | 53 MB |
| FedAvg 50 %, top-10 % updates | 7,844 ± 33 | 51 MB |
| FedProx (μ = 0.01), 50 % | 7,807 ± 29 | 85 MB |
| FedAvg, all 12 areas per round (30 rounds) | 7,223 ± 138 | 85 MB |
| FedAvg, all 12 areas per round, E = 2 (60 rounds) | 7,355 ± 323 | 170 MB |
| *heuristic: centralised greedy (tuned on the heterogeneous validation areas)* | *8,522 ± 224* | — |
| *heuristic: decentralised greedy* | *7,832 ± 212* | — |
| *heuristic: straight line* | *6,743 ± 194* | — |

Paired by seed:
- **FedAvg (50 %) − local: +695 ± 108 bit/J (+9.7 %), 5 of 5 seeds** — with non-IID areas the
  benefit of federating is twice as large as in §4 (+4.9 %).
- FedAvg (50 %) − pooled: +134 ± 193 bit/J (+1.7 %, not significant; 4 of 5 seeds) — FedAvg is
  **at least as good as pooling all raw data**, with 13× less cross-area traffic.
- 8-bit / top-10 % uploads − full precision: −26 ± 122 / −51 ± 69 bit/J (−0.3 / −0.7 %, not
  significant) for **38–40 % less cross-area traffic**.
- FedProx − FedAvg: −87 ± 80 bit/J (−1.1 %, 0 of 5 seeds better): the proximal term does not
  help here.
- All areas per round − 50 %: −672 ± 149 bit/J (−8.5 %, 0 of 5 seeds). At the same budget,
  full participation means half as many averaging rounds (30 vs 60) — but that is **not**
  the explanation: a control with all areas per round and E = 2 (60 rounds, same budget)
  is still −540 ± 386 bit/J (−6.8 %, 0 of 5 seeds better than 50 %) and only +1.8 % (n.s.)
  over the 30-round version. Likely reason (not separately tested): at a fixed budget,
  sampling half the areas lets each participant run 4 local updates per round in 60 rounds,
  i.e. 240 local steps of progress for the global model, against 120 for both full-
  participation variants (60 × 2 or 30 × 4). Partial participation is therefore not just a
  necessity but, here, the better use of the same compute.
- Local area models violate the separation rule more often (1.39 per mission vs 0.00–0.09 for
  every federated variant): the federated model is also the safer one.

**Few-shot adaptation in a new area** (`fedhet/finetune/`). Each unseen area's ground station
adapts a model using only its own 320 missions (40 PPO updates; for the federated model the
fresh critic is first trained alone for 8 updates):

| Starting point (5 seeds, mean over the 8 unseen areas) | before | after 320 missions |
|---|---|---|
| **global FedAvg model** | 7,895 ± 73 | **7,998 ± 56** |
| random initialisation (training from scratch) | 4,669 ± 23 | 6,884 ± 41 |

- Starting from the federated model is **+1,114 ± 56 bit/J (+16.2 %) better** than training
  from scratch on the same missions (5 of 5 seeds): a new area is useful from the first
  mission instead of after thousands.
- After adaptation the learned policy is above the decentralised heuristic in every seed
  (7,949–8,071 vs 7,832, +2.1 % on average — modest, within the heuristic's per-mission
  spread) and still below the centralised full-information heuristic (8,522).

## 5. Where should the RIS be? (learned flight vs heuristics)

(`results_v9/fig_ris_value.png`)

| RIS | Learned MAPPO-CTDE (5 seeds) | Centralised greedy | Decentralised greedy | Flight energy (CTDE) |
|---|---|---|---|---|
| None | 7,201 ± 151 | 7,564 | 7,208 | 42.7 kJ |
| **Building (rooftop, 16 × 16)** | **8,119 ± 61** | **8,675** | **8,199** | 42.6 kJ |
| Aerial (carried by a 4th UAV, energy charged) | 5,070 ± 101 | 5,757 | 5,199 | 60.0 kJ |

- **Building RIS under learned flight: +919 ± 155 bit/J (+12.8 %)** over no RIS (paired by
  seed); courtyard data delivered 50.5 % → 70.0 %.
- **Aerial RIS: −3,049 ± 123 bit/J (−37.6 %)** vs the building RIS. The carrier adds ~17 kJ of
  flight energy per mission (+41 %) and the extra bits do not pay for it. This holds for every
  policy.

---

## 6. Pilot experiments (how the training settings were chosen; validation only)

CTDE, 2 seeds; validation EE = mean of the last 5 evaluations.

| Variant | Val EE |
|---|---|
| A: Dinkelbach price on | 8,046 |
| B: fixed energy price | 7,984 |
| **C: Dinkelbach + learning-rate annealing (chosen)** | **8,198** |
| D: 16 missions per update | 8,071 (unstable between seeds) |
| E: C with 6,000 missions | 8,055 |
| F: E with a 2× wider network | 7,954 |
| G / H / I: 12 / 24 / 8 devices in view (I = control on the final environment) | 8,149 / 8,164 / 8,058 |

Differences of ~100 bit/J are within seed-to-seed noise. Locked: Dinkelbach + LR annealing,
4,000 missions, 8 devices in view. (`results_v9/train/pilots/`)

---

## 7. Sensitivity sweeps (trained policies, no retraining)

(`results_v9/sweeps/building/fig_sweeps.png` and CSVs.) Policies trained on the default
scene are flown in modified scenes on the same 100 test missions. Learned values are means
over 5 seeds; heuristics keep their default-scene tuning.

| Parameter | Values | Centralised greedy | Decentralised greedy | Straight | CTDE | IPPO | FedAvg E=4 |
|---|---|---|---|---|---|---|---|
| RIS elements L | 64 / 144 / **256** / 400 | 8096 / 8379 / 8675 / 8911 | 7512 / 7855 / 8199 / 8453 | 6192 / 6675 / 7065 / 7412 | 7421 / 7811 / 8119 / 8353 | 7327 / 7708 / 8027 / 8232 | 6891 / 7360 / 7742 / 8017 |
| Phase bits | 1 / **2** / 3 / continuous | 8511 / 8675 / 8720 / 8706 | 7955 / 8199 / 8231 / 8239 | 6821 / 7065 / 7125 / 7149 | 7948 / 8119 / 8179 / 8180 | 7831 / 8027 / 8059 / 8081 | 7516 / 7742 / 7781 / 7812 |
| Phase error (std, °) | **0** / 10 / 20 / 40 | 8675 / 8684 / 8610 / 8548 | 8199 / 8201 / 8199 / 8059 | 7065 / 7055 / 7024 / 6890 | 8119 / 8110 / 8093 / 8007 | 8027 / 8023 / 7978 / 7857 | 7742 / 7729 / 7704 / 7578 |
| Device power (dBm) | 0 / 5 / **10** / 15 / 20 | 5787 / 7199 / 8675 / 10048 / 11077 | 5270 / 6701 / 8199 / 9703 / 10828 | 4203 / 5600 / 7065 / 8629 / 10066 | 5314 / 6737 / 8119 / 9435 / 10435 | 5158 / 6568 / 8027 / 9377 / 10417 | 4957 / 6340 / 7742 / 9094 / 10199 |
| Data per device (×) | 0.5 / **1** / 1.5 / 2 | 5852 / 8675 / 9224 / 9511 | 5772 / 8199 / 8671 / 8893 | 5488 / 7065 / 7497 / 7676 | 5472 / 8119 / 8609 / 8779 | 5461 / 8027 / 8463 / 8635 | 5388 / 7742 / 8235 / 8419 |
| RIS setting | none / random / common / **timeshare** | 7698 / 7701 / 8689 / 8675 | 6850 / 6875 / 8234 / 8199 | 5616 / 5645 / 7061 / 7065 | 6548 / 6910 / 8150 / 8119 | 6626 / 6812 / 8071 / 8027 | 6049 / 6271 / 7761 / 7742 |

(bold = default)

**What this shows**
- **Bigger RIS helps steadily** (CTDE +12.6 % from 64 to 400 elements); the ranking of the
  methods never changes.
- **2-bit phases are enough:** 1 bit costs 2–3.5 %, 3 bits or continuous add < 1 %.
- **Robust to RIS phase noise:** (random per-element phase errors on top of perfect channel
  knowledge; imperfect channel estimates are studied separately in §11) a 20° error costs < 1 %, 40° about
  1.4–2.5 %.
- **Transmit power and data volume move everything together;** with half the data
  (5–15 Mbit per device) all policies are within ~8 % of each other (the straight line even
  edges out the learned policies, 5,488 vs 5,472 for CTDE) because there is little left to
  compete for — scarcity is what makes trajectories matter, and policies trained for the
  default data volume do not adapt to a much smaller one without retraining.
- **A policy trained with the RIS, flown without it, does worse** (CTDE 6,548) than one
  trained without the RIS (7,201): the learned behaviour relies on the RIS being there.
- Apart from the half-data point, the ordering is the same everywhere: centralised greedy >
  decentralised greedy ≈ CTDE ≳ IPPO > FedAvg (single area) > straight line.

---

## 7b. Optimisation additions (v9.2)

**Optimised RIS time shares** (`arbitration="opt"`, `env._opt_shares`): instead of giving
every RIS configuration an equal share of the slot, the shares solve
$\max_\tau \sum_i \min(D_i^{\text{left}}, \delta\sum_c \tau_c r_{ci})$ (concave; greedy over
1/20-slot slices). Same test missions, policies not retrained:

| Policy | equal shares | optimised shares | gain |
|---|---|---|---|
| centralised greedy | 8,675 | 8,730 | +0.6 % |
| decentralised greedy | 8,199 | 8,221 | +0.3 % |
| MAPPO-CTDE | 8,119 | 8,165 | +0.6 % |
| IPPO | 8,027 | 8,071 | +0.5 % |
| FedAvg (E = 4) | 7,742 | 7,778 | +0.5 % |
| straight line | 7,065 | 7,111 | +0.7 % |

A consistent but small gain: usually only one courtyard device needs the RIS at a time, so the
simple equal split is already close to optimal.

**Imperfect channel knowledge** (`csi_nmse_db`): scheduling and RIS alignment use channel
estimates with normalised MSE e (all links, every slot); the bits are computed with the true
channels.

| Policy | perfect | e = −20 dB | −10 dB | −5 dB | 0 dB |
|---|---|---|---|---|---|
| centralised greedy | 8,675 | 8,418 | 8,177 | 7,909 | 7,578 |
| decentralised greedy | 8,199 | 7,833 | 7,519 | 7,314 | 6,922 |
| MAPPO-CTDE | 8,119 | 7,933 | 7,602 | 7,382 | 6,935 |
| IPPO | 8,027 | 7,827 | 7,480 | 7,272 | 6,833 |
| FedAvg (E = 4) | 7,742 | 7,544 | 7,306 | 7,080 | 6,632 |
| straight line | 7,065 | 6,823 | 6,617 | 6,431 | 6,020 |

EE falls by 2–4 % at −20 dB, 6–8 % at −10 dB and 13–16 % with very poor estimates (0 dB).
The learned CTDE policy degrades slightly more gracefully than the decentralised heuristic
(it is ahead of it at every error level from −20 dB on). Pilot overhead for estimating the
257 cascaded coefficients per device is < 0.1 % of a slot (§9) and is neglected.

## 8. What changed from v9.0, and why the numbers moved

- RIS model: Tang et al. element pattern and an aimed panel (stronger, more realistic RIS) →
  larger RIS gains than v9.0 (+13.8–25.8 % vs +11–16 %).
- Heuristics: tuned on validation, decentralised variant added, separation enforced → stronger
  and fairer rivals than v9.0's single untuned greedy.
- Energy: composite Simpson integration; separating autopilot.
- FL methods judged on the federated (averaged) model; Dinkelbach price actually updated.
- Statistics: 100 fresh test missions, 5 seeds, 95 % CIs, paired comparisons.
- v9.0's headline "learned MAPPO beats greedy" does **not** survive: against tuned and fair
  heuristics the learned policies are within ~1 % of (slightly below) the like-for-like rival
  and 6 % below the full-information one.

## 9. Limitations

Perfect channel knowledge at the RIS controller in the main results (imperfect estimates in
§11; pilot overhead for estimating the 257 cascaded coefficients is ≈ 257 symbols × 5.6 µs ≈
1.4 ms per device per 2.5 s slot, i.e. < 0.1 %, and is neglected); buildings
are boxes with a fixed +20 dB loss; fixed altitude, Shannon rates, no protocol overhead; only
propulsion energy; data size, bandwidth and one receiver per UAV are design choices within
NB-IoT ranges; FL simulated on one machine (it fixes what is exchanged, not real links); the
heuristic references are strong because they were tuned per scenario, which learned policies
did not need; 5 seeds and 100 test missions per number.

## 10. One-paragraph summary for the report

On a 500 m urban scenario with a shadowed courtyard, a building-mounted 256-element RIS with
closed-form 2-bit phase control raises energy efficiency by 14–26 % for every flight policy
(+12.8 % under learned flight), whereas the same panel carried by a fourth UAV lowers it by
~38 % because of the carrier's flight energy; in the original 20 m scene the RIS changes EE
by only 0.5 %. Learned MAPPO trajectories beat straight-line flight by 15 % and come within
1 % of a tuned decentralised heuristic, but remain 6 % below a centralised full-information
planner.
Federated learning does not help three UAVs within one area, but across city areas FedAvg of
the area actors generalises to eight unseen layouts 4.9 % better than training each area
alone — not because it sees more data (4× more own data gives +1 %) — recovering 64 % of the
benefit of pooling all raw data with 8.8× less cross-area traffic and without sharing raw
data. With twelve heterogeneous areas and 50 % participation the gain grows to 9.7 %,
FedAvg matches pooling with 13× less cross-area traffic, 8-bit or top-10 % compressed
uploads cost nothing measurable, and the federated model fine-tuned on 320 missions in a
new area beats training from scratch by 16 % and the like-for-like heuristic by 2 %.
