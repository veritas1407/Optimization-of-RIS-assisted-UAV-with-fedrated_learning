# v9 Audit — verified findings (Nov 2026)

Four independent reviews (physics/env code, RL/FL code, numbers-vs-data, methodology).
Every finding marked **verified** below was reproduced a second time, independently, from
the code or saved checkpoints. Items marked *reviewer-verified* were reproduced by the
reviewing agent only. Scripts: session scratchpad (`verify/`, `agent_*/`).

What is **sound** (checked and correct): the physics core (units, noise, Al-Hourani path loss,
Rician fading, blockage test, RIS geometry, aligned phases, composite channel, quantisation);
scheduling and bit accounting; kinematics; the autopilot reaches the final zone for every
policy; PPO maths (stored pre-squash action ⇒ exact ratio, summed log-probs, GAE, reward
scaling, CTDE batching); FedAvg/FedPer/FedProx aggregation and key names; communication byte
formulas; train/validation/test seeds disjoint; baselines and learned methods paired on the same
test episodes; every saved checkpoint reproduces its reported test number exactly.

## Critical — a headline claim is wrong

| ID | Finding | Evidence | Claim affected |
|---|---|---|---|
| C1 | **A tuned greedy heuristic beats every learned method.** Tuning 5 constants of the existing greedy policy on validation episodes (speed cap 12 m/s, margin 1.0, uniform weights) gives 8,325 on the official test set. On 100 fresh episodes, learned − tuned is CTDE −199 ± 58, IPPO −236 ± 62, no-RIS CTDE −373 ± 67 (paired SE). Reviewer: tuned greedy 7,652 on the 8 unseen multi-area test areas vs FedAvg 7,542. **Caveat:** greedy is centralized with full information (sees all 24 devices, assigns devices jointly); the learned policies are decentralized (each UAV sees its 5 nearest devices). | verified | "Learning beats hand-made flight"; "learning helps most without a RIS"; §5 learned-vs-greedy gaps |
| C2 | **Single-area FedAvg/FedProx were evaluated on each UAV's locally drifted model, not on the federated (averaged) model.** Checkpoints taken between averaging rounds; selection on validation picks drifted copies. Averaged model at the same checkpoints: FedAvg E=16 **7,325** (reported 7,973), E=4 **7,482** (reported 7,848). Multi-area evaluates the averaged model correctly. | verified | "FedAvg E=16 reaches 99 % of CTDE with 13 % of its communication"; E=1 vs E=16 ranking; FedProx/FedPer comparisons |
| C3 | **In the old 20 m scene the RIS was switched off.** The RIS lies flat on the ground (z=0, normal up) and devices are on the ground, so the strict front-half-space test excludes every device (0 of 12 eligible, RIS channel ≡ 0). The "+0.0 %" was produced by a dead RIS. With the RIS working the reviewer measured +0.6 % — still tiny, so the conclusion survives qualitatively, but the evidence must be regenerated. | verified | §1 table, `fig_old_vs_new.png`, README gate bullet |

## Major — biases results or is clearly wrong

| ID | Finding | Evidence |
|---|---|---|
| M1 | Energy per slot = average of start/end power × dt over-charges when speed changes (P(v) is non-linear): +4.9 % for 0→12.5 m/s, +10.3 % for 20→7.5 m/s, 0 % at constant speed. Policy-dependent: greedy EE understated ~1.3 %, CTDE overstated ~0.4 %. Fix: Simpson's rule. | verified |
| M2 | RIS elements are isotropic and the panel hangs vertically: courtyard devices are 68–84° off broadside. With a cos element pattern the 33 % shadow-coverage figure becomes 12 % (cos¹), 25 % (cos¹ + 5 dB element gain) or 4 % (cos³). The RIS-gain magnitudes depend heavily on this unmodelled choice. | reviewer-verified; geometry verified |
| M3 | All 12 multi-area runs used `eval_every=250` (only 3 checkpoint candidates: 1000/2000/2496); the committed code says 200, so §3 cannot be reproduced from the current code. | verified |
| M4 | Communication claims mix different things. "13× less traffic" is cross-area only; 488 MB of pooled's 790 MB are model downloads caused by pooled doing 4 updates per round. Intra-area traffic (~669 MB, needed by both) is excluded; system total is ~2× not 13×. Single-area "0 MB / 27 MB" ignore per-step coordination (team reward, other UAVs' positions, central scheduler) that every method needs. | reviewer-verified (formulas verified) |
| M5 | The reward uses a fixed price λ = 6,000 (no Dinkelbach iteration) while achieved EE is ~8,000, so the agents are trained to over-value bits relative to energy — the learned policies indeed spend more energy than the tuned greedy. | verified (lam_update=False in all runs) |
| M6 | The RIS gain is regime-specific: reviewer measured greedy RIS gain 13.7 % → 1.5 % with 2 receivers per UAV, 7.0 % without the extra 20 dB building loss, 6.1 % at 360 kHz. None of these are swept. | reviewer-verified |
| M7 | Statistics: 3 seeds, population std (ddof=0), 20 test episodes. Many reported differences are within noise (IPPO vs CTDE, FedAvg variants, FedAvg vs pooled, critic averaging +0.7 %, FedProx/FedPer vs FedAvg). Robust: building RIS vs no RIS under learned flight (+608, 95 % CI [268, 919]); local-only < pooled. | reviewer-verified |

## Minor (all verified)

- Phase errors draw from the fading random stream: a 1e-9° "error" changes greedy EE by −1.12 %, so the phase-error sweep is noise-dominated.
- "Open" devices can land inside the shadowed courtyard (42/200 test episodes) and are labelled unshadowed (affects shadow metrics only).
- `random_policy` keeps its RNG as a default argument: same seeds give 5,572 then 5,383.
- At the area boundary velocity is not zeroed, so a UAV pressed against the edge is charged flight energy while not moving (aerial carrier only, ≤2.5 % fleet energy).
- Aerial preset: two of four landing slots lie outside the area; collision push-apart is 2-D while the distance test is 3-D.
- Advantage normalisation includes autopilot-overridden steps (small bias).
- `v9/README.md` still says the aerial carrier flies at 100 m and costs +37 % energy (now 70 m, +40 %).
- Three bar charts start the y-axis far above zero (gaps look 2–3× larger); some error bars are unlabeled; local-only "±15" is the spread of 4-model averages, the 12 individual models spread ±176.
- Unsupported statements in the results doc: "the no-RIS policy compensates by detouring over the courtyard" (data show the opposite: 57 % vs 63 % shadow delivery), "14 layouts, 2–5 % vs 29–39 %" (no saved file), the aerial static-analysis sentence (no saved output).
- Overstated wording: "every learned method beats the best heuristic", "~0 collisions" (FedPer 1.5/episode), "clear lead from 500 to ~2000", "diminishing returns", "gap grows with scarcity", "equal to pooled", "aerial delivers about as much data" (learned: 67.8 % vs 72.8 %).
- Test seeds were reused across development rounds; final numbers should be reported on fresh seeds.
