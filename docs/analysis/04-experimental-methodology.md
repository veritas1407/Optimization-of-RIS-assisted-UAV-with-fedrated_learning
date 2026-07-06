# 04 — Experimental Methodology

Protocol for the redesigned system ([03-redesign-proposal.md](03-redesign-proposal.md)).
Design principle: every claim in the paper maps to one experiment with error bars; no
figure exists that a reviewer can invalidate by asking "how many seeds?".

---

## 1. Baselines (all evaluated with the *same* environment, channel generators, and seeds)

| # | Name | Trajectory | RIS | Purpose |
|---|---|---|---|---|
| B1 | No-RIS | straight line | — | floor; isolates RIS contribution |
| B2 | Random-RIS | straight line | uniform phases/slot | sanity floor for RIS control |
| B3 | **Aligned-RIS (corrected)** | straight line | single shared Θ[t], closed-form alignment to the scheduled weakest device (verified against theoretical max, cf. audit F1 script) | honest model-based RIS bound on a fixed path |
| B4 | Sum-gain RIS | straight line | Θ[t] optimizing Σᵢ effective gain (element-wise phase of the dominant eigen-direction, or manifold/SDR if time permits) | stronger model-based bound; cite as such |
| B5 | Joint-PPO | learned (one agent, (2K+L)-dim action) | learned | shows why multi-agent decomposition is needed |
| B6 | IPPO | learned, independent per UAV, no exchange | RIS agent local | lower bound of the coordination axis |
| B7 | MAPPO-CTDE | learned, centralized critic | RIS agent | upper bound of the coordination axis |
| B8 | MADDPG | learned | learned | the standard MARL alternative reviewers ask for |
| B9 | **Fed-MAPPO / Fed-MAPPO-P** (proposed) | learned, FedProx every E | RIS agent as FL client | the contribution |
| B10 | MAPPO-traj + Aligned-RIS | learned trajectories | analytic B3 phases | ablation: is *learning* the RIS worth anything? (M3 gate) |

Notes: B3 replaces v8's broken `opt_phase_gpu` baseline and must be unit-tested against
the coherent-combining maximum. If B10 ≥ B9's RIS-agent variant, the paper's RIS-learning
claim is dropped and the story becomes trajectory-learning + analytic RIS + FL — decide
at the M3 gate, not after the writeup.

## 2. Statistical protocol

- **≥5 training seeds per method** (seed controls networks, exploration, and the
  train-episode channel stream). Report mean ± 95% CI (t-based, n=5) on every curve and
  bar; shaded bands on learning curves.
- **Evaluation:** 100 held-out episodes (disjoint seed range, S4) per trained model;
  deterministic policy. Metrics averaged over episodes then over seeds.
- **Comparisons:** Welch's t-test (or bootstrap CI on the difference) whenever the paper
  says "X outperforms Y"; report p or CI in the text.
- **No post-processing of measured values.** Monotonicity enforcement (audit F6) and any
  cousin of it is banned; non-monotone curves get error bars and an explanation.
- All sweeps evaluate the *same fully-trained* method per point (no 500-vs-1500-episode
  sub-agent confounds, audit F12); where a per-point retrain is inherent (e.g., per-L),
  every point gets the identical training budget and all seeds.

## 3. Metrics

| Metric | Definition | Unit |
|---|---|---|
| Energy efficiency | B·Σₜ Σᵢ R_{i,k*(i,t)} Δt ÷ Σₜ Σₖ P_prop Δt | bits/J |
| Sum throughput | numerator above | Mbit |
| Propulsion energy | denominator above | kJ |
| Device fairness | Jain index over per-device delivered bits | — |
| Safety | min inter-UAV distance; violation rate with projection disabled (S7) | m, % |
| Convergence | episodes (and wall-clock) to 95% of final EE | eps, min |
| **Communication overhead** | uplink+downlink bytes per FL round × rounds; for CTDE: per-step state exchange bytes × steps (count both, same units) | MB |
| Sample efficiency | env steps to 95% of final EE | steps |

## 4. Experiment matrix

| Exp | Question | Sweep | Methods | Est. runs |
|---|---|---|---|---|
| E1 Main comparison | Does Fed-MAPPO approach CTDE at a fraction of the communication? | — | B1–B10 | 10 methods × 5 seeds |
| E2 EE–communication frontier | trade-off curve | E ∈ {1,5,10,25}; FedAvg vs FedProx vs Fed-MAPPO-P | B9 variants | 12 × 5 |
| E3 RIS size | value of the RIS | L ∈ {16,36,64,100,144} | B1,B3,B9 | 15 × 5 (B1/B3 cheap) |
| E4 Phase quantization | hardware realism | b ∈ {1,2,3,∞} | B3,B9 | 8 × 5 |
| E5 Scalability | does the framework scale? | K ∈ {2,3,5}; I ∈ {10,20,40} | B6,B7,B9 | 15 × 5 |
| E6 Deployment generalization (redone correctly, cf. audit F2) | zero-shot transfer across device layouts | 4 layout modes, train on uniform only | B9 | 4 evals × 5 |
| E7 Robustness | noisy CSI features σ ∈ {0, 3, 6 dB}; UAV wind perturbation | B9 vs B7 | 6 × 5 |
| E8 Reward sensitivity | shaping-weight sensitivity (v8's "most important lesson", made rigorous) | w_p ∈ {0.02,0.05,0.1,0.2} | B9 | 4 × 5 |

Ablations folded in: RIS agent on/off (B10 vs B9), association on/off (S2 vs v8-style
counting — one row in a table to *show* the inflation), personalized heads (E2),
codebook vs continuous phases (D2).

**Budget check:** with the vectorized simulator target (≤50 ms/episode, D5), a
1500-episode training ≈ 2–3 min; the full matrix ≈ 350 runs ≈ 15–20 GPU-hours plus
evaluation — feasible on Colab across a few sessions, with per-run checkpointing so
sessions can resume. Without D5 (v8 speed) the same matrix is ≈ 900 GPU-hours —
this is why vectorization is a roadmap gate, not an optimization nicety.

## 5. Convergence analysis (reported, not just plotted)

- Learning curves (train EE and held-out EE) with CI bands for B5–B9.
- For Fed-MAPPO: EE trajectory across FL rounds; post-aggregation dips quantified
  (mean drop and recovery time) — this is the honest instability disclosure and doubles
  as the FedProx-μ justification.
- Entropy and clip-fraction curves in an appendix/repo (diagnostics reviewers trust).

## 6. Reproducibility checklist (repo deliverables)

- Config-driven runs: `experiments/*.yaml`; one command per figure:
  `python -m experiments.run --config configs/e1_main.yaml --seed 0..4`.
- Seed manifest committed; train/eval seed ranges documented.
- `requirements.txt` with pinned versions; Python ≥3.10; CPU fallback tested.
- Unit tests (`tests/`): hover power = 168.5 W analytic value; Rician sample moments vs
  K-factor; **coherent alignment achieves the analytic maximum** (regression test for
  audit F1); association invariants (each device counted once); EE parity between env
  path and baseline path on identical inputs (regression for F2/F5); PPO ratio = 1 at
  first epoch (regression for F7).
- Raw per-seed results saved as CSV alongside figures; plotting scripts read CSVs only
  (no recomputation inside plot code — the v8 pattern of training sub-agents inside
  plotting functions is retired).
- Fixed-seed smoke run (<2 min) in CI to catch environment regressions.
