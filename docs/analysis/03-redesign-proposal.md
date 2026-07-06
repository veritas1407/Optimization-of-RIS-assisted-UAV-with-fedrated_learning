# 03 — Redesign Proposal: System Model, Architecture, and Fed-MAPPO

This document specifies the redesigned system (items 5–7 of the analysis request): the
corrected physical model, the heterogeneous multi-agent architecture, the federated
learning pipeline, and the comparison of FL algorithms with a recommendation. Every major
decision carries its rationale, trade-offs, implementation complexity, and publication
value. The research topic is unchanged: **energy-efficiency optimization of a
RIS-assisted, UAV-enabled IoT network with federated learning**.

Design constraints taken as fixed:
- Topic: RIS + multi-UAV + IoT uplink + EE (propulsion-aware) + FL.
- Compute: Colab T4-class GPU; every experiment must fit the budget in
  [04-experimental-methodology.md](04-experimental-methodology.md).
- Venue: IEEE conference (ICC/GLOBECOM/WCNC), 6 pages — one clear contribution, not five.

---

## 1. Corrected system model (decisions S1–S8)

### S1 — One shared RIS configuration per slot *(fixes audit F3; non-negotiable)*

Exactly one Θ[t] = diag(e^{jθ₁[t]}, …, e^{jθ_L[t]}) exists per time slot and applies to
every device–UAV pair. All rate computations for slot t use the same Θ[t].

*Trade-off:* EE numbers will drop relative to v8 (each UAV loses its private RIS), and
the learning problem becomes genuinely harder — that is the point; the coupling this
creates is the multi-agent problem the paper claims to study.
*Complexity:* small (environment change).
*Publication value:* mandatory for correctness; enables Gap B (arbitration).

### S2 — Device–UAV association; each device's rate counted once *(fixes audit F8)*

Baseline rule: in slot t, device i is associated to the UAV with the highest effective
channel gain, `k*(i,t) = argmax_k |h_tot_ik[t]|²`; its rate is counted only at k*. System
EE follows the report's own Eq. (3.7) as a **ratio of sums**:

```
EE[t] = Σ_i R_{i,k*(i,t)}[t]  /  Σ_k P_prop(v_k[t])
```

TDMA within a UAV (τ = 1/|assigned set|) and full per-device power (P_i = P_max, since
channels are orthogonal) — pick this pair and state it in both code and paper; the v8
report/code contradiction (Pmax vs P/n) must not survive.

*Trade-off:* argmax association is a modeling choice, not learned; a learned scheduling
action is deferred to future work to keep the action space small (see D2).
*Complexity:* small.
*Publication value:* removes triple counting; makes cooperation *necessary* (a device
covered by two UAVs is wasted coverage), so fairness/coordination results become
meaningful.

### S3 — Physically defensible geometry and link budget *(fixes audit F11)*

| Parameter | v8 | Redesign | Why |
|---|---|---|---|
| Area | 20×20 m | **500×500 m** | UAV data-collection scale used in [1], [2]; makes trajectory design matter |
| UAV altitude | 5 m | **50–100 m** | LoS probability argument becomes real; matches propulsion-model regime |
| Mission | 20 slots × 1 s | **50 slots × 2 s** | trajectory has room to express; propulsion energy is meaningful |
| Carrier / λ | λ=0.1 m implied, spacing 1.0λ | **fc = 2.4 GHz (λ=0.125 m), spacing λ/2** | removes grating lobes; standard |
| RIS elements | 50 (1-D response, x-projection only) | **L = 64–100, planar (e.g., 8×8/10×10), 2-D array response** | valid far field: aperture ≈ 0.5–0.63 m ⇒ Fraunhofer 2D²/λ ≈ 4–6 m ≪ link distances |
| ρ₀ | 5×10⁻² (tuned) | **10⁻³ (−30 dB at 1 m)** | standard anchor; no tuned physics |
| Bandwidth / noise | absent / σ²=10⁻¹⁰ W | **B = 1 MHz, N₀ = −170 dBm/Hz + NF 10 dB ⇒ σ² = −110 dBm** | rates in bits/s, EE in bits/J — real units |
| P_max | 1 W | **0.1 W (20 dBm)** typical IoT | link budget consistent with the above |
| Path-loss exponents | 3.5 direct / 2.2 RIS | keep (justify via air-to-ground LoS/NLoS literature) | already reasonable |

*Validation gate:* before any training, a static link-budget notebook must show (a)
direct-link SNR distribution, (b) RIS gain at true optimal phases vs L, (c) that the RIS
adds 20–40 % rate for cell-edge devices at these parameters *without touching ρ₀*. If the
RIS benefit is invisible at honest parameters, the honest response is to place the RIS
where it matters (deep-shadow region between device cluster and UAV corridor), not to
inflate ρ₀.

### S4 — Channel stochasticity done properly *(fixes audit F4, F5)*

- NLoS components redrawn **every episode** (block fading per slot: redrawn each slot for
  the direct link; device→RIS static per episode since both ends are fixed; RIS→UAV
  redrawn per slot as the UAV moves).
- One `numpy.random.Generator` per environment instance, seeded once per episode from a
  master seed sequence; **identical generator path for env and all baselines** (baselines
  call the same channel functions on the same env).
- Disjoint seed ranges for train and evaluation; final results on held-out seeds only.

### S5 — Observations that make the task learnable *(fixes audit F10)*

Per-UAV observation (UAV agents): own position/velocity, distances *and bearings* to the
κ nearest **assigned** devices, distance/bearing to RIS, other UAVs' positions, remaining
mission time, distance to terminal point. Per-RIS-agent observation: UAV positions,
device cluster summary (e.g., centroid + count per sector), previous-slot per-link
SNRs. Positions are sufficient statistics for the LoS phase geometry (which carries
K/(K+1) of the link power), so geometric phase control becomes learnable in principle;
optional noisy CSI features are an ablation, not the default.

### S6 — Discrete-phase realism as a first-class ablation

b-bit phase quantization (b ∈ {1,2,3} vs continuous) applied in the environment.
*Publication value:* addresses the report's own limitation §5.3 and the Björnson et al.
hardware critique [7]; cheap to implement; reviewers expect it in 2026.

### S7 — Collision handling that is honest *(fixes audit F9)*

Count the violation **before** correcting it: if a proposed step would bring two UAVs
within d_min, register the collision event (penalty applies), then apply the projection.
Report both the penalty-on rate during training and the violation rate of the final
policy with the projection disabled. Claims about learned avoidance are made only if the
latter is ~0.

### S8 — EE metric and units

EE in **bits/J** = (B · Σ log₂(1+SNR) · Δt) / (Σ P_prop · Δt), reported with B explicit.
Secondary metrics: sum throughput (Mbit), total propulsion energy (kJ), Jain fairness
over per-device throughput (not per-UAV EE — fairness to *devices* is what matters after
S2), collision-violation rate.

---

## 2. Agent architecture (decisions D1–D5)

### D1 — Heterogeneous agents: K UAV agents + 1 RIS-controller agent

```
                     ┌──────────────────────────────────────┐
                     │  CENTRALIZED CRITIC (training only)   │
                     │  V(s): s = [obs_UAV1..K ‖ obs_RIS]    │   ← CTDE baseline
                     └──────────────────────────────────────┘
   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌─────────────────┐
   │ UAV agent │   │ UAV agent │   │ UAV agent │   │  RIS agent       │
   │ π_u(o_k)  │   │ π_u(o_k)  │   │ π_u(o_k)  │   │  π_r(o_RIS)      │
   │ a = (vx,vy)│  │ a = (vx,vy)│  │ a = (vx,vy)│  │  a = θ_1..θ_L    │
   └───────────┘   └───────────┘   └───────────┘   │  (or codebook id)│
                                                    └─────────────────┘
   One shared UAV policy π_u (parameter sharing, agent-ID feature in obs)
   One RIS policy π_r      → exactly one Θ[t] per slot (S1)
```

Rationale:
- **Physics**: one RIS ⇒ one controller ⇒ one Θ[t]. The v8 design (each UAV outputs 50
  phases) is unphysical and also saddles every UAV policy with a 52-dim action, of which
  50 dims dominate the exploration budget. Splitting reduces the UAV action to 2 dims —
  the trajectory policy trains dramatically more easily — and concentrates the phase
  problem in one agent that can be given phase-relevant observations (S5).
- **Where the RIS agent runs**: physically, RIS control is a controller co-located with
  the surface (standard assumption); its policy can be trained federatively along with
  the UAVs or trained centrally — both variants are part of the FL study (Section 3).
- **Alternatives considered and rejected:**
  - *Per-UAV phase proposals + averaging/voting*: averaging unit-modulus proposals is
    physically meaningless (phasor averaging changes magnitudes) — rejected.
  - *Time-slicing the RIS among UAVs within a slot*: valid, but reduces to K independent
    single-UAV problems again (kills the coordination question) — kept only as a baseline.
  - *Analytic RIS (align to the scheduled weakest device each slot), learn trajectories
    only*: strong, simple, and honest — **kept as a mandatory baseline** ("MAPPO-traj +
    aligned RIS"); if the learned RIS agent cannot beat it, the paper's RIS-learning
    claim dies, and we must know that early (roadmap M3 gate).
  - *One centralized agent with a (2K+L)-dim action*: baseline for the scalability
    argument; expected to train poorly at K≥3 — include as "Joint-PPO" baseline.

### D2 — Action spaces

UAV agent: a ∈ [−1,1]², tanh-squashed Gaussian (proper change-of-variables log-prob),
mapped to velocity ≤ v_max. RIS agent, two variants: (i) continuous θ ∈ [−1,1]^L
(squashed Gaussian), (ii) discrete codebook (e.g., 64 pre-computed steering
configurations toward sector centroids + identity) with a categorical policy — variant
(ii) is both hardware-realistic (S6) and much easier to explore; make (ii) the default
and (i) the ablation.

### D3 — PPO correctness fixes *(fixes audit F7; all mandatory)*

1. Joint log-prob = **sum** over action dims (not mean); entropy likewise summed.
2. tanh-squashed Gaussian with log-det-Jacobian correction; store the pre-squash action;
   never clamp post-hoc.
3. LR schedule horizon = actual number of PPO updates (or drop annealing; PPO with fixed
   3e-4 + entropy floor works at this scale).
4. Value-clipping "old value" = stored behavior-time value from the buffer, not
   `returns − normalized advantages`.
5. Per-agent GAE with a shared reward is fine; keep advantage normalization per batch.
6. Keep: orthogonal init, grad clipping, entropy annealing with floor, best-model
   checkpointing on *held-out-seed* evaluation (S4).

### D4 — Reward

```
r[t] = EE[t] / EE_ref                      (EE_ref = fixed-trajectory baseline EE, precomputed)
     + w_p · Σ_k Δd_goal_k[t] / D_diag     (w_p ≈ 0.05–0.1, re-tuned; navigation shaping)
     − w_c · n_violations[t]               (counted pre-projection, S7)
```

Normalizing by a precomputed reference EE removes the magic "×100" scale constant and
makes w_p interpretable across parameter sweeps. The v8 finding that progress-weight
balance is critical (README "single most important lesson") transfers; the sensitivity
sweep is an ablation in [04](04-experimental-methodology.md).

### D5 — Vectorized simulator *(fixes audit F14)*

Channels as batched complex tensors: h_dev→RIS ∈ ℂ^{I×L} (per episode), h_RIS→UAV ∈
ℂ^{K×L} (per slot), h_direct ∈ ℂ^{I×K} (per slot); rates for all (i,k) pairs in one
einsum + logsumexp-free vectorized log2. No Python loops over devices, no `.item()` in
the hot path, physics in float32 on CPU or GPU (T4 fp64 is 1/32 rate). Target: ≤50 ms per
episode (≈75× vs v8), which is what makes the multi-seed campaign affordable.
Restructure the single script into a package (`env/`, `agents/`, `fl/`, `baselines/`,
`experiments/`, `tests/`) with YAML/dataclass configs — sweeps must construct configured
objects, never mutate globals (v8 mutates `V_MAX`/`P_MAX`).

---

## 3. Federated learning pipeline (the paper's central axis)

### 3.1 Why FL here is not decoration

In v8, "centralized training" means: every step, all observations are concatenated into
a global state for one critic, and one shared actor is updated from pooled experience —
i.e., full raw-experience sharing. In a real swarm, that is a per-step communication
requirement. The FL question with actual content is:

> **How much EE do we lose, and how much communication do we save, when UAVs train
> locally and exchange only model parameters every E episodes?**

This gives the paper its measured trade-off (Gap A in
[02-novelty-and-gap.md](02-novelty-and-gap.md)) and honors the project title.

### 3.2 Fed-MAPPO training loop (proposed)

```
Each UAV k keeps: local actor π_k, local critic V_k (input: own obs history,
                  optionally + last known positions of others — state what is assumed)
RIS controller:   π_r trained at the RIS/GCS node, participates as an FL client

repeat for round r = 1, 2, …:
    each client trains locally for E episodes (PPO on its own experience)
    clients upload Δweights to the aggregator (GCS or elected UAV)
    aggregator: w ← Σ_k (n_k / n) · w_k          (FedAvg)
                with proximal term μ‖w_k − w^{r−1}‖² in each local loss (FedProx)
    broadcast w; optionally keep personal output heads local (personalization ablation)
Communication accounting: bytes_up + bytes_down per round × rounds  (report alongside EE)
```

Local critics see only local information — this is the honest decentralization; the
centralized-critic MAPPO remains in the paper as the **upper bound**, and independent
PPO with no exchange as the **lower bound**. The paper's main figure is EE (and
convergence speed) vs communication budget across {IPPO, Fed-MAPPO(E), MAPPO-CTDE}.

### 3.3 FL algorithm comparison (calibrated to this setting: K=3–5 clients, on-policy RL, spatially non-IID experience, UAV energy/link constraints)

| Method | Core idea | Comm/round | Fit here | Verdict |
|---|---|---|---|---|
| **FedAvg** | weighted parameter average | 1× model | Simple; known to drift when clients' data distributions diverge — and UAV experience *is* non-IID by construction (each UAV patrols a different region) | Baseline — always include |
| **FedProx** | FedAvg + proximal term μ‖w−w_global‖² in local loss | 1× model | Directly targets client drift; in on-policy RL, unconstrained local PPO between rounds can walk policies far apart, making naive averaging destructive — the proximal term is a principled damper; one scalar hyperparameter (μ ∈ {10⁻³…10⁻¹}) | **Recommended default** |
| **SCAFFOLD** | control variates correct client drift | **2× model** (variates travel too) | Designed for many clients with partial participation; at K=3–5 full participation the variance-reduction benefit is marginal while doubling the very communication the paper is minimizing | Rejected (discuss in paper) |
| **FedNova** | normalize updates by local step counts | 1× model + scalars | Only matters when clients do *heterogeneous amounts* of local work; relevant if UAV energy budgets differ | Optional ablation (heterogeneous-E scenario) |
| **Hierarchical FL** | edge aggregators below a cloud | ≥1× at two tiers | Needs a hierarchy; with one swarm and one GCS there is none. Relevant only for multi-swarm future work | Rejected |
| **Asynchronous FL** | clients update the server whenever ready | 1× model, staggered | Solves straggler problems that K=3 synchronized UAVs on a shared mission don't have; staleness interacts badly with on-policy PPO (stale policies ⇒ off-policy data) | Rejected (note in paper) |
| **Personalized FL** (shared trunk + private head / Ditto / FedPer) | share representation, keep per-client output layers | <1× model | Attractive: UAV roles genuinely specialize spatially; sharing the encoder while keeping personal heads often beats full averaging under non-IID; costs nothing extra in comm | **Ablation** (Fed-MAPPO-P) |

**Recommendation: Fed-MAPPO = local PPO + FedProx aggregation every E episodes, with a
personalized-head variant as the headline ablation.** Sweep E ∈ {1, 5, 10, 25} to trace
the EE-vs-communication frontier. Justification summary: FedProx is the minimal-overhead
method that addresses the one failure mode this setting is guaranteed to have
(spatially non-IID on-policy experience → client drift), while SCAFFOLD/async/hierarchical
solve problems this setting does not have and would dilute a 6-page paper.

Known risk to state honestly: federated averaging of **on-policy** RL policies is less
mature than supervised FL; naive averaging can transiently collapse performance after
each round. Mitigations, in order: small E, FedProx μ, trunk-only averaging
(personalization), and averaging critics more frequently than actors. If instability
persists, the fallback design is federated critics + fully local actors — still a valid
paper (the trade-off story is unchanged), and this fallback should be decided at the M4
gate (see roadmap).

### 3.4 What stays centralized, and saying so

Offline phase (this paper): environment is a simulator, so "federated" refers to the
*information structure* imposed on training (who sees what, what is exchanged), not to
physically distributed hardware. The paper must state this plainly — reviewers respect
an honest simulation-scope statement and punish pretending otherwise.

---

## 4. Summary of every major design decision

| # | Decision | Fixes | Cost | Publication value |
|---|---|---|---|---|
| S1 | One shared Θ[t] | F3 | env rewrite (small) | mandatory correctness; creates the real MARL coupling |
| S2 | Association + ratio-of-sums EE | F8 | small | makes cooperation/fairness results meaningful |
| S3 | Realistic geometry/link budget | F11 | parameter work + validation notebook | removes tuned-physics and near-field criticisms |
| S4 | Per-episode fading, shared generators, held-out eval seeds | F4, F5 | small | statistical validity of all claims |
| S5 | Phase-relevant observations | F10 | small | makes RIS learning possible in principle |
| S6 | Quantized phases ablation | report §5.3 | small | expected by reviewers |
| S7 | Pre-projection collision counting | F9 | trivial | honest safety claims |
| S8 | Real units (B, N₀) | F11 | trivial | bits/J means bits/J |
| D1 | K UAV agents + 1 RIS agent | F3, F10 | moderate (new agent type) | the architecture contribution (Gap B) |
| D2 | Codebook RIS action default | — | small | hardware realism + explorability |
| D3 | PPO math fixes | F7 | small | correctness |
| D4 | Normalized reward | scale hack | small | interpretable shaping |
| D5 | Vectorized package refactor | F14 | moderate | enables the whole experimental campaign |
| FL | Fed-MAPPO (FedProx, sweep E) + personalization ablation | F13 | moderate–high | **the paper's novelty (Gap A)** |
