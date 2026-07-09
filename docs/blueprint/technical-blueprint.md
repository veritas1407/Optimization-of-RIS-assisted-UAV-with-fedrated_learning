# Technical Blueprint — Fed-MAPPO for Energy-Efficient Data Collection over a Shared RIS

**Project:** Energy-efficiency optimization of RIS-assisted, UAV-enabled IoT data collection
with federated multi-agent reinforcement learning.
**Status:** Design document (no code). Implementation-level specification: an engineer or an
LLM should be able to build every component from this document without additional
architectural decisions.
**Companion documents:** the v8 audit and defect register (`docs/analysis/01-technical-audit.md`,
findings F1–F14), novelty audit (`02`), initial redesign sketch (`03`), methodology (`04`),
roadmap (`05`). This blueprint supersedes `03` where they differ, because it incorporates the
two reference papers now verified in full:

- **[J]** Jiang et al., "Energy-Efficiency Optimization for RIS-Assisted UAV-Enabled IoT
  Networks," *IEEE IoT Journal* 12(20), Oct 2025 (AO + Dinkelbach/SCA, single UAV).
- **[A]** Abualhayja'a et al., "Federated Learning-Empowered RIS-Assisted UAV Networks for
  IoT Data Collection and Optimisation," TechRxiv, May 2025 (per-area PPO + BCD + FedAvg).

---

## Stage 1 — Understanding of the Research

### 1.1 Problem

Maximize the energy efficiency (bits delivered per joule of UAV propulsion energy) of an IoT
uplink data-collection mission in which K rotary-wing UAVs fly over a field of I
battery-limited ground devices, assisted by a passive RIS that strengthens links into
shadowed regions. Decision variables: UAV kinematics (trajectory/velocity), RIS phase
configuration over time, device scheduling/association, and (optionally) per-device transmit
power and time allocation. Constraints: kinematic limits, terminal positions, inter-UAV
separation, unit-modulus (and in practice quantized) RIS phases, per-device power caps.

### 1.2 What the two reference papers actually do (verified, not paraphrased from the report)

| Aspect | [J] Jiang et al. 2025 | [A] Abualhayja'a et al. 2025 | BTP v8 (audited) |
|---|---|---|---|
| UAVs | 1 (rotary-wing, 3-D kinematics **with acceleration**, Eq. 3–4) | K=5, one per *disjoint* service area, discrete moves {L,R,F,B,Stop} | 3 in one area, continuous 2-D velocity |
| RIS | 1 ULA, L elements; **per-device phase configs applied in TDMA time fractions τᵢ[m]** — physically valid because transmissions are time-separated | 1 RIS **per area** (M=100), 2-bit phases via BCD | 1 RIS but **each UAV applies its own config simultaneously** — physically invalid (audit F3) |
| Objective | EE = time-averaged sum rate ÷ total propulsion power (ratio of sums, Eq. 12) | Number of devices fully served (no propulsion, no EE) | mean-of-ratios EE (contradicts its own report) |
| Method | AO: closed-form phases (Eq. 17) → convex (τ,P) via ς=τP relaxation → Dinkelbach+SCA trajectory | PPO per area for trajectory+scheduling; BCD for phases; **FedAvg of PPO weights** across areas | MAPPO-CTDE, 52-dim action per UAV |
| CSI | Perfect at receiver (stated assumption) | Channel gain in state | magnitudes of 4 direct links only |
| Scale | **RIS [5,9,0], devices [3,11],[6,13],[9,4],[12,6], UAV [0,0,5]→[16,16,5]** (Table II sim) | 300×300 m, UAV at 100 m, N=50, data 50 bits | same toy scale as [J] |

Three insights follow that were not visible before reading the papers:

1. **The toy geometry is inherited, not invented.** The BTP copied [J]'s simulation scene
   verbatim (same RIS position, same 4 device positions, same endpoints, 5 m altitude).
   The physical-realism criticism therefore applies to the published reference as well —
   which is an *opportunity*: a redesigned study at defensible scale is itself a visible
   improvement over the closest published work.
2. **The origin of audit finding F1 is now identified.** [J] Eq. (17) gives
   θ_opt = arg(h_diu) − arg(h_ru^H(l)) − arg(h_dir(l)) — correct *for their composite*
   h_ru^H Θ h_dir (note the Hermitian). The BTP code copied the conj into the phase formula
   but implemented the composite *without* the Hermitian, producing the verified
   misalignment. The blueprint's channel module must define the composite and the aligned
   phases from one shared convention, with a unit test asserting achievement of the
   analytic maximum.
3. **Per-device RIS configs are legitimate under TDMA** ([J]), because only one device
   transmits at a time. They stop being legitimate the moment K UAVs receive
   *simultaneously* on orthogonal channels — a passive RIS is frequency-flat and reflects
   everything with one phase profile. **The multi-UAV extension therefore changes the
   physics, not just the dimensionality.** This is the precise, defensible research gap.

### 1.3 Hidden assumptions and unrealistic simplifications in the current line of work

Carried by v8 (full register in the audit): private-RIS-per-UAV (F3), frozen fading (F4),
no device association (F8), toy scale with tuned ρ₀ and near-field-invalid steering (F11),
observation insufficient for phase control (F10). Carried by **[J] itself**: perfect CSI,
1-D ULA response depending only on the x-axis cosine, continuous phases, toy coordinates,
single UAV. Carried by **[A]**: no propulsion or EE model at all, disjoint areas (so FL
clients are physically uncoupled and FedAvg is essentially multi-task averaging), discrete
grid motion, BCD phase search per element (O(L·2^b) per iteration) assumed free of latency.

### 1.4 Why the inherited scene cannot demonstrate utility (quantified)

The motivation for the Stage-3 scenario redesign, in numbers (these arguments double as
system-model justification text for the paper):

1. **No scarcity.** Arena diagonal √(20²+20²) ≈ 22.6 m; at v_max = 15 m/s a UAV crosses
   it in ~1.5 s of a 20 s episode — ~92 % of the mission is unconstrained loitering.
   Trajectory optimization has nothing to trade off.
2. **Position-insensitive reward.** Without association, every UAV collects from all
   devices from anywhere; at 5–25 m distances with the inflated ρ₀ every link sits in the
   flat high-SNR regime of log₂(1+SNR), so moving buys almost nothing. The reward
   landscape is nearly flat in position — learned path curvature is noise around a flat
   optimum.
3. **The RIS has no job.** No device is shadowed; nothing exists that the direct link
   cannot serve. This is exactly why ρ₀ had to be raised 17 dB "for clearly visible RIS
   benefit" (v8 README). A RIS shows utility only where direct links fail.
4. **Degenerate EE.** Propulsion is ~138–168 W across the whole speed range while the
   numerator barely depends on position (point 2), so the near-optimal policy is "fly at
   the power-minimum speed (~10 m/s), anywhere" — not a demonstrable capability.
5. **The 5 m altitude is not a decision.** It is copied verbatim from [J]'s numerical
   section (UAV [0,0,5]→[16,16,5]); in [J] the scene is best read as an abstract,
   unitless algorithm benchmark. A real rotary-wing survey UAV flies 30–120 m
   (regulatory ceiling ≈120 m), which is what makes air-to-ground LoS and the propulsion
   budget physically meaningful.

Each defect maps to a Stage-3 mechanism: scale + duty cycles + association create
scarcity (1, 2); the blockage field gives the RIS a structural reason to exist (3);
realistic altitude and kinematics restore physical stakes (4, 5).

---

## Stage 2 — Literature-Level Critique and Positioning

### 2.1 Reviewer-style challenges to the current design, with verdicts

- **Is the RIS location realistic?** In [J]/v8 it is an arbitrary point in a toy field.
  Verdict: place the RIS where an RIS is economically rational — mounted on a structure
  adjacent to a *shadowed device cluster* (NLoS to most of the sky), so that it fixes a
  problem the direct link cannot. The simulator must contain an explicit blockage field
  (Section 3.6) so that "RIS-needed" regions exist by construction, not by tuning ρ₀.
- **Should the RIS be UAV-mounted?** A UAV-mounted RIS (as in parts of the literature,
  e.g., flying-RIS MEC studies cited in [J] §I-A3) adds a second mobility layer and
  couples RIS orientation to flight dynamics. It is a *different paper*; for an EE study
  of data collection, a fixed RIS isolates the phenomenon under study. Verdict: fixed RIS
  in the main study; UAV-mounted RIS listed as future work. (Rationale: one contribution
  per conference paper; mounting the RIS on a UAV would also break the shared-RIS
  arbitration question, which is the novelty.)
- **Should there be multiple RISs?** [A] uses one per area, which dissolves the sharing
  problem. Verdict: exactly one RIS shared by all K UAVs *is the point*. A two-RIS
  scalability experiment (E5 extension) is optional.
- **Should FL happen on UAVs? Should an edge server exist?** Yes and yes. Local PPO
  training runs on each UAV's compute (Jetson-class assumption, §3.9); aggregation runs
  at the ground control station (GCS), which physically exists anyway as the mission
  controller and RIS backhaul. This matches [A]'s topology but with *coupled* clients.
- **Should communication be synchronous?** FL rounds: synchronous with straggler dropout
  handling (§3.8) — at K=3–5 full participation is realistic and async adds staleness
  that hurts on-policy PPO. Control loop (RIS configuration): synchronous per slot with a
  configurable 1-slot actuation delay ablation (§4.6), because real RIS controllers have
  reconfiguration latency.
- **Should PPO be replaced?** Candidates: MADDPG (off-policy, sample-efficient,
  notoriously unstable — kept as baseline), QMIX (discrete actions only — unsuitable for
  continuous kinematics unless motion is discretized), SAC-based MARL (viable, but the
  MAPPO literature [Yu et al. 2022] and both reference papers use PPO machinery; staying
  with PPO maximizes comparability). Verdict: PPO/MAPPO retained; the *novel* algorithmic
  content is the federated layer and the RIS-arbitration agent, not the policy-gradient
  core.
- **Could another aggregation method beat FedAvg?** [A] uses plain FedAvg over
  *independent* areas. Ours are *coupled* clients with spatially non-IID on-policy data —
  precisely the client-drift regime; FedProx is the minimal fix, personalization
  (shared trunk, local heads) the strongest cheap variant. Full comparison in Stage 6.7 /
  Stage 7. SCAFFOLD doubles communication (control variates) for marginal benefit at
  K≤5 full participation; FedNova matters only under heterogeneous local-step counts
  (kept as one ablation with heterogeneous UAV energy budgets); async and hierarchical FL
  solve problems this topology does not have.

### 2.2 Adjacent recent work (searched July 2026) and the surviving gap

Recent literature confirms the crowdedness of "MARL + RIS + UAV" and the openness of the
specific combination proposed here: multi-agent DRL for RIS-aided UAV communications
(IEEE 2025), joint 3-D trajectory + phase-shift DRL (Physical Communication 2024),
federated DRL for RIS-assisted indoor multi-robot systems (arXiv 2022), federated DRL for
multi-functional RIS LEO networks (arXiv 2025), an energy-efficient RIS-UAV survey
(arXiv 2025), and RIS sharing via time-division among users/base stations in terrestrial
settings (Sensors 2024; Sensors 2022). None of these combine: **(i)** one passive RIS
*simultaneously* shared by multiple receiving UAVs with an explicit arbitration mechanism,
**(ii)** propulsion-aware EE as the objective, and **(iii)** federated (not centralized)
multi-agent training with the communication budget measured. Sources:

- https://ieeexplore.ieee.org/document/11318351/ (MADRL RIS-aided UAV comms)
- https://www.sciencedirect.com/science/article/abs/pii/S1874490724001745 (DRL 3-D trajectory + phase)
- https://arxiv.org/pdf/2207.08056 (federated DRL, RIS indoor multi-robot)
- https://arxiv.org/html/2501.11079v1 (federated DRL, MF-RIS LEO)
- https://arxiv.org/pdf/2504.02162 (survey: energy-efficient RIS-UAV networks)
- https://doi.org/10.3390/s24010178 (RIS-assisted cooperative TDMA)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9370948/ (RIS sharing among multiple BSs)

**2025–26 frontier positioning (cite, do not build):** Beyond-Diagonal RIS — elements
interconnected rather than independently phased — is being pushed for 6G IoT and UAV
non-terrestrial networks (https://arxiv.org/html/2502.03637v1,
https://arxiv.org/pdf/2409.06073, survey https://arxiv.org/pdf/2503.08826). Its extra
degrees of freedom are exactly what would let one surface serve multiple simultaneous
links better, so the related-work section should frame our element-partitioned
arbitration mode (§3.3-3) as the passive-diagonal approximation of what BD-RIS enables,
and name BD-RIS arbitration as future work. Likewise active/STAR-RIS UAV EE work
(e.g., https://www.mdpi.com/2504-446X/9/8/520) is thematically adjacent (amplifier power
enters the EE denominator, as our FL-airtime ablation does) but doubles the modeling
scope — cited, not implemented. A 2025 review of RIS-UAV system models
(https://www.sciencedirect.com/science/article/pii/S2405959525001602) anchors the
related-work matrix required by `02-novelty-and-gap.md` §5.

**Claim of record** (to be finalized after full-text due diligence per `02-novelty-and-gap.md` §5):

> Fed-MAPPO: federated multi-agent PPO in which K rotary-wing UAVs learn
> trajectories and a dedicated controller learns the *arbitration and configuration of a
> single shared RIS*, maximizing propulsion-aware EE of IoT uplink collection; we
> quantify the EE-vs-communication trade-off of federated training against
> centralized-critic CTDE and model-based baselines, under quantized phases and
> realistic geometry.

---

## Stage 3 — System Architecture (redesigned from first principles)

### 3.1 Network topology and node hierarchy

```
                        ┌────────────────────────────┐
                        │  GCS / Edge server          │
                        │  • mission control           │
                        │  • FL aggregator (FedProx)   │
                        │  • RIS-agent host (default)  │
                        └───────┬──────────┬──────────┘
              C2 + FL link (sub-6GHz)   fiber/wireless backhaul
                        ┌───────┴───┐  ┌───┴────────┐
   ┌────────────┐  ┌────┴─────┐ ┌───┴──┴───┐   ┌────┴─────┐
   │  UAV 1     │  │  UAV 2   │ │  UAV K   │   │ RIS ctrl │──[RIS: N_y×N_z planar,
   │ local PPO  │  │ local PPO│ │ local PPO│   │ (applies │   L elements, b-bit]
   │ client k=1 │  │ client 2 │ │ client K │   │  Θ[t])   │
   └─────┬──────┘  └────┬─────┘ └────┬─────┘   └──────────┘
         │ FDMA uplink (orthogonal per device)
   ┌─────┴──────────────┴────────────┴─────────────────────┐
   │  I IoT devices; subset inside a blockage/shadow zone  │
   └────────────────────────────────────────────────────────┘
```

Roles and rationale:

- **UAV k (k=1..K):** flies, receives uplink from its associated devices, runs local PPO
  training (actor π_u^k, critic V^k) on its own experience. *Why local training:* the FL
  premise; raw observations (device positions, channel measurements, trajectories) never
  leave the UAV — reduced backhaul and a privacy narrative consistent with [A].
- **RIS controller:** a microcontroller co-located with the surface (standard hardware
  assumption), applies Θ[t] each slot; the RIS *policy* π_r executes at the GCS (which has
  the backhaul to the controller) — this is the physically honest placement, since the
  passive RIS has no sensing. *Why a policy rather than BCD ([A]) or closed form ([J]):*
  with K simultaneous receivers there is no closed-form optimum, and BCD per slot needs
  global CSI at the controller each slot; a learned policy needs only the compact
  observation of §4.4. The analytic per-slot arbitration schemes remain as baselines.
- **GCS/edge:** FL aggregator; also the natural home of the centralized critic used ONLY
  in the CTDE *baseline*. *Why GCS aggregation rather than a designated UAV:* [A] allows
  either; GCS avoids single-UAV compute burden and models the common deployment. A
  UAV-aggregator variant is a robustness ablation (aggregator loss, §3.8).
- **Devices:** transmit-only sources with duty cycles (active windows per [A]'s activation
  model), FDMA orthogonal subchannels (as in [A]; removes intra-slot interference and
  keeps rate expression (Stage 6.2) exact).

### 3.2 Physical scenario (all values fixed here; config file §5.4 carries them)

| Parameter | Value | Why |
|---|---|---|
| Area | 500 × 500 m | UAV data-collection scale; trajectories become meaningful (v8's 20 m arena crossed in 1.5 s — audit §1.2) |
| UAV altitude | fixed 80 m (3-D optional extension) | air-to-ground LoS regime; consistent with [A]'s 100 m; keeps action 2-D for the main study |
| Mission | T = 100 s, M = 50 slots, δt = 2 s | slot ≪ channel drift at 20 m/s; horizon long enough for coverage patterns |
| UAVs | K = 3 (scalability: 2, 5) | matches BTP; scaling axis in E5 |
| Kinematics | v_max = 20 m/s, a_max = 5 m/s² (velocity-and-acceleration model, Eq. (3)-(4) of [J]) | restore the acceleration constraint v8 dropped; smooth, flyable paths |
| Separation | d_min = 50 m horizontal | wake/collision safety at this scale |
| Devices | I = 20 (sweep 10–40), P_max = 20 dBm, B = 1 MHz per FDMA subchannel | IoT-typical |
| Duty cycle | each device active in a contiguous window of 30 s placed u.a.r. in T (ablation: always-on) | imports [A]'s activation realism; forces temporal planning |
| Carrier | f_c = 2.4 GHz (λ = 0.125 m) | sub-6 GHz IoT band |
| RIS | 10 × 10 = 100 elements, spacing λ/2, planar, wall-mounted at 15 m height on the blockage edge | aperture 0.625 m ⇒ Fraunhofer 2D²/λ ≈ 6.3 m ≪ all link distances ⇒ far-field model valid (fixes audit F11) |
| Phases | b = 2 bits default (ablation: 1, 3, ∞) | hardware-realistic ([A] uses b=2) |
| Path loss | ρ₀ = −30 dB at 1 m; exponents: device–UAV 3.2 (NLoS-prone), device–RIS 2.8, RIS–UAV 2.2 | standard anchors; no tuned physics |
| Blockage | rectangular obstacle cluster (e.g., 100×100 m, height 25 m) shadowing a device subset: direct links crossing it get +20 dB loss and Rayleigh fading; unblocked direct links are Rician κ=5 dB | gives the RIS a *reason to exist* (Stage 2.1); geometry-driven, binary LoS test by segment–box intersection |
| Fading | RIS links Rician (device–RIS κ=3 dB, RIS–UAV κ=10 dB); blocked direct links Rayleigh | elevation-dependent LoS heuristic, standard |
| Noise | N₀ = −170 dBm/Hz, NF = 10 dB ⇒ σ² = −110 dBm over 1 MHz | real units; EE is genuinely bits/J (fixes audit F11 units) |
| Propulsion | Zeng et al. rotary-wing model, constants as in v8 (`Pp_gpu` verified correct) | continuity with [J]/v8 |

**Validation gate (before any RL):** a static link-budget notebook must show (i) blocked
devices' direct SNR below the useful range, (ii) ideally-aligned RIS restoring 20–40 % of
their rate, (iii) diminishing returns in L. If not, adjust *geometry* (blockage size, RIS
placement), never ρ₀.

### 3.3 RIS sharing: the arbitration design space (core architectural contribution)

A passive RIS is frequency-flat: with K UAVs receiving simultaneously on orthogonal
subchannels, all links see the *same* Θ[t]. Three physically valid arbitration schemes:

1. **Common configuration (default, learned):** one Θ[t] chosen to serve the system
   objective. The RIS agent's action (§4.4) selects it. Devices/UAVs whose geometry
   aligns benefit; others rely on direct links.
2. **Time-slicing (baseline, analytic):** partition each slot into K fractions; in
   fraction k, Θ aligns to the weakest scheduled device of UAV k ([J]'s per-device
   alignment reused, now valid because time-separated). Rates scale by the fraction.
3. **Element partitioning (ablation):** partition the L elements into K groups; group k
   is phase-aligned for UAV k's scheduled device; other groups contribute quasi-random
   phasors to that link. Valid because alignment, not filtering, is per-group.

*Why this is the right research axis:* scheme 2 reduces the problem to K independent
single-UAV problems (the [J] world); schemes 1 and 3 create genuine inter-UAV coupling.
Comparing them head-to-head under one EE objective is the arbitration study missing from
the literature (Stage 2.2).

### 3.4 Communication architecture and protocol assumptions

- **Uplink (devices→UAV):** FDMA, one subchannel per scheduled device, TDMA within a
  UAV's served set when it exceeds subchannels; Shannon-rate abstraction (no MCS tables) —
  standard in [J]/[A].
- **Control plane (GCS↔UAV, GCS↔RIS):** out-of-band sub-6 GHz C2 link, rate model
  R_c2(d) from the same channel machinery (LoS Rician); carries per-slot RIS observation
  uplink (UAV positions — ~48 B/slot) and Θ index downlink (⌈log₂|codebook|⌉ bits/slot).
- **FL plane (UAV↔GCS):** model deltas, uploaded between episodes over R_c2; **the bytes
  and the airtime energy of FL uploads are metered** (Stage 6.8) — communication overhead
  is a first-class metric, and an ablation charges its energy against the EE budget.
- **Synchronization:** slot-synchronous simulation; FL rounds barrier-synchronized with a
  timeout: clients missing the deadline are skipped that round (§3.8).

### 3.5 Mobility model

UAVs: double-integrator per [J] Eq. (3)–(4): position and velocity states, acceleration
action (‖a‖ ≤ a_max, ‖v‖ ≤ v_max), fixed altitude; terminal-region constraint
‖q[M] − q_F‖ ≤ r_F soft-enforced via shaping (Stage 6.5). Devices: static (mobile-device
extension noted as future work, as in [J]). Wind: optional zero-mean Ornstein–Uhlenbeck
velocity perturbation for the robustness experiment E7 only.

### 3.6 Channel model (single source of truth for env AND baselines — fixes audit F5)

Planar-array response replacing the 1-D x-projection of [J]/v8: for direction (azimuth φ,
elevation ψ) and element (n_y, n_z),
a(φ,ψ)[n_y,n_z] = exp(j·(2π/λ)·(d_y·n_y·cosψ·sinφ + d_z·n_z·sinψ)); LoS phases derive from
exact geometry. Rician mixing and path-loss anchors as §3.2. NLoS components redrawn per
episode (device–RIS static within an episode; RIS–UAV and direct links redrawn per slot —
block fading). One `numpy.random.Generator` owned by the environment instance; baselines
receive the same instance (audit F4/F5 fix). CSI realism: agents observe geometry
(positions) and *delayed, noisy* per-link SNR measurements (previous slot, additive
log-domain noise σ_csi ∈ {0, 3, 6} dB in E7) — never the complex coefficients.

### 3.7 Energy model

UAV: propulsion (Zeng model) + communication electronics P_rx = 5 W constant (small,
included for honesty) + FL transmit energy E_fl = bytes·8 / R_c2(d) · P_tx,c2 (P_tx,c2 =
1 W), charged in the E-comm ablation. Devices: transmit energy tracked as a secondary
fairness metric (bits delivered per device-joule), not in the main objective — the paper's
EE is UAV-side, as in [J]. RIS: passive; controller draw noted as negligible constant.

### 3.8 Failure handling (simulated, because reviewers ask "what if a UAV drops?")

- **FL client dropout:** per-round Bernoulli(p_drop) skip; aggregator averages present
  clients (weighted by steps). Experiment: p_drop ∈ {0, 0.2}.
- **UAV failure mid-episode (E7):** one UAV frozen at a random slot; remaining agents keep
  executing (their policies see the frozen position). Metric: EE degradation vs K−1
  system trained from scratch — measures graceful degradation of decentralized execution.
- **Aggregator loss:** round skipped entirely; training continues locally (tests that the
  FL schedule is not a single point of failure).

### 3.9 Security and hardware assumptions (stated, not overclaimed)

Honest-but-curious aggregator; FL keeps raw trajectories/observations local — a data-
locality benefit, **not** a formal privacy guarantee (no DP noise; DP-FedProx cited as
future work). Model updates assumed authenticated (C2 link security out of scope). UAV
compute: Jetson-class edge GPU (parameter budget of §4.4 networks is ≪1 M — trivially
feasible). RIS: b-bit PIN-diode elements, reconfiguration ≤ ms (1-slot-delay ablation
covers slower controllers).

---

## Stage 4 — Simulation Design (implementation-level, no code)

### 4.1 Entities and relationships

| Entity | State variables | Owned by |
|---|---|---|
| `Device i` | position q_di (3,), active window [t_on, t_off], transmit power P_i, buffer of delivered bits | Environment |
| `UAV k` | position q_k (2-D + fixed z), velocity v_k, cumulative propulsion energy, served-device set S_k[t] | Environment |
| `RIS` | element grid (N_y,N_z), current Θ[t] (L phases, b-bit), codebook (§4.4) | Environment |
| `BlockageField` | list of axis-aligned boxes; method `is_blocked(p, q) → bool` | Environment |
| `ChannelState` | per-episode device–RIS NLoS draws; per-slot RIS–UAV and direct draws; cached gains | Environment |
| `Association A[t]` | device→UAV map, per-UAV TDMA fractions | Environment (rule §4.3) |
| `FLClient k` | actor/critic weights, optimizer state, local buffer, bytes counters | Trainer |
| `FLServer` | global weights, round index, straggler mask, aggregation strategy | Trainer |

### 4.2 Timeline and episode lifecycle

```
EPISODE:
  reset(seed_ep):
    draw device–RIS NLoS; place UAVs at q_I offsets; zero velocities/energies;
    device windows from seed; emit initial observations
  for t = 1..M:                       (slot = 2 s)
    1. AGENT PHASE   : UAV agents emit accelerations a_k[t]; RIS agent emits Θ index
                       (applied at t or t+1 per actuation-delay flag)
    2. KINEMATICS    : integrate double-integrator; project separation violations
                       AFTER counting them (audit F9 fix); clip to area
    3. CHANNEL PHASE : redraw per-slot fading; compute all (i,k) gains under Θ[t]
    4. ASSOCIATION   : rule of §4.3 over *active* devices
    5. RATE/ENERGY   : per-device rates (Stage 6.2); propulsion power from ‖v_k[t]‖
    6. REWARD        : Stage 6.5; emit (obs, state, reward, done, info)
  done at t = M; info carries per-episode metrics (§4.7)

TRAINING RUN (Fed-MAPPO):
  for round r = 1..R:
    each client k: E episodes of local PPO (buffer 10 episodes → 1 update, 4 epochs)
    RIS agent: trained at GCS on its own experience stream, same schedule
    upload deltas → FedProx aggregate → broadcast; log bytes, wall time
  evaluation every r_eval rounds on HELD-OUT seed range (deterministic policies)
```

### 4.3 Association rule (environment-owned, not learned in v1)

Active devices sorted by effective gain to each UAV; each device joins
argmax_k gain(i,k) subject to per-UAV capacity N_max = ⌈I/K⌉+1; UAV k's TDMA fraction per
served device = 1/|S_k[t]|. Each device's rate counted exactly once (audit F8 fix).
Learned scheduling is deliberately out of scope for v1 (action-space growth); revisit
after E1 results.

### 4.4 Observations and actions (dimensions fixed)

**UAV agent k (obs dim 24):** own position (2, normalized), velocity (2), remaining time
(1), distance+bearing to RIS (2), distance+bearing to goal (2), other UAVs' relative
positions (2×(K−1)=4), the 3 nearest *active assigned* devices: distance, bearing,
prev-slot SNR (3×3=9), count of unserved active devices (1), agent-ID one-hot folded via
embedding is avoided — parameter-shared policy with ID feature (1), pad to 24.
**UAV action (2):** acceleration (a_x, a_y) ∈ [−1,1]², scaled by a_max; tanh-squashed
Gaussian.

**RIS agent (obs dim 4K+8):** all UAV positions+velocities (4K), centroid+count of active
blocked devices (3), previous Θ index (1), remaining time (1), prev-slot mean SNR of
blocked devices (1), pad. **RIS action:** categorical over a codebook of size
C = K·N_sector + 2: steering configurations toward N_sector = 8 sector centroids per UAV
route corridor (precomputed from geometry: for each sector-centroid/UAV-position pair,
the b-bit-quantized aligned configuration), plus "identity" and "hold". *Why codebook:*
explorable (|C| ≈ 26 for K=3), hardware-realistic, and the continuous-phase variant
(action dim L=100, squashed Gaussian) remains as an ablation. Element-partitioning mode
(§3.3-3) doubles the codebook with partitioned variants.

### 4.5 Random-seed strategy (audit F4/F12 fix)

Master seed s₀ per run → `numpy.random.SeedSequence(s₀)` spawns: net-init stream,
exploration stream, env-train stream (episode e uses child e), env-eval stream (children
10⁶+j, j = eval episode). Train seeds s₀ ∈ {0..4}; eval children disjoint by
construction. Every random draw in the simulator flows from the env stream — no module-
level `RandomState`, no fresh generators inside loops (audit F5 fix).

### 4.6 Stopping criteria and schedules

Training: fixed budget (1 500 episodes/client v1; revisit after M1 speedups), best
checkpoint by held-out eval EE (every 50 episodes, 20 eval episodes). No reward-based
early stopping (v8's is deleted — it selects on train noise). Actuation-delay flag,
CSI-noise σ, p_drop, arbitration mode: all config switches; defaults 0/0/0/common.

### 4.7 Logging, evaluation, visualization

Per-slot record (parquet/CSV): t, per-UAV position/velocity/power, Θ index, per-device
rate & association, reward terms (separately!), collision-violation count. Per-episode:
EE, throughput, energy, Jain(device bits), min-separation, comm bytes cumulative. Per-run
manifest: config hash, git commit, seed tree, wall time. Figures generated ONLY from
saved CSVs (no computation in plot code — retires the v8 pattern of training sub-agents
inside plotting functions).

**Paper figures (ranked by reviewer value):**
1. *Coverage/SNR heatmap* of the arena with blockage overlay, side-by-side no-RIS vs
   aligned-RIS — the figure that justifies the RIS's existence and the scenario design.
2. *RIS beam/phase-pattern snapshots synchronized with UAV positions* at 4–6 mission
   instants — makes the arbitration behavior (who the surface serves, when it switches)
   directly visible; no prior RIS-UAV paper found shows arbitration dynamics this way.
3. *EE-vs-communication Pareto frontier* with per-seed scatter and CI ellipses (the
   headline plot), zoomed inset on post-aggregation dips.
4. *Stacked energy-budget bars* (propulsion / receive electronics / FL airtime) per
   method — ties the FL-cost story to the EE story in one glance.
5. *Device-fairness map* — device dots colored by delivered bits, per method; plus
   learning curves with CI bands, trajectory maps over the blockage field, and
   EE-vs-{L, K, I, b, E} sweeps.

**Repo/defense (not paper):** animated episode (mp4/GIF) of trajectories + active beam +
association edges; optional interactive Plotly dashboard over the logged CSVs. The
per-slot logging schema above is deliberately sufficient to build all of these without
re-running experiments.

---

## Stage 5 — Software Architecture

### 5.1 Directory layout

```
risfed/
  config/            # dataclasses + YAML: scenario.yaml, agents.yaml, fl.yaml, exp/*.yaml
  physics/
    geometry.py      # positions, bearings, blockage boxes, LoS test, sectors
    channels.py      # array response, Rician/Rayleigh draws, gain tensor (I,K), codebook build
    propulsion.py    # Zeng model; hover-power constant exported for tests
  env/
    entities.py      # Device, UAV, RIS, Association dataclasses
    ris_env.py       # SharedRISEnv: reset/step per §4.2; owns Generator; returns dict obs
    association.py   # rule §4.3 (pure function: gains → assignment, fractions)
  agents/
    nets.py          # MLP trunks; SquashedGaussianHead (log-det-Jacobian); CategoricalHead
    ppo.py           # PPOLearner: GAE, clipped loss (log-prob SUM), value clip w/ stored values
    mappo.py         # CTDE wrapper: centralized critic on concat obs (baseline only)
    buffers.py       # per-agent rollout buffer; episode-aligned
  fl/
    client.py        # FLClient: local train E episodes, delta upload, byte metering
    server.py        # FLServer: FedAvg/FedProx/personalized (trunk-only) aggregate; dropout mask
    strategies.py    # aggregation strategy registry
  baselines/
    analytic_ris.py  # aligned phases (unit-tested vs analytic max), time-slicing, partitioning
    fixed_traj.py    # straight-line kinematics under double-integrator limits
    ippo.py, joint_ppo.py, maddpg.py
  experiments/
    runner.py        # one config+seed → artifacts dir (checkpoints, CSVs, manifest)
    registry.py      # experiment matrix E1..E8 definitions
    stats.py         # CIs, Welch tests, bootstrap
  viz/plots.py       # CSV → figures only
  tests/             # unit tests §5.3
```

### 5.2 Key interfaces (contracts the implementer must honor)

- `SharedRISEnv.step(actions: dict[str, np.ndarray]) -> (obs: dict, state: np.ndarray,
  reward: float, done: bool, info: dict)` — `actions` keyed `"uav_0".."uav_{K-1}", "ris"`;
  `info` carries every §4.7 per-slot field. The env NEVER reads module globals; all
  parameters come from its `ScenarioConfig`.
- `channels.gains(positions, theta, episode_state, slot_rng) -> np.ndarray (I,K)` — pure
  function of explicit inputs; used identically by env and every baseline (single source
  of truth, audit F2/F5 fix).
- `FLServer.aggregate(deltas: list[StateDict|None], weights: list[int]) -> StateDict` —
  `None` = dropped client; strategy chosen by config.
- `runner.run(config_path, seed) -> ArtifactDir` — the only entry point; sweeps are
  lists of configs, never global mutation (retires v8 `V_MAX`/`P_MAX` pattern).

### 5.3 Testing strategy (CI, <2 min)

Hover power = analytic constant; Rician sample moments vs κ (KS test); aligned codebook
entry achieves ≥ (quantization-loss bound)·analytic max — regression for audit F1
(the stdlib reproduction script from the audit becomes a fixture); association: every
active device counted once, capacity respected; env determinism: same seed ⇒ identical
trajectory of rewards; PPO: ratio ≡ 1 on first epoch of first update; squashed-Gaussian
log-prob vs numerical integration on a 1-D case; FL: FedAvg of identical clients is a
no-op; byte meter matches parameter-count arithmetic; smoke run: 3 episodes end-to-end.

### 5.4 Configuration & versioning

Frozen dataclasses mirrored by YAML; every artifact dir stores resolved config + git SHA.
Semantic versioning of the *scenario* (scenario-v1.0 = §3.2 values) so results are
citable as "scenario-v1.0, seed 0–4, commit …". Checkpoints: best-eval and final, per
client and per aggregation round.

---

## Stage 6 — Mathematical Framework

### 6.1 Composite channel (one convention, used everywhere)

h_tot,ik[t] = h_diu,ik[t] + h_ru,k[t]ᵀ Θ[t] h_dir,i,  Θ[t] = diag(e^{jθ₁[t]},…,e^{jθ_L[t]}),
θ_l[t] ∈ {0, 2π/2^b, …} (b-bit). Aligned configuration for target link (i,k):
θ_l* = quantize_b( arg(h_diu,ik) − arg(h_ru,k(l)) − arg(h_dir,i(l)) ) — **no Hermitian
anywhere**, matching the composite above (this pair of definitions is the F1 regression
test).

### 6.2 Rates, association, EE

R_i[t] = α_ik[t] τ_ik[t] B log₂(1 + P_i |h_tot,ik[t]|² / σ²), α_ik ∈ {0,1} association
(Σ_k α_ik ≤ 1), τ_ik = 1/|S_k[t]| TDMA fraction, active-window gate implicit in α.

**Objective (ratio of sums, per [J] Eq. 12):**
maximize η = [Σ_t Σ_i R_i[t] δt] / [Σ_t Σ_k P_prop(‖v_k[t]‖) δt]  (bits/J)
s.t. double-integrator kinematics, ‖v‖≤v_max, ‖a‖≤a_max, ‖q_k[M]−q_F‖≤r_F,
‖q_k[t]−q_j[t]‖≥d_min, b-bit phase set, P_i ≤ P_max.

### 6.3 Dec-POMDP

⟨N, S, {A^n}, P, r, {O^n}, γ⟩ with N = K+1 agents (K UAV + 1 RIS), shared reward r
(§6.5), γ = 0.99. Observations as §4.4 — deliberately *not* the full state (positions +
noisy delayed SNRs only), making the decentralized-execution claim honest.

### 6.4 Policy classes and losses

UAV: π_u(a|o) = tanh-squashed diagonal Gaussian; log π(a|o) = Σ_d [log N(u_d) −
log(1−tanh²(u_d))], a = tanh(u) — log-prob **summed** over dims (audit F7 fix). RIS:
categorical over codebook. PPO clipped loss with GAE(γ=0.99, λ=0.95), entropy bonus
annealed 0.01→0.001 (dimension-correct scale now that log-probs are sums), value loss
clipped against **stored behavior values** (audit F7.4 fix), grad-norm 0.5, Adam 3e-4/1e-3
constant (no cosine — v8's never engaged anyway).

### 6.5 Reward (shared, per slot)

r[t] = EE_slot[t]/EE_ref + w_p Σ_k Δd_goal,k[t]/D_diag − w_c n_viol[t] − w_T·1[t=M]·Σ_k
max(0, ‖q_k[M]−q_F‖−r_F)/D_diag, with EE_slot = Σ_i R_i[t]/Σ_k P_prop,k[t], EE_ref = the
fixed-trajectory baseline's mean slot EE (precomputed per scenario), w_p ≈ 0.05 (E8
sensitivity sweep), w_c = 1, w_T = 1. Rationale: normalization by EE_ref replaces v8's
"×100" magic constant and keeps term scales meaningful across sweeps.

### 6.6 Fed-MAPPO local objective and aggregation

Client k, round r, FedProx local loss: L_k(w) = L_PPO,k(w) + (μ/2)‖w − w^{(r)}‖²,
μ ∈ {0, 10⁻³, 10⁻², 10⁻¹} (μ=0 ⇒ FedAvg). Aggregation:
w^{(r+1)} = Σ_{k∈K_r} (n_k/Σn) w_k, n_k = local env steps, K_r = surviving clients.
Personalized variant: split w = (w_trunk, w_head); aggregate trunk only.
RIS agent participates as client K+1 (its policy benefits from averaging only in the
personalized/trunk sense with UAV clients — **do not average heterogeneous heads**; RIS
trunk aggregation with UAV trunks is OFF by default, ON as an exploratory ablation).

### 6.7 Communication and latency accounting

Per FL round per client: U = 2·|w|·4 bytes (up+down, fp32; delta compression noted as
future work). CTDE baseline equivalent: per-slot state exchange = Σ_n |o^n|·4 bytes ×
M slots × episodes — reported in the same MB units (this is the paper's frontier x-axis).
FL upload airtime: t_up = U·8/R_c2(d_k); energy E_fl = t_up·P_tx,c2, charged to the EE
denominator in the E-comm ablation. RIS actuation latency: Θ chosen at t applies at
t+Δ_act, Δ_act ∈ {0,1} slots.

---

## Stage 7 — Experimental Plan

Inherits `04-experimental-methodology.md` §§2–6 verbatim (≥5 seeds, 95 % CIs, held-out
eval seeds, Welch/bootstrap tests, no post-processing of measurements) with these
paper-informed updates:

- **Baselines (final set):** B1 No-RIS; B2 Random-Θ; B3 Time-sliced analytic RIS (§3.3-2 —
  this is "[J]-style, made multi-UAV-valid"); B4 Common-Θ greedy (align to weakest blocked
  device each slot); B5 Element-partitioned analytic; B6 Fixed straight-line trajectory ×
  {B1–B4}; B7 Joint-PPO (one agent, all actions); B8 IPPO; B9 MAPPO-CTDE (upper bound);
  B10 MADDPG (timeboxed); B11 **Fed-MAPPO** (FedAvg / FedProx / personalized × E ∈
  {1,5,10,25}); B12 MAPPO-traj + B4 analytic RIS (the "is learning the RIS worth it?"
  gate).
- **Headline figures:** (1) EE-vs-communication frontier {B8, B11 grid, B9}; (2)
  arbitration comparison {B3, B4, B5, learned common-Θ} vs L; (3) EE vs K and I; (4)
  quantization b sweep; (5) trajectories over blockage map; (6) FL-round convergence with
  post-aggregation dips quantified.
- **Tables:** main comparison (EE, throughput, energy, fairness, comm MB, violation rate —
  mean±CI, 5 seeds); ablation table (association on/off shows the F8 triple-counting
  inflation explicitly; actuation delay; CSI noise; duty cycle on/off; E-comm energy
  charging).
- **Reviewer-expectation checklist:** every "outperforms" claim carries a test; the B12
  gate decides whether RIS-learning is claimed; scenario-v1.0 frozen before final runs;
  negative results (e.g., if FedAvg ≈ FedProx at K=3) reported, not hidden.

## Stage 8 — Research Contributions, Ranked

| # | Contribution | Novelty | Difficulty | Impact | Effort | Verdict |
|---|---|---|---|---|---|---|
| C1 | Shared-RIS arbitration among simultaneous UAV receivers (common/time-sliced/partitioned, analytic + learned) under propulsion-EE | **High** (gap verified against [J], [A], 2024–26 search) | Medium | High (physics-correct multi-agent coupling; reusable problem definition) | Medium | **Core** |
| C2 | Fed-MAPPO with measured EE-vs-communication frontier (FedProx, personalization, dropout) | **High** in this domain (FL exists for disjoint areas [A]; not for coupled agents w/ comm metering) | Medium-High (on-policy FL instability risk; mitigation ladder §3.3 of doc 03) | High (title-consistent; venue-aligned) | High | **Core** |
| C3 | Physically defensible scenario (blockage-driven RIS raison-d'être, far-field-valid array, real units, quantized phases, duty cycles) | Medium (individually standard; jointly rare in this line — [J] itself is toy-scale) | Low-Medium | Medium-High (kills the standard rejections) | Medium | **Required** |
| C4 | Honest evaluation framework (held-out seeds, CIs, association-inflation ablation, F1-class regression tests) | Low (should be standard) | Low | Medium (credibility) | Low | **Required** |
| C5 | Learned continuous phase control (L-dim action) | Low (exists in single-agent literature) | High | Low-Medium | Medium | Ablation only |
| C6 | Learned device scheduling | Medium | High | Medium | High | **Removed** from v1 (future work) |
| C7 | UAV-mounted / multiple RIS, 3-D altitude, NOMA, imperfect-CSI estimation pipeline | Medium | High | Medium | High | Future-work section |

What is *incremental* and must not be oversold: MAPPO itself, PPO machinery, the
propulsion model, Rician channel assembly. What is *removed* relative to v8: per-UAV
52-dim actions, private-RIS physics, monotonicity post-processing, reward ×100 scaling,
in-plot sub-agent training, global-mutation sweeps.

## Stage 9 — Implementation Roadmap and Dependency Graph

Module-level plan (supersedes doc 05's M0–M5 with the blueprint's structure; timeline
unchanged at ~12–14 weeks for two part-time students + T4-class GPU):

| Order | Module | Inputs | Outputs | Depends on | Key challenge | Validation |
|---|---|---|---|---|---|---|
| 1 | `config` | — | frozen dataclasses, YAML load | — | none | round-trip test |
| 2 | `physics.geometry` | config | LoS tests, bearings, sectors | 1 | box-segment intersection edge cases | hand-computed fixtures |
| 3 | `physics.propulsion` | config | P(v) | 1 | none | hover constant |
| 4 | `physics.channels` | 1,2 | gain tensor (I,K), codebook | 2 | vectorization correctness; F1-class conventions | moments KS-test; aligned≥bound test; loop-vs-vector parity |
| 5 | `env.association` | 4 | assignment, fractions | 4 | tie-breaking determinism | count-once invariant |
| 6 | `env.ris_env` | 3,4,5 | step/reset per §4.2 | 3–5 | reward-term bookkeeping; seed tree | determinism test; smoke run |
| 7 | `baselines.analytic_ris`, `fixed_traj` | 4,6 | B1–B6 numbers | 6 | none (pure evaluation) | link-budget notebook gate (§3.2) |
| 8 | `agents.nets`, `ppo`, `buffers` | 1 | PPOLearner | 1 | squashed-Gaussian log-prob | ratio≡1 test; CartPole-style sanity env |
| 9 | `agents.mappo` (CTDE baseline), `ippo`, `joint_ppo` | 6,8 | B7–B9 trained | 6,8 | reward re-tuning on new scenario | learning curves converge; beats B6 |
| 10 | `fl.client/server/strategies` | 8 | Fed-MAPPO (B11) | 8,9 | post-aggregation dips | no-op FedAvg test; byte meter test; E2 frontier smooth across seeds |
| 11 | `baselines.maddpg` | 6,8 | B10 | 9 | stability — timebox 1 week | disclosed tuning budget |
| 12 | `experiments.runner/registry/stats`, `viz` | all | E1–E8 artifacts, figures | 6–10 | orchestration on Colab session limits | figure-from-CSV-only check; full matrix dry run at 1 seed |

```
config ─┬─ geometry ──┬─ channels ──┬─ association ──┐
        ├─ propulsion ┤             │                ├─ ris_env ──┬─ analytic/fixed baselines (gate)
        └─ nets/ppo ──┴─────────────┴────────────────┘            ├─ mappo-CTDE / ippo / joint-ppo
                                                                  ├─ fl (Fed-MAPPO)  ← ppo
                                                                  ├─ maddpg
                                                                  └─ experiments/runner → stats → viz
```

Go/no-go gates (unchanged in spirit from doc 05): link-budget gate after module 7;
RIS-learning gate after module 9+ (B12 comparison); FL-viability gate after module 10
(fallback: federated critics + local actors — pre-approved design change, not a crisis).

---

### Final note on the problem statement (the user's direct question)

The current problem statement — "maximize propulsion-aware EE of RIS-assisted multi-UAV
IoT collection with MARL" — is the *right topic* but the *wrong structure*: as inherited
from [J]'s single-UAV world, it silently assumes a per-link-configurable RIS, toy
geometry (which [J] itself uses), free CSI, and uncounted coordination communication.
The restructured statement this blueprint implements:

> Given one passive, b-bit, *shared* RIS and K UAVs collecting from duty-cycled IoT
> devices in a blockage-afflicted area, jointly choose UAV kinematics and the per-slot
> RIS arbitration/configuration to maximize bits-per-joule of propulsion energy — using
> agents trained *federatively*, with the training communication itself measured (and
> optionally charged against the energy budget).

Every clause of that sentence is a fix to a verified defect or a verified gap, and the
whole of it remains squarely the original research topic.
