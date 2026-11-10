# 01 — Technical Audit (v8)

All code references are to **`ris_uav_mappo_v8.py`** (the current version, per the authors).
Earlier versions (v4–v7) are treated as development history only and are not audited.
Report references are to `BTP_Report_Final_v3.pdf` (chapters/equations as numbered there).

Contents:

1. [Architecture and implementation as built](#1-architecture-and-implementation-as-built)
2. [Validation of the mathematical models](#2-validation-of-the-mathematical-models)
3. [Defect register F1–F14](#3-defect-register)
4. [Impact-on-claims matrix](#4-impact-on-claims-matrix)

---

## 1. Architecture and implementation as built

### 1.1 Component overview

The entire system is a single 1,870-line script with six functional blocks:

```
┌────────────────────────────────────────────────────────────────────┐
│ SECTION 2–3  Physics layer                                         │
│   constants (K=3, I=12, L=50, 20×20 m, M=20 slots, Δt=1 s)         │
│   generate_iot_positions()      4 deployment modes                  │
│   _rician_gpu / _rayleigh_gpu   per-link fading draws               │
│   precompute_channels_gpu()     channels for ALL global devices     │
│   Pp_gpu()                      rotary-wing propulsion (Zeng et al.)│
│   opt_phase_gpu()               "optimal" RIS phases (BROKEN — F1)  │
│   compute_ee_single()           per-UAV EE = Σᵢ τᵢ·rateᵢ / P_prop   │
├────────────────────────────────────────────────────────────────────┤
│ SECTION 4  RISSwarmEnv (lines 319–512)                             │
│   obs: 18-dim per UAV   state: 54-dim concat   action: 52-dim      │
│   step(): move UAVs → push-apart collisions → recompute channels    │
│           → EE per UAV → reward = 100·mean_EE + 0.4·progress − 5·coll│
├────────────────────────────────────────────────────────────────────┤
│ SECTION 5–7  MAPPO (lines 515–813)                                 │
│   ONE shared Actor(18→52) used by all 3 UAVs (parameter sharing)    │
│   ONE centralized Critic(54→1)                                      │
│   RolloutBuffer(200) → GAE(γ=.99, λ=.95) → clipped PPO, 4 epochs    │
│   reward normalization, entropy anneal 0.05→0.005, cosine LR,       │
│   value clipping, grad-norm clip 0.5                                │
├────────────────────────────────────────────────────────────────────┤
│ SECTION 8  Benchmarks (lines 816–989)                              │
│   eval_agent (deterministic policy, phase resampling across L)      │
│   fixed straight-line trajectory × {optimal RIS, random RIS, no RIS}│
│   MAPPO trajectory + random RIS                                     │
├────────────────────────────────────────────────────────────────────┤
│ SECTION 9  train_mappo() (lines 992–1111)                          │
│   1500 episodes; eval every 50 (5 det. episodes); best-model ckpt;  │
│   early stop if 100-ep reward falls >25 % below best                │
├────────────────────────────────────────────────────────────────────┤
│ SECTION 10–11  Sweeps & plots (lines 1114–1871)                    │
│   per-L sub-agents (500 eps each, L∈{4,8,16,32,64,128})             │
│   per-I sub-agents (500 eps each, I∈{2,4,6,8,10})                   │
│   9 figures incl. 8-panel summary; scenario comparison (4 modes)    │
└────────────────────────────────────────────────────────────────────┘
```

Execution flow of `main()` (lines 1841–1867): seed everything with 42 → `train_mappo()`
→ reload best checkpoint → generate all plots, which internally train 11 additional
sub-agents (~142 min of the 183 min total runtime).

### 1.2 Key structural facts a reviewer will notice immediately

- **One actor is shared by all three UAVs** (a single `Actor` instance in `MAPPOAgent`,
  lines 661–663; `select_actions` calls `self.actor` for every k, lines 701–713). The
  report (§3.6.1) states "Each agent has its own actor network with separate learned
  parameters" — this is **not what the code does**. Parameter sharing is a legitimate
  MAPPO variant (Yu et al. use it), but the report's description is wrong, and it changes
  the interpretation of "each agent develops a slightly different behavioral pattern"
  (report §4.4): behavioral differences come only from observations, not from parameters.
- **Each UAV outputs its own 50 RIS phases** and the environment applies each UAV's
  phases to *its own private copy* of the RIS (see F3). There is no arbitration.
- **The mission is trivial relative to the dynamics**: v_max=15 m/s crosses the 20 m arena
  diagonal (~22.6 m) in ~1.5 s, but the episode is 20 s. The goal constraint consumes a
  tiny fraction of the mission; most of the episode is unconstrained loitering. This makes
  the "navigation vs. EE trade-off" framing weaker than the report suggests.
- **The environment is deterministic given positions** (see F4): all fading is drawn from
  seeds that do not depend on the episode. Training, evaluation, best-model selection, and
  all reported results use the *same frozen channel realization field*. There is no
  train/test split of randomness anywhere in the pipeline.

### 1.3 Observation, action, reward as implemented (vs. report)

| Element | Code (v8) | Report | Match? |
|---|---|---|---|
| obs[0–3] | own x,y,vx,vy normalized | same | ✅ |
| obs[4–6] | distance to devices **0,1,2 by index** (lines 484–485: `self.q_iot[i]` for `i in range(min(n,3))`) | "distances to the three **nearest** IoT devices" (§3.5.2) | ❌ |
| obs[7] | `n_devices/12` (constant within an episode) | "count of how many devices are still unserved" (§3.5.2) | ❌ — nothing is ever "served"; there is no serving state |
| obs[8] | distance to RIS | same | ✅ |
| obs[9–12] | log‖h_direct‖ to devices **0–3 by index** (lines 495–497) | "channel estimates to the four **nearest** devices" | ❌ |
| obs[13–16] | positions of **both** other UAVs | Eq. (3.14) lists **one** other UAV (and its listed entries total 16, not 18) | ❌ report internally inconsistent |
| obs[17] | distance to goal | same | ✅ |
| action | [vx,vy, θ₁..θ₅₀] ∈ [−1,1]⁵² | same | ✅ |
| reward | `100·mean_k(EEₖ) + 0.4·Σ progress − 5·n_coll` (lines 452–459) | same (Eq. 3.18) | ✅ |
| EE | **mean over k of** (Σᵢ τᵢ Rᵢₖ / Pₖ) | Eq. (3.7): **ratio of sums** ΣₖΣᵢRᵢₖ / ΣₖPₖ | ❌ different quantity |
| per-device power | `p_i = P_MAX/n`, `τ_i = 1/n` (lines 305–308) | "each device transmits at fixed maximum power Pmax = 1.0 W on its own orthogonal channel" (§3.1); §4.8: "no shared power budget" | ❌ direct contradiction |

The last two rows matter: the report's Eq. (3.5) has no τ and no power splitting, so **the
report's math does not describe the code that produced the report's figures**.

---

## 2. Validation of the mathematical models

### 2.1 What is correct

- **Propulsion power (report Eq. 3.6 / `Pp_gpu`, lines 257–264).** Matches the standard
  rotary-wing model of Zeng, Xu & Zhang (TWC 2018) with the paper's own constants
  (P_B=79.8563 W, P_I=88.6279 W, U_tip=120 m/s, v₀=4.03 m/s, d₀=0.6, s=0.05, ρ=1.225,
  A=0.503 m²). Hover power ≈ 168.5 W is verified at startup. ✅
  **Erratum (2026-11-10): this check was wrong.** Only hover power was verified. The
  induced-power term in `Pp_gpu` computes P_I·√(1 + v⁴/4v₀⁴ − v²/2v₀²) instead of Zeng's
  P_I·(√(1 + v⁴/4v₀⁴) − v²/2v₀²)^½ (missing inner square root). Hover is identical
  (168.5 W), but at 10 m/s v8 gives 331.8 W instead of 126.0 W, and at 15 m/s 689.5 W instead
  of 138.5 W, so v8 strongly over-penalises forward flight and every v8 EE value is affected.
  v9 (`v9/physics.py::propulsion_power`) implements the correct formula; unit tests pin hover
  168.5 W and minimum 126.0 W at 10.2 m/s. ❌
- **Rician fading normalization (`_rician_gpu`, lines 219–227).** LoS/NLoS mixing
  √(K/(K+1))·h_LoS + √(1/(K+1))·CN(0,1) with unit-power NLoS — correct second-order
  statistics. ✅
- **Composite channel structure (report Eq. 3.4 / `gamma_all_gpu`, lines 275–282).**
  h_total = h_direct + Σₗ h_RIS→UAV,ₗ·φₗ·h_dev→RIS,ₗ — the standard cascaded-RIS model. ✅
- **GAE and clipped-PPO skeleton (lines 638–653, 734–792).** The advantage recursion,
  done-masking, advantage normalization, clipping structure, entropy bonus, and gradient
  clipping are all structurally standard. (Specific defects inside this skeleton: F7.)
- **CTDE shape.** A centralized critic over the concatenated observations with
  decentralized actors is a faithful MAPPO layout (modulo the single shared actor noted
  above).

### 2.2 What is not physically defensible even before the bugs

- **Geometry.** 20 m × 20 m arena, 5 m altitude, 3 m separation, RIS 50 elements. With
  element spacing d = `D_SEP` = 0.1 m and λ = `LAM` = 0.1 m (lines 107–108):
  - spacing = **1.0 λ** → grating lobes; standard designs use ≤ λ/2;
  - aperture D ≈ 50 × 0.1 = 5 m → Fraunhofer distance 2D²/λ = **500 m**, while every link
    in the scenario is 5–25 m. All receivers are deep in the **near field**, where the
    far-field steering-vector model used in `_los_gpu` (lines 211–216) is invalid.
  - `_los_gpu` computes the array response from the **x-axis projection only**
    (`phi = (rx[0]−tx[0])/d`), i.e., a 1-D ULA with azimuth-cosine from one coordinate.
    A real RIS is planar (2-D) and the response depends on both azimuth and elevation.
- **Path-loss anchor.** ρ₀ = 5×10⁻² (−13 dB at 1 m, line 110) versus the conventional
  −30 dB (10⁻³). The README openly states it was "raised in v7 for clearly visible RIS
  benefit." Tuning a physical constant until the proposed method's advantage becomes
  visible is exactly the kind of thing reviewers reject papers over; at minimum it must
  be justified against a link-budget calculation, and the honest fix is a realistic
  geometry where the RIS gain appears at physical parameter values (see redesign doc).
- **Units.** Rates are `log2(1+SNR)` in bits/s/Hz (no bandwidth anywhere), yet EE is
  reported in "bits/J". The numbers are actually bits/s/Hz per watt. σ² = 10⁻¹⁰ W is
  quoted without a bandwidth, so the noise figure/bandwidth pair it implies is undefined.
- **Interference.** Orthogonal channels are assumed (fine), but then the power-splitting
  `p_i = P/n` contradicts orthogonality's usual point (each device could use full power).
  Either model is defensible; the code and the report must pick one and agree.

### 2.3 Verification evidence for the phase-alignment bug (F1)

Self-contained reproduction (Python stdlib only; no dependencies):

```python
import random, cmath, math
random.seed(0)
L = 50
def cn():  # CN(0,1)
    return complex(random.gauss(0,1), random.gauss(0,1))/math.sqrt(2)
h_dir = [cn() for _ in range(L)]   # device->RIS
h_ru  = [cn() for _ in range(L)]   # RIS->UAV
h_diu = cn()                       # direct device->UAV

def gain(theta):
    tot = h_diu + sum(hr*cmath.exp(1j*t)*hd for hr,t,hd in zip(h_ru,theta,h_dir))
    return abs(tot)**2

# Code's formula (opt_phase_gpu): theta = ang(h_diu) - ang(conj(h_ru)) - ang(h_dir)
th_code = [cmath.phase(h_diu) - cmath.phase(h_ru[l].conjugate()) - cmath.phase(h_dir[l]) for l in range(L)]
# Correct coherent alignment:      theta = ang(h_diu) - ang(h_ru) - ang(h_dir)
th_true = [cmath.phase(h_diu) - cmath.phase(h_ru[l]) - cmath.phase(h_dir[l]) for l in range(L)]
gmax = (abs(h_diu) + sum(abs(h_ru[l])*abs(h_dir[l]) for l in range(L)))**2
rnd  = sum(gain([random.uniform(0,2*math.pi) for _ in range(L)]) for _ in range(300))/300
print(f"code 'optimal' gain : {gain(th_code):10.2f}")
print(f"true optimal gain   : {gain(th_true):10.2f}")
print(f"theoretical maximum : {gmax:10.2f}")
print(f"random-phase mean   : {rnd:10.2f}")
```

Output:

```
code 'optimal' gain :      14.10
true optimal gain   :    1566.14
theoretical maximum :    1566.14
random-phase mean   :      46.70
```

Derivation: the composite reflected term is h_ru,ₗ·e^{jθₗ}·h_dir,ₗ, whose phase is
∠h_ru,ₗ + θₗ + ∠h_dir,ₗ. Coherent combining with the direct path requires
θₗ = ∠h_diu − ∠h_ru,ₗ − ∠h_dir,ₗ. The code (lines 267–272) computes
`ang_ru = angle(conj(h_ru)) = −∠h_ru`, hence θₗ = ∠h_diu **+** ∠h_ru,ₗ − ∠h_dir,ₗ,
leaving a residual error of 2·∠h_ru,ₗ per element — the terms do not align. In the actual
Rician-K=5 setting the damage is partially masked (LoS phases are structured), but the
baseline is still fundamentally mis-steered.

---

## 3. Defect register

Severity: 🔴 fatal (invalidates reported results) · 🟠 major (biases results or blocks
publication) · 🟡 minor (should fix; limited quantitative impact).

### 🔴 F1 — "Fixed trajectory + Optimal RIS" baseline is mathematically wrong

`opt_phase_gpu`, lines 267–272. Conjugate applied to the wrong term (§2.3 above). The
strongest baseline in every EE-vs-L figure is not an optimal-RIS bound; in i.i.d. fading it
performs **below random phases**. Consequences: (a) the baseline curves in
`mappo_ee_vs_L.png` and README tables are meaningless; (b) the fact that this *broken*
baseline still beats MAPPO at L ∈ {8, 32, 64} (README table) is silently devastating for
the learned policy; (c) any claim of the form "MAPPO approaches the optimal RIS
configuration" is unsupported.

### 🔴 F2 — Reward and reported EE computed on the wrong device set in all sweeps

`env.step` → `compute_ee_single` (line 434) → `precompute_channels_gpu` (lines 297–298),
which builds channels from the **module-level** `Q_IOT` / `I_DEV = 12` (lines 246–253),
ignoring `env.q_iot` and `env.n_devices`. `n_dev=None` at the call site means `n_i = 12`
always. Therefore, in every environment constructed with non-default devices:

- **Scenario comparison** (`plot_scenario_comparison`, lines 1571–1618): "uniform /
  clustered / edge / gaussian" all have their EE computed on the *same fixed 12-device
  default layout*. The observations change with the scenario; the reward and the plotted
  metric do not. The report's headline generalization result ("spread < 0.002 bits/J
  across four deployment modes", §4.10/§5.2) is an artifact of this bug, not evidence of
  generalization.
- **Device sweep** (`plot_ee_vs_devices`, lines 1439–1485): the MAPPO curve is evaluated
  with 12 devices at P/12 each regardless of the nominal I, while the fixed-trajectory
  baseline (`benchmark_fixed_traj_optimal_ris(L, n_devices=n_dev)`) correctly uses I
  devices at P/I. The "Fixed dominant for I<12, MAPPO wins at I=12" narrative in the
  README compares two different physical systems at every point except I=12.

For the main configuration (default 12 devices) the bug has no effect, so the training
run itself is internally consistent — but §4.8 and §4.10 of the report are invalid.

### 🔴 F3 — The single physical RIS is simulated as multiple private RISs

Each UAV k applies **its own** θ-vector: `step()` collects per-UAV `thetas[k]` (lines
405–413) and evaluates `compute_ee_single(..., thetas[k], ...)` per UAV (lines 430–435).
Three different phase configurations are active simultaneously in the same time slot on
the same surface — physically impossible for one passive RIS. The baseline goes further:
`opt_phase_gpu` returns Θ of shape (I, ·, L), i.e., **every device** gets a private
optimal configuration simultaneously (used at lines 896–897).

This is not a nitpick; it deletes the paper's core premise. The stated reason multi-agent
learning is needed here is that "the RIS phase that is best for UAV 1's position might
not be good for UAV 2" (report §1.2). In the code that conflict never exists, so (a) the
agents have nothing to coordinate over via the RIS, (b) the reported EE is an unphysical
upper bound, and (c) the "implicit coordination through shared reward" claim (§2.5) is
untested.

### 🔴 F4 — No channel stochasticity across episodes; observation/reward channel mismatch

- Observation-side channels: `_recompute_channels` seeds `42+k·7+slot·31` and
  `99+k·13+slot·31` (lines 360–364) — deterministic in (k, slot), **identical in every
  episode**.
- Reward-side channels: `compute_ee_single` is called with `seed_ris=42+k·7`,
  `seed_direct=99+k·13` (lines 434–435) — no slot term at all, so the fading used for EE
  is **frozen across slots and episodes**; the only thing that varies is geometry.
- The two sets use different seeds, so the channel magnitudes the agent observes are
  *not* the channels that generate its reward.

Consequences: training, evaluation, best-model selection, early stopping, and every
reported number all use one frozen fading realization (train = test); "Rayleigh/Rician
fading" in the report describes a distribution that is sampled once, ever; robustness and
generalization claims have no statistical basis; and the best-model checkpointing (eval
every 50 eps on the same frozen field) is selection on the training distribution.

### 🔴 F5 — Baseline channels are statistically different from environment channels

In `benchmark_fixed_traj_optimal_ris` (lines 887–895) and `benchmark_no_ris`
(lines 951–954), a **fresh `np.random.RandomState(42+k·7)`** is constructed *inside* the
per-device list comprehension, so every device receives the *same* NLoS draw (and h_ru is
drawn from the same generator seed again). In the environment (`_recompute_channels`,
lines 365–372) a single generator is passed through the comprehension, so devices get
independent draws. The MAPPO curves and the baseline curves are therefore evaluated under
different channel statistics — the comparison is not apples-to-apples even where the
formulas are right.

### 🔴 F6 — Post-hoc "monotonicity enforcement" rewrites result data

Lines 1322–1325:

```python
# Enforce monotonicity (smoothing artefacts from short training)
for i in range(1, len(ee_mappo)):
    if ee_mappo[i] < ee_mappo[i-1]:
        ee_mappo[i] = ee_mappo[i-1]
```

Measured EE-vs-L values for the proposed method are overwritten to force a monotone
increasing curve before plotting, and this is disclosed nowhere in the report. Under any
venue's standards this is data manipulation. It must be removed and never replaced with
anything but honest error bars. (Note the README's own table — 0.08427 at L=16 vs
0.08420 at L=32 — contradicts the plotted monotone curve.)

### 🟠 F7 — PPO mathematics deviates from PPO in three ways

1. **Joint log-probability divided by the action dimension.**
   `dist.log_prob(action).mean(-1)` (line 556 in `select_action`; line 763 in `update`)
   instead of `.sum(-1)`. The importance ratio becomes
   exp((Σ Δlogπ)/52): policy changes are under-measured by a factor of 52 in the
   exponent, so the clip at ±0.2 corresponds to an enormous true policy change. Training
   "works" but it is not PPO's trust region; combined with the separate log-ratio clamp
   (line 766) the update rule is ad hoc.
2. **Clamped-action / log-prob mismatch.** `select_action` computes log-prob at the
   *unclamped* sample, then clamps the action to [−1,1] (lines 555–559) and the clamped
   action is stored and replayed in `update` (line 763). Old and new log-probs are
   evaluated at different points whenever the sample saturates — which is often, since
   the mean head is tanh-squashed toward ±1 and σ≈0.6. Standard fixes: store the
   pre-clamp action, or use a tanh-squashed Gaussian with the change-of-variables
   correction.
3. **The advertised LR schedule never happens.** Cosine schedulers with `T_max=1000`
   *updates* (lines 667–670), stepped once per PPO update (line 798). Total updates =
   1500 episodes × 20 steps / 200 buffer ≈ **150**, so the LR decays only ~5 % of the way
   down the cosine; it never approaches the documented 1×10⁻⁵ floor (README, report
   Table 3.1 both claim it does).
4. (Sub-item) **Value clipping uses a wrong "old value".** `v_old = returns − advantages`
   (line 745) — but `advantages` were *normalized* in `compute_gae` (line 652) after
   returns were formed, so `v_old` is not the behavior-time value estimate; the stored
   `values` are available in the buffer and unused for this purpose.

### 🟠 F8 — No device–UAV association: throughput is triple-counted and cooperation is unincentivized

Every UAV sums the rates of **all** devices (lines 306–308), and the mean over UAVs is
rewarded. Physically, a device's uplink stream received by three UAVs is not 3× the data.
Formally, the report's own objective Eq. (3.7) also sums Rᵢₖ over all (i,k) pairs — the
formulation, not just the code, needs an association or scheduling variable. Consequences:
(a) the EE numerator is inflated ~3×; (b) no division of labor is ever required — each
UAV independently maximizes the same quantity, so the "cooperation" and "load balancing"
findings (report §4.9: near-equal per-UAV EE, Jain ≈ 0.999) are what symmetric geometry
produces by default, not evidence of learned coordination; (c) code computes
mean-of-ratios while Eq. (3.7) defines ratio-of-sums — different objectives with
different optima (a UAV hovering at zero speed changes the two very differently).

### 🟠 F9 — Collision avoidance is done by the environment, not the policy

`step()` applies a geometric push-apart that guarantees separation (lines 416–427) and
*then* counts collisions (lines 439–444), so `n_collisions` is almost always 0 and the
−5 penalty almost never fires (the only path to a counted collision is the post-push
boundary re-clip). The report's claims that "the UAVs maintain safe separation …the reward
penalty plus the push-apart correction are enough" (§4.4) and "It does not need a separate
collision avoidance controller" (§4.11) are contradicted by the code: the push-apart *is*
a separate collision controller, and the learning signal for avoidance is effectively off.

### 🟠 F10 — The observation cannot support the RIS-control task

Agents output 50 phases but observe only four direct-link log-magnitudes (lines 494–497)
— no phases of any link, no cascaded-channel information, not even magnitudes of the
device→RIS or RIS→UAV links. Per-realization coherent alignment is therefore
**information-theoretically impossible** for this policy class; the best any policy can do
is geometry-based (LoS) alignment inferred from its own position. This single fact
explains the small and erratic learned-RIS gains (and why the broken baseline still often
wins). Any redesign must either provide phase-relevant features (positions are sufficient
for the LoS component; K=5 means ~83 % of link power is LoS) or explicitly frame the
problem as statistical/blind phase control and calibrate claims accordingly.

### 🟠 F11 — Scale, units, and report/code contradictions

Collected from §1.3 and §2.2: near-field geometry with far-field model, λ-spaced elements,
tuned ρ₀ = 5×10⁻² with the tuning motive documented in the README, "bits/J" without
bandwidth, power-splitting vs. per-device-Pmax contradiction, "nearest devices" vs.
by-index devices, Eq. (3.14) dimension inconsistency (16 listed entries vs 18 used),
report §4.3 training EE ≈ 0.043 vs README/report §4.5–4.7 values ≈ 0.082–0.101 with no
reconciliation, report §4.8 sweep {2,4,6} vs v8 code {2,4,6,8,10,12}, and §3.6.1
"separate learned parameters" vs a single shared actor.

### 🟠 F12 — Evaluation methodology cannot support the claims

- **Single seed (42) for everything** (acknowledged in README §Known Limitations 5) and
  2–10 evaluation episodes per plotted point; no error bars on any curve except the
  per-UAV bars; no significance tests.
- **Confounded sweeps**: per-L and per-I points come from *different agents trained for
  500 episodes* while the L=50/I=12 point is the 1500-episode main agent (lines
  1305–1310, 1444–1449) — curve shape mixes convergence differences with the swept
  variable (README's Known Limitation 3 admits this for I).
- **Speed sweep is out-of-distribution in two ways** (`plot_ee_vs_speed`, lines
  1346–1370): mutating global `V_MAX` rescales both the action→velocity map and the
  velocity observation normalization, so a policy trained at 15 m/s is evaluated with
  silently rescaled dynamics and observations. The monotone EE-vs-speed curve (report
  §4.6) measures this distortion as much as any physical effect.
- **Two different methods produce "the same" figure**: `mappo_ee_vs_L.png` uses per-L
  sub-agents, while summary panel (c) (lines 1716–1728) evaluates the main agent on other
  L via `select_action_resample` linear interpolation of phases (lines 563–576) — a
  physically meaningless operation on phase vectors. The two versions of the figure will
  disagree.
- **By the project's own numbers the learned policy loses to a straight line at most
  operating points**: Fixed+Opt ≥ MAPPO at L ∈ {4, 8, 32, 64} (README EE-vs-L table) and
  Fixed > MAPPO for all I < 12 (README device table). Headline gains (7–12 %) are quoted
  only against No-RIS. Reviewers will read the tables.

### 🔴 F13 — The federated learning component does not exist

The repository is named "Optimization-of-RIS-assisted-UAV-with-fedrated_learning" and the
research topic includes FL; reference [11] (the only cited multi-UAV work) is an FL paper.
There is no federated anything in the code — one shared actor, one centralized critic,
centralized training (report §5.3 admits "Training is fully centralized"). As it stands
the project does not implement its own title. This is the single largest gap between
promise and delivery, and — as argued in [02-novelty-and-gap.md](02-novelty-and-gap.md) —
also the most promising direction for genuine novelty.

### 🟡 F14 — Performance/scalability bottlenecks block the required experimental campaign

Channel generation is per-device, per-UAV, per-slot Python loops with `.item()` GPU
syncs and per-call `RandomState` construction (lines 219–254, 360–375); rates are
accumulated in a Python loop over devices (lines 306–308). Result: ~3.8 s/episode,
~183 min per full run on a T4. The corrected methodology needs ≥5 seeds × ~10
configurations; at current speed that is ~150 GPU-hours. Vectorizing channels as
(I×L) complex tensors and rates as batched ops is straightforward and should yield
10–100×. Minor additional issues in the same category: `torch.cuda.empty_cache()`
sprinkled after every plot (an anti-pattern), float64 physics on GPU (T4 has 1/32-rate
fp64 — physics would be faster on CPU or in fp32), and global-variable mutation
(`V_MAX`, `P_MAX`) as the sweep mechanism, which makes the script non-reentrant and
non-parallelizable.

---

## 4. Impact-on-claims matrix

| Claim (report/README) | Where | Status after audit |
|---|---|---|
| "RIS consistently helps once L ≥ 16" | §4.5, README | ❓ Unverifiable — F1 (broken opt baseline), F5 (mismatched statistics), F6 (data rewritten), F12 (single seed) |
| "Generalizes across 4 deployment scenarios without retraining (<0.002 spread)" | §4.10, §5.2 | ❌ Invalid — F2: metric computed on the same device layout in all scenarios |
| "MAPPO beats No-RIS at every P_max (+7–12 %)" | README | ⚠️ Plausible but unproven — comparison is against straight-line trajectories under different channel statistics (F5), single frozen realization (F4), single seed (F12) |
| "MAPPO wins vs fixed trajectory at I=12" | README | ⚠️ Only point not invalidated by F2, but same F4/F5/F12 caveats; and MAPPO loses at all I<12 |
| "Agents cooperate; load is balanced (Jain ≈ 0.999)" | §4.9 | ❌ Unsupported — F8: no association ⇒ no cooperation is needed; symmetry explains the result |
| "Policy learned collision avoidance" | §4.4, §4.11 | ❌ Unsupported — F9: environment enforces separation |
| "Joint UAV-trajectory + RIS-phase learning" | throughout | ⚠️ Partially — trajectories are learned; per-realization phase control is impossible from the given observation (F10) and the RIS is not shared (F3) |
| "Realistic channel and propulsion models" | Abstract, §3 | ⚠️ Propulsion model yes but v8 code has a formula bug (see erratum in §2.1); channels structurally yes but frozen (F4), near-field-invalid (F11), tuned ρ₀ (F11) |
| "Training is stable over 1500 episodes" | §4.3 | ✅ Supported by the learning curve (with the caveat that the objective it optimizes has the defects above) |

**Bottom line:** the pipeline runs and trains stably, and the engineering iteration
discipline (46 documented changes) is genuinely good practice — but F1–F6 mean the
central quantitative results cannot be trusted, and F13 means the titled contribution is
absent. Everything in the fix path is tractable; see
[03-redesign-proposal.md](03-redesign-proposal.md) and
[05-roadmap-and-readiness.md](05-roadmap-and-readiness.md).
