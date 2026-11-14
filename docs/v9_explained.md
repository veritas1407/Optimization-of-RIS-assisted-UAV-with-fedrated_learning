# The v9 project, explained from the ground up

*Written 2026-11-09; Part 13 updated with the final 5-seed campaign results. Everything here
describes the code on branch `claude/btp-project-analysis-425r7n` (package `v9/`). Numbers
quoted are measured, not assumed; where a number will only exist after the campaign, this
document says so.*

How to read this: each part starts with the one-paragraph intuition ("explain it to a
friend"), then gives the precise version ("explain it to an examiner"). If you only read the
first paragraph of every part you will understand the project; the rest lets you defend it.

---

## Part 0 — The whole project in one story

A city block has a courtyard enclosed by tall buildings. Sensors on the ground (IoT devices)
have data to upload — say a batch of camera images. Three drones (UAVs) fly over the city for
100 seconds, collect the data by radio, and then land at a collection point. Flying costs
energy, so we want **as many bits as possible per joule of flight energy**. That ratio is the
*energy efficiency* (EE, in bit/J) and it is the single number everything is judged by.

Problem: sensors inside the courtyard are *shadowed*. The buildings block the radio path to a
drone unless the drone is almost directly overhead. So a drone would have to detour to every
courtyard, which wastes energy.

Fix: bolt a **Reconfigurable Intelligent Surface (RIS)** — a 1 m × 1 m panel of 256 tiny
programmable reflectors — high on the courtyard wall. The RIS catches the weak signal from the
courtyard and re-aims it at the drone, wherever the drone is. Now the drones can collect
courtyard data from far away.

Two decisions remain: **where should each drone fly** (a hard sequential decision → we use
reinforcement learning, RL), and **how should the RIS reflectors be set** (an easy one-shot
calculation → we use a closed-form formula). And because the project is about **federated
learning (FL)**, we also ask: can the drones (or several city areas) learn *together* by
sharing only their neural-network weights, never their raw data, and how good is that
compared to a central learner that sees everything?

Then we check everything honestly against strong hand-written rivals, with many random
episodes, several training seeds and confidence intervals.

---

## Part 1 — Why v9 exists (what was wrong before)

**Intuition.** The original code (v4–v8) copied a published toy scene: a 20 m × 20 m square,
drones at 5 m height, no buildings, devices with unlimited data. In such a tiny scene every
drone is always close to every device, so the RIS has nothing to fix, and every flight path is
roughly as good as every other. You cannot show that anything matters there.

**Precise.** The v8 audit (`docs/analysis/01-technical-audit.md`) found, among others: a phase
alignment bug in the RIS formula, PPO math issues (log-probabilities not summed, clamped
actions breaking the probability ratio), per-UAV private RIS phases that a single physical
panel cannot have, a tuned path-loss constant that made bit/J meaningless, single-seed results
and post-processed curves. (Found later: v8's propulsion code also had a formula bug — the
induced-power term misses an inner square root, so forward flight cost up to 7× too much;
hover was correct, which is why the first audit missed it. v9 uses the correct formula.)
v9 is a new package that keeps v8's correct parts (the propulsion
model's constants, the general PPO structure) and rebuilds the rest. The old scene is still available as
preset `legacy_small` so we can *show* that it could not demonstrate the RIS's value
(measured: +0.5 % EE from the RIS there; see Part 13).

---

## Part 2 — The world (the "building" scenario)

**Intuition.** A 500 m × 500 m square of city. Drones start in the south-west corner and must
end in the north-east corner. On the way there is one urban block with a courtyard. 24
sensors are scattered around; 6 of them sit inside the courtyard.

**Precise** (`v9/config.py`, `ScenarioConfig`):

| Item | Value | Why this value |
|---|---|---|
| Area | 500 × 500 m | big enough that *where* you fly matters |
| Mission | 40 slots × 2.5 s = 100 s | |
| UAVs | 3 collectors at 60 m altitude | typical low-altitude IoT collection |
| Speed / acceleration | ≤ 20 m/s, ≤ 5 m/s² | commercial quadrotor limits |
| Minimum separation | 20 m between any two UAVs | safety rule |
| Start | (20, 20), (45, 20), (70, 20) | 25 m apart |
| Final zone | circle of radius 40 m around (480, 480) | each UAV has its own landing slot inside it |
| Buildings | two 140 m × 35 m rows, 45 m tall, plus two 40 m end caps | form a 140 m × 30 m courtyard |
| Devices | 18 open-field + 6 in the courtyard, **re-drawn every episode** | the policy must generalise, not memorise |
| Data per device | 10–30 Mbit (1.25–3.75 MB) | e.g. an image or log batch |
| Active window | each device is awake for one 30 s window at a random time | sensors duty-cycle to save battery |

Distances matter: the start→final diagonal is about 650 m. At the drone's most economical
speed (10.2 m/s, Part 4) that takes ~64 s of the 100 s mission. So drones have ~36 s of
"spare" time to detour — they must choose where to spend it. That scarcity is deliberate (the
"design rules" checked in the gate, Part 13).

---

## Part 3 — Radio physics, from scratch

### 3.1 What decides how fast a sensor can upload?

**Intuition.** A radio link is like shouting across a field. How loud it arrives (received
power) versus the background hiss (noise) is the *signal-to-noise ratio*, SNR. Shannon's
formula turns SNR into a maximum data rate:

$$\text{rate} = B \log_2(1 + \text{SNR}) \quad \text{bits/s}$$

with bandwidth $B$ = 180 kHz (one NB-IoT carrier). Double the SNR and you gain roughly one
extra bit per second per hertz.

**Precise.** SNR $= P\,|h|^2 / N$ where $P$ = 10 dBm (10 mW) transmit power, $N$ = noise power
= −174 dBm/Hz + 10 log₁₀(180 kHz) + 7 dB noise figure = **−114.45 dBm**, and $|h|^2$ is the
*channel gain* — how much of the transmitted power arrives. All of the physics below is about
computing $h$.

### 3.2 Path loss — signals fade with distance

In free space the power falls with the square of distance: $|h|^2 = (\lambda / 4\pi d)^2$ at
2.4 GHz ($\lambda$ = 12.5 cm wavelength). At 1 m that is −40 dB; at 300 m it is −90 dB.

### 3.3 Cities: line of sight or not?

**Intuition.** In a city the drone is often hidden behind *some* building, especially when it
is far away and therefore low on the horizon. Close and high up → probably visible; far and low
→ probably not.

**Precise.** We use the standard urban air-to-ground model of **Al-Hourani et al. (2014)**:
probability of line of sight $P_\text{LoS} = 1/(1 + a\,e^{-b(\phi - a)})$ with elevation angle
$\phi$ in degrees and urban parameters $a$ = 9.61, $b$ = 0.16. A LoS link gets 1 dB extra loss,
a non-LoS link gets 20 dB. On top of that, the *specific* buildings of our block are modelled
as boxes: if the straight line between sensor and drone passes through a box (a "segment–box
slab test", `physics.segments_blocked`), the link gets another +20 dB. That is what shadows the
courtyard.

### 3.4 Fading — the random wobble

Even at fixed positions, reflections make the received signal fluctuate. We draw it from a
**Rician** distribution: a strong direct part plus random scattered parts (Rician factor
κ = 5 dB for drone links, 10 dB for the RIS links which are mostly line-of-sight). Every slot
gets a fresh random draw, so nothing is perfectly predictable.

---

## Part 4 — Flight energy (the denominator of EE)

**Intuition.** A helicopter-style drone burns a lot of power just to hover (it has to push air
down to stay up). Flying forward at a moderate speed is actually *cheaper* than hovering,
because the forward motion brings fresh air to the rotors. Flying very fast becomes expensive
again (air drag). So there is a sweet spot speed.

**Precise.** The rotary-wing propulsion model of **Zeng, Xu & Zhang (IEEE TWC 2019)**:

$$P(v) = P_B\left(1 + \frac{3v^2}{U_\text{tip}^2}\right) + P_I\left(\sqrt{1 + \frac{v^4}{4v_0^4}} - \frac{v^2}{2v_0^2}\right)^{1/2} + \tfrac12 d_0 \rho s A v^3$$

(blade profile + induced + parasite power). With the paper's constants: **hover 168.5 W,
minimum 126.0 W at 10.2 m/s, 178.3 W at 20 m/s**.

Within a 2.5 s slot the velocity changes linearly (acceleration-limited), so the speed is not
constant. v9.1 integrates $P(|v(t)|)$ over the slot with composite Simpson's rule (8
sub-intervals, `physics.slot_energy`), accurate to < 0.2 % of a fine numerical integral even
when the drone accelerates through hover, where $P$ curves sharply. (v9.0 used a cruder rule;
the audit caught it — Part 12.)

**EE definition.** $\text{EE} = \dfrac{\text{total bits delivered by all devices}}{\text{total propulsion energy of all UAVs}}$ — a ratio of sums over the whole mission, in real units (bit/J).

---

## Part 5 — The RIS, from scratch

### 5.1 What a RIS is

**Intuition.** Imagine a wall covered with 256 small mirrors, each of which can be rotated —
except they don't rotate physically; each one delays the reflected wave by a chosen amount
(its *phase*). If you choose the delays so that all 256 reflected waves arrive at the drone
**in step** (crests on crests), they add up constructively: 256 weak echoes become one strong
signal. Choose them badly and they cancel each other out.

**Precise.** The received signal at UAV $k$ from device $i$ is the sum of the direct path and
the reflected paths:

$$h_\text{tot} = h_d + \sum_{l=1}^{L} h_{ru,l}\, e^{j\theta_l}\, h_{ir,l}$$

$h_d$ = direct device→UAV channel, $h_{ir,l}$ = device→element $l$, $h_{ru,l}$ = element
$l$→UAV, $\theta_l$ = the phase we set. Our panel: $L$ = 16 × 16 = 256 elements at λ/2 =
6.25 cm spacing, so 1 m × 1 m.

### 5.2 Setting the phases (closed form — no learning needed)

Each term has an angle. Make every reflected term point the same way as the direct one:

$$\theta_l = \angle h_d - \angle h_{ru,l} - \angle h_{ir,l}$$

Then all magnitudes simply add: $|h_\text{tot}| = |h_d| + \sum_l |h_{ru,l}||h_{ir,l}|$. This
is the textbook optimum for one link (it is Jiang et al.'s Eq. 17 with the sign convention
corrected — v8 had it wrong). Real panels cannot set any angle; ours has **2-bit phase
shifters**, i.e. only 0°, 90°, 180°, 270°, so each $\theta_l$ is rounded to the nearest of
those (`physics.quantize`). On average rounding costs about 0.9 dB.

That is why we **do not learn the RIS phases with RL**: for a given link the best answer is a
formula. RL is used only where no formula exists — the flight paths. Correct wording for the
report: *"hybrid: RL for trajectories + closed-form RIS configuration"*.

### 5.3 How strong is the reflected path? (the element pattern)

**Intuition.** Each little element is not a perfect point mirror: it reflects best straight
out of the panel and worse at grazing angles, like a torch that is brightest in front.

**Precise (v9.1).** We use the far-field model of **Tang et al. (IEEE TWC 2021)**: each
element has a power pattern $F(\vartheta) = \cos^q\vartheta$ and gain $G = 2(q+1)$; the
cascaded path gets the factor $\pi G \cdot F(\vartheta_\text{in}) F(\vartheta_\text{out})$. We
use $q$ = 0.57, which gives $G = \pi$ — exactly the effective area of a λ/2 × λ/2 element
(Ellingson 2021). Waves only reflect into the *front* half-space of the panel.

**The panel is aimed** like an installer would aim it: its facing direction bisects the
direction down into the courtyard and the direction toward the middle of the flight zone, so
both are seen at ~40° off-axis (`config.aim_normal`).

### 5.4 A sanity check you can do by hand

Device 25–50 m from the panel, drone 300 m away: the cascaded path loss works out to about
−105 dB, i.e. SNR ≈ 10 dBm − 105 dB + 114.45 dB ≈ 20 dB → rate ≈ 180 kHz × log₂(1 + 90) ≈
1.2 Mbit/s → a 20 Mbit batch in ~17 s, inside the 30 s window. The blocked direct path (building loss + non-LoS)
is roughly 25 dB weaker. This is why the RIS is so useful here: it sits *close to the devices* (a known
property: a RIS helps most near the transmitter or the receiver) and *high up*, so its link to
the drone is line-of-sight.

### 5.5 One panel, several drones: arbitration

**Intuition.** There is only one physical panel. It can be set for one link at a time. If two
courtyard sensors are uploading to two drones in the same slot, who gets the RIS?

**Precise** (`env._communicate`, setting `arbitration`):
- **timeshare** (default): within the slot the panel cycles through up to 8 configurations,
  one per RIS-dependent link, each for an equal share of the slot (a TDMA-style split, as in
  Jiang et al.). Devices not targeted see whatever the active configuration does to them.
- **common**: one configuration per slot, aimed at the most urgent RIS-dependent link.
- **random**: random phases (control: shows that an *unaligned* RIS does nothing).
- **none**: no RIS.

Measured: timeshare and common give practically the same EE (8675 vs 8689 bit/J for the tuned
heuristic), because usually only one courtyard sensor is active at a time.

### 5.6 The aerial alternative

Preset `aerial`: instead of a wall, the same panel hangs under a 4th drone at 70 m, facing
down and tilted toward the collectors. It can move to wherever it is needed — but **its own
flight energy is added to the EE denominator** (with a 1.2× payload factor for carrying the
panel). The question it answers: *is a movable RIS worth its flight energy?*

---

## Part 6 — One slot of the simulation, step by step

Everything happens in `RISUAVEnv.step` (`v9/env.py`). Each 2.5 s slot:

1. **Read the commands.** Each UAV's policy outputs a 2-D velocity command in [−1, 1]²
   (× 20 m/s).
2. **Safety autopilot (return-to-home shield).** If a UAV has *just* enough time left to reach
   its landing slot (planning with 90 % of top speed), the autopilot overrides it and flies it
   home. Since v9.1 the autopilot also keeps the 20 m separation from the other UAVs. These
   overridden steps are *masked out* of the learning signal (the policy did not choose them).
3. **Move.** Velocity changes by at most a_max·dt = 12.5 m/s toward the command; position
   follows the average of old and new velocity; walls of the area stop the UAV.
4. **Energy.** Propulsion energy over the slot (Part 4).
5. **Separation check.** Any pair closer than 20 m (3-D) is counted as a *separation
   violation* ("collision" in the code) and pushed apart.
6. **Who uploads to whom (scheduling).** Each UAV has one receiver (one NB-IoT carrier), so it
   can serve **one device per slot**. Awake devices with data left are sorted by *urgency* =
   remaining bits ÷ time left in their window; the most urgent device is assigned to the UAV
   with the best estimated rate, then the next, until the receivers are used up. Each device's
   bits are counted once.
7. **Set the RIS** for the RIS-dependent links (Part 5.5) and compute the actual channels with
   fresh fading.
8. **Upload.** Bits = rate × 2.5 s, capped by what the device still holds.
9. **Reward** (Part 7) and the next **observation**.

At the end of the mission: EE, bits, energy, fraction of data delivered (overall and for the
courtyard devices), separation violations, and whether every UAV ended inside the final zone.

---

## Part 7 — What the drones are told to optimise (reward and Dinkelbach)

**Intuition.** We want to maximise a *ratio* (bits ÷ energy). RL is good at maximising a
*sum* of rewards. Trick: put a "price" λ on energy. Each slot, reward = bits earned − λ ×
energy spent. If λ equals the best achievable EE, then maximising "bits − λ·energy" is exactly
the same as maximising "bits ÷ energy". We don't know the best EE in advance, so we keep
nudging λ toward the EE the drones currently achieve. This is **Dinkelbach's method** for
fractional programming — the same method Jiang et al. use.

**Precise.** Per slot:

$$r_t = \frac{\text{bits}_t - \lambda\,E_t}{\lambda\,E_\text{ref}} + \text{shaping}_t - 1\cdot(\text{violations}_t)$$

$E_\text{ref}$ = 3 UAVs × 126 W × 2.5 s scales the reward to order 1. The *shaping* term
($w$ = 5) gently penalises being further from the final zone than the remaining time can
cover (a potential-based shaping, which does not change which policy is optimal). λ starts at
6000 bit/J and after every PPO update moves 10 % of the way to the EE of the last 8 training
episodes: $\lambda \leftarrow 0.9\lambda + 0.1\,\text{EE}_\text{recent}$. (v9.0 accidentally
kept λ fixed; the audit caught it.)

---

## Part 7½ — The optimisation problem, written formally (P1)

**Intuition.** Before choosing any algorithm, write down exactly what we want and what is
not allowed. Everything else in the project is a way of solving this one problem.

**Variables** (for slots $t = 1..M$, UAVs $k = 1..K$, devices $i = 1..I$):
- trajectories: positions $\mathbf q_k[t]$ and velocities $\mathbf v_k[t]$;
- scheduling: $a_{ik}[t] \in \{0,1\}$ (device $i$ uploads to UAV $k$ in slot $t$);
- RIS: configurations $\boldsymbol\theta_c[t] \in \{0, \tfrac{\pi}{2}, \pi, \tfrac{3\pi}{2}\}^L$ and their
  time shares $\tau_c[t] \ge 0$, $\sum_c \tau_c[t] = 1$.

$$
\textbf{(P1)}\quad \max_{\mathbf q, \mathbf v, \mathbf a, \boldsymbol\theta, \boldsymbol\tau}\;
\mathrm{EE} = \frac{\sum_{t}\sum_{i} b_i[t]}{\sum_{t}\sum_{k} E_k[t]},
\qquad b_i[t] = \sum_k a_{ik}[t]\,\delta \sum_c \tau_c[t]\, B\log_2\!\Big(1 + \tfrac{P\,|h_{d,ik} + \mathbf h_{ru,k}^{T}\mathrm{diag}(e^{j\boldsymbol\theta_c})\mathbf h_{ir,i}|^2}{N}\Big)
$$

subject to

| | Constraint | Meaning |
|---|---|---|
| C1 | $\lVert\mathbf v_k[t]\rVert \le v_{\max}$ = 20 m/s | speed limit |
| C2 | $\lVert\mathbf v_k[t] - \mathbf v_k[t-1]\rVert \le a_{\max}\delta$ | acceleration limit |
| C3 | $\mathbf q_k[t] = \mathbf q_k[t-1] + \tfrac{\delta}{2}(\mathbf v_k[t-1] + \mathbf v_k[t])$, inside the area | kinematics |
| C4 | $\lVert\mathbf q_k[t] - \mathbf q_j[t]\rVert \ge d_{\min}$ = 20 m | separation |
| C5 | $\mathbf q_k[0]$ = start, $\lVert\mathbf q_k[M] - \mathbf q_F\rVert \le r_F$ | take-off and landing |
| C6 | $\sum_i a_{ik}[t] \le N_{\text{sub}}$, $\sum_k a_{ik}[t] \le 1$, $a_{ik}[t] = 0$ outside device $i$'s window | one receiver per UAV, each device to one UAV, duty cycle |
| C7 | $\sum_t b_i[t] \le D_i$ | a device cannot send more than it holds |
| C8 | $\theta_{c,l} \in$ 2-bit set, $\tau_c[t] \ge 0$, $\sum_c\tau_c[t] = 1$ | discrete phases, one panel shared in time |

with flight energy $E_k[t] = \int_{\text{slot}} P(\lVert\mathbf v_k(s)\rVert)\,ds$ (Part 4).

**Why it cannot be solved directly.** It is a *fractional* (ratio) objective, *mixed-integer*
(scheduling, discrete phases), *non-convex* (trajectories inside the log and the path loss),
*stochastic* (fading) and *online*: device positions, data sizes and wake-up times are only
revealed as the mission unfolds. No off-the-shelf solver handles all of that at once.

**How v9 solves it — one solver per block of variables (the solution architecture):**

| Block | Solved by | Optimal? |
|---|---|---|
| Ratio objective | **Dinkelbach**: maximise $\sum b - \lambda\sum E$, update $\lambda$ toward the achieved EE (Part 7) | standard equivalence |
| $\boldsymbol\theta_c$ given who is served | **closed-form alignment** + 2-bit rounding (Part 5.2) | optimal per link (continuous), ≤ 0.9 dB loss from rounding |
| $\boldsymbol\tau$ given the configurations | equal shares (default) or **optimised shares** (`arbitration="opt"`): concave problem $\max_\tau \sum_i \min(D_i^{\text{left}}, \delta\sum_c \tau_c r_{ci})$, greedy over 1/20-slot slices | optimal up to the slice size |
| $\mathbf a$ given positions | deadline-aware greedy (most urgent device to its best UAV) | heuristic |
| $\mathbf q, \mathbf v$ (C1–C5) | **multi-agent RL policy** (MAPPO), one decision per slot from local observations; C1–C3 enforced by the dynamics, C5 by the autopilot shield, C4 by reward penalty + separation layer | learned |
| Training the RL policy across city areas | **federated learning** (FedAvg / FedProx, partial participation, compressed updates) | — |

So "optimisation of RIS-assisted UAVs" = solving P1; "with federated learning" = the way the
trajectory part of the solver is trained across many areas without moving raw data.

---

## Part 8 — Reinforcement learning, from scratch

### 8.1 The idea

**Intuition.** RL is learning by trial and error. The *agent* (a drone's brain) sees a
description of the situation (the *observation*), chooses an *action* (where to fly), and gets
a *reward*. After thousands of practice missions it learns which actions lead to high total
reward. The brain is a small neural network called the **actor** (or policy). A second
network, the **critic**, learns to predict "how much reward will I still collect from here?"
— this prediction tells the actor whether an action turned out better or worse than expected.

### 8.2 What each drone sees (observation, 99 numbers)

Own position, velocity and time left; vector to its landing slot and how much slack it has to
get there; where the other two drones are (relative); where the RIS is; and the **8 most
relevant devices** (nearest, awake ones first), each described by 10 numbers: relative position,
data left, awake or not, time left in its window, whether buildings block it from me, whether
the RIS can help it, last measured SNR, whether *I* served it last slot, whether *another*
drone did. Plus total data left overall and a one-hot "which drone am I" (so drones that share
one network can still take different roles).

Important for fairness: a drone does **not** see all 24 devices or the other drones'
velocities. It is a *decentralised* decision maker.

### 8.3 PPO — the learning algorithm

**Intuition.** Collect a batch of practice missions with the current policy, then improve the
policy a little — but *not too much at once*, because the batch only tells you about actions
near the current behaviour. PPO ("Proximal Policy Optimisation") enforces "not too much" by
clipping how far the probability of any action may change in one update.

**Precise** (`v9/ppo.py`):
- Policy: Gaussian over a 2-D "pre-action" $u$, squashed by tanh into [−1, 1]². We store $u$,
  so the probability ratio new/old is exact (the tanh correction cancels). v8 clamped actions
  after sampling, which broke this ratio.
- Advantage (how much better than expected) with GAE, γ = 0.99, λ_GAE = 0.95, normalised over
  the steps the policy actually controlled (autopilot steps excluded).
- Clipped objective, clip 0.2; 5 epochs over minibatches of 256; entropy bonus 0.003;
  learning rates 3·10⁻⁴ (actor) and 10⁻³ (critic), **linearly decayed to 10 %** over training
  (chosen by the pilots, Part 13); 8 missions per update; rewards scaled by a running std of
  the return.

### 8.4 Multi-agent flavours

- **MAPPO-CTDE** ("centralised training, decentralised execution"): the three drones share one
  actor (with the "which drone am I" input). During *training* a central critic sees all three
  observations at once (297 numbers), which makes learning easier. During *flight* each drone
  only uses its own actor and its own observation. Cost: every training step, every drone's
  experience has to be sent to the central learner. **This is the "upper bound" style method.**
- **IPPO** ("independent PPO"): every drone has its own actor and its own critic, trained only
  on its own experience. Nothing is exchanged. **Lower bound** on communication.

---

## Part 9 — Federated learning, from scratch

### 9.1 The idea

**Intuition.** Several learners each train on their own data. Instead of sending the data to a
central server, they periodically send only their *model weights* (the numbers inside the
neural network). The server averages the weights and sends the average back. Everyone
continues from the average. Data never leaves its owner; only models travel. That is
**FedAvg** (McMahan et al., 2017).

### 9.2 Inside one area: drones as FL clients

Each of the 3 drones trains its own actor + critic (like IPPO) on its own experience. Every
**E** local PPO updates the drones upload their weights; the aggregator averages them and
broadcasts the result.
- **E** trades communication for agreement: E = 1 averages after every update (most traffic),
  E = 16 rarely (least traffic, clients drift apart in between). We run E = 1, 4, 16.
- **FedProx** (Li et al., 2020): FedAvg plus a penalty μ/2·‖w − w_global‖² (μ = 0.01) that stops
  clients drifting too far from the last average between rounds.
- **FedPer** (Arivazhagan et al., 2019): average only the shared hidden layers; each drone keeps
  its own output layer ("personal head").
- **What gets evaluated** for FL methods is the *federated model* (the average), because that
  is what the server would deploy. v9.0 evaluated the drifted local copies by mistake; the
  audit caught it.

### 9.3 Across city areas: hierarchical FL (the main FL experiment)

**Intuition.** One area's drones can only learn about one courtyard layout. A city has many
areas, each with its own ground station (GCS) and its own drone team. If every area trains
alone, each learns only its own layout. If all areas ship their raw flight data to one cloud
server, that server can learn a general policy — but that is a lot of data and a privacy
problem. FL in between: each area learns locally and only shares its model.

**Precise** (`v9/multiarea.py`):
- City areas are generated procedurally (`config.random_area(seed)`): a courtyard block of
  random size, position and orientation, with the RIS on the far row. **4 training areas, 4
  validation areas (used only to pick checkpoints), 8 test areas (never seen in training).**
- Inside each area the ground station trains its 3 drones with MAPPO-CTDE (centralised inside
  the area — the GCS is right there).
- Modes:
  - **local** — every area alone; nothing crosses area boundaries; we report all 4 area models.
  - **fedavg** — every E = 4 updates the area GCSs upload their *actor* weights to the cloud,
    which averages and broadcasts (critics stay local; an ablation "both" averages critics
    too).
  - **pooled** — every area ships its raw trajectories to one cloud learner (maximal data
    sharing; the "upper bound").
- All are judged on the 8 **unseen** test areas → this measures *generalisation*.
- Traffic is metered: **cross-area** (GCS ↔ cloud) and **intra-area** (drones ↔ their GCS).

Expected and honest framing: FL is not expected to beat pooling all raw data; it is expected
to get close at a fraction of the traffic, without moving raw data, and to beat areas
learning alone.

---

### 9.4 The harder FL setting (v9.2): non-IID areas, partial participation, compression

**Intuition.** Real city areas differ in more than layout: some have many sensors, some few;
some sensors hold a lot of data, some little; transmitters and panels differ. And in a real
federation not every ground station is online every round. FL is supposed to shine exactly
here — and that is where we now test it.

**Precise** (`v9/fedhet.py`, `config.hetero_area`):
- 12 training areas that differ in device count (15–31), data per device (×0.6–1.5), device
  power (6–14 dBm) and RIS size (10×10 to 20×20); 4 validation and 8 test areas.
- Each round the cloud samples **50 %** of the areas; each starts from the global actor, does
  4 local MAPPO-CTDE updates and uploads its *update* (new weights − global weights).
- Uploads can be **compressed**: 8-bit quantisation (1 byte per weight instead of 4), or
  **top-10 %** sparsification (send only the 10 % largest changes, keep the rest in a local
  "residual" that is added next time — error feedback — so nothing is lost for good).
- Every mode (local, pooled, FedAvg, FedProx, compressed) uses the **same total number of
  training missions**, so no mode wins by simply training more.
- **Few-shot adaptation:** a new area's ground station takes the global model and fine-tunes
  it on 320 of its own missions; compared with training from scratch on the same missions.

## Part 10 — The rivals (baselines) and why they matter

**Intuition.** A learned policy is only impressive if it beats sensible hand-written rules.
So we built rivals, from silly to strong, and tuned the strong ones as carefully as we tuned
the learning.

`v9/baselines.py`:
- **random** — random velocity commands (sanity floor).
- **straight line** — fly at constant speed straight to the landing slot (Jiang et al.'s
  "fly-through" baseline).
- **Greedy** — fly to the most valuable device (data left × "does it need me nearby" × "is it
  awake" ÷ distance), circle it at the economical speed, leave for the final zone in time.
  Two versions:
  - **centralised greedy (full information)**: knows all 24 devices and assigns drones to
    devices jointly so two drones never chase the same one. *A strong reference — it has more
    information than the learned drones.*
  - **decentralised greedy (same information as RL)**: each drone sees only its 8 devices and
    who served what last slot, knows only where the others *are* (not where they fly), and
    decides alone. *The like-for-like rival.*
- **Tuning** (`v9/tune.py`): random search over 6–7 parameters on **validation** episodes
  only, then re-checking the best 10 on more validation episodes. A candidate must reach the
  final zone in ≥ 99 % of episodes and have ≤ 0.5 separation violations per episode.
- **Separation layer** (`env.separate`): pushes predicted end-of-slot positions apart so the
  heuristics obey the same 20 m rule the learned drones are penalised for (Part 12).

---

## Part 11 — How we measure, so that results are trustworthy

**Intuition.** One lucky run proves nothing. We separate the data we learn from, the data we
use to choose, and the data we report on; we repeat everything with several random seeds; and
we put error bars on everything.

**Precise.**
- **Three disjoint seed streams.** Training missions use seeds 10,000,000·(seed+1)+episode;
  checkpoint selection and heuristic tuning use *validation* seeds 2,000,000+j (20 missions);
  every reported number uses *test* seeds 5,000,000+j (**100 missions**), a fresh set created in
  v9.1 that was never looked at during development. Learned and heuristic policies are tested
  on the **same** 100 missions (same device positions, data sizes, windows, fading), so they
  can be compared *paired*, mission by mission.
- **Checkpoint selection.** During training, the policy is evaluated on validation every 200
  missions; the best one (that reaches the final zone ≥ 90 % of the time) is kept and then
  tested once.
- **5 training seeds per method.** Learning is random (initial weights, sampling); results are
  reported as mean ± **95 % confidence interval** over seeds (Student-t, `v9/stats.py`).
- **Paired comparisons.** "Method A vs heuristic B" = for each seed, the mean over the 100
  shared test missions of (A − B); then a CI over seeds. This removes mission-to-mission
  luck from the comparison.
- **Figures** have axes starting at 0 and say what the error bars mean.

---

## Part 12 — The audit and what it changed (v9.0 → v9.1)

After the first full version (v9.0) we ran a four-reviewer audit and verified every finding
before accepting it (`docs/analysis/06-v9-audit.md`). What it found and what we did:

| Finding | Why it mattered | Fix |
|---|---|---|
| **C1** A tuned greedy heuristic beat the learned agents, and the greedy had full information while the agents were decentralised | the headline "RL beats heuristics" was not supported | strong *and* fair rivals (centralised + decentralised greedy), tuned on validation; RL strengthened (Dinkelbach price, LR annealing, pilots) |
| **C2** FedAvg results were measured on the clients' drifted copies, not the averaged model | overstated FL | evaluate/checkpoint the federated model (`Learner.eval_view`) |
| **C3** The RIS in the old 20 m scene was effectively off | the "old scene shows nothing" comparison was unfair | fixed; old-scene RIS now active (+0.5 %, still nothing) |
| **M1** energy integration inaccurate when accelerating | biased EE | composite Simpson integration |
| **M2** RIS elements modelled as ideal point scatterers, panel facing straight out | unrealistic RIS strength/geometry | Tang et al. element pattern + aimed panel |
| **M5** energy price λ never updated | not really Dinkelbach | λ update on |
| **M7** std with ddof=0, no CIs, test seeds reused during development | weak statistics | ddof=1, 95 % CIs, fresh test seeds, 5 seeds |
| plus: random RIS draws shifted the fading stream, open devices could spawn inside buildings, landing slots outside the area in the aerial case, 2-D collision push, boundary handling | small biases | all fixed, each with a regression test (`v9/tests/test_audit_fixes.py`) |

**Two more problems found while re-running (after the audit):**
1. **Separation violations.** The return-to-home autopilot and the heuristics had no
   separation logic; the tuned greedy broke the 20 m rule ~6 times per mission, the
   decentralised one ~17 times, while the learned drones are penalised for it (~0.1–0.5).
   Unfair to RL. Fix: separation in the autopilot and in the heuristics; tuner enforces ≤ 0.5.
2. **Old-scene landing slots** sat exactly on the edge of the 2 m final zone, so "reached
   final" was misreported. Fix: slots strictly inside (final zone 3 m).

Lesson worth saying in the viva: *every* number in v9.0 was re-measured after these fixes; the
v9.0 results are archived in `results_v9/archive_v9.0/` and are not used.

---

## Part 13 — Results so far (measured, v9.1)

### 13.1 The gate: does the scenario make sense, and does the RIS matter?

(`results_v9/gate/`, 100 fresh test missions, 95 % CI; tuned heuristics.)

**Shadowing is real and the RIS fixes it.** A courtyard device can finish its 20 Mbit
upload within its 30 s window from **3.4 %** of the area without the RIS and from **90 %** of
the area with it (`fig_coverage.png`). Upload time for 20 Mbit from a courtyard device:

| Drone distance | 50 m | 200 m | 400 m |
|---|---|---|---|
| No RIS | 55 s | 194 s | 513 s |
| With RIS | 11 s | 17 s | 23 s |
| (open-field device, no RIS) | 7.5 s | 16 s | 31 s |

**EE of the heuristics (bit/J):**

| Policy | No RIS | Random RIS phases | Aligned RIS (time-share) | RIS gain (paired) |
|---|---|---|---|---|
| straight line | 5616 ± 129 | 5645 | 7065 ± 126 | **+25.8 %** |
| decentralised greedy (same info as RL) | 7208 ± 194 | 6875 | 8199 ± 160 | **+13.8 %** |
| centralised greedy (full info, tuned) | 7564 ± 143 | 7701 | 8675 ± 162 | **+14.7 %** |

- Random RIS phases ≈ no RIS: the panel helps only when *aligned*.
- All 16 procedurally generated city areas tell the same story: RIS gain +24–32 % for the
  straight line, +8–19 % for the tuned greedy (`area_feasibility.csv`).
- **Old 20 m scene**: RIS gain only **+0.5 %** — the old scene could not show the RIS's value.
- **Aerial RIS** (carrier energy charged): **−34 % to −42 %** vs the building RIS for every
  policy (e.g. 5757 vs 8675 for the tuned greedy). A movable RIS is not worth a 4th drone's
  flight energy in this setting.
- Time-share vs common arbitration: no meaningful difference.

### 13.2 Pilots: choosing how to train (validation only, 2 seeds each)

| Variant | Val EE (mean of last 5 evaluations) |
|---|---|
| A: Dinkelbach price on | 8046 |
| B: fixed price | 7984 |
| **C: Dinkelbach + learning-rate annealing (chosen)** | **8198** |
| D: 16 missions per update | 8071 (unstable across seeds) |
| E: C trained 6000 missions instead of 4000 | 8055 |
| F: E with a 2× wider network | 7954 |
| G / H: C with 12 / 24 devices in view (vs I: 8, re-run) | 8149 / 8164 (vs 8058) |

Differences of ~100 bit/J are within seed-to-seed noise. **Locked configuration:** Dinkelbach
price + LR annealing, 4000 training missions, 8 devices in view.

### 13.3 Final campaign results (all 65 runs, 5 seeds each)

Full tables: [`docs/v9_results.md`](v9_results.md). The short version:

**Single area (building RIS), test EE in bit/J, 95 % CI over 5 seeds:**

| Method | EE | vs decentralised greedy (8,199) | vs centralised greedy (8,675) | Training traffic |
|---|---|---|---|---|
| MAPPO-CTDE | 8,119 ± 61 | −1.0 % (within 1 %, slightly below) | −6.4 % | 375 MB |
| IPPO | 8,027 ± 262 | −2.1 % | −7.5 % | 0 |
| FedPer (E = 4) | 7,989 ± 113 | −2.6 % | −7.9 % | 176 MB |
| FedAvg E = 1 / 4 / 16 | 7,829 / 7,742 / 7,647 | −4.5 / −5.6 / −6.7 % | −9.7 / −10.8 / −11.8 % | 708 / 177 / 44 MB |
| FedProx (E = 4) | 7,827 ± 115 | −4.5 % | −9.8 % | 177 MB |
| straight line (heuristic) | 7,065 | | | |

**Multi-area FL, EE on 8 unseen city areas:** local 7,581 ± 115 → **FedAvg (actors)
7,953 ± 62** (88 MB cross-area) → pooled raw data 8,162 ± 59 (771 MB). FedAvg beats local in
all 5 seeds (+4.9 %) and recovers 64 % of pooling's gain with 8.8× less traffic. Averaging the
critics too: +0.2 % for 5.8× the traffic.

**RIS placement under learned flight (CTDE):** no RIS 7,201 → building RIS 8,119
(+12.8 %) → aerial RIS 5,070 (−37.6 % vs building).

**Sweeps:** 2-bit phases are enough (1 bit −2 to −3.5 %); 20° phase error costs < 1 %; a
bigger RIS helps steadily (64 → 400 elements: +12.6 % for CTDE).

### 13.4 The honest reading (what to say in the presentation)

1. The new scenario makes the RIS matter (+13–26 %), and the old one could not (+0.5 %).
2. A building-mounted RIS is far better than a UAV-carried one once the carrier's energy is
   counted.
3. The learned trajectories beat straight flight clearly and come within 1 % of a carefully
   tuned heuristic with the same information. They do not beat a centralised planner that sees
   everything; more training, a wider network or more devices in view did not change that
   (pilots, 13.2).
4. Federated learning's value here is **generalisation across areas without sharing raw
   data**, not better performance inside one area: inside one area averaging three UAVs'
   models costs a little; across areas, FedAvg clearly beats training each area alone at a
   small fraction of pooling's traffic — +4.9 % with similar areas, **+9.7 % with
   heterogeneous areas and 50 % participation**, where it even matches pooling all raw data,
   and it survives 8-bit or top-10 % compression of the uploads.
5. A federated model gives a new area a working policy immediately (+16 % over training from
   scratch on the same missions) and, after a short local fine-tune, beats the like-for-like
   heuristic — the one place where learning beats hand-tuning in this project.

### 13.6 v9.2 additions (after the independent review)

| Question | Experiment | Answer |
|---|---|---|
| Does FedAvg win only because its model saw more data? | each area trained alone on 4× its missions (`local_x4`) | No: 4× own data gives +1.1 % (n.s.); FedAvg is still +3.8 ± 1.2 % better (5/5 seeds) |
| Do local models really overfit their own layout? | cross-area matrix (each area's model in every area) | Yes: on the same area, its own model is +6.6 % better than the others' |
| Does FL help more when areas really differ? | 12 non-IID areas, 50 % participation | Yes: FedAvg +9.7 % over local (5/5), at least as good as pooling raw data with 13× less traffic |
| Can the uploads be compressed? | 8-bit / top-10 % updates | Yes: −0.3 / −0.7 % (n.s.) for 38–40 % less traffic |
| Does FedProx help? | μ = 0.01 | No (−1.1 %) |
| Is the federated model useful in a brand-new area? | fine-tune 320 missions vs scratch | +16.2 % over scratch; after adaptation above the like-for-like heuristic in every seed (+2.1 %) |
| Can the RIS time-sharing be optimised? | concave time-share optimisation | +0.3–0.7 %: the equal split is already near-optimal |
| What if channel estimates are imperfect? | NMSE −20 … 0 dB in scheduling and RIS alignment | −2–4 % at −20 dB, −6–8 % at −10 dB, −13–16 % at 0 dB; learned policy degrades slightly more gracefully than the heuristic |
| Is FL communication energy significant? | 0.4 J/MB UAV radio | ~0.04 J per mission vs ~42.6 kJ flight: negligible |

## Part 14 — Limitations (say these before an examiner does)

1. The RIS controller knows the channels perfectly when it sets the phases (as in Jiang et
   al.); a ±phase-error sweep shows the sensitivity.
2. Buildings are boxes with a fixed +20 dB loss; no diffraction or wall reflections.
3. Fixed flight altitude (2-D motion), Shannon rates, perfect synchronisation, no protocol
   overhead; only propulsion energy is counted.
4. Data sizes, bandwidth and one receiver per drone are design choices within realistic
   ranges (NB-IoT), chosen so that the problem is non-trivial; the data-size sweep shows the
   conclusions do not hinge on them.
5. FL is simulated on one computer: it fixes *what is exchanged*, not real wireless links or
   delays.
6. The learned policies match, but do not beat, a well-tuned decentralised heuristic, and stay
   below a centralised full-information planner. The contribution is the honest comparison and
   the FL framework, not "RL beats everything".
7. 5 training seeds, 100 test missions per number: enough for a BTP; a journal paper would
   want more seeds and scenario variety.

---

## Part 15 — Where everything is in the code

| File | What it does |
|---|---|
| `v9/config.py` | all parameters; presets `building`, `building_noris`, `aerial`, `legacy_small`; `random_area(seed)` |
| `v9/physics.py` | geometry, blockage, LoS probability, fading, RIS elements + pattern, phase alignment + quantisation, propulsion power + slot energy, Shannon rate |
| `v9/env.py` | the simulator (`RISUAVEnv`): autopilot, kinematics, energy, separation, scheduling, RIS arbitration, reward, observations; `separate()` |
| `v9/baselines.py` | random / straight / Greedy (centralised or decentralised), `run_episode`, `evaluate`, `tuned_greedy` |
| `v9/tune.py` | tunes the heuristics on validation missions |
| `v9/ppo.py` | actor, critic, GAE, PPO update (+ FedProx term), LR-annealing flag |
| `v9/trainers.py` | single-area CTDE / IPPO / FedAvg / FedProx / FedPer, communication metering, checkpointing, testing |
| `v9/multiarea.py` | multi-area local / FedAvg / pooled |
| `v9/stats.py` | 95 % CIs, paired differences |
| `v9/run.py` | one experiment from the command line (gate, train, multiarea, sweeps) |
| `v9/launch.py` | groups of experiments in parallel (pilots, campaign, post) |
| `v9/plots.py` | all figures, from saved files only |
| `v9/tests/` | 66 unit and regression tests |
| `results_v9/` | gate, tuning, pilots, (soon) campaign results; `archive_v9.0/` = superseded results |

---

## Part 16 — Glossary

- **bit/J (EE)** — bits delivered per joule of flight energy; higher is better.
- **slot** — one 2.5 s decision step; a mission has 40.
- **episode / mission** — one 100 s flight from start to final zone.
- **LoS / NLoS** — line of sight / not.
- **Rician fading** — random signal wobble with a dominant direct component.
- **RIS element / phase** — one tiny reflector / the delay it applies.
- **2-bit quantisation** — each phase can only be 0°, 90°, 180° or 270°.
- **arbitration** — how one panel is shared among several links in a slot.
- **actor / critic** — policy network / value-prediction network.
- **PPO** — the RL algorithm; small, clipped policy updates.
- **CTDE** — centralised training (shared critic), decentralised execution.
- **IPPO** — each agent learns alone.
- **FedAvg / FedProx / FedPer** — average weights / plus a pull-to-average penalty / average
  all but the personal output layer.
- **E** — local updates between federated averaging rounds.
- **GCS** — ground control station of one city area.
- **Dinkelbach** — turning "maximise a ratio" into "maximise a difference with a price".
- **validation vs test** — data for choosing vs data for reporting; never mixed.
- **95 % CI** — the range that would contain the true mean in 95 % of repeated experiments.
- **paired comparison** — comparing two methods on the *same* missions.
- **separation violation** — two drones closer than 20 m.

---

## Part 17 — Questions an examiner is likely to ask (with short answers)

1. **Why not learn the RIS phases with RL too?** For a given link the optimum is a closed-form
   formula (align all reflected paths with the direct path); learning it would be slower and
   worse. RL is used where no formula exists: trajectories.
2. **Is the RIS gain realistic?** It comes from a standard published model (Tang et al. 2021)
   with a 1 m² panel tens of metres from the devices; the hand check (Part 5.4) gives
   ~1.2 Mbit/s at 300 m. The gain is large because the RIS is near the devices and elevated.
3. **Why does the old scene show nothing?** Everything is within a few metres; the direct link
   is already excellent, so the RIS adds +0.5 %.
4. **Does your RL beat the heuristics?** It matches a tuned decentralised heuristic with the
   same information to within 1 % (slightly below) and stays 6.4 % below a centralised
   full-information planner. We report
   this honestly; the heuristics were tuned as carefully as the RL.
5. **Then why use RL?** No hand-tuning per scenario (the heuristics needed separate tuning for
   every preset), and it is the substrate on which FL is studied; the multi-area experiment
   tests whether learned policies transfer to unseen layouts.
6. **What does FL buy you?** Raw data stays in each area, and much less traffic than
   pooling. Across similar areas FedAvg generalises 4.9 % better than each area alone (not
   because of more data: 4× more own data gives only +1 %); across heterogeneous areas with
   50 % participation +9.7 %, matching pooling with 13× less cross-area traffic; and a new
   area gets a ready policy that fine-tunes past the like-for-like heuristic.
7. **Why is the aerial RIS worse?** A 4th drone burns ~126–178 W continuously (+20 % for the
   payload); the extra bits do not pay for it.
8. **How do you know results are not luck?** Disjoint train/validation/test seeds, 100 fresh
   test missions, 5 training seeds, 95 % CIs, paired comparisons on identical missions.
9. **What is the return-to-home autopilot and doesn't it help the RL?** A safety shield that
   guarantees the drones land in time; it applies equally to heuristics and RL, and the steps
   it controls are excluded from learning.
10. **What would you do with more time?** Learned RIS scheduling across drones, 3-D flight,
    imperfect channel knowledge, more seeds and more varied cities, real FL communication
    delays.
