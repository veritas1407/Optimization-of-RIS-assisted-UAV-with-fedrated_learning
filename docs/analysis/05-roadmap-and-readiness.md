# 05 — Implementation Roadmap and Publication Readiness

## Part A — Milestone roadmap

Sized for two students working part-time with Colab T4-class compute. Each milestone has
a validation gate; do not start the next milestone until the gate passes. Defect IDs
(F1–F14) refer to [01-technical-audit.md](01-technical-audit.md); design IDs (S/D/FL) to
[03-redesign-proposal.md](03-redesign-proposal.md).

### M0 — Repository restructure and test scaffold (~1 week)

- Split the monolith into `env/`, `agents/`, `baselines/`, `fl/`, `experiments/`,
  `tests/`; dataclass/YAML configs; no module-level mutable globals (retire the
  `V_MAX`/`P_MAX` mutation pattern).
- Port v8 behavior as-is first (bugs included) so refactor correctness can be verified
  by output parity on a fixed seed; keep `ris_uav_mappo_v8.py` untouched as the archive.
- Set up the unit-test skeleton from [04 §6](04-experimental-methodology.md).
- *Challenge:* resisting the urge to fix bugs during the refactor (breaks parity check).
- **Gate:** fixed-seed parity between package and v8 script; tests run in CI (<2 min).

### M1 — Correctness fixes (F1–F9) + vectorization (D5) (~2 weeks)

- Fix, in order: F1 (phase formula — one line + regression test), F5 (generator
  discipline), F2 (env device set plumbed through EE), F4 (per-episode fading, S4),
  F7 (PPO log-prob sum, squashed Gaussian, scheduler horizon, value-clip old-value),
  F9 (pre-projection collision counting, S7), F8 partially (association S2 — the full
  EE redefinition lands with M2 physics).
- Vectorize channels/rates (D5); target ≤50 ms/episode.
- *Challenges:* the squashed-Gaussian log-prob is fiddly (test: ratios = 1 on first
  epoch); vectorized channel code must reproduce loop code exactly (test on fixed seed).
- **Gate:** all unit tests green; ≥20× episode speedup; a 1500-episode retrain on the
  *old* toy geometry still converges (sanity that fixes didn't break learning).

### M2 — Realistic physical model (S1–S3, S8) and re-tuning (~2 weeks)

- Shared Θ[t] (S1), association + ratio-of-sums EE (S2), 500 m geometry / 2-D planar
  RIS response / real units (S3, S8), link-budget validation notebook.
- Re-tune reward weights (D4) on the new scale; expect the v8 shaping lesson to recur.
- *Challenges:* RIS benefit may be small at honest parameters — respond with RIS
  placement/shadowed-region design, never with ρ₀ (the notebook decides placement before
  any RL runs); reward re-tuning may take several iterations.
- **Gate:** link-budget notebook shows ≥20 % rate gain from ideally-aligned RIS for
  cell-edge devices at honest parameters; B1–B4 baselines produce sane, explainable
  numbers with error bars.

### M3 — Heterogeneous agents + full baseline suite (~2–3 weeks)

- RIS-controller agent (D1, D2 codebook default), retrain MAPPO-CTDE (B7), implement
  B5/B6/B8/B10, statistical harness (5 seeds, held-out eval, CIs).
- *Challenges:* MADDPG (B8) is famously finicky — timebox it, and if it won't stabilize,
  report best-effort with tuning budget disclosed; codebook design for the RIS agent
  needs a few iterations.
- **Gate — the go/no-go decision of the project:** does learned RIS control (B7 with RIS
  agent) beat analytic aligned RIS on learned trajectories (B10)? If yes → the paper
  claims joint learning. If no → the paper's claim narrows to trajectory learning +
  analytic RIS + FL (still publishable; decided *now*, not at writeup).

### M4 — Fed-MAPPO (FL §3) (~2–3 weeks)

- Local actor–critic per client, FedAvg → FedProx → personalized-head variants;
  aggregation interval sweep; communication accounting instrumented from day one.
- *Challenges:* post-aggregation performance dips (known on-policy FL failure mode);
  mitigation ladder is specified in [03 §3.3](03-redesign-proposal.md) — small E, μ,
  trunk-only averaging; fallback design (federated critics + local actors) is
  pre-approved here so the decision is quick.
- **Gate:** Fed-MAPPO within ~10 % of CTDE EE at ≥10× less communication than per-step
  state sharing (target, not guarantee — whatever the measured frontier is, it must be
  smooth and reproducible across seeds).

### M5 — Experimental campaign + paper (~3–4 weeks)

- Run the full matrix E1–E8 ([04 §4](04-experimental-methodology.md)); freeze CSVs;
  camera-ready figures from CSVs only.
- Write the 6-page paper: system model (M2), Fed-MAPPO (M4), main figure = EE vs
  communication frontier, honesty subsections (quantization E4, robustness E7,
  limitations).
- Fresh literature pass per [02 §5](02-novelty-and-gap.md); position against the ten
  closest works.
- **Gate:** internal red-team review against this audit's checklist — every claim has an
  experiment, every figure has CIs, no claim survives that the audit's matrix (01 §4)
  would kill.

**Total: ~12–14 weeks.** Venue timing (verify exact CFPs — they move): from July 2026 the
realistic targets are **WCNC 2027** (paper deadline typically ~Sep–Oct 2026 — tight but
feasible if M0–M2 go smoothly) or **ICC 2027** (~Oct 2026), with **GLOBECOM 2027**
(~Apr 2027) as the comfortable fallback. A workshop paper (e.g., ICC/GLOBECOM workshops,
deadlines later than main tracks) is achievable after M3 alone if an early publication is
wanted.

## Part B — Publication readiness assessment

### Verdict on the current state (v8 + report)

**Not submittable to any peer-reviewed venue in its current form.** This is a bug/validity
problem, not a polish problem: F1–F6 individually invalidate headline figures, and
jointly they mean no quantitative claim in Chapter 4 of the report survives; F13 means
the titled contribution (federated learning) is absent. Submitting as-is risks a desk
reject at best; at worst, the undisclosed monotonicity rewriting (F6) reads as data
fabrication if a reviewer inspects the repo.

### Strengths to preserve (these are real)

- A complete, running end-to-end pipeline: environment, MAPPO with the full stability
  toolkit (reward normalization, entropy annealing with floor, value clipping,
  best-model checkpointing), baseline harness, and an unusually rich plotting suite.
- Documented engineering discipline: 46 changes across 8 versions with stated rationale;
  the progress-weight sensitivity finding is genuine experimental insight worth keeping
  (formalized as E8).
- Correct propulsion model, correct Rician normalization, correct GAE/PPO skeleton —
  the physics and RL *scaffolding* is right even where specific formulas are wrong.
- The report's own Limitations section (§5.3) is honest about altitude, CSI, phase
  quantization, and the missing FL — self-awareness that makes the fix path credible.
- Writing quality of the report is above par for a BTP; the narrative structure ports
  directly to a conference paper.

### Weaknesses (ranked by how a reviewer would weight them)

1. Fatal correctness defects F1–F6 (results invalid).
2. Missing FL (title–content mismatch; also the novelty opportunity).
3. Unphysical scenario scale + tuned ρ₀ (F11) — "toy world tuned to show the effect".
4. Single seed, no CIs, confounded sweeps, OOD speed sweep (F12).
5. Weak/incorrect baselines: no classical AO/SCA-flavored comparison, no MARL
   alternatives, and the one "optimal" baseline is broken (F1) yet *still* beats the
   learned policy at most operating points — the current evidence, read honestly,
   argues *against* the method.
6. Thin novelty as claimed (see [02](02-novelty-and-gap.md)); conjunction-style claim
   resting on an anonymous preprint.
7. Report–code contradictions (F11 list) — even the correct parts don't describe each
   other.

### Technical depth

The RL engineering depth is decent (stability mechanisms, CTDE, reward shaping study).
The wireless depth is currently shallow — channel models are assembled reasonably but
not validated, and the RIS is mis-modeled at the level of what a surface physically is
(F3) and what information phase control requires (F10). The redesign (S1–S8, D1–D2)
raises the wireless side to the level reviewers at ICC/GLOBECOM expect; the FL layer
(M4) adds the distributed-AI depth the title promises.

### What "ready" looks like (submission checklist)

- [ ] All 14 audit findings resolved or explicitly scoped out in the paper's limitations
- [ ] Every figure: ≥5 seeds, CI bands, held-out evaluation seeds, raw CSVs in repo
- [ ] Corrected aligned-RIS baseline verified against the analytic maximum (F1 test)
- [ ] Shared Θ[t] and device association in the model section, matching the code exactly
- [ ] Fed-MAPPO frontier figure (EE vs MB communicated) with FedAvg/FedProx/personalized
- [ ] M3 gate decision reflected honestly in the claims (RIS learning in or out)
- [ ] Related-work matrix vs ten closest 2023–2026 works; [1] and [11] read in full
- [ ] Reproducibility: one command per figure, pinned requirements, seed manifest
- [ ] No sentence in the paper that the code contradicts (the F11 class of errors)

With M0–M5 executed as specified, this is a credible full-paper submission to
WCNC/ICC/GLOBECOM: correct system model, one clear measured contribution
(EE–communication trade-off of federated MARL over a genuinely shared RIS), honest
baselines, and statistics that survive review. The current codebase is the right
starting point for that work — but as scaffolding, not as evidence.
