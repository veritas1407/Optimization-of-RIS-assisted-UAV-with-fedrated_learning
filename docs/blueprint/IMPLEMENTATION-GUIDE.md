# Implementation Guide — How to Build This Project from the Docs

Audience: a future implementation session (human engineer or LLM coding assistant) that
was NOT part of the design conversations. Everything you need is in this repository; this
page tells you what to read, in what order, and the rules that keep the implementation
faithful to the design.

## Read in this order

1. **`docs/blueprint/technical-blueprint.md`** — THE build specification (Stages 3–6 are
   the system/simulator/software/math specs; Stage 9 is the module order). If any other
   document disagrees with the blueprint, the blueprint wins.
2. **`docs/analysis/01-technical-audit.md`** — the defect register (F1–F14) of the legacy
   v8 code. You are not fixing v8; you are building the new `risfed/` package. But every
   F-item is a mistake you must not reproduce, and several have mandated regression tests.
3. **`docs/analysis/04-experimental-methodology.md`** — statistical protocol and
   experiment matrix (updated baseline list is in blueprint Stage 7).
4. **`docs/analysis/05-roadmap-and-readiness.md`** — milestone gates and go/no-go
   criteria.
5. Reference only if needed: `02-novelty-and-gap.md` (paper positioning),
   `03-redesign-proposal.md` (superseded by the blueprint where they differ).

## Ground rules (non-negotiable)

- **Do not modify or delete** `ris_uav_mappo_v4..v8.py` or `mappo_results_*/` — they are
  the archived legacy line. All new code lives in a new `risfed/` package
  (layout: blueprint §5.1).
- **Build order = blueprint Stage 9 table.** Do not start a module before its
  dependencies' tests pass. The three gates (link-budget gate after baselines;
  RIS-learning gate B12; FL-viability gate) are decision points — stop and surface
  results at each gate rather than pushing through.
- **One channel convention.** The composite channel and aligned-phase formula are defined
  once in blueprint §6.1 (no Hermitian). `physics/channels.py` is the single source of
  truth used by the environment AND every baseline. The F1 regression test (aligned
  config reaches the analytic maximum, up to the b-bit quantization bound) must exist
  before anything consumes the channel module.
- **No module-level mutable globals; no global mutation for sweeps.** All parameters flow
  from `ScenarioConfig` (frozen dataclass + YAML). Sweeps construct configs.
- **Seed discipline** per blueprint §4.5: one `SeedSequence` tree; train and eval streams
  disjoint; no `RandomState` constructed inside loops.
- **PPO correctness:** log-probs SUMMED over action dims; tanh-squashed Gaussian with
  log-det-Jacobian; value clipping against stored behavior values; ratio ≡ 1 test on the
  first epoch. (These reverse audit F7 — the most common silent RL bugs.)
- **No post-processing of measured results, ever** (audit F6). Non-monotone curves get
  error bars, not edits.
- **Figures are generated only from saved CSVs.** No training or evaluation inside plot
  code.
- **Tests first for physics.** `tests/` list is in blueprint §5.3; the physics tests
  (hover power, Rician moments, alignment maximum, association count-once, env
  determinism) are cheap and catch the classes of bugs that invalidated v8.

## Definition of done, per milestone (condensed from roadmap M0–M5)

| Milestone | Done means |
|---|---|
| M0 scaffold | package + configs + CI skeleton; smoke run (3 episodes) passes |
| M1 core sim | modules 1–6 of Stage 9 built; all unit tests green; ≤50 ms/episode |
| M2 baselines + gate | B1–B6 evaluated with CIs; link-budget notebook shows blocked devices need the RIS at honest parameters (adjust geometry, never ρ₀) |
| M3 MARL | B7–B9 trained, 5 seeds; MAPPO-CTDE beats fixed trajectory with statistical significance; B12 gate decided |
| M4 Fed-MAPPO | FL strategies + byte metering; E2 frontier smooth across seeds; FL-viability gate decided (fallback: federated critics + local actors) |
| M5 campaign | experiment matrix E1–E8 artifacts frozen as CSVs; all paper figures reproducible by one command each |

## Compute assumptions

Colab T4-class GPU (or CPU for physics — the simulator is NumPy-dominated). Training runs
must checkpoint and resume across session limits. Full matrix budget ≈ 15–20 GPU-hours
*after* the ≤50 ms/episode target is met — treat that target as a hard requirement, not
an optimization.

## What to ask the humans (the two decision points that are theirs)

1. The B12 gate outcome (claim learned RIS control, or narrow to trajectory + analytic
   RIS + FL).
2. The FL-viability fallback (switch to federated-critics design) if post-aggregation
   instability persists after the mitigation ladder (small E → FedProx μ → trunk-only
   averaging).
