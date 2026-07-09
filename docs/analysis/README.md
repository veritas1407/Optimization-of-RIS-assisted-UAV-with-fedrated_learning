# Project Analysis — RIS-Assisted UAV-IoT MAPPO (v8)

Critical-reviewer-grade audit of the BTP project (code v8, report `BTP_Report_Final_v3.pdf`),
redesign proposal, and publication roadmap targeting an IEEE conference
(ICC / GLOBECOM / WCNC). Produced July 2026 on branch `claude/btp-project-analysis-425r7n`.

**Scope of this analysis:** documents only — no simulation code was modified. All findings
cite `ris_uav_mappo_v8.py` line numbers and are reproducible; one physics bug is verified by
a self-contained script included in the audit.

## Documents

| Doc | Contents | Maps to requested items |
|-----|----------|-------------------------|
| [01-technical-audit.md](01-technical-audit.md) | Architecture & execution-flow walkthrough as built; validation of the mathematical models; full defect register F1–F14 with evidence and impact-on-claims | 1, 2, 3 |
| [02-novelty-and-gap.md](02-novelty-and-gap.md) | Honest novelty audit against the cited literature; where the defensible research gap actually is | 4 |
| [03-redesign-proposal.md](03-redesign-proposal.md) | Corrected system model, heterogeneous-agent architecture, Fed-MAPPO pipeline, FL-method comparison and recommendation, RL implementation fixes — with rationale and trade-offs for every decision | 5, 6, 7 |
| [04-experimental-methodology.md](04-experimental-methodology.md) | Baselines, ablations, metrics, statistical protocol, scalability & communication-overhead experiments, reproducibility checklist | 8 |
| [05-roadmap-and-readiness.md](05-roadmap-and-readiness.md) | Milestone-based implementation roadmap (M0–M5) with validation strategy per milestone; publication-readiness verdict and venue strategy | 9, 10 |

> **Update (July 2026):** after both reference papers ([J] Jiang et al. IoT-J 2025 and
> [A] Abualhayja'a et al. TechRxiv 2025) were obtained and read in full, a complete
> implementation-level design document was produced that supersedes `03` where they
> differ: [`../blueprint/technical-blueprint.md`](../blueprint/technical-blueprint.md).

## One-paragraph summary

The v8 implementation is a working, end-to-end MAPPO pipeline with disciplined iteration
history, but it is **not currently publishable**: six correctness defects (F1–F6) invalidate
the headline results — most notably, the "optimal RIS" baseline's phase formula has a sign
error that makes it *worse than random phases* (verified numerically), the reward is computed
against the wrong IoT device set in every scenario/device sweep, and the single physical RIS
is simulated as three independent private copies (one per UAV). The federated-learning
component promised by the project title does not exist in the code. The fix path is clear and
tractable: repair the physics and PPO math (M1), rescale the scenario to physically defensible
dimensions with a genuinely shared RIS and device–UAV association (M2–M3), and add a
Fed-MAPPO layer with FedProx aggregation as the paper's central novelty (M4). With the
proposed experimental protocol (M5), this becomes a credible ICC/GLOBECOM/WCNC submission
in roughly three months of focused work.

## Known limitations of this analysis

- The full texts of references [1] (Jiang et al.), [9] (Cao et al.), and [11] (anonymous
  FL-RIS-UAV preprint) were not provided; the novelty audit relies on the report's own
  descriptions of them plus general literature knowledge, and flags this where it matters.
- Code for v1–v3 is not in the repository (history starts at v4); statements about early
  versions come from the README and walkthrough files.
- Compute assumptions (Colab T4-class GPU) come from the README's training summary.
