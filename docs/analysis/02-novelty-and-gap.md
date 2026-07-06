# 02 — Novelty Audit and the Real Research Gap

> Caveat stated up front: the full texts of [1] (Jiang et al., IoT-J 2025), [9] (Cao et
> al., TVT 2021) and [11] (anonymous FL-RIS-UAV TechRxiv preprint) were not provided with
> this repository. The assessment below uses the report's own descriptions of those works
> (report §§1.3, 2.2–2.6, Table 2.1) plus general knowledge of the 2020–2025 literature.
> Before submission, the novelty claims must be re-checked against the actual texts and a
> fresh literature search.

## 1. What the project currently claims as novel

Report §2.6 / Table 2.1: *"None of the existing works combine all four things at once:
multiple cooperating UAVs, RIS phase control as part of the learned action, an EE
objective that accounts for propulsion, and a MARL-CTDE framework."*

This is a **conjunction novelty claim** — "first to combine A+B+C+D" — and it is the
weakest form of novelty argument a paper can make, for three reasons:

1. **It is fragile.** A single counterexample kills it, and the space of RIS+UAV+DRL
   papers from 2021–2025 is very crowded: there are published works on multi-agent DRL
   for RIS-assisted multi-UAV networks (MADDPG- and MAPPO-based), on DRL for joint UAV
   trajectory + RIS phase design, and on energy-efficiency objectives with the same Zeng
   propulsion model. The report's own Table 2.1 already contains a work ([11]) with
   multi-UAV ✓, RIS ✓, learning ✓ — the differentiation reduces to "they used
   BCD + federated DRL, we use MAPPO-CTDE," which is an algorithm swap, not a problem gap.
   A GLOBECOM/ICC reviewer in this area will name two or three near-misses from memory.
2. **The distinguishing element is undermined by the implementation.** The one genuinely
   under-explored piece of the conjunction — *multiple UAVs contending for one shared
   RIS* — is exactly what the code does not model (audit F3: each UAV gets a private RIS
   copy). As implemented, the multi-agent dimension adds no coupling through the RIS at
   all, so the work is functionally "three independent single-UAV problems with a shared
   reward," which *is* solved in prior art.
3. **The comparison table's only multi-UAV row is an anonymous, non-peer-reviewed
   preprint** ([11], TechRxiv, cited as "Anonymous"). Building the gap argument on a work
   that cannot be verified is itself a review risk.

## 2. What is genuinely not novel (and should never be claimed)

- MAPPO/CTDE itself (Yu et al. 2022), GAE, PPO machinery — commodity.
- RL for RIS phase control (single-agent) — established since ~2020 (report §2.5 admits
  a "growing set of papers").
- Propulsion-aware EE for UAV trajectory design — Zeng et al. 2018 and a large
  follow-on literature; Jiang et al. [1] already do it *with* an RIS.
- "DRL avoids needing perfect CSI / re-solving optimization each slot" — standard
  motivation, appears verbatim in dozens of papers.

## 3. Where a defensible gap actually exists

Three candidate gaps, in decreasing order of strategic value for this project:

### Gap A (recommended): Federated MARL for shared-RIS multi-UAV EE optimization, with the communication cost measured

The project's own title promises FL; the code has none (audit F13). The literature has:
classical optimization for RIS-UAV EE [1]; single-agent DRL for RIS-UAV; MARL-CTDE for
multi-UAV; FL-flavored DRL for RIS-UAV [11, unverified]. What is thin on the ground is a
work that:

1. models **one physically shared RIS** serving K UAVs (arbitration is the interesting
   multi-agent problem — see redesign doc),
2. trains the multi-agent policy **without a centralized critic at deployment scale**,
   replacing centralized training with **federated aggregation of actor–critic
   parameters over the UAV swarm**, and
3. **quantifies the trade** between the CTDE upper bound (centralized critic, full state
   exchange every step) and federated variants (parameter exchange every E episodes) in
   *both* EE achieved *and* bytes communicated — i.e., treats communication overhead as a
   first-class metric, which is natural in a paper whose objective is energy efficiency.

This is a real, articulable question — *"how much of the centralized-critic benefit
survives when coordination is limited to periodic federated averaging, and what does it
cost?"* — with a built-in ablation axis (aggregation interval E, FedAvg vs FedProx vs
personalization) and a natural fit to the venue. It also rescues the title.

### Gap B: The shared-RIS arbitration problem itself

Who sets the phases when K UAVs (or K device clusters) want different configurations from
one passive surface? Options — time-slicing the RIS among UAVs, a dedicated RIS-controller
agent, phase configurations optimized for the sum objective — have materially different
performance and are rarely compared head-to-head in a learning framework. This is the
scientifically cleanest contribution enabled by fixing audit-F3, and it can live inside
the same paper as Gap A (the RIS agent is part of the redesigned architecture).

### Gap C: Honest characterization of what blind RIS control can and cannot learn

Audit F10 showed the current observation makes per-realization phase alignment
impossible. Turned around, there is a publishable sub-question: with only geometric
information (positions ⇒ LoS phases) and K-factor κ, the achievable fraction of the
coherent-combining gain is bounded (LoS carries κ/(κ+1) of the link power), and a learned
geometric policy can be compared against that bound. This makes a strong *analysis
subsection* rather than a paper on its own.

## 4. Recommended novelty statement for the paper

> *We propose Fed-MAPPO, a federated multi-agent PPO framework in which K rotary-wing
> UAVs jointly learn trajectories and the configuration of a single shared RIS to
> maximize propulsion-aware energy efficiency of IoT uplink data collection. Unlike
> centralized-critic CTDE, agents exchange only periodic model updates (FedProx
> aggregation every E episodes), and we quantify the resulting EE–communication trade-off
> against centralized MAPPO, independent PPO, and corrected model-based baselines.*

Every element of that sentence is implementable within the roadmap
([05-roadmap-and-readiness.md](05-roadmap-and-readiness.md)), and none of it depends on a
fragile "first ever" conjunction: the contribution is a *measured trade-off* plus a
*correctly modeled shared resource*, both of which survive the discovery of adjacent
related work.

## 5. Due diligence still required before submission

- Obtain and read [1] and [11] in full; if [11] turns out to do federated DRL over a
  shared RIS with multi-UAV EE, the differentiation must sharpen to the arbitration
  mechanism + communication accounting (Gap B emphasis).
- Run a fresh search (IEEE Xplore, arXiv cs.IT/eess.SP, 2023–2026) for: "federated
  reinforcement learning RIS UAV", "MAPPO RIS", "multi-UAV intelligent reflecting
  surface energy efficiency", "shared RIS multiple users phase scheduling". Log the
  ten closest works in a related-work matrix with explicit deltas.
- Decide the claim of record *before* running final experiments, so the experimental
  design (see [04-experimental-methodology.md](04-experimental-methodology.md)) proves
  exactly that claim.
