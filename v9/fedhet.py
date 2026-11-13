"""Heterogeneous multi-area federated learning (v9.2).

The harder, more realistic FL setting the v9.1 review asked for:

* 12 training areas (clients) that differ in layout AND deployment: number of devices,
  data per device, device transmit power and RIS size (``config.hetero_area``) -> non-IID.
* Partial participation: each communication round the cloud samples a fraction of the
  areas; only those train (E local MAPPO-CTDE updates starting from the global actor) and
  upload their actor update.
* Aggregation: FedAvg (mean of the updates) or FedProx (local loss + mu/2 ||w - w_global||^2).
* Communication-efficient uploads: 8-bit quantised updates ("q8") or top-10 % sparsified
  updates with client-side error feedback ("top10").
* Baselines at the SAME total number of training episodes: every area alone ("local") and
  one cloud learner on everyone's raw trajectories ("pooled").
* ``finetune``: few-shot adaptation of the global model inside an unseen area (the area's
  ground station fine-tunes it on its own missions), against training from scratch there.

All models are judged on 8 unseen test areas with checkpoints chosen on 4 validation areas.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .baselines import TEST_SEED_BASE
from .config import HET_TEST, HET_TRAIN, HET_VAL, hetero_area
from .env import RISUAVEnv
from .multiarea import collect, eval_on_areas
from .ppo import PPOConfig, RunningScale, n_params
from .trainers import TRAIN_SEED_BASE, VAL_SEED_BASE, Learner, TrainConfig, _write


@dataclass
class FedHetConfig:
    mode: str = "fedavg"            # local | fedavg | fedprox | pooled
    participation: float = 0.5      # fraction of areas sampled per round (fedavg / fedprox)
    total_episodes: int = 11_520    # training episodes summed over all areas (equal for all modes)
    E: int = 4                      # local PPO updates per round per participating area
    mu: float = 0.01                # FedProx weight
    compress: str = "none"          # none | q8 | top10 (uploads only)
    seed: int = 0
    n_evals: int = 12               # validation checkpoints over training
    n_val_eps: int = 5              # per validation area
    n_test_eps: int = 12            # per test area (8 areas -> 96 missions)
    lam_rate: float = 0.1
    train_areas: tuple = HET_TRAIN
    val_areas: tuple = HET_VAL
    test_areas: tuple = HET_TEST
    ppo: PPOConfig = field(default_factory=lambda: PPOConfig(lr_anneal=True))


# ------------------------------------------------------------------ update compression

def compress_update(delta: dict, method: str, residual: dict | None) -> tuple[dict, float, dict | None]:
    """Returns (decompressed update as the server sees it, upload bytes, new residual)."""
    if method == "none":
        return delta, sum(v.numel() for v in delta.values()) * 4.0, residual
    out, nbytes = {}, 0.0
    new_res = {} if method == "top10" else None
    for k, v in delta.items():
        if method == "q8":
            scale = v.abs().max().clamp(min=1e-12) / 127.0
            out[k] = torch.round(v / scale).clamp(-127, 127) * scale
            nbytes += v.numel() * 1.0 + 4.0                     # int8 values + one fp32 scale
        elif method == "top10":
            x = v + (residual[k] if residual is not None else 0.0)   # error feedback
            kk = max(1, int(0.1 * x.numel()))
            idx = torch.topk(x.abs().flatten(), kk).indices
            sparse = torch.zeros_like(x).flatten()
            sparse[idx] = x.flatten()[idx]
            sparse = sparse.view_as(x)
            out[k] = sparse
            new_res[k] = x - sparse
            nbytes += kk * 8.0                                  # fp32 value + int32 index
        else:
            raise ValueError(method)
    return out, nbytes, new_res


# ------------------------------------------------------------------ helpers

def _actor_sd(lr: Learner) -> dict:
    return {k: v.detach().clone() for k, v in lr.actors[0].state_dict().items()}


def _set_lr(learners, frac):
    for lr in learners:
        lr.set_lr_frac(frac)


def _local_round(lr: Learner, env: RISUAVEnv, scale, n_updates: int, epu: int, seed_base: int,
                 lam_rate: float, update_actor: bool = True):
    """n_updates PPO updates on this area's own missions. Returns (episode metrics, transitions)."""
    ms_all, steps = [], 0
    for u in range(n_updates):
        traj, ms, st = collect(env, lr, scale, epu, lambda e, u=u: seed_base + u * epu + e)
        lr.update(traj, update_actor=update_actor)
        env.lam = (1 - lam_rate) * env.lam + lam_rate * float(np.mean([m["ee"] for m in ms]))
        ms_all += ms
        steps += st
    return ms_all, steps


# ------------------------------------------------------------------ training

def train_fedhet(fc: FedHetConfig, out_dir: Path, verbose: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    rng = np.random.default_rng(1_000 + fc.seed)
    train_cfgs = [hetero_area(s) for s in fc.train_areas]
    val_cfgs = [hetero_area(s) for s in fc.val_areas]
    test_cfgs = [hetero_area(s) for s in fc.test_areas]
    n = len(train_cfgs)
    epu = fc.ppo.episodes_per_update
    tc = TrainConfig(algo="ctde", seed=fc.seed, ppo=fc.ppo)
    t0 = time.time()

    envs = [RISUAVEnv(c) for c in train_cfgs]
    base = Learner(envs[0], tc)
    learners = [base] if fc.mode == "pooled" else [base] + [copy.deepcopy(base) for _ in range(n - 1)]
    if fc.mode != "pooled":
        for lr, env in zip(learners, envs):
            lr.env = env
    scales = [RunningScale(fc.ppo.gamma) for _ in learners]
    actor_b = n_params(base.actors[0]) * 4.0
    step_b = envs[0].n_agents * (envs[0].obs_dim + 4) * 4.0
    lam0 = envs[0].lam

    # rounds: every mode spends the same total number of training episodes
    if fc.mode in ("fedavg", "fedprox"):
        m = max(1, int(round(fc.participation * n)))
        per_round = m * fc.E * epu
    elif fc.mode == "local":
        m = n
        per_round = n * fc.E * epu
    else:   # pooled: one learner, fc.E updates of epu missions from a rotating set of areas
        m = n
        per_round = fc.E * epu
    n_rounds = max(1, fc.total_episodes // per_round)
    eval_rounds = set(np.unique(np.linspace(1, n_rounds, fc.n_evals).round().astype(int)).tolist())

    global_sd = _actor_sd(base)
    residuals = [None] * n
    comm_cross = comm_intra = 0.0
    best: dict = {}
    log_rows, eval_rows = [], []
    episodes_done = 0
    cursor = [0]

    def models_now():
        if fc.mode == "local":
            return {f"area{a}": lr.actors[0] for a, lr in enumerate(learners)}
        if fc.mode == "pooled":
            return {"global": learners[0].actors[0]}
        g = copy.deepcopy(base.actors[0])
        g.load_state_dict(global_sd)
        return {"global": g}

    for rnd in range(1, n_rounds + 1):
        if fc.ppo.lr_anneal:
            _set_lr(learners, max(0.1, 1.0 - (rnd - 1) / n_rounds))
        recent = []
        if fc.mode == "pooled":
            sb = TRAIN_SEED_BASE * (fc.seed + 1) + 9_000_000 + (rnd - 1) * per_round
            for u in range(fc.E):
                traj = None
                for e in range(epu):   # one batch mixes missions from several areas
                    a = cursor[0] % n
                    cursor[0] += 1
                    tr, ms, st = collect(envs[a], learners[0], scales[0], 1,
                                         lambda _e, u=u, e=e: sb + u * epu + e)
                    envs[a].lam = (1 - fc.lam_rate) * envs[a].lam + fc.lam_rate * ms[0]["ee"]
                    traj = tr if traj is None else {k: traj[k] + tr[k] for k in traj}
                    recent += ms
                    comm_cross += st * step_b                      # raw transitions to the cloud
                    comm_intra += st * step_b
                learners[0].update(traj)
            comm_cross += n * actor_b                              # model back to every area
            comm_intra += n * envs[0].n_agents * actor_b
        else:
            chosen = (np.arange(n) if fc.mode == "local"
                      else np.sort(rng.choice(n, size=m, replace=False)))
            updates = []
            for a in chosen:
                lr, env = learners[a], envs[a]
                if fc.mode != "local":
                    lr.actors[0].load_state_dict(global_sd)                    # download
                    comm_cross += actor_b
                    lr.prox = ({"actor." + k: v for k, v in global_sd.items()}
                               if fc.mode == "fedprox" else None)
                    lr.mu = fc.mu if fc.mode == "fedprox" else 0.0
                sb = TRAIN_SEED_BASE * (fc.seed + 1) + a * 1_000_000 + (rnd - 1) * fc.E * epu
                ms, st = _local_round(lr, env, scales[a], fc.E, epu, sb, fc.lam_rate)
                recent += ms
                comm_intra += st * step_b + fc.E * env.n_agents * actor_b
                if fc.mode != "local":
                    local_sd = _actor_sd(lr)
                    delta = {k: local_sd[k] - global_sd[k] for k in global_sd}
                    sent, nb, residuals[a] = compress_update(delta, fc.compress, residuals[a])
                    updates.append(sent)
                    comm_cross += nb                                            # upload
            if updates:   # equal sample counts -> equal weights
                global_sd = {k: global_sd[k] + torch.stack([u[k] for u in updates]).mean(0) for k in global_sd}
        episodes_done += per_round
        log_rows.append({"round": rnd, "episodes": episodes_done, "time_s": time.time() - t0,
                         "comm_cross_MB": comm_cross / 1e6, "comm_intra_MB": comm_intra / 1e6,
                         "train_ee": float(np.mean([x["ee"] for x in recent]))})

        if rnd in eval_rounds:
            row = {"round": rnd, "episodes": episodes_done, "comm_cross_MB": comm_cross / 1e6}
            for key, actor in models_now().items():
                ev = eval_on_areas(actor, val_cfgs, fc.n_val_eps, VAL_SEED_BASE, lam0)
                row[f"val_ee_{key}"] = ev["ee"]
                sel = (ev["reached_final"] >= 0.9, ev["ee"])
                if key not in best or sel > best[key][0]:
                    best[key] = (sel, copy.deepcopy(actor.state_dict()), rnd, comm_cross / 1e6, episodes_done)
            row["val_ee_mean"] = float(np.mean([v for k, v in row.items() if k.startswith("val_ee_")]))
            eval_rows.append(row)
            if verbose:
                print(f"  [{fc.mode} p{fc.participation} {fc.compress} s{fc.seed}] round {rnd:3d}/{n_rounds}"
                      f"  episodes {episodes_done:6d}  val EE {row['val_ee_mean']:7.0f}"
                      f"  cross-area {comm_cross / 1e6:7.1f} MB  ({time.time() - t0:5.0f}s)", flush=True)

    tests = {}
    template = copy.deepcopy(base.actors[0])
    for key, (_, sd, rnd_b, comm_b, ep_b) in best.items():
        template.load_state_dict(sd)
        ev = eval_on_areas(template, test_cfgs, fc.n_test_eps, TEST_SEED_BASE, lam0)
        ev.update({"best_round": rnd_b, "comm_cross_MB_at_best": comm_b, "episodes_at_best": ep_b})
        tests[key] = ev
        torch.save(sd, out_dir / f"best_{key}.pt")
    ees = [t["ee"] for t in tests.values()]
    summary = {
        "mode": fc.mode, "participation": fc.participation if fc.mode in ("fedavg", "fedprox") else None,
        "compress": fc.compress, "seed": fc.seed, "n_rounds": n_rounds, "episodes": episodes_done,
        "comm_cross_MB": comm_cross / 1e6, "comm_intra_MB": comm_intra / 1e6,
        "test_ee": float(np.mean(ees)),
        "test_ee_per_area": list(np.mean([t["ee_per_area"] for t in tests.values()], axis=0)),
        "test_ee_across_models_std": float(np.std(ees, ddof=1)) if len(ees) > 1 else 0.0,
        "test_delivered": float(np.mean([t["delivered_frac"] for t in tests.values()])),
        "test_reached": float(np.mean([t["reached_final"] for t in tests.values()])),
        "test_collisions": float(np.mean([t["collisions"] for t in tests.values()])),
        "train_time_s": time.time() - t0, "models": tests,
    }
    _write(out_dir / "train_log.csv", log_rows)
    _write(out_dir / "eval_log.csv", eval_rows)
    (out_dir / "test.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "config.json").write_text(json.dumps(asdict(fc), indent=2, default=str))
    if verbose:
        print(f"  [{fc.mode} s{fc.seed}] TEST on {len(test_cfgs)} unseen heterogeneous areas: "
              f"EE {summary['test_ee']:.0f}  cross-area {comm_cross / 1e6:.1f} MB")
    return summary


# ------------------------------------------------------------------ few-shot adaptation

def finetune(global_actor_path: Path | None, seed: int, out_dir: Path, episodes: int = 320,
             warmup_updates: int = 8, test_areas: tuple = HET_TEST, n_test_eps: int = 12,
             verbose: bool = True) -> dict:
    """Adapt a model inside each unseen test area using only that area's own missions.

    ``global_actor_path`` = None trains from scratch (control). Otherwise the global actor is
    loaded, a fresh critic is first trained alone for ``warmup_updates`` updates (actor
    frozen), then both are fine-tuned. Same number of missions in both cases."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    ppo = PPOConfig(lr_anneal=True)
    epu = ppo.episodes_per_update
    n_upd = episodes // epu
    rows = []
    for i, s in enumerate(test_areas):
        cfg = hetero_area(s)
        env = RISUAVEnv(cfg)
        lr = Learner(env, TrainConfig(algo="ctde", seed=seed, ppo=ppo))
        if global_actor_path is not None:
            lr.actors[0].load_state_dict(torch.load(global_actor_path))
        zero = eval_on_areas(lr.actors[0], [cfg], n_test_eps, TEST_SEED_BASE, env.lam)["ee"]
        scale = RunningScale(ppo.gamma)
        sb = TRAIN_SEED_BASE * (seed + 1) + 50_000_000 + i * 1_000_000
        for u in range(n_upd):
            lr.set_lr_frac(max(0.1, 1.0 - u / n_upd))
            frozen = global_actor_path is not None and u < warmup_updates
            _local_round(lr, env, scale, 1, epu, sb + u * epu, 0.1, update_actor=not frozen)
        after = eval_on_areas(lr.actors[0], [cfg], n_test_eps, TEST_SEED_BASE, env.lam)["ee"]
        rows.append({"area": s, "ee_before": zero, "ee_after": after})
        if verbose:
            print(f"  [finetune {'global' if global_actor_path else 'scratch'} s{seed}] area {s}: "
                  f"{zero:7.0f} -> {after:7.0f}", flush=True)
    res = {"seed": seed, "init": "global" if global_actor_path else "scratch", "episodes": episodes,
           "ee_before": float(np.mean([r["ee_before"] for r in rows])),
           "ee_after": float(np.mean([r["ee_after"] for r in rows])), "per_area": rows}
    _write(out_dir / "per_area.csv", rows)
    (out_dir / "test.json").write_text(json.dumps(res, indent=2))
    return res
