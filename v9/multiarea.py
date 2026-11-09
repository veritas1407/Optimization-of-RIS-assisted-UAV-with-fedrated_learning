"""Multi-area federated learning (hierarchical: CTDE inside an area, FL across areas).

Each city area has its own ground station (GCS) that trains its 3-UAV swarm with
MAPPO-CTDE on local experience. Areas differ in building layout and RIS placement
(non-IID clients). Modes:

* local  - every area trains alone; nothing crosses area boundaries.
* fedavg - every E local PPO updates, the area GCSs upload actor + critic weights to a
           cloud aggregator, which averages and broadcasts them (no raw data leaves an area).
* pooled - every area ships its raw trajectories to one cloud learner that trains a single
           model on all areas (upper bound; maximal data sharing).

All models are judged on held-out, never-seen areas (generalization), with checkpoints
selected on separate validation areas.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .baselines import TEST_SEED_BASE, run_episode
from .config import TEST_AREAS, TRAIN_AREAS, VAL_AREAS, random_area
from .env import RISUAVEnv
from .ppo import PPOConfig, RunningScale, n_params
from .trainers import TRAIN_SEED_BASE, VAL_SEED_BASE, Learner, TrainConfig, _write


@dataclass
class MultiAreaConfig:
    mode: str = "fedavg"                 # local | fedavg | pooled
    episodes_per_area: int = 3000
    E: int = 4
    fed_parts: str = "actor"             # fedavg: "actor" (critics stay at each GCS) | "both"
    seed: int = 0
    eval_every: int = 200                # in per-area episodes (multiple of episodes_per_update)
    n_val_eps: int = 5                   # episodes per validation area (4 areas -> 20)
    n_test_eps: int = 12                 # episodes per test area (8 areas -> 96)
    lam_rate: float = 0.1                # Dinkelbach price update rate (as in trainers)
    train_areas: tuple = TRAIN_AREAS
    val_areas: tuple = VAL_AREAS
    test_areas: tuple = TEST_AREAS
    ppo: PPOConfig = field(default_factory=PPOConfig)


class ActorPolicy:
    def __init__(self, actor):
        self.actor = actor

    def policy(self, env, obs):
        return np.array([self.actor.act(o, deterministic=True)[0] for o in obs])


def eval_on_areas(actor, area_cfgs, n_eps: int, seed_base: int, lam: float) -> dict:
    pol = ActorPolicy(actor)
    per_area, rows = [], []
    for cfg in area_cfgs:
        env = RISUAVEnv(cfg)
        env.lam = lam
        rs = [run_episode(env, pol.policy, seed_base + j) for j in range(n_eps)]
        rows += rs
        per_area.append(float(np.mean([r["ee"] for r in rs])))
    out = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    out["ee_per_area"] = per_area
    out["ee_per_episode"] = [float(r["ee"]) for r in rows]
    return out


def collect(env: RISUAVEnv, learner: Learner, scale: RunningScale, n_eps: int, seed_fn,
            area_cfgs=None, area_cursor=None):
    """Run n_eps episodes; returns (trajectory dict, list of episode metrics, transitions)."""
    traj = {k: [] for k in ("obs", "state", "u", "logp", "val", "rew", "done", "mask")}
    metrics, steps = [], 0
    for e in range(n_eps):
        if area_cfgs is not None:
            env.set_scenario(area_cfgs[area_cursor[0] % len(area_cfgs)])
            area_cursor[0] += 1
        obs, state = env.reset(seed=seed_fn(e))
        done = False
        while not done:
            act, u, lp = learner.act(obs)
            val = learner.values(obs, state)
            nobs, nstate, r, done, info = env.step(act)
            traj["obs"].append(obs), traj["state"].append(state), traj["u"].append(u)
            traj["logp"].append(lp), traj["val"].append(val)
            traj["rew"].append(scale(r, done)), traj["done"].append(float(done))
            traj["mask"].append(1.0 - info["overridden"].astype(np.float32))
            obs, state = nobs, nstate
            steps += 1
        metrics.append(info["episode"])
    return traj, metrics, steps


def _avg_into(learners: list[Learner], parts: str = "actor"):
    """FedAvg across area learners of the collector actor (role 0), and of the
    centralized critic too if parts == "both"."""
    for part in (("actor", "critic") if parts == "both" else ("actor",)):
        mods = [lr.actors[0] if part == "actor" else lr.critic for lr in learners]
        sds = [m.state_dict() for m in mods]
        avg = {k: torch.stack([sd[k] for sd in sds]).mean(0) for k in sds[0]}
        for m in mods:
            m.load_state_dict(avg)


def train_multiarea(mc: MultiAreaConfig, out_dir: Path, verbose: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    np.random.seed(mc.seed)
    train_cfgs = [random_area(s) for s in mc.train_areas]
    val_cfgs = [random_area(s) for s in mc.val_areas]
    test_cfgs = [random_area(s) for s in mc.test_areas]
    tc = TrainConfig(algo="ctde", seed=mc.seed, ppo=mc.ppo)
    epu = mc.ppo.episodes_per_update
    n_areas = len(train_cfgs)
    t0 = time.time()
    comm = 0.0          # cross-area traffic (between area ground stations and the cloud)
    comm_intra = 0.0    # within-area traffic (UAVs <-> their ground station), all modes

    if mc.mode == "pooled":
        envs = [RISUAVEnv(train_cfgs[0])]
        learners = [Learner(envs[0], tc)]
    else:
        envs = [RISUAVEnv(c) for c in train_cfgs]
        learners = [Learner(envs[0], tc)]
        for env in envs[1:]:   # identical initialisation in every area (cloud broadcasts it)
            lr = Learner(env, tc)
            lr.load_state_dict(learners[0].state_dict())
            lr.critic.load_state_dict(learners[0].critic.state_dict())
            learners.append(lr)
    scales = [RunningScale(mc.ppo.gamma) for _ in learners]
    lam = envs[0].lam
    actor_b = n_params(learners[0].actors[0]) * 4.0
    critic_b = n_params(learners[0].critic) * 4.0
    step_b = envs[0].n_agents * (envs[0].obs_dim + 4) * 4.0

    def global_actor():
        if mc.mode == "local":
            return None
        if mc.mode == "pooled":
            return learners[0].actors[0]
        g = copy.deepcopy(learners[0].actors[0])
        sds = [lr.actors[0].state_dict() for lr in learners]
        g.load_state_dict({k: torch.stack([sd[k] for sd in sds]).mean(0) for k in sds[0]})
        return g

    best = {}        # model key -> (sel_key, state_dict, episode)
    log_rows, eval_rows = [], []
    n_rounds = mc.episodes_per_area // epu
    cursor = [0]
    for rnd in range(1, n_rounds + 1):
        ep_done = rnd * epu
        recent = []
        if mc.mode == "pooled":
            # same total experience per round as the per-area learners, one update per round
            base = TRAIN_SEED_BASE * (mc.seed + 1) + 9_000_000
            k0 = cursor[0]
            traj, ms, steps = collect(envs[0], learners[0], scales[0], n_areas * epu,
                                      lambda e: base + k0 + e, train_cfgs, cursor)
            learners[0].update(traj)
            comm += steps * step_b + n_areas * actor_b       # raw data up, model down to each area
            comm_intra += steps * step_b + n_areas * envs[0].n_agents * actor_b
            envs[0].lam = (1 - mc.lam_rate) * envs[0].lam + mc.lam_rate * float(np.mean([m["ee"] for m in ms]))
            recent += ms
        else:
            for a, (env, lr) in enumerate(zip(envs, learners)):
                base = TRAIN_SEED_BASE * (mc.seed + 1) + a * 1_000_000 + (rnd - 1) * epu
                traj, ms, steps = collect(env, lr, scales[a], epu, lambda e: base + e)
                lr.update(traj)
                comm_intra += steps * step_b + env.n_agents * actor_b
                env.lam = (1 - mc.lam_rate) * env.lam + mc.lam_rate * float(np.mean([m["ee"] for m in ms]))
                recent += ms
            if mc.mode == "fedavg" and rnd % mc.E == 0:
                _avg_into(learners, mc.fed_parts)
                comm += 2.0 * (actor_b + (critic_b if mc.fed_parts == "both" else 0.0)) * n_areas
        log_rows.append({"episodes_per_area": ep_done, "time_s": time.time() - t0, "comm_MB": comm / 1e6,
                         "comm_intra_MB": comm_intra / 1e6, "lam": float(np.mean([e.lam for e in envs])),
                         "train_ee": float(np.mean([m["ee"] for m in recent])),
                         "train_reached": float(np.mean([m["reached_final"] for m in recent]))})

        if ep_done % mc.eval_every == 0 or rnd == n_rounds:
            models = ({f"area{a}": lr.actors[0] for a, lr in enumerate(learners)} if mc.mode == "local"
                      else {"global": global_actor()})
            row = {"episodes_per_area": ep_done, "comm_MB": comm / 1e6}
            for key, actor in models.items():
                ev = eval_on_areas(actor, val_cfgs, mc.n_val_eps, VAL_SEED_BASE, lam)
                row[f"val_ee_{key}"] = ev["ee"]
                sel = (ev["reached_final"] >= 0.9, ev["ee"])
                if key not in best or sel > best[key][0]:
                    best[key] = (sel, copy.deepcopy(actor.state_dict()), ep_done)
            row["val_ee_mean"] = float(np.mean([v for k, v in row.items() if k.startswith("val_ee_")]))
            eval_rows.append(row)
            if verbose:
                print(f"  [{mc.mode}{'_E' + str(mc.E) if mc.mode == 'fedavg' else ''} s{mc.seed}] "
                      f"ep/area {ep_done:5d}  val EE (unseen areas) {row['val_ee_mean']:7.0f}  "
                      f"train EE {log_rows[-1]['train_ee']:7.0f}  cross-area comm {comm / 1e6:8.1f} MB"
                      f"  ({time.time() - t0:5.0f}s)", flush=True)

    # test every selected model on the unseen test areas
    tests = {}
    template = copy.deepcopy(learners[0].actors[0])
    for key, (_, sd, ep) in best.items():
        template.load_state_dict(sd)
        ev = eval_on_areas(template, test_cfgs, mc.n_test_eps, TEST_SEED_BASE, lam)
        ev["best_episode"] = ep
        if mc.mode == "local":   # also how the area's model does on its own area
            a = int(key[4:])
            ev["own_area_ee"] = eval_on_areas(template, [train_cfgs[a]], mc.n_test_eps,
                                              TEST_SEED_BASE, lam)["ee"]
        tests[key] = ev
        torch.save(sd, out_dir / f"best_{key}.pt")
    ees = [t["ee"] for t in tests.values()]
    summary = {
        "mode": mc.mode, "E": mc.E if mc.mode == "fedavg" else None, "seed": mc.seed,
        "comm_MB": comm / 1e6, "comm_intra_MB": comm_intra / 1e6,
        "test_ee": float(np.mean(ees)), "test_ee_across_models_std": float(np.std(ees, ddof=1)) if len(ees) > 1 else 0.0,
        "test_delivered": float(np.mean([t["delivered_frac"] for t in tests.values()])),
        "test_reached": float(np.mean([t["reached_final"] for t in tests.values()])),
        "test_collisions": float(np.mean([t["collisions"] for t in tests.values()])),
        "train_time_s": time.time() - t0, "models": tests,
    }
    _write(out_dir / "train_log.csv", log_rows)
    _write(out_dir / "eval_log.csv", eval_rows)
    (out_dir / "test.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "config.json").write_text(json.dumps(asdict(mc), indent=2, default=str))
    if verbose:
        print(f"  [{mc.mode} s{mc.seed}] TEST on {len(test_cfgs)} unseen areas: EE {summary['test_ee']:.0f}"
              f"  delivered {summary['test_delivered']:.2f}  reached {summary['test_reached']:.2f}"
              f"  cross-area comm {comm / 1e6:.1f} MB")
    return summary
