"""Non-learning reference policies and a common evaluation loop.

A policy is a callable ``policy(env, obs) -> actions (n_agents, 2)`` that may read the
environment's public state (positions, devices, windows). Baselines use the same
environment, channels and seeds as the learned agents.
"""
from __future__ import annotations

import numpy as np

from . import physics as ph
from .env import RISUAVEnv

EVAL_SEED_BASE = 1_000_000


def _vel_to_action(v, v_max):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n > v_max:
        v = v * v_max / n
    return v / v_max


def _final_slot_targets(env):
    cfg = env.cfg
    f = np.asarray(cfg.final_xy, dtype=float)
    n = env.n_agents
    offs = (np.arange(n) - (n - 1) / 2.0) * min(cfg.d_min * 1.2, cfg.r_final)
    perp = np.array([1.0, -1.0]) / np.sqrt(2.0)
    return f[None, :] + offs[:, None] * perp[None, :]


def random_policy(env, obs, rng=np.random.default_rng(0)):
    return rng.uniform(-1.0, 1.0, (env.n_agents, 2))


def straight_policy(env, obs):
    """Constant-velocity straight line to the final point (Jiang et al.'s FT baseline)."""
    cfg = env.cfg
    tl = max(env._time_left(), cfg.dt)
    tgt = _final_slot_targets(env)
    acts = np.zeros((env.n_agents, 2))
    for a in range(env.n_agents):
        acts[a] = _vel_to_action((tgt[a] - env.pos[a, :2]) / tl, cfg.v_max)
    return acts


def greedy_policy(env, obs, cruise=None, margin=1.15):
    """Heuristic: each collector flies to the most valuable reachable device and orbits it.

    Devices that need proximity (blocked and not reachable through the RIS) are worth more.
    Every UAV leaves for the final point once the remaining time requires it. The aerial
    RIS UAV (if any) parks over the shadowed region.
    """
    cfg = env.cfg
    cruise = cruise or max(env.v_pmin, 8.0)
    t0 = env.slot * cfg.dt
    tl = env._time_left()
    tgt_final = _final_slot_targets(env)
    alive = (env.remaining > 1.0) & (env.win_end > t0)
    active = alive & (env.win_start <= t0)
    ris_on = env.use_ris and cfg.arbitration in ("timeshare", "common")
    taken = set()
    acts = np.zeros((env.n_agents, 2))
    for a in range(env.n_agents):
        p = env.pos[a, :2]
        d_final = np.linalg.norm(tgt_final[a] - p)
        if d_final * margin >= cfg.v_max * 0.9 * (tl - cfg.dt):
            acts[a] = _vel_to_action((tgt_final[a] - p) / max(tl - cfg.dt, cfg.dt) * margin, cfg.v_max)
            continue
        if a >= env.K:  # aerial RIS carrier: hover-orbit above the shadowed region
            x0, y0, x1, y1 = cfg.shadow_region
            goal = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        else:
            dist = np.linalg.norm(env.dev[:, :2] - p, axis=1)
            needs = np.where(env.blocked[:, a] & ~(ris_on & env.ris_elig), 1.0, 0.35)
            when = np.where(active, 1.0, 0.5)
            score = (env.remaining / (cfg.data_mbit[1] * 1e6)) * needs * when / (1.0 + dist / 150.0)
            score = np.where(alive, score, -np.inf)
            for i in taken:
                score[i] = -np.inf
            if not np.isfinite(score).any():
                acts[a] = _vel_to_action((tgt_final[a] - p) / max(tl, cfg.dt), cfg.v_max)
                continue
            i = int(np.argmax(score))
            taken.add(i)
            goal = env.dev[i, :2]
        to = goal - p
        dist = np.linalg.norm(to)
        if dist > 15.0:
            v = to / dist * min(cfg.v_max, max(cruise, dist / (2 * cfg.dt)))
        else:  # orbit at cruise speed: cheaper than hovering
            v = np.array([-to[1], to[0]]) if dist > 1e-6 else np.array([1.0, 0.0])
            v = v / np.linalg.norm(v) * cruise
        acts[a] = _vel_to_action(v, cfg.v_max)
    return acts


POLICIES = {"random": random_policy, "straight": straight_policy, "greedy": greedy_policy}


def run_episode(env: RISUAVEnv, policy, seed: int, record: bool = False):
    obs, _ = env.reset(seed=seed)
    done = False
    log = []
    ret = 0.0
    while not done:
        act = policy(env, obs)
        obs, _, r, done, info = env.step(act)
        ret += r
        if record:
            log.append(info)
    m = dict(info["episode"])
    m["return"] = ret
    return (m, log) if record else m


def evaluate(cfg, policy, n_episodes: int = 20, seed_base: int = EVAL_SEED_BASE) -> dict:
    env = RISUAVEnv(cfg)
    rows = [run_episode(env, policy, seed_base + j) for j in range(n_episodes)]
    keys = rows[0].keys()
    return {k: (float(np.nanmean([r[k] for r in rows])), float(np.nanstd([r[k] for r in rows])))
            for k in keys}
