"""Non-learning reference policies and a common evaluation loop.

A policy is a callable ``policy(env, obs) -> actions (n_agents, 2)``. Baselines use the
same environment, channels and seeds as the learned agents.

* ``random_policy``   - uniform random commands (reproducible per episode).
* ``straight_policy`` - constant-velocity straight line (Jiang et al.'s FT baseline).
* ``Greedy``          - parameterised heuristic. ``centralized=True`` sees every device and
  assigns devices to UAVs jointly (a strong reference with full information);
  ``centralized=False`` gives each UAV only what a learned UAV observes (its top-K devices
  and who served them last slot) and no joint assignment (the like-for-like rival).
  Parameters are tuned on validation episodes by ``v9.tune``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .env import RISUAVEnv

TEST_SEED_BASE = 5_000_000       # fresh test set (v9.1): never used during development
OLD_TEST_SEED_BASE = 1_000_000   # v9.0 test seeds (reused during development)
EVAL_SEED_BASE = TEST_SEED_BASE
N_TEST = 100
TUNED_DIR = Path("results_v9/tune")


def _vel_to_action(v, v_max):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n > v_max:
        v = v * v_max / n
    return v / v_max


def random_policy(env, obs):
    rng = np.random.default_rng([env.episode_seed, env.slot])
    return rng.uniform(-1.0, 1.0, (env.n_agents, 2))


def straight_policy(env, obs):
    """Constant-velocity straight line to the final point (Jiang et al.'s FT baseline)."""
    cfg = env.cfg
    tl = max(env._time_left(), cfg.dt)
    acts = np.zeros((env.n_agents, 2))
    for a in range(env.n_agents):
        acts[a] = _vel_to_action((env.final_targets[a] - env.pos[a, :2]) / tl, cfg.v_max)
    return acts


@dataclass(frozen=True)
class GreedyParams:
    margin: float = 1.15        # head home when distance x margin exceeds the reachable range
    vcap: float = 20.0          # maximum approach speed (m/s)
    need_w: float = 0.35        # weight of devices that do NOT need proximity
    dscale: float = 150.0       # distance discount scale (m)
    inactive_w: float = 0.5     # weight of devices whose window has not opened yet
    orbit_r: float = 15.0       # orbit radius around the target device (m)


class Greedy:
    """Heuristic: fly to the most valuable device and orbit it at the minimum-power speed;
    leave for the final point when the remaining time requires it."""

    def __init__(self, params: GreedyParams = GreedyParams(), centralized: bool = True):
        self.p = params
        self.centralized = centralized

    def __call__(self, env, obs):
        cfg, P = env.cfg, self.p
        cruise = max(env.v_pmin, 8.0)
        t0 = env.slot * cfg.dt
        tl = env._time_left()
        alive = (env.remaining > 1.0) & (env.win_end > t0)
        active = alive & (env.win_start <= t0)
        ris_on = env.use_ris and cfg.arbitration in ("timeshare", "common")
        taken = set()
        acts = np.zeros((env.n_agents, 2))
        for a in range(env.n_agents):
            p = env.pos[a, :2]
            tf = env.final_targets[a]
            d_final = np.linalg.norm(tf - p)
            if d_final * P.margin >= cfg.v_max * 0.9 * (tl - cfg.dt):
                acts[a] = _vel_to_action((tf - p) / max(tl - cfg.dt, cfg.dt) * P.margin, cfg.v_max)
                continue
            if a >= env.K:  # aerial RIS carrier: orbit above the shadowed region
                x0, y0, x1, y1 = cfg.shadow_region
                goal = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            else:
                dist = np.linalg.norm(env.dev[:, :2] - p, axis=1)
                needs = np.where(env.blocked[:, a] & ~(ris_on & env.ris_elig), 1.0, P.need_w)
                when = np.where(active, 1.0, P.inactive_w)
                score = (env.remaining / (cfg.data_mbit[1] * 1e6)) * needs * when / (1.0 + dist / P.dscale)
                score = np.where(alive, score, -np.inf)
                if self.centralized:
                    for i in taken:
                        score[i] = -np.inf
                else:
                    # only the devices a learned UAV would observe; avoid devices another
                    # UAV served in the last slot (the information in the observation)
                    key = np.where(alive, dist / cfg.area + 0.3 * (~active), np.inf)
                    visible = np.zeros(env.I, bool)
                    visible[np.argsort(key)[: env.top_k]] = True
                    other = (env.last_served >= 0) & (env.last_served != a)
                    score = np.where(visible & ~other, score, -np.inf)
                if not np.isfinite(score).any():
                    acts[a] = _vel_to_action((tf - p) / max(tl, cfg.dt), cfg.v_max)
                    continue
                i = int(np.argmax(score))
                taken.add(i)
                goal = env.dev[i, :2]
            to = goal - p
            dist = np.linalg.norm(to)
            if dist > P.orbit_r:
                v = to / dist * min(P.vcap, max(cruise, dist / (2 * cfg.dt)))
            else:  # orbit at cruise speed: cheaper than hovering
                v = np.array([-to[1], to[0]]) if dist > 1e-6 else np.array([1.0, 0.0])
                v = v / np.linalg.norm(v) * cruise
            acts[a] = _vel_to_action(v, cfg.v_max)
        return acts

    def describe(self) -> dict:
        return {"centralized": self.centralized, **asdict(self.p)}


greedy_policy = Greedy()   # untuned defaults (the v9.0 heuristic)


def tuned_greedy(target: str, centralized: bool = True) -> Greedy:
    """Greedy with the parameters chosen by ``v9.tune`` on validation episodes for ``target``
    (a preset name or "multiarea"); falls back to the defaults if no tuning file exists."""
    f = TUNED_DIR / f"greedy_{target}_{'central' if centralized else 'decentral'}.json"
    params = GreedyParams(**json.loads(f.read_text())["params"]) if f.exists() else GreedyParams()
    return Greedy(params, centralized=centralized)

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


def episode_values(cfg, policy, n: int, seed_base: int, key: str = "ee") -> np.ndarray:
    env = RISUAVEnv(cfg)
    return np.array([run_episode(env, policy, seed_base + j)[key] for j in range(n)])


def evaluate(cfg, policy, n_episodes: int = N_TEST, seed_base: int = TEST_SEED_BASE) -> dict:
    env = RISUAVEnv(cfg)
    rows = [run_episode(env, policy, seed_base + j) for j in range(n_episodes)]
    keys = rows[0].keys()
    return {k: (float(np.nanmean([r[k] for r in rows])), float(np.nanstd([r[k] for r in rows], ddof=1)))
            for k in keys}
