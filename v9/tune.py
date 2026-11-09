"""Tune the Greedy heuristic on validation episodes (never on test episodes).

    python -m v9.tune --target building            # centralized + decentralized
    python -m v9.tune --target multiarea           # on the 4 validation areas

Two stages: random search over a grid (short evaluation), then the best candidates are
re-evaluated on more validation episodes. Selection: reach the final region in >= 99 % of
episodes with (almost) no collisions, then highest mean EE. Results go to
results_v9/tune/greedy_<target>_<central|decentral>.json and are picked up by
``baselines.tuned_greedy``.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

from .baselines import TUNED_DIR, Greedy, GreedyParams, run_episode
from .config import VAL_AREAS, get_preset, random_area
from .env import RISUAVEnv

VAL_SEED_BASE = 2_000_000   # same validation seeds as the RL checkpoint selection

GRID = {
    "margin": [0.95, 1.0, 1.05, 1.15, 1.3],
    "vcap": [8.0, 10.0, 12.0, 15.0, 20.0],
    "need_w": [0.2, 0.35, 0.6, 1.0],
    "dscale": [75.0, 150.0, 300.0, 600.0, 1000.0, 3000.0],
    "inactive_w": [0.1, 0.25, 0.5, 0.75, 1.0],
    "orbit_r": [5.0, 10.0, 15.0, 25.0, 40.0],
}


def target_cfgs(target: str):
    if target == "multiarea":
        return [random_area(s) for s in VAL_AREAS]
    return [get_preset(target)]


def _eval(args):
    target, params, centralized, n_eps = args
    pol = Greedy(GreedyParams(**params), centralized=centralized)
    cfgs = target_cfgs(target)
    rows = []
    for c, cfg in enumerate(cfgs):
        env = RISUAVEnv(cfg)
        per = max(1, n_eps // len(cfgs))
        rows += [run_episode(env, pol, VAL_SEED_BASE + c * 10_000 + j) for j in range(per)]
    return {k: float(np.mean([r[k] for r in rows])) for k in ("ee", "reached_final", "collisions",
                                                              "delivered_frac")}


def _key(m):
    return (m["reached_final"] >= 0.99 and m["collisions"] <= 0.05, m["ee"])


def tune(target: str, centralized: bool, n_random: int = 150, n1: int = 20, n2: int = 80,
         top: int = 10, seed: int = 0, workers: int = 4) -> dict:
    rng = np.random.default_rng(seed)
    cands = [asdict(GreedyParams()),
             dict(margin=1.0, vcap=12.0, need_w=1.0, dscale=1000.0, inactive_w=0.5, orbit_r=15.0)]
    while len(cands) < n_random:
        c = {k: float(rng.choice(v)) for k, v in GRID.items()}
        if c not in cands:
            cands.append(c)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        s1 = list(ex.map(_eval, [(target, c, centralized, n1) for c in cands]))
        order = sorted(range(len(cands)), key=lambda i: _key(s1[i]), reverse=True)[:top]
        if 0 not in order:
            order.append(0)       # always re-evaluate the untuned defaults for reference
        s2 = list(ex.map(_eval, [(target, cands[i], centralized, n2) for i in order]))
    best = max(range(len(order)), key=lambda j: _key(s2[j]))
    default = s2[order.index(0)]
    out = {"target": target, "centralized": centralized, "params": cands[order[best]],
           "val": s2[best], "val_default_params": default, "n_candidates": len(cands),
           "n_val_stage1": n1, "n_val_stage2": n2, "seconds": time.time() - t0,
           "stage2": [{"params": cands[i], **s2[j]} for j, i in enumerate(order)]}
    TUNED_DIR.mkdir(parents=True, exist_ok=True)
    tag = "central" if centralized else "decentral"
    (TUNED_DIR / f"greedy_{target}_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"  {target:15s} {tag:9s} val EE {s2[best]['ee']:7.0f} (defaults {default['ee']:7.0f})  "
          f"reached {s2[best]['reached_final']:.2f}  params {cands[order[best]]}  "
          f"({out['seconds']:.0f}s)", flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="building")
    p.add_argument("--mode", default="both", choices=["both", "central", "decentral"])
    p.add_argument("--n-random", type=int, default=150)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    for cen in {"both": (True, False), "central": (True,), "decentral": (False,)}[a.mode]:
        tune(a.target, cen, n_random=a.n_random, workers=a.workers)


if __name__ == "__main__":
    main()
