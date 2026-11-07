"""Command-line entry point for v9 experiments.

    python -m v9.run --exp gate
    python -m v9.run --exp ctde --seed 0
    python -m v9.run --exp fed --algo fedavg --E 5 --seed 0

Every experiment writes CSV/NPZ/JSON files into ``results_v9/<exp>/``; figures are
drawn from those files by ``v9.plots``.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .analysis import coverage_map, upload_time_table
from .baselines import POLICIES, evaluate
from .config import get_preset

RESULTS = Path("results_v9")


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def baseline_rows(cfg, policies, arbitrations, n_eval):
    rows = []
    for arb in arbitrations:
        c = replace(cfg, arbitration=arb)
        for pol in policies:
            m = evaluate(c, POLICIES[pol], n_eval)
            row = {"preset": cfg.name, "policy": pol, "arbitration": arb}
            for k, (mu, sd) in m.items():
                row[f"{k}_mean"], row[f"{k}_std"] = mu, sd
            rows.append(row)
            print(f"  {cfg.name:12s} {pol:8s} {arb:9s} EE={m['ee'][0]:9.0f} ± {m['ee'][1]:6.0f}"
                  f"  delivered={m['delivered_frac'][0]:.2f}", flush=True)
    return rows


def exp_gate(args):
    out = RESULTS / "gate"
    out.mkdir(parents=True, exist_ok=True)
    cfg = get_preset("building")
    x0, y0, x1, y1 = cfg.shadow_region
    shadow_dev = ((x0 + x1) / 2, (y0 + y1) / 2)
    print("coverage maps ...")
    cmap = coverage_map(cfg, shadow_dev, grid_n=100, n_draws=8)
    np.savez(out / "coverage_shadow.npz", **cmap)
    omap = coverage_map(cfg, (150.0, 300.0), grid_n=100, n_draws=8)
    np.savez(out / "coverage_open.npz", **omap)
    d_mid = 0.5 * sum(cfg.data_mbit) * 1e6
    _write_csv(out / "upload_time.csv", upload_time_table(cfg, d_mid))
    print("baselines ...")
    rows = baseline_rows(cfg, ["random", "straight", "greedy"],
                         ["none", "random", "common", "timeshare"], args.n_eval)
    rows += baseline_rows(get_preset("aerial"), ["straight", "greedy"], ["timeshare"], args.n_eval)
    rows += baseline_rows(get_preset("legacy_small"), ["random", "straight", "greedy"],
                          ["none", "random", "timeshare"], args.n_eval)
    _write_csv(out / "baselines.csv", rows)
    summary = {
        "data_mbit": cfg.data_mbit,
        "upload_window_s": cfg.window_s,
        "area_frac_shadow_feasible_noris": float(np.mean(cmap["rate_dir"] * cfg.window_s > d_mid)),
        "area_frac_shadow_feasible_ris": float(np.mean(cmap["rate_ris"] * cfg.window_s > d_mid)),
        "config": asdict(cfg),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


def exp_train(args):
    from .ppo import PPOConfig
    from .trainers import TrainConfig, train

    overrides = json.loads(args.scenario) if args.scenario else {}
    cfg = get_preset(args.preset, **overrides)
    tc = TrainConfig(algo=args.algo, episodes=args.episodes, E=args.E, mu=args.mu, seed=args.seed,
                     lam_update=args.lam_update, eval_every=args.eval_every,
                     ppo=PPOConfig(**(json.loads(args.ppo) if args.ppo else {})))
    tag = args.tag or (f"{args.algo}" + (f"_E{args.E}" if args.algo.startswith("fed") else ""))
    out = RESULTS / "train" / args.preset / tag / f"seed{args.seed}"
    if args.skip_existing and (out / "test.json").exists():
        print(f"  [{tag} s{args.seed}] already done, skipping")
        return
    train(cfg, tc, out)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, choices=["gate", "train"])
    p.add_argument("--n-eval", type=int, default=20)
    p.add_argument("--preset", default="building")
    p.add_argument("--algo", default="ctde", choices=["ctde", "ippo", "fedavg", "fedprox", "fedper"])
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--E", type=int, default=1)
    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--lam-update", action="store_true")
    p.add_argument("--scenario", default="", help="JSON dict of ScenarioConfig overrides")
    p.add_argument("--ppo", default="", help="JSON dict of PPOConfig overrides")
    p.add_argument("--tag", default="")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args(argv)
    t = time.time()
    {"gate": exp_gate, "train": exp_train}[args.exp](args)
    print(f"done in {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
