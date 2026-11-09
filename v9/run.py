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
from .baselines import N_TEST, POLICIES, TEST_SEED_BASE, evaluate
from .config import get_preset

RESULTS = Path("results_v9")


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def gate_policies(preset: str, arb: str, names):
    """Policy callables for the gate. Tuned heuristics use the parameters chosen on validation
    episodes for this exact configuration (no-RIS runs use the no-RIS tuning)."""
    from .baselines import tuned_greedy

    target = "building_noris" if (preset == "building" and arb == "none") else preset
    make = {"random": lambda: POLICIES["random"], "straight": lambda: POLICIES["straight"],
            "greedy": lambda: POLICIES["greedy"],
            "greedy_tuned": lambda: tuned_greedy(target, True),
            "greedy_dec": lambda: tuned_greedy(target, False)}
    return {n: make[n]() for n in names}


def baseline_rows(cfg, policies, arbitrations, n_eval, per_episode: dict):
    from .baselines import TEST_SEED_BASE, run_episode
    from .env import RISUAVEnv
    from .stats import mean_ci

    rows = []
    for arb in arbitrations:
        c = replace(cfg, arbitration=arb)
        env = RISUAVEnv(c)
        for pname, pol in gate_policies(cfg.name, arb, policies).items():
            eps = [run_episode(env, pol, TEST_SEED_BASE + j) for j in range(n_eval)]
            row = {"preset": cfg.name, "policy": pname, "arbitration": arb}
            for k in eps[0]:
                st = mean_ci([e[k] for e in eps])
                row[f"{k}_mean"], row[f"{k}_std"], row[f"{k}_ci95"] = st["mean"], st["std"], st["ci95"]
            row["n"] = n_eval
            rows.append(row)
            per_episode[f"{cfg.name}|{pname}|{arb}"] = np.array([e["ee"] for e in eps])
            print(f"  {cfg.name:12s} {pname:12s} {arb:9s} EE={row['ee_mean']:9.0f} ± {row['ee_ci95']:5.0f} (95% CI)"
                  f"  delivered={row['delivered_frac_mean']:.2f}  reached={row['reached_final_mean']:.2f}",
                  flush=True)
    return rows


def area_feasibility(n_eval: int = 10) -> list[dict]:
    """Per-layout link budget and RIS value for all 16 procedurally generated areas."""
    from .baselines import TEST_SEED_BASE, run_episode, straight_policy, tuned_greedy
    from .config import TEST_AREAS, TRAIN_AREAS, VAL_AREAS, random_area
    from .env import RISUAVEnv

    rows = []
    for split, seeds in (("train", TRAIN_AREAS), ("val", VAL_AREAS), ("test", TEST_AREAS)):
        for s in seeds:
            cfg = random_area(s)
            x0, y0, x1, y1 = cfg.shadow_region
            cm = coverage_map(cfg, ((x0 + x1) / 2, (y0 + y1) / 2), grid_n=40, n_draws=4)
            d_mid = 0.5 * sum(cfg.data_mbit) * 1e6
            row = {"split": split, "layout_seed": s,
                   "frac_area_shadow_feasible_noris": float(np.mean(cm["rate_dir"] * cfg.window_s > d_mid)),
                   "frac_area_shadow_feasible_ris": float(np.mean(cm["rate_ris"] * cfg.window_s > d_mid))}
            for pname, pol in (("straight", straight_policy), ("greedy_tuned", tuned_greedy("multiarea"))):
                ee = {}
                for arb in ("none", "timeshare"):
                    env = RISUAVEnv(replace(cfg, arbitration=arb))
                    ee[arb] = np.array([run_episode(env, pol, TEST_SEED_BASE + j)["ee"] for j in range(n_eval)])
                row[f"{pname}_ee_noris"] = float(ee["none"].mean())
                row[f"{pname}_ee_ris"] = float(ee["timeshare"].mean())
                row[f"{pname}_ris_gain_pct"] = float(100 * (ee["timeshare"].mean() / ee["none"].mean() - 1))
            rows.append(row)
            print(f"  area {s:3d} ({split:5s}) shadow-feasible area {row['frac_area_shadow_feasible_noris']:.3f}"
                  f" -> {row['frac_area_shadow_feasible_ris']:.3f}   RIS gain straight "
                  f"{row['straight_ris_gain_pct']:+5.1f} %  greedy {row['greedy_tuned_ris_gain_pct']:+5.1f} %",
                  flush=True)
    return rows


def exp_gate(args):
    from .stats import paired

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
    per_ep: dict = {}
    pols = ["random", "straight", "greedy", "greedy_tuned", "greedy_dec"]
    rows = baseline_rows(cfg, pols, ["none", "random", "common", "timeshare"], args.n_eval, per_ep)
    rows += baseline_rows(get_preset("aerial"), ["straight", "greedy_tuned", "greedy_dec"], ["timeshare"],
                          args.n_eval, per_ep)
    rows += baseline_rows(get_preset("legacy_small"), ["random", "straight", "greedy"],
                          ["none", "random", "timeshare"], args.n_eval, per_ep)
    _write_csv(out / "baselines.csv", rows)
    np.savez(out / "baselines_per_episode.npz", **per_ep)
    # paired RIS value: same policy family, same test episodes, RIS (time-share) vs no RIS
    ris_value = {}
    for preset, pol in (("building", "straight"), ("building", "greedy_tuned"), ("building", "greedy_dec"),
                        ("legacy_small", "straight"), ("legacy_small", "greedy")):
        ris_value[f"{preset}|{pol}"] = paired(per_ep[f"{preset}|{pol}|timeshare"], per_ep[f"{preset}|{pol}|none"])
    for pol in ("straight", "greedy_tuned", "greedy_dec"):
        ris_value[f"aerial_vs_building|{pol}"] = paired(per_ep[f"aerial|{pol}|timeshare"],
                                                        per_ep[f"building|{pol}|timeshare"])
    if not args.skip_areas:
        print("per-area feasibility ...")
        _write_csv(out / "area_feasibility.csv", area_feasibility())
    summary = {
        "data_mbit": cfg.data_mbit,
        "upload_window_s": cfg.window_s,
        "area_frac_shadow_feasible_noris": float(np.mean(cmap["rate_dir"] * cfg.window_s > d_mid)),
        "area_frac_shadow_feasible_ris": float(np.mean(cmap["rate_ris"] * cfg.window_s > d_mid)),
        "ris_value_paired": ris_value,
        "n_test_episodes": args.n_eval,
        "config": asdict(cfg),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2, default=str))


def exp_train(args):
    from .ppo import PPOConfig
    from .trainers import TrainConfig, train

    overrides = json.loads(args.scenario) if args.scenario else {}
    cfg = get_preset(args.preset, **overrides)
    tc = TrainConfig(algo=args.algo, episodes=args.episodes, E=args.E, mu=args.mu, seed=args.seed,
                     lam_update=not args.fixed_lam, eval_every=args.eval_every, hidden=args.hidden,
                     ppo=PPOConfig(**(json.loads(args.ppo) if args.ppo else {})))
    tag = args.tag or (f"{args.algo}" + (f"_E{args.E}" if args.algo.startswith("fed") else ""))
    out = RESULTS / "train" / args.preset / tag / f"seed{args.seed}"
    if args.skip_existing and (out / "test.json").exists():
        print(f"  [{tag} s{args.seed}] already done, skipping")
        return
    train(cfg, tc, out)


def exp_multiarea(args):
    from .multiarea import MultiAreaConfig, train_multiarea
    from .ppo import PPOConfig

    mc = MultiAreaConfig(mode=args.mode, episodes_per_area=args.episodes, E=args.E, seed=args.seed,
                         fed_parts=args.fed_parts,
                         ppo=PPOConfig(**(json.loads(args.ppo) if args.ppo else {})))
    tag = args.tag or (args.mode + (f"_E{args.E}" + ("_both" if args.fed_parts == "both" else "")
                                    if args.mode == "fedavg" else ""))
    out = RESULTS / "multiarea" / tag / f"seed{args.seed}"
    if args.skip_existing and (out / "test.json").exists():
        print(f"  [{tag} s{args.seed}] already done, skipping")
        return
    train_multiarea(mc, out)


def load_policy(preset: str, tag: str, seed: int, cfg=None):
    """Deterministic policy callable from a saved single-area checkpoint (network sizes are
    read from the run's config.json)."""
    import torch

    from .env import RISUAVEnv
    from .trainers import Learner, TrainConfig

    run = RESULTS / "train" / preset / tag / f"seed{seed}"
    tcfg = json.loads((run / "config.json").read_text())["train"]
    cfg = cfg or get_preset(preset)
    learner = Learner(RISUAVEnv(cfg), TrainConfig(algo=tcfg["algo"], hidden=tcfg["hidden"],
                                                  critic_hidden_ctde=tcfg["critic_hidden_ctde"]))
    learner.load_state_dict(torch.load(run / "best.pt"))
    return learner.policy


SWEEPS = {
    # name: (values, function value -> scenario overrides)
    "ris_size": ([64, 144, 256, 400], lambda v: {"ris": ("ny", "nz", int(np.sqrt(v)))}),
    "phase_bits": ([1, 2, 3, 0], lambda v: {"ris": ("bits", v)}),
    "phase_err_deg": ([0, 10, 20, 40], lambda v: {"ris": ("phase_err_deg", v)}),
    "p_tx_dbm": ([0, 5, 10, 15, 20], lambda v: {"p_tx_dbm": float(v)}),
    "data_scale": ([0.5, 1.0, 1.5, 2.0], lambda v: {"data_mbit": (10.0 * v, 30.0 * v)}),
    "arbitration": (["none", "random", "common", "timeshare"], lambda v: {"arbitration": v}),
}


def _apply(cfg, ov: dict):
    for k, v in ov.items():
        if k == "ris":
            if v[0] == "ny":
                cfg = replace(cfg, ris=replace(cfg.ris, ny=v[2], nz=v[2]))
            else:
                cfg = replace(cfg, ris=replace(cfg.ris, **{v[0]: v[1]}))
        else:
            cfg = replace(cfg, **{k: v})
    return cfg


def exp_sweep(args):
    """Evaluate trained policies (no retraining) and the greedy heuristic across one parameter."""
    from .baselines import run_episode, straight_policy, tuned_greedy
    from .env import RISUAVEnv
    from .stats import mean_ci

    names = list(SWEEPS) if args.sweep == "all" else [args.sweep]
    tags = args.tags.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    for name in names:
        values, fn = SWEEPS[name]
        rows = []
        for v in values:
            cfg = _apply(get_preset(args.preset), fn(v))
            # fixed policies across the sweep (no retuning / retraining), same test episodes per value
            pols = {"greedy_tuned": tuned_greedy(args.preset, True),
                    "greedy_dec": tuned_greedy(args.preset, False), "straight": straight_policy}
            for tag in tags:
                for s in seeds:
                    if (RESULTS / "train" / args.preset / tag / f"seed{s}" / "best.pt").exists():
                        pols[f"{tag}/seed{s}"] = load_policy(args.preset, tag, s, cfg)
            for pname, pol in pols.items():
                env = RISUAVEnv(cfg)
                ms = [run_episode(env, pol, TEST_SEED_BASE + j) for j in range(args.n_eval)]
                ee_ci = mean_ci([m["ee"] for m in ms])
                rows.append({"param": name, "value": v, "policy": pname.split("/")[0],
                             "seed": pname.split("seed")[-1] if "/" in pname else "",
                             "ee": ee_ci["mean"], "ee_ci95": ee_ci["ci95"], "n": args.n_eval,
                             "delivered_frac": float(np.mean([m["delivered_frac"] for m in ms])),
                             "shadow_delivered_frac": float(np.mean([m["shadow_delivered_frac"] for m in ms]))})
            summary = {p: np.mean([r["ee"] for r in rows if r["value"] == v and r["policy"] == p])
                       for p in dict.fromkeys(r["policy"] for r in rows if r["value"] == v)}
            print(f"  {name}={v}: " + "  ".join(f"{p} {e:.0f}" for p, e in summary.items()), flush=True)
        _write_csv(RESULTS / "sweeps" / args.preset / f"{name}.csv", rows)


def exp_multiarea_baselines(args):
    """Heuristics on the unseen test areas, same episodes as the multi-area test (12 per area)."""
    from .baselines import run_episode, straight_policy, tuned_greedy
    from .config import TEST_AREAS, random_area
    from .env import RISUAVEnv
    from .multiarea import MultiAreaConfig
    from .stats import mean_ci

    n = MultiAreaConfig().n_test_eps
    rows = []
    for pname, arb in (("straight", "none"), ("straight", "timeshare"), ("greedy_tuned", "none"),
                       ("greedy_tuned", "timeshare"), ("greedy_dec", "timeshare")):
        pol = straight_policy if pname == "straight" else tuned_greedy("multiarea", pname == "greedy_tuned")
        ees, per_area, dl, rch = [], [], [], []
        for s in TEST_AREAS:
            env = RISUAVEnv(replace(random_area(s), arbitration=arb))
            ms = [run_episode(env, pol, TEST_SEED_BASE + j) for j in range(n)]
            ees += [m["ee"] for m in ms]
            per_area.append(float(np.mean([m["ee"] for m in ms])))
            dl += [m["delivered_frac"] for m in ms]
            rch += [m["reached_final"] for m in ms]
        st = mean_ci(ees)
        rows.append({"policy": pname, "arbitration": arb, "test_ee": st["mean"], "test_ee_ci95": st["ci95"],
                     "test_ee_std_over_areas": float(np.std(per_area, ddof=1)),
                     "test_delivered": float(np.mean(dl)), "test_reached": float(np.mean(rch)),
                     "n_episodes": len(ees)})
        print(f"  {pname:12s} {arb:9s} EE on unseen areas {st['mean']:7.0f} ± {st['ci95']:4.0f} (95% CI)"
              f"  area-to-area std {rows[-1]['test_ee_std_over_areas']:5.0f}")
    _write_csv(RESULTS / "multiarea" / "baselines.csv", rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, choices=["gate", "train", "multiarea", "multiarea_baselines", "sweep"])
    p.add_argument("--mode", default="fedavg", choices=["local", "fedavg", "pooled"])
    p.add_argument("--fed-parts", default="actor", choices=["actor", "both"])
    p.add_argument("--n-eval", type=int, default=N_TEST)
    p.add_argument("--skip-areas", action="store_true")
    p.add_argument("--preset", default="building")
    p.add_argument("--algo", default="ctde", choices=["ctde", "ippo", "fedavg", "fedprox", "fedper"])
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--E", type=int, default=1)
    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--hidden", type=int, default=128, help="actor (and local critic) hidden width")
    p.add_argument("--fixed-lam", action="store_true", help="keep the energy price fixed (no Dinkelbach)")
    p.add_argument("--scenario", default="", help="JSON dict of ScenarioConfig overrides")
    p.add_argument("--ppo", default="", help="JSON dict of PPOConfig overrides")
    p.add_argument("--tag", default="")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--sweep", default="all")
    p.add_argument("--tags", default="ctde,ippo,fedavg_E4")
    p.add_argument("--seeds", default="0,1,2")
    args = p.parse_args(argv)
    t = time.time()
    {"gate": exp_gate, "train": exp_train, "multiarea": exp_multiarea,
     "multiarea_baselines": exp_multiarea_baselines, "sweep": exp_sweep}[args.exp](args)
    print(f"done in {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
