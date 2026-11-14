"""Run a list of v9.run jobs in parallel (one process per job, one thread each).

    python -m v9.launch pilots --seeds 0 1          # CTDE pilots that chose the training settings
    python -m v9.launch campaign --episodes 6000     # full 5-seed campaign (multi-area, presets, single)
    python -m v9.launch single --seeds 0 --episodes 16 --dry-run
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LOGS = Path("results_v9/logs")


def pilots(seeds, episodes):
    # A: v9.1 defaults (Dinkelbach price on)   B: fixed energy price
    # C: learning-rate annealing                D: 16 episodes per PPO update
    variants = {"ctde_pA": [], "ctde_pB": ["--fixed-lam"], "ctde_pC": ["--ppo", '{"lr_anneal": true}'],
                "ctde_pD": ["--ppo", '{"episodes_per_update": 16}']}
    return [(f"pilots/{tag}_s{s}", ["--exp", "train", "--skip-existing", "--preset", "building", "--algo", "ctde",
                                    "--episodes", str(episodes), "--eval-every", "200", "--tag", tag,
                                    "--seed", str(s)] + extra)
            for s in seeds for tag, extra in variants.items()]


def pilots2(seeds, episodes):
    # E: C (Dinkelbach + LR annealing) trained longer   F: E with a wider actor (256)
    variants = {"ctde_pE": [], "ctde_pF": ["--hidden", "256"]}
    return [(f"pilots/{tag}_s{s}", ["--exp", "train", "--skip-existing", "--preset", "building", "--algo", "ctde",
                                    "--episodes", str(episodes), "--eval-every", "200", "--tag", tag,
                                    "--seed", str(s), "--ppo", '{"lr_anneal": true}'] + extra)
            for s in seeds for tag, extra in variants.items()]


def pilots3(seeds, episodes):
    # G / H: setting C with more devices in each UAV's view (top_k 12 / 24 = all devices);
    # I: setting C again (top_k 8) as the control, on the environment with the separating autopilot
    variants = {"ctde_pG": ["--scenario", '{"top_k": 12}'], "ctde_pH": ["--scenario", '{"top_k": 24}'],
                "ctde_pI": []}
    return [(f"pilots/{tag}_s{s}", ["--exp", "train", "--skip-existing", "--preset", "building", "--algo", "ctde",
                                    "--episodes", str(episodes), "--eval-every", "200", "--tag", tag,
                                    "--seed", str(s), "--ppo", '{"lr_anneal": true}'] + extra)
            for s in seeds for tag, extra in variants.items()]


# settings locked from the pilots (validation only): Dinkelbach price + learning-rate annealing
PPO = '{"lr_anneal": true}'
METHODS = [("ctde", []), ("ippo", []), ("fedavg_E1", ["--algo", "fedavg", "--E", "1"]),
           ("fedavg_E4", ["--algo", "fedavg", "--E", "4"]), ("fedavg_E16", ["--algo", "fedavg", "--E", "16"]),
           ("fedprox_E4", ["--algo", "fedprox", "--E", "4", "--mu", "0.01"]),
           ("fedper_E4", ["--algo", "fedper", "--E", "4"])]


def _train(preset, tag, seed, episodes, extra):
    algo = ["--algo", tag.split("_")[0]] if "--algo" not in extra else []
    return (f"{preset}/{tag}_s{seed}", ["--exp", "train", "--skip-existing", "--preset", preset, "--tag", tag,
                                         "--episodes", str(episodes), "--eval-every", "200", "--seed", str(seed),
                                         "--ppo", PPO] + algo + extra)


def single(seeds, episodes):
    """All single-area methods on the building scene."""
    return [_train("building", tag, s, episodes, extra) for s in seeds for tag, extra in METHODS]


def presets(seeds, episodes):
    """CTDE on the other RIS cases (no RIS, aerial RIS)."""
    return [_train(p, "ctde", s, episodes, []) for p in ("aerial", "building_noris") for s in seeds]


def multiarea(seeds, episodes_per_area):
    modes = [("pooled", ["--mode", "pooled"]), ("fedavg_E4", ["--mode", "fedavg", "--E", "4"]),
             ("fedavg_E4_both", ["--mode", "fedavg", "--E", "4", "--fed-parts", "both"]),
             ("local", ["--mode", "local"])]
    return [(f"multiarea/{tag}_s{s}", ["--exp", "multiarea", "--skip-existing", "--episodes", str(episodes_per_area),
                                        "--seed", str(s), "--ppo", PPO] + extra)
            for s in seeds for tag, extra in modes]


def campaign(seeds, episodes, ma_episodes=3000):
    """Everything, longest jobs first so the parallel slots drain evenly."""
    return multiarea(seeds, ma_episodes) + presets(seeds, episodes) + single(seeds, episodes)


def post(seeds, episodes):
    """After training: heuristics on the unseen test areas, and the parameter sweeps (one job each)."""
    from .run import SWEEPS
    seed_arg = ",".join(str(s) for s in seeds)
    return [("post/multiarea_baselines", ["--exp", "multiarea_baselines"])] + \
        [(f"post/sweep_{n}", ["--exp", "sweep", "--sweep", n, "--tags", "ctde,ippo,fedavg_E4", "--seeds", seed_arg])
         for n in SWEEPS]


def control(seeds, episodes_per_area=12000):
    """Equal-data control: each area trains alone on 4x the episodes, i.e. as much experience
    as the FedAvg/pooled global model sees in total."""
    return [(f"multiarea/local_x4_s{s}", ["--exp", "multiarea", "--skip-existing", "--mode", "local",
                                           "--episodes", "12000", "--tag", "local_x4", "--seed", str(s),
                                           "--ppo", PPO]) for s in seeds]


def fedhet(seeds, episodes=11_520):
    """Heterogeneous multi-area FL: 12 non-IID areas, equal total training episodes per mode."""
    modes = [("local", ["--mode", "local"]), ("pooled", ["--mode", "pooled"]),
             ("fedavg_p0.5", ["--mode", "fedavg", "--participation", "0.5"]),
             ("fedavg_p1", ["--mode", "fedavg", "--participation", "1"]),
             ("fedprox_p0.5", ["--mode", "fedprox", "--participation", "0.5", "--mu", "0.01"]),
             ("fedavg_p0.5_q8", ["--mode", "fedavg", "--participation", "0.5", "--compress", "q8"]),
             ("fedavg_p0.5_top10", ["--mode", "fedavg", "--participation", "0.5", "--compress", "top10"])]
    return [(f"fedhet/{tag}_s{s}", ["--exp", "fedhet", "--skip-existing", "--E", "4", "--episodes", str(episodes),
                                     "--seed", str(s), "--tag", tag] + extra)
            for s in seeds for tag, extra in modes]


def finetune(seeds, episodes=320):
    """Few-shot adaptation in each unseen area: global FedAvg model vs scratch (run after fedhet)."""
    return [(f"fedhet/finetune_s{s}", ["--exp", "finetune", "--skip-existing", "--tag", "fedavg_p0.5",
                                        "--episodes", str(episodes), "--seed", str(s)]) for s in seeds]


def rounds(seeds, episodes=11_520):
    """Control for the full-participation result: all 12 areas per round but E = 2 local
    updates, so there are as many averaging rounds (60) as FedAvg with 50 % and E = 4, at the
    same total number of training missions."""
    return [(f"fedhet/fedavg_p1_E2_s{s}", ["--exp", "fedhet", "--skip-existing", "--E", "2", "--episodes",
                                            str(episodes), "--seed", str(s), "--tag", "fedavg_p1_E2",
                                            "--mode", "fedavg", "--participation", "1"]) for s in seeds]


GROUPS = {"rounds": rounds, "fedhet": fedhet, "finetune": finetune, "control": control, "pilots": pilots, "pilots2": pilots2, "pilots3": pilots3, "single": single, "presets": presets,
          "multiarea": multiarea, "campaign": campaign, "post": post}


def run(jobs, n_par: int, dry: bool):
    def one(job):
        name, args = job
        cmd = [sys.executable, "-m", "v9.run"] + args
        if dry:
            return f"{name}: {shlex.join(cmd)}"
        log = LOGS / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        last = log.read_text().strip().splitlines()[-2:]
        return f"{name}: rc={rc} {' | '.join(last)}"

    with ThreadPoolExecutor(n_par) as ex:
        for line in ex.map(one, jobs):
            print(line, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("group", choices=list(GROUPS))
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--n-par", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    run(GROUPS[a.group](a.seeds, a.episodes), a.n_par, a.dry_run)


if __name__ == "__main__":
    main()
