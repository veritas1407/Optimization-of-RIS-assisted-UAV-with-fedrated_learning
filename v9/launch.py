"""Run a list of v9.run jobs in parallel (one process per job, one thread each).

    python -m v9.launch pilots            # CTDE pilot runs that choose the training settings
    python -m v9.launch pilots --seeds 0 --episodes 16 --dry-run
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


GROUPS = {"pilots": pilots}


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
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--n-par", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    run(GROUPS[a.group](a.seeds, a.episodes), a.n_par, a.dry_run)


if __name__ == "__main__":
    main()
