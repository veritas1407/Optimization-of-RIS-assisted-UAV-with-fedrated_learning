"""Top-down animation of one mission, two cases side by side on the SAME mission
(same devices, data, wake-up windows and fading seed).

    python -m v9.animate                                  # learned CTDE: building RIS vs no RIS
    python -m v9.animate --left building:ctde:0 --right building:greedy_tuned --seed 5000003

Each case is "preset:policy[:training seed]"; policy is a trained tag under
results_v9/train/<preset>/ or one of straight / greedy_tuned / greedy_dec.
Writes results_v9/animation/<name>.mp4 (and a .gif if --gif).

What is drawn: buildings (grey), the RIS (gold star), devices as circles whose size shows the
data still waiting (red = in the shadowed courtyard, blue = open field; filled while the
device's 30 s window is open), UAV trails, a line from each served device to its UAV
(green = direct link, gold = through the RIS, with the RIS bounce drawn), and live
counters for bits delivered, flight energy and EE.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .baselines import run_episode, straight_policy, tuned_greedy  # noqa: E402
from .config import get_preset  # noqa: E402
from .env import RISUAVEnv  # noqa: E402

OUT = Path("results_v9/animation")
UAV_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
LABELS = {"ctde": "learned MAPPO-CTDE", "greedy_tuned": "centralised greedy", "greedy_dec": "decentralised greedy",
          "straight": "straight line"}


def make_policy(preset: str, policy: str, seed: int, cfg):
    if policy == "straight":
        return straight_policy
    if policy in ("greedy_tuned", "greedy_dec"):
        return tuned_greedy(preset, policy == "greedy_tuned")
    from .run import load_policy
    return load_policy(preset, policy, seed, cfg)


def record(case: str, mission_seed: int) -> dict:
    parts = case.split(":")
    preset, policy = parts[0], parts[1]
    tseed = int(parts[2]) if len(parts) > 2 else 0
    cfg = get_preset(preset)
    env = RISUAVEnv(cfg)
    pol = make_policy(preset, policy, tseed, cfg)
    env.reset(seed=mission_seed)
    start = env.pos.copy()
    m, log = run_episode(env, pol, mission_seed, record=True)
    ris = env.ris_center if (env.use_ris and not env.aerial) else None
    title = f"{LABELS.get(policy, policy)} — " + ("no RIS" if cfg.arbitration == "none" else
                                                  "aerial RIS (purple UAV carries the panel)" if env.aerial
                                                  else "building RIS")
    return {"cfg": cfg, "env": env, "log": log, "start": start, "metrics": m, "ris": ris, "title": title}


def animate(left: str, right: str, mission_seed: int, name: str, sub: int = 6, fps: int = 12, gif: bool = False):
    cases = [record(left, mission_seed), record(right, mission_seed)]
    M = len(cases[0]["log"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.6))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.86, bottom=0.04, wspace=0.08)
    arts = []
    for ax, c in zip(axes, cases):
        cfg, env = c["cfg"], c["env"]
        for b in cfg.buildings:
            ax.add_patch(Rectangle((b.x0, b.y0), b.x1 - b.x0, b.y1 - b.y0, fc="0.6", ec="0.3", zorder=2))
        if c["ris"] is not None:
            ax.plot(*c["ris"][:2], marker="*", ms=17, color="gold", mec="k", zorder=6)
        ax.add_patch(plt.Circle(cfg.final_xy, cfg.r_final, fill=False, ls="--", color="m", zorder=3))
        ax.set_xlim(0, cfg.area)
        ax.set_ylim(0, cfg.area)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(c["title"], fontsize=12)
        dev_col = np.where(env.shadow, "#d62728", "#1f77b4")
        devs = ax.scatter(env.dev[:, 0], env.dev[:, 1], s=40, c=dev_col, edgecolors=dev_col, zorder=4)
        n = env.n_agents
        trails = [ax.plot([], [], "-", color=UAV_COLORS[k], lw=1.5, alpha=0.7, zorder=5)[0] for k in range(n)]
        uavs = ax.scatter(c["start"][:, 0], c["start"][:, 1], s=110, marker="^", c=UAV_COLORS[:n],
                          edgecolors="k", zorder=8)
        links = [ax.plot([], [], "-", lw=2.2, zorder=7)[0] for _ in range(n)]
        bounce = [ax.plot([], [], ":", color="goldenrod", lw=2.0, zorder=7)[0] for _ in range(n)]
        text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10, family="monospace",
                       bbox=dict(fc="white", ec="0.6", alpha=0.85), zorder=9)
        arts.append(dict(devs=devs, trails=trails, uavs=uavs, links=links, bounce=bounce, text=text))
    fig.suptitle(f"Same mission (seed {mission_seed}): circles = data still waiting "
                 "(red: shadowed courtyard, blue: open field; filled = awake); "
                 "green link = direct, gold = via RIS", fontsize=10)

    def frame(f):
        t = min(f // sub, M - 1)
        a = (f % sub + 1) / sub
        out = []
        for c, ar in zip(cases, arts):
            env, log = c["env"], c["log"]
            prev = c["start"] if t == 0 else log[t - 1]["pos"]
            pos = prev + a * (log[t]["pos"] - prev)
            hist = np.array([c["start"][:, :2]] + [x["pos"][:, :2] for x in log[:t]] + [pos[:, :2]])
            for k, tr in enumerate(ar["trails"]):
                tr.set_data(hist[:, k, 0], hist[:, k, 1])
            ar["uavs"].set_offsets(pos[:, :2])
            rem = log[t]["remaining"]
            ar["devs"].set_sizes(15 + 220 * rem / max(c["cfg"].data_mbit[1] * 1e6, 1.0))
            tmid = (t + 0.5) * c["cfg"].dt
            awake = (env.win_start <= tmid) & (tmid < env.win_end) & (rem > 1.0)
            ar["devs"].set_facecolors(np.where(awake[:, None], ar["devs"].get_edgecolors(), [[1, 1, 1, 1]]))
            served = log[t]["served"]
            via = {i for i, _ in log[t]["ris_targets"]}
            for k in range(len(ar["links"])):
                ar["links"][k].set_data([], [])
                ar["bounce"][k].set_data([], [])
            for i in np.where(served >= 0)[0]:
                k = int(served[i])
                if k >= len(ar["links"]):
                    continue
                r = pos[env.K] if env.aerial else c["ris"]        # aerial: the panel rides on UAV K
                if i in via and r is not None:
                    ar["bounce"][k].set_data([env.dev[i, 0], r[0], pos[k, 0]], [env.dev[i, 1], r[1], pos[k, 1]])
                else:
                    ar["links"][k].set_data([env.dev[i, 0], pos[k, 0]], [env.dev[i, 1], pos[k, 1]])
                    ar["links"][k].set_color("green")
            bits = sum(x["bits"] for x in log[:t + 1])
            energy = sum(x["energy"] for x in log[:t + 1])
            ar["text"].set_text(f"t = {(t + a) * c['cfg'].dt:5.1f} s\n"
                                f"data  {bits / 8e6:6.1f} MB\n"
                                f"energy {energy / 1e3:5.1f} kJ\n"
                                f"EE   {bits / max(energy, 1e-9):6.0f} bit/J")
            out += [ar["devs"], ar["uavs"], ar["text"], *ar["trails"], *ar["links"], *ar["bounce"]]
        return out

    anim = FuncAnimation(fig, frame, frames=M * sub + 2 * fps, interval=1000 / fps, blit=False)
    OUT.mkdir(parents=True, exist_ok=True)
    mp4 = OUT / f"{name}.mp4"
    anim.save(mp4, writer=FFMpegWriter(fps=fps, bitrate=2400))
    if gif:
        anim.save(OUT / f"{name}.gif", writer=PillowWriter(fps=fps))
    frame(M * sub - 1)
    fig.savefig(OUT / f"{name}_final.png", dpi=130)
    plt.close(fig)
    for c in cases:
        print(f"  {c['title']}: EE {c['metrics']['ee']:.0f} bit/J, delivered {c['metrics']['delivered_frac']:.2f}, "
              f"courtyard {c['metrics']['shadow_delivered_frac']:.2f}")
    print("written", mp4)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--left", default="building:ctde:0")
    p.add_argument("--right", default="building_noris:ctde:0")
    p.add_argument("--seed", type=int, default=5_000_003)
    p.add_argument("--name", default="")
    p.add_argument("--gif", action="store_true")
    a = p.parse_args(argv)
    name = a.name or f"{a.left.replace(':', '_')}__vs__{a.right.replace(':', '_')}"
    animate(a.left, a.right, a.seed, name, gif=a.gif)


if __name__ == "__main__":
    main()
