"""Figures for v9, drawn only from files saved by ``v9.run``.

    python -m v9.plots gate
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .config import get_preset  # noqa: E402

RESULTS = Path("results_v9")
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def draw_scene(ax, cfg, ris=True, final=True):
    for b in cfg.buildings:
        ax.add_patch(Rectangle((b.x0, b.y0), b.x1 - b.x0, b.y1 - b.y0, facecolor="0.55",
                               edgecolor="0.25", alpha=0.85, zorder=3))
    if ris and cfg.ris.kind == "building":
        ax.plot(cfg.ris.center[0], cfg.ris.center[1], marker="*", ms=16, color="gold",
                mec="k", zorder=6, label="RIS")
    ax.plot(*cfg.start_xy, marker=">", ms=10, color="limegreen", mec="k", zorder=6, label="start")
    if final:
        ax.add_patch(plt.Circle(cfg.final_xy, cfg.r_final, fill=False, ls="--", color="m", zorder=5))
        ax.plot(*cfg.final_xy, marker="X", ms=10, color="m", mec="k", zorder=6, label="final")
    ax.set_xlim(0, cfg.area)
    ax.set_ylim(0, cfg.area)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def plot_gate():
    d = RESULTS / "gate"
    cfg = get_preset("building")
    m = np.load(d / "coverage_shadow.npz")
    data = 0.5 * sum(cfg.data_mbit) * 1e6
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    ext = [0, cfg.area, 0, cfg.area]
    vmin, vmax = 1e4, max(m["rate_ris"].max(), m["rate_dir"].max())
    for ax, key, title in ((axes[0], "rate_dir", "(a) Shadowed device, no RIS"),
                           (axes[1], "rate_ris", "(b) Shadowed device, with RIS (2-bit)")):
        im = ax.imshow(np.maximum(m[key], vmin), origin="lower", extent=ext, cmap="viridis",
                       norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.contour(m["x"], m["x"], m[key] * cfg.window_s, levels=[data], colors="w", linewidths=1.5)
        draw_scene(ax, cfg)
        ax.plot(*m["dev"], marker="o", ms=7, color="red", mec="w", zorder=7, label="device")
        frac = np.mean(m[key] * cfg.window_s > data)
        ax.set_title(f"{title}\n20 Mbit in 30 s possible from {frac:.1%} of the area", fontsize=10)
    fig.colorbar(im, ax=axes[:2], label="achievable rate (bit/s)", shrink=0.85)
    gain = 10 * np.log10(np.maximum(m["rate_ris"], 1) / np.maximum(m["rate_dir"], 1))
    im2 = axes[2].imshow(gain, origin="lower", extent=ext, cmap="magma")
    draw_scene(axes[2], cfg)
    axes[2].set_title("(c) Rate gain from the RIS (dB)")
    fig.colorbar(im2, ax=axes[2], shrink=0.85)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.savefig(d / "fig_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    rows = _read_csv(d / "upload_time.csv")
    dist = [float(r["distance_m"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.semilogy(dist, [float(r["open_noris_s"]) for r in rows], "o-", label="open-field device")
    ax.semilogy(dist, [float(r["shadow_noris_s"]) for r in rows], "s-", label="shadowed device, no RIS")
    ax.semilogy(dist, [float(r["shadow_ris_s"]) for r in rows], "^-", label="shadowed device, with RIS")
    ax.axhline(cfg.window_s, color="k", ls="--", lw=1, label="device active window (30 s)")
    ax.set_xlabel("UAV horizontal distance from device (m)")
    ax.set_ylabel("time to upload 20 Mbit (s)")
    ax.set_title("Who can be served from where")
    ax.legend(fontsize=8)
    fig.savefig(d / "fig_upload_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    rows = _read_csv(d / "baselines.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, preset, title in ((axes[0], "legacy_small", "Old 20 m scene (v8 / Jiang et al.)"),
                              (axes[1], "building", "New 500 m urban scene (v9)")):
        rs = [r for r in rows if r["preset"] == preset and r["arbitration"] in ("none", "random", "timeshare")]
        pols = list(dict.fromkeys(r["policy"] for r in rs))
        arbs = ["none", "random", "timeshare"]
        labels = {"none": "no RIS", "random": "random RIS phases", "timeshare": "aligned RIS"}
        w = 0.26
        for j, arb in enumerate(arbs):
            vals = [next(float(r["ee_mean"]) for r in rs if r["policy"] == p and r["arbitration"] == arb) for p in pols]
            errs = [next(float(r["ee_std"]) for r in rs if r["policy"] == p and r["arbitration"] == arb) for p in pols]
            ax.bar(np.arange(len(pols)) + (j - 1) * w, vals, w, yerr=errs, capsize=3, label=labels[arb])
        ax.set_xticks(range(len(pols)))
        ax.set_xticklabels(pols)
        ax.set_ylabel("energy efficiency (bit/J)")
        ax.set_title(title)
        lo = min(float(r["ee_mean"]) for r in rs)
        ax.set_ylim(lo * 0.8, None)
        ax.legend(fontsize=8)
    fig.suptitle("Does the RIS matter? Baseline flight policies, mean ± std over 20 held-out episodes")
    fig.savefig(d / "fig_old_vs_new.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("gate figures written to", d)


if __name__ == "__main__":
    {"gate": plot_gate}[sys.argv[1]]()
