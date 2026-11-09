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
POLICY_ORDER = ["random", "straight", "greedy", "greedy_dec", "greedy_tuned"]
POLICY_LABELS = {"random": "random", "straight": "straight line", "greedy": "greedy\n(untuned v9.0)",
                 "greedy_dec": "greedy, decentral.\n(same info as RL)",
                 "greedy_tuned": "greedy, central.\n(full info, tuned)"}
ARB_COLORS = {"none": "C7", "random": "C1", "common": "C4", "timeshare": "C0"}
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, preset, title in ((axes[0], "legacy_small", "Old 20 m scene (v8 / Jiang et al.)"),
                              (axes[1], "building", "New 500 m urban scene (v9)")):
        rs = [r for r in rows if r["preset"] == preset and r["arbitration"] in ("none", "random", "timeshare")]
        pols = [p for p in POLICY_ORDER if any(r["policy"] == p for r in rs)]
        arbs = ["none", "random", "timeshare"]
        labels = {"none": "no RIS", "random": "random RIS phases", "timeshare": "aligned RIS (time-share)"}
        w = 0.26
        for j, arb in enumerate(arbs):
            sel = [next((r for r in rs if r["policy"] == p and r["arbitration"] == arb), None) for p in pols]
            vals = [float(r["ee_mean"]) if r else np.nan for r in sel]
            errs = [float(r["ee_ci95"]) if r else 0.0 for r in sel]
            ax.bar(np.arange(len(pols)) + (j - 1) * w, vals, w, yerr=errs, capsize=3, label=labels[arb],
                   color=ARB_COLORS[arb])
        ax.set_xticks(range(len(pols)))
        ax.set_xticklabels([POLICY_LABELS.get(p, p) for p in pols], fontsize=8)
        ax.set_ylabel("energy efficiency (bit/J)")
        ax.set_title(title)
        ax.set_ylim(0, None)
        ax.legend(fontsize=8)
    n = rows[0].get("n", "?")
    fig.suptitle(f"Does the RIS matter? Baseline flight policies, mean ± 95 % CI over {n} test episodes")
    fig.savefig(d / "fig_old_vs_new.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("gate figures written to", d)


LABELS = {"ctde": "MAPPO-CTDE (centralized)", "ippo": "IPPO (no exchange)",
          "fedavg_E1": "FedAvg (E=1)", "fedavg_E4": "FedAvg (E=4)", "fedavg_E16": "FedAvg (E=16)",
          "fedprox_E4": "FedProx (E=4, μ=0.01)", "fedprox_E1": "FedProx (E=1)",
          "fedper_E4": "FedPer (E=4, personal heads)"}


def _runs(preset):
    root = RESULTS / "train" / preset
    out = {}
    if not root.exists():
        return out
    for tag_dir in sorted(root.iterdir()):
        if tag_dir.name.startswith("smoke"):
            continue
        seeds = [s for s in sorted(tag_dir.glob("seed*")) if (s / "test.json").exists()]
        if seeds:
            out[tag_dir.name] = seeds
    return out


def _gate_episodes(key):
    f = RESULTS / "gate" / "baselines_per_episode.npz"
    if not f.exists():
        return None
    d = np.load(f)
    return d[key] if key in d else None


def summarize(preset="building"):
    """Per-method test results over training seeds: mean, std (ddof=1), 95 % CI over seeds, and
    paired differences to the heuristics on the same 100 test episodes."""
    import json

    from .stats import mean_ci
    rows = []
    gp, arb = ("building", "none") if preset == "building_noris" else (preset, "timeshare")
    ref = {k: _gate_episodes(f"{gp}|{k}|{arb}") for k in ("greedy_tuned", "greedy_dec", "straight")}
    for tag, seeds in _runs(preset).items():
        tests = [json.loads((s / "test.json").read_text()) for s in seeds]
        row = {"method": tag, "n_seeds": len(tests)}
        for k in ("ee", "delivered_frac", "shadow_delivered_frac", "energy", "reached_final",
                  "collisions", "comm_MB"):
            st = mean_ci([t[k] for t in tests])
            row[f"{k}_mean"], row[f"{k}_std"], row[f"{k}_ci95"] = st["mean"], st["std"], st["ci95"]
        for name, base in ref.items():
            if base is None or "ee_per_episode" not in tests[0]:
                continue
            # per seed: mean over the same test episodes of (method - heuristic); CI over seeds
            d = [np.mean(np.asarray(t["ee_per_episode"]) - base[:len(t["ee_per_episode"])]) for t in tests]
            st = mean_ci(d)
            row[f"vs_{name}_mean"], row[f"vs_{name}_ci95"] = st["mean"], st["ci95"]
            row[f"vs_{name}_pct"] = 100.0 * st["mean"] / float(np.mean(base))
        rows.append(row)
    if rows:
        out = RESULTS / "train" / preset / "summary.csv"
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return rows


def _baseline_ee(preset):
    path = RESULTS / "gate" / "baselines.csv"
    if not path.exists():
        return {}
    rs = [r for r in _read_csv(path) if r["preset"] == preset]
    arb = "none" if preset == "building_noris" else "timeshare"
    if preset == "building_noris":
        rs = [r for r in _read_csv(path) if r["preset"] == "building"]
    pick = {("straight", arb): "straight line", ("greedy_dec", arb): "greedy, decentralized (same info)",
            ("greedy_tuned", arb): "greedy, centralized (full info, tuned)"}
    return {pick[(r["policy"], r["arbitration"])]: float(r["ee_mean"]) for r in rs
            if (r["policy"], r["arbitration"]) in pick}


def plot_train(preset="building"):
    runs = _runs(preset)
    if not runs:
        print("no finished runs for", preset)
        return
    rows = summarize(preset)
    out = RESULTS / "train" / preset
    base = _baseline_ee(preset)

    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, seeds in runs.items():
        logs = [_read_csv(s / "eval_log.csv") for s in seeds]
        n = min(len(lg) for lg in logs)
        x = np.array([float(r["episode"]) for r in logs[0][:n]])
        y = np.array([[float(r["ee"]) for r in lg[:n]] for lg in logs])
        mu, sd = y.mean(0), y.std(0)
        ax.plot(x, mu, lw=2, label=f"{LABELS.get(tag, tag)} ({len(seeds)} seeds)")
        ax.fill_between(x, mu - sd, mu + sd, alpha=0.2)
    for (name, v), ls in zip(base.items(), [":", "-.", "--"]):
        ax.axhline(v, color="k", ls=ls, lw=1, label=f"baseline: {name}")
    ax.set_xlabel("training episode")
    ax.set_ylabel("validation EE (bit/J)")
    ax.set_ylim(0, None)
    ax.set_title(f"Learning curves ({preset} scene); line = mean, band = ± 1 std over seeds")
    ax.legend(fontsize=8)
    fig.savefig(out / "fig_learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    floor = 1e-1
    for r in rows:
        x = max(r["comm_MB_mean"], floor)
        ax.errorbar(x, r["ee_mean"], yerr=r["ee_ci95"], fmt="o", ms=8, capsize=4,
                    label=LABELS.get(r["method"], r["method"]))
        ax.annotate(f"{r['comm_MB_mean']:.0f} MB" if r["comm_MB_mean"] >= 1 else "0 MB",
                    (x, r["ee_mean"]), textcoords="offset points", xytext=(6, 6), fontsize=8)
    for (name, v), ls in zip(base.items(), [":", "-.", "--"]):
        ax.axhline(v, color="k", ls=ls, lw=1, label=f"baseline: {name}")
    ax.set_xscale("log")
    ax.set_xlabel("training communication (MB, log scale; IPPO plotted at 0.1)")
    ax.set_ylabel("test EE (bit/J), mean ± 95 % CI over seeds")
    ax.set_title("Energy efficiency vs. communication cost of training")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig(out / "fig_ee_vs_comm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    for r in rows:
        print(f"  {r['method']:12s} seeds={r['n_seeds']}  EE {r['ee_mean']:7.0f} ± {r['ee_ci95']:5.0f} (95% CI)  "
              f"delivered {r['delivered_frac_mean']:.2f}  final {r['reached_final_mean']:.2f}  "
              f"comm {r['comm_MB_mean']:8.1f} MB")


MA_LABELS = {"local": "Local only\n(each area alone)", "fedavg_E4": "FedAvg across areas\n(actor only, E=4)",
             "fedavg_E4_both": "FedAvg across areas\n(actor+critic, E=4)",
             "pooled": "Pooled raw data\n(cloud, upper bound)"}


def plot_area_maps():
    from .config import TEST_AREAS, TRAIN_AREAS, random_area
    out = RESULTS / "multiarea"
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(TRAIN_AREAS) + list(TEST_AREAS)
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.4))
    for ax, s in zip(axes.ravel(), seeds):
        cfg = random_area(s)
        draw_scene(ax, cfg)
        x0, y0, x1, y1 = cfg.shadow_region
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="red", alpha=0.35, zorder=4))
        ax.plot([0, cfg.area], [0, cfg.area], "k:", lw=0.8)
        kind = "TRAIN" if s in TRAIN_AREAS else "TEST (unseen)"
        ax.set_title(f"{kind} area {s}", fontsize=9, color="C0" if s in TRAIN_AREAS else "C3")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=7)
    fig.suptitle("City areas: grey = buildings, red = shadowed courtyard, star = RIS, dotted = start→final diagonal")
    fig.tight_layout()
    fig.savefig(out / "fig_areas.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_multiarea():
    import json
    root = RESULTS / "multiarea"
    runs = {}
    for d in sorted(root.iterdir()) if root.exists() else []:
        if d.is_dir():
            seeds = [s for s in sorted(d.glob("seed*")) if (s / "test.json").exists()]
            if seeds:
                runs[d.name] = seeds
    if not runs:
        print("no finished multi-area runs")
        return
    order = [k for k in ("local", "fedavg_E4", "fedavg_E4_both", "pooled") if k in runs] + \
            [k for k in runs if k not in MA_LABELS]
    from .stats import mean_ci
    rows = []
    for tag in order:
        ts = [json.loads((s / "test.json").read_text()) for s in runs[tag]]
        st = mean_ci([t["test_ee"] for t in ts])
        rows.append({"method": tag, "n_seeds": len(ts), "test_ee_mean": st["mean"], "test_ee_std": st["std"],
                     "test_ee_ci95": st["ci95"],
                     # local: the 4 area models differ; mean over seeds of their spread (ddof=1)
                     "local_model_spread_std": (np.mean([t["test_ee_across_models_std"] for t in ts])
                                                if tag == "local" else np.nan),
                     "comm_intra_MB_mean": np.mean([t.get("comm_intra_MB", np.nan) for t in ts]),
                     "test_delivered_mean": np.mean([t["test_delivered"] for t in ts]),
                     "test_reached_mean": np.mean([t["test_reached"] for t in ts]),
                     "comm_MB_mean": np.mean([t["comm_MB"] for t in ts]),
                     "own_area_ee_mean": (np.mean([np.mean([m.get("own_area_ee", np.nan)
                                                            for m in t["models"].values()]) for t in ts])
                                          if tag == "local" else np.nan)})
    with (root / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    base = {}
    if (root / "baselines.csv").exists():
        for r in _read_csv(root / "baselines.csv"):
            base[(r["policy"], r["arbitration"])] = float(r["test_ee"])

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(rows))
    bars = ax.bar(x, [r["test_ee_mean"] for r in rows], yerr=[r["test_ee_ci95"] for r in rows], capsize=5,
                  color=["C7" if r["method"] == "local" else "C2" if r["method"].startswith("fed") else "C0"
                         for r in rows])
    for b, r in zip(bars, rows):
        ax.annotate(f"{r['test_ee_mean']:.0f}\ncross-area {r['comm_MB_mean']:.0f} MB",
                    (b.get_x() + b.get_width() / 2, b.get_height() + r["test_ee_ci95"]),
                    textcoords="offset points", xytext=(0, 4), ha="center", va="bottom", fontsize=8)
    names = {("greedy_tuned", "timeshare"): "greedy, central. (full info, tuned) + RIS",
             ("greedy_dec", "timeshare"): "greedy, decentral. (same info) + RIS",
             ("straight", "timeshare"): "straight + RIS"}
    for (k, v), ls in zip([(k, base[k]) for k in names if k in base], ["--", ":", "-."]):
        ax.axhline(v, color="k", ls=ls, lw=1, label=f"heuristic: {names[k]}")
    ax.set_xticks(x)
    ax.set_xticklabels([MA_LABELS.get(r["method"], r["method"]) for r in rows], fontsize=9)
    ax.set_ylabel("EE on 8 never-seen city areas (bit/J)")
    ax.set_ylim(0, None)
    ax.set_title("Generalization to unseen city areas (mean ± 95 % CI over training seeds)")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(root / "fig_generalization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for tag in order:
        logs = [_read_csv(s / "eval_log.csv") for s in runs[tag]]
        n = min(len(lg) for lg in logs)
        xs = np.array([float(r["episodes_per_area"]) for r in logs[0][:n]])
        ys = np.array([[float(r["val_ee_mean"]) for r in lg[:n]] for lg in logs])
        ax.plot(xs, ys.mean(0), lw=2, label=MA_LABELS.get(tag, tag).replace("\n", " "))
        ax.fill_between(xs, ys.mean(0) - ys.std(0, ddof=1), ys.mean(0) + ys.std(0, ddof=1), alpha=0.2)
    ax.set_xlabel("training episodes per area")
    ax.set_ylabel("EE on unseen validation areas (bit/J)")
    ax.set_ylim(0, None)
    ax.set_title("Learning curves on never-seen validation areas (band = ± 1 std over seeds)")
    ax.legend(fontsize=8)
    fig.savefig(root / "fig_multiarea_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    for r in rows:
        own = f"  own-area {r['own_area_ee_mean']:.0f}" if not np.isnan(r["own_area_ee_mean"]) else ""
        print(f"  {r['method']:16s} seeds={r['n_seeds']}  unseen-area EE {r['test_ee_mean']:7.0f} ± "
              f"{r['test_ee_ci95']:5.0f} (95% CI)  cross-area comm {r['comm_MB_mean']:7.1f} MB"
              f"  intra-area {r['comm_intra_MB_mean']:7.1f} MB{own}")


def plot_ris_value():
    """No RIS vs building RIS vs aerial RIS, for the tuned heuristic and learned MAPPO."""
    import json

    from .stats import mean_ci
    gate = _read_csv(RESULTS / "gate" / "baselines.csv") if (RESULTS / "gate" / "baselines.csv").exists() else []

    def heur(preset, arb):
        r = [x for x in gate if x["preset"] == preset and x["policy"] == "greedy_tuned" and x["arbitration"] == arb]
        return (float(r[0]["ee_mean"]), float(r[0]["ee_ci95"]), float(r[0]["energy_mean"]), int(r[0]["n"])) \
            if r else None

    def learned(preset):
        ts = [json.loads(p.read_text()) for p in sorted((RESULTS / "train" / preset / "ctde").glob("seed*/test.json"))]
        if not ts:
            return None
        st = mean_ci([t["ee"] for t in ts])
        return st["mean"], st["ci95"] if len(ts) > 1 else 0.0, float(np.mean([t["energy"] for t in ts])), len(ts)

    cases = [("No RIS", heur("building", "none"), learned("building_noris")),
             ("Building RIS\n(rooftop, 16×16)", heur("building", "timeshare"), learned("building")),
             ("Aerial RIS\n(carried by a 4th UAV)", heur("aerial", "timeshare"), learned("aerial"))]
    nh = next((c[1][3] for c in cases if c[1]), "?")
    nl = next((c[2][3] for c in cases if c[2]), "?")
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.38
    for j, (lab, col) in enumerate(((f"tuned centralized greedy (± 95 % CI, {nh} test episodes)", "C7"),
                                    (f"learned MAPPO-CTDE (± 95 % CI, {nl} training seeds)", "C0"))):
        for i, (_, h, lv) in enumerate(cases):
            v = h if j == 0 else lv
            if v is None:
                continue
            b = ax.bar(i + (j - 0.5) * w, v[0], w, yerr=v[1], capsize=4, color=col,
                       label=lab if i == 0 else None)
            ax.annotate(f"{v[0]:.0f}\n{v[2] / 1e3:.0f} kJ", (b[0].get_x() + w / 2, v[0] + v[1]),
                        textcoords="offset points", xytext=(0, 3), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([c[0] for c in cases])
    ax.set_ylabel("EE (bit/J)  [label: EE, total flight energy]")
    ax.set_title("Where should the RIS be? (aerial carrier's propulsion energy is charged)")
    ax.set_ylim(0, None)
    ax.legend(fontsize=8, loc="lower right")
    out = RESULTS / "fig_ris_value.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    for name, h, lv in cases:
        print(f"  {name.splitlines()[0]:14s} heuristic {h[0] if h else float('nan'):7.0f}   "
              f"learned {lv[0] if lv else float('nan'):7.0f}")


SWEEP_TITLES = {"ris_size": "RIS elements L", "phase_bits": "RIS phase bits (0 = continuous)",
                "phase_err_deg": "random phase error (deg, std)", "p_tx_dbm": "device transmit power (dBm)",
                "data_scale": "data per device (× 10–30 Mbit)", "arbitration": "RIS setting"}


def plot_sweeps(preset="building"):
    from .stats import mean_ci
    root = RESULTS / "sweeps" / preset
    files = sorted(root.glob("*.csv")) if root.exists() else []
    if not files:
        print("no sweeps")
        return
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, f in zip(axes.ravel(), files):
        rows = _read_csv(f)
        name = rows[0]["param"]
        vals = list(dict.fromkeys(r["value"] for r in rows))
        cat = name == "arbitration"
        x = np.arange(len(vals)) if cat or name == "phase_bits" else np.array([float(v) for v in vals])
        pols = list(dict.fromkeys(r["policy"] for r in rows))
        for i, pol in enumerate(pols):
            sel = [[r for r in rows if r["value"] == v and r["policy"] == pol] for v in vals]
            mu = np.array([np.mean([float(r["ee"]) for r in rs]) for rs in sel])
            if len(sel[0]) > 1:      # learned: one row per training seed -> CI over seeds
                sd = np.array([mean_ci([float(r["ee"]) for r in rs])["ci95"] for rs in sel])
            else:                    # heuristic: CI over the test episodes
                sd = np.array([float(rs[0].get("ee_ci95", 0.0) or 0.0) for rs in sel])
            lab = LABELS.get(pol, POLICY_LABELS.get(pol, pol).replace("\n", " "))
            if cat:
                w = 0.8 / len(pols)
                ax.bar(x + (i - (len(pols) - 1) / 2) * w, mu, w, yerr=sd, capsize=2, label=lab)
            else:
                ax.errorbar(x, mu, yerr=sd, marker="o", capsize=3, label=lab)
        if cat or name == "phase_bits":
            ax.set_xticks(x)
            ax.set_xticklabels(["∞" if (name == "phase_bits" and v == "0") else v for v in vals])
        ax.set_xlabel(SWEEP_TITLES.get(name, name))
        ax.set_ylabel("EE (bit/J)")
        ax.set_ylim(0, None)
        ax.set_title(SWEEP_TITLES.get(name, name))
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(files):]:
        ax.axis("off")
    n = _read_csv(files[0])[0].get("n", "?")
    fig.suptitle(f"Sensitivity of trained policies (no retraining) and heuristics; {n} test episodes per point; "
                 "error bars: 95 % CI over training seeds (learned) or episodes (heuristics)")
    fig.tight_layout()
    fig.savefig(root / "fig_sweeps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("sweep figure written to", root)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "ris_value":
        plot_ris_value()
    if cmd == "sweeps":
        plot_sweeps(sys.argv[2] if len(sys.argv) > 2 else "building")
    if cmd == "multiarea":
        plot_area_maps()
        plot_multiarea()
    if cmd == "gate":
        plot_gate()
    elif cmd == "train":
        plot_train(sys.argv[2] if len(sys.argv) > 2 else "building")
