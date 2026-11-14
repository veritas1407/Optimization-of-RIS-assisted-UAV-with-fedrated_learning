"""Export one recorded mission as JSON for external 3-D rendering.

    python -m v9.export_trace --case building:ctde:0 --seed 5000002

Writes results_v9/animation/trace_<preset>_<policy>_<seed>.json with the static scene
(area, buildings, RIS panel, devices, final zone) and, for every 2.5 s slot, UAV positions
and velocities at the slot end, who uploaded to whom, which links used the RIS, data left,
bits and energy. Within a slot the velocity changes linearly, so a smooth curve (e.g.
Catmull-Rom) through the slot-end positions is a faithful interpolation for rendering.
Also writes results_v9/animation/fl_areas.json: the 24 heterogeneous city areas of the
federated-learning study (layout, RIS, deployment) for an FL overview scene.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .animate import make_policy
from .baselines import run_episode
from .config import get_preset
from .env import RISUAVEnv


def export(case: str, seed: int) -> Path:
    torch.set_num_threads(1)
    preset, policy, *rest = case.split(":")
    cfg = get_preset(preset)
    env = RISUAVEnv(cfg)
    pol = make_policy(preset, policy, int(rest[0]) if rest else 0, cfg)
    env.reset(seed=seed)
    start_pos, start_vel = env.pos.copy(), env.vel.copy()
    m, log = run_episode(env, pol, seed, record=True)
    vel_hist, prev = [], start_pos
    for x in log:
        vel_hist.append(((x["pos"][:, :2] - prev[:, :2]) / cfg.dt).tolist())   # mean velocity over the slot
        prev = x["pos"]
    ris = None
    if env.use_ris:
        ris = {"kind": cfg.ris.kind, "elements": [cfg.ris.ny, cfg.ris.nz],
               "element_spacing_m": float(env.spacing), "side_m": float(cfg.ris.ny * env.spacing),
               "phase_bits": cfg.ris.bits}
        if not env.aerial:
            ris |= {"center": env.ris_center.tolist(), "normal": list(cfg.ris.normal)}
        else:
            ris |= {"carrier_uav_index": int(env.K), "note": "panel hangs under UAV K, facing down/toward collectors"}
    out = {
        "project": "RIS-assisted multi-UAV IoT data collection (v9)",
        "case": case, "mission_seed": seed, "units": "metres, seconds, bits, joules",
        "area_m": cfg.area, "slot_s": cfg.dt, "n_slots": cfg.M, "uav_altitude_m": cfg.uav_alt,
        "n_collectors": int(env.K), "n_agents": int(env.n_agents), "min_separation_m": cfg.d_min,
        "final_zone": {"center": list(cfg.final_xy), "radius_m": cfg.r_final},
        "landing_slots": env.final_targets.tolist(),
        "buildings": [{"x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1, "height_m": b.h} for b in cfg.buildings],
        "courtyard_region": list(cfg.shadow_region), "ris": ris,
        "devices": [{"id": i, "xyz": env.dev[i].tolist(), "in_courtyard": bool(env.shadow[i]),
                     "data_bits": float(env.data0[i]), "awake_from_s": float(env.win_start[i]),
                     "awake_until_s": float(env.win_end[i])} for i in range(env.I)],
        "start": {"pos": start_pos.tolist(), "vel": start_vel.tolist()},
        "slots": [{"t_end_s": (k + 1) * cfg.dt, "pos": x["pos"].tolist(), "mean_vel_xy": vel_hist[k],
                   "uploads": [{"device": int(i), "uav": int(x["served"][i]),
                                "via_ris": any(int(i) == r[0] for r in x["ris_targets"])}
                               for i in np.where(x["served"] >= 0)[0]],
                   "bits_this_slot": float(x["bits"]), "energy_this_slot_J": float(x["energy"]),
                   "data_left_bits": x["remaining"].tolist()} for k, x in enumerate(log)],
        "mission_result": {k: float(v) for k, v in m.items()},
    }
    path = Path("results_v9/animation") / f"trace_{preset}_{policy}_{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return path


def export_areas() -> Path:
    from .config import HET_TEST, HET_TRAIN, HET_VAL, hetero_area
    areas = []
    for split, seeds in (("train", HET_TRAIN), ("validation", HET_VAL), ("test_unseen", HET_TEST)):
        for s in seeds:
            c = hetero_area(s)
            areas.append({"seed": s, "split": split, "n_devices": c.n_devices, "n_courtyard_devices": c.n_shadow,
                          "data_mbit_range": list(c.data_mbit), "device_power_dbm": c.p_tx_dbm,
                          "ris_elements": [c.ris.ny, c.ris.nz], "ris_center": list(c.ris.center),
                          "ris_normal": list(c.ris.normal), "courtyard_region": list(c.shadow_region),
                          "buildings": [{"x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1, "height_m": b.h}
                                        for b in c.buildings]})
    path = Path("results_v9/animation/fl_areas.json")
    path.write_text(json.dumps({"area_m": 500.0, "areas": areas}, indent=1))
    return path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--case", action="append", default=[])
    p.add_argument("--seed", type=int, default=5_000_002)
    a = p.parse_args(argv)
    for c in a.case or ["building:ctde:0", "building_noris:ctde:0", "aerial:ctde:0"]:
        path = export(c, a.seed)
        r = json.loads(path.read_text())["mission_result"]
        print(f"  {path}  EE {r['ee']:.0f} bit/J  delivered {r['delivered_frac']:.2f}")
    print(" ", export_areas())


if __name__ == "__main__":
    main()
