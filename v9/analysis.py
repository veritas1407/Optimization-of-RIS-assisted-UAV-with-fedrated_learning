"""Static link-budget analysis used by the gate experiment (no learning involved)."""
from __future__ import annotations

import numpy as np

from . import physics as ph


def coverage_map(cfg, dev_xy, grid_n: int = 80, n_draws: int = 8, seed: int = 0) -> dict:
    """Mean achievable rate (bit/s) from one ground device to a UAV at every grid point.

    Returns rates without RIS and with a 2-D grid of RIS-aligned (quantized) phases,
    averaged over ``n_draws`` small-scale fading draws.
    """
    rng = np.random.default_rng(seed)
    boxes = ph.boxes_array(cfg.buildings)
    p_w, n_w = ph.tx_power_w(cfg), ph.noise_power_w(cfg)
    xs = np.linspace(0.0, cfg.area, grid_n)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    uav = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, cfg.uav_alt)])
    dev = np.array([[dev_xy[0], dev_xy[1], 0.0]])
    spacing = cfg.ris.spacing_wl * ph.wavelength(cfg.fc)
    center = np.asarray(cfg.ris.center, float)
    elems = ph.ris_elements(center, cfg.ris.normal, cfg.ris.ny, cfg.ris.nz, spacing)
    r_dir = np.zeros(X.size)
    r_ris = np.zeros(X.size)
    for _ in range(n_draws):
        h_d, blocked = ph.direct_channel(dev, uav, cfg, boxes, rng)          # (1, N)
        h_d = h_d[0]
        h_ir, _ = ph.ris_side_channel(dev, elems, center, cfg.ris.normal, cfg, boxes, rng, "los")
        h_ru, _ = ph.ris_side_channel(uav, elems, center, cfg.ris.normal, cfg, boxes, rng, "los")
        theta = ph.quantize(np.angle(h_d)[:, None] - np.angle(h_ru) - np.angle(h_ir[0])[None, :],
                            cfg.ris.bits)
        tot = h_d + np.sum(h_ru * np.exp(1j * theta) * h_ir[0][None, :], axis=1)
        r_dir += ph.rate_bps(np.abs(h_d) ** 2, cfg, p_w, n_w)
        r_ris += ph.rate_bps(np.abs(tot) ** 2, cfg, p_w, n_w)
    shape = X.shape
    return {"x": xs, "rate_dir": (r_dir / n_draws).reshape(shape),
            "rate_ris": (r_ris / n_draws).reshape(shape),
            "blocked": blocked[0].reshape(shape), "dev": np.asarray(dev_xy, float)}


def upload_time_table(cfg, data_bits: float, distances=(0.0, 50.0, 100.0, 200.0, 300.0, 400.0),
                      n_draws: int = 64, seed: int = 1) -> list[dict]:
    """Seconds to upload ``data_bits`` from an open device and from a shadowed device
    (with and without the RIS), for a UAV at horizontal distance d heading toward the
    open field (away from the courtyard)."""
    rng = np.random.default_rng(seed)
    boxes = ph.boxes_array(cfg.buildings)
    p_w, n_w = ph.tx_power_w(cfg), ph.noise_power_w(cfg)
    x0, y0, x1, y1 = cfg.shadow_region
    sh = np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0.0])
    op = np.array([150.0, 300.0, 0.0])
    spacing = cfg.ris.spacing_wl * ph.wavelength(cfg.fc)
    center = np.asarray(cfg.ris.center, float)
    elems = ph.ris_elements(center, cfg.ris.normal, cfg.ris.ny, cfg.ris.nz, spacing)
    rows = []
    for d in distances:
        out = {"distance_m": d}
        for name, dev, direction in (("open", op, np.array([0.0, 1.0])),
                                     ("shadow", sh, np.array([-0.6, 0.8]))):
            uav = np.array([[dev[0] + direction[0] * d, dev[1] + direction[1] * d, cfg.uav_alt]])
            r_d = r_r = 0.0
            for _ in range(n_draws):
                h_d, _ = ph.direct_channel(dev[None], uav, cfg, boxes, rng)
                h_ir, _ = ph.ris_side_channel(dev[None], elems, center, cfg.ris.normal, cfg, boxes, rng, "los")
                h_ru, _ = ph.ris_side_channel(uav, elems, center, cfg.ris.normal, cfg, boxes, rng, "los")
                th = ph.aligned_phases(h_d[0, 0], h_ru[0], h_ir[0], cfg.ris.bits)
                g = ph.composite(h_d, h_ru, h_ir, th[None])[0, 0, 0]
                r_d += ph.rate_bps(abs(h_d[0, 0]) ** 2, cfg, p_w, n_w)
                r_r += ph.rate_bps(g, cfg, p_w, n_w)
            out[f"{name}_noris_s"] = data_bits / (r_d / n_draws)
            if name == "shadow":
                out["shadow_ris_s"] = data_bits / (r_r / n_draws)
        rows.append(out)
    return rows
