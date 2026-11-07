"""Physical layer for v9: geometry, blockage, channels, RIS alignment, propulsion.

Conventions (used by the environment AND all baselines):

* Channel amplitudes are complex, in units of sqrt(linear power gain).
* Composite channel of device i to UAV k under RIS phases theta:
      h_tot[i, k] = h_d[i, k] + sum_l h_ru[k, l] * exp(j theta_l) * h_ir[i, l]
  (no Hermitian anywhere).
* Aligned phases for the target link (i, k):
      theta_l = arg h_d[i, k] - arg h_ru[k, l] - arg h_ir[i, l]
  which makes every reflected term co-phased with the direct term, so that
  |h_tot| = |h_d| + sum_l |h_ru[k, l]| |h_ir[i, l]|.
"""
from __future__ import annotations

import numpy as np

C_LIGHT = 299_792_458.0


# ---------------------------------------------------------------- radio helpers

def wavelength(fc: float) -> float:
    return C_LIGHT / fc


def pl0_db(fc: float) -> float:
    """Free-space path loss at 1 m (dB)."""
    return 20.0 * np.log10(4.0 * np.pi * fc / C_LIGHT)


def noise_power_w(cfg) -> float:
    dbm = cfg.n0_dbm_hz + 10.0 * np.log10(cfg.bw_hz) + cfg.nf_db
    return 10.0 ** ((dbm - 30.0) / 10.0)


def tx_power_w(cfg) -> float:
    return 10.0 ** ((cfg.p_tx_dbm - 30.0) / 10.0)


def db_to_lin(x):
    return 10.0 ** (np.asarray(x) / 10.0)


# ---------------------------------------------------------------- propulsion

def propulsion_power(v, cfg) -> np.ndarray:
    """Rotary-wing propulsion power (W) at horizontal speed v (m/s), Zeng et al."""
    v = np.asarray(v, dtype=float)
    blade = cfg.P_B * (1.0 + 3.0 * v ** 2 / cfg.U_tip ** 2)
    parasite = 0.5 * cfg.d0 * cfg.rho * cfg.s * cfg.A * v ** 3
    inner = np.sqrt(1.0 + v ** 4 / (4.0 * cfg.v0 ** 4)) - v ** 2 / (2.0 * cfg.v0 ** 2)
    induced = cfg.P_I * np.sqrt(np.maximum(inner, 0.0))
    return blade + parasite + induced


def min_power_speed(cfg) -> tuple[float, float]:
    vs = np.linspace(0.0, cfg.v_max, 2001)
    p = propulsion_power(vs, cfg)
    i = int(np.argmin(p))
    return float(vs[i]), float(p[i])


# ---------------------------------------------------------------- geometry

def boxes_array(buildings) -> np.ndarray:
    if not buildings:
        return np.zeros((0, 6))
    return np.array([[b.x0, b.y0, 0.0, b.x1, b.y1, b.h] for b in buildings], dtype=float)


def segments_blocked(p: np.ndarray, q: np.ndarray, boxes: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """True where segment p->q passes through any box (slab test).

    p, q: (..., 3) arrays with matching leading shapes. boxes: (B, 6) [xmin ymin zmin xmax ymax zmax].
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    shape = np.broadcast_shapes(p.shape, q.shape)[:-1]
    if boxes.shape[0] == 0:
        return np.zeros(shape, dtype=bool)
    p = np.broadcast_to(p, shape + (3,))[..., None, :]          # (..., 1, 3)
    d = np.broadcast_to(q, shape + (3,))[..., None, :] - p       # (..., 1, 3)
    lo = boxes[:, :3]
    hi = boxes[:, 3:]
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (lo - p) * inv
        t2 = (hi - p) * inv
    tmin = np.minimum(t1, t2)
    tmax = np.maximum(t1, t2)
    # Axis parallel to slab: inside -> (-inf, inf); outside -> empty interval.
    par = np.abs(d) < 1e-12
    inside = (p >= lo) & (p <= hi)
    tmin = np.where(par, np.where(inside, -np.inf, np.inf), tmin)
    tmax = np.where(par, np.where(inside, np.inf, -np.inf), tmax)
    t_enter = np.max(tmin, axis=-1)
    t_exit = np.min(tmax, axis=-1)
    hit = (t_enter <= t_exit) & (t_exit > eps) & (t_enter < 1.0 - eps)
    return np.any(hit, axis=-1)


def inside_any_box(xy: np.ndarray, boxes: np.ndarray, margin: float = 0.0) -> np.ndarray:
    xy = np.atleast_2d(xy)
    if boxes.shape[0] == 0:
        return np.zeros(len(xy), dtype=bool)
    x = xy[:, None, 0]
    y = xy[:, None, 1]
    hit = ((x >= boxes[:, 0] - margin) & (x <= boxes[:, 3] + margin)
           & (y >= boxes[:, 1] - margin) & (y <= boxes[:, 4] + margin))
    return hit.any(axis=1)


def ris_frame(normal) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(ref, n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return n, e1, e2


def ris_elements(center, normal, ny: int, nz: int, spacing: float) -> np.ndarray:
    """(L, 3) element positions of a planar array centred at ``center``."""
    _, e1, e2 = ris_frame(normal)
    a = (np.arange(ny) - (ny - 1) / 2.0) * spacing
    b = (np.arange(nz) - (nz - 1) / 2.0) * spacing
    A, B = np.meshgrid(a, b, indexing="ij")
    return (np.asarray(center, dtype=float)[None, :]
            + A.reshape(-1, 1) * e1[None, :] + B.reshape(-1, 1) * e2[None, :])


def in_front(points: np.ndarray, center, normal) -> np.ndarray:
    n, _, _ = ris_frame(normal)
    return (np.asarray(points) - np.asarray(center)) @ n > 0.0


# ---------------------------------------------------------------- channels

def cn(rng: np.random.Generator, shape) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def a2g_link_params(tx: np.ndarray, rx: np.ndarray, cfg, boxes: np.ndarray):
    """Large-scale parameters of ground/air links tx->rx (broadcast over leading dims).

    Returns (path_loss_db, kappa_lin, blocked, distance).
    """
    d = np.linalg.norm(rx - tx, axis=-1)
    d = np.maximum(d, 1.0)
    blocked = segments_blocked(tx, rx, boxes)
    if cfg.clutter == "alhourani":
        dz = np.abs(rx[..., 2] - tx[..., 2])
        elev = np.degrees(np.arcsin(np.clip(dz / d, 0.0, 1.0)))
        p_los = 1.0 / (1.0 + cfg.alh_a * np.exp(-cfg.alh_b * (elev - cfg.alh_a)))
        p_los = np.where(blocked, 0.0, p_los)
        pl = (pl0_db(cfg.fc) + 20.0 * np.log10(d)
              + p_los * cfg.eta_los_db + (1.0 - p_los) * cfg.eta_nlos_db)
        kappa = p_los * db_to_lin(cfg.kappa_direct_db)
    else:
        pl = pl0_db(cfg.fc) + 10.0 * cfg.exp_direct_fixed * np.log10(d)
        kappa = np.zeros_like(d)
    pl = pl + np.where(blocked, cfg.building_loss_db, 0.0)
    return pl, kappa, blocked, d


def rician(pl_db, kappa, los_phase, nlos) -> np.ndarray:
    amp = np.sqrt(db_to_lin(-np.asarray(pl_db)))
    k = np.asarray(kappa)
    return amp * (np.sqrt(k / (k + 1.0)) * np.exp(1j * los_phase) + np.sqrt(1.0 / (k + 1.0)) * nlos)


def direct_channel(dev: np.ndarray, uav: np.ndarray, cfg, boxes, rng) -> tuple[np.ndarray, np.ndarray]:
    """h_d (I, K) and geometric blockage (I, K)."""
    tx = dev[:, None, :]
    rx = uav[None, :, :]
    pl, kappa, blocked, d = a2g_link_params(tx, rx, cfg, boxes)
    lam = wavelength(cfg.fc)
    h = rician(pl, kappa, -2.0 * np.pi * d / lam, cn(rng, pl.shape))
    return h, blocked


def ris_side_channel(points: np.ndarray, elems: np.ndarray, ris_center, ris_normal,
                     cfg, boxes, rng, model: str):
    """Channels between ``points`` (N, 3) and every RIS element: (N, L), plus eligibility (N,).

    model="los": elevated building RIS, LoS Rician with the RIS exponent when the segment to
    the RIS is clear; model="a2g": use the device-UAV air-to-ground model (aerial RIS).
    The LoS phase uses the exact distance to each element (spherical wavefront).
    """
    center = np.asarray(ris_center, dtype=float)
    elig = in_front(points, center, ris_normal)
    lam = wavelength(cfg.fc)
    d_el = np.linalg.norm(points[:, None, :] - elems[None, :, :], axis=-1)   # (N, L)
    los_phase = -2.0 * np.pi * d_el / lam
    if model == "los":
        blocked = segments_blocked(points, center[None, :], boxes)
        d_c = np.maximum(np.linalg.norm(points - center, axis=-1), 1.0)
        pl = pl0_db(cfg.fc) + 10.0 * cfg.ris.exp * np.log10(d_c)
        kappa = np.full(len(points), db_to_lin(cfg.ris.kappa_db))
        elig = elig & ~blocked
    else:
        pl, kappa, _, _ = a2g_link_params(points, center[None, :], cfg, boxes)
    h = rician(pl[:, None], kappa[:, None], los_phase, cn(rng, d_el.shape))
    return h * elig[:, None], elig


def aligned_phases(h_d: np.ndarray, h_ru: np.ndarray, h_ir: np.ndarray, bits: int,
                   err_deg: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Phases (L,) aligning the reflected path of one link with its direct path."""
    theta = np.angle(h_d) - np.angle(h_ru) - np.angle(h_ir)
    return quantize(theta, bits, err_deg, rng)


def quantize(theta: np.ndarray, bits: int, err_deg: float = 0.0,
             rng: np.random.Generator | None = None) -> np.ndarray:
    theta = np.mod(theta, 2.0 * np.pi)
    if bits and bits > 0:
        step = 2.0 * np.pi / (2 ** bits)
        theta = np.mod(np.round(theta / step) * step, 2.0 * np.pi)
    if err_deg > 0.0 and rng is not None:
        theta = theta + np.radians(err_deg) * rng.standard_normal(theta.shape)
    return theta


def composite(h_d: np.ndarray, h_ru: np.ndarray, h_ir: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """|h_tot|^2 for every config: thetas (C, L) -> gains (C, I, K)."""
    phi = np.exp(1j * thetas)                                   # (C, L)
    refl = np.einsum("kl,cl,il->cik", h_ru, phi, h_ir)
    return np.abs(h_d[None, :, :] + refl) ** 2


def best_aligned_gain(h_d: np.ndarray, h_ru: np.ndarray, h_ir: np.ndarray) -> np.ndarray:
    """Upper bound |h_d| + sum |h_ru||h_ir| squared, for every (i, k): (I, K)."""
    refl = np.abs(h_ir) @ np.abs(h_ru).T                        # (I, K)
    return (np.abs(h_d) + refl) ** 2


def rate_bps(gain, cfg, p_w: float, noise_w: float) -> np.ndarray:
    return cfg.bw_hz * np.log2(1.0 + p_w * np.asarray(gain) / noise_w)
