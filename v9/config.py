"""Scenario and training configuration for v9.

All physical and scenario parameters live here as frozen dataclasses. Nothing in the
simulator reads module-level mutable state: sweeps build new configs with
``dataclasses.replace``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Box:
    """Axis-aligned building footprint with height (metres)."""

    x0: float
    y0: float
    x1: float
    y1: float
    h: float


@dataclass(frozen=True)
class RISConfig:
    kind: str = "building"            # "building" | "aerial"
    center: tuple = (370.0, 95.5, 48.0)  # building: mount point; aerial: start (x, y, altitude)
    normal: tuple = (0.0, 1.0, 0.0)   # outward normal of the reflecting face
    ny: int = 16                      # elements along the first in-plane axis
    nz: int = 16                      # elements along the second in-plane axis
    spacing_wl: float = 0.5           # element spacing in wavelengths
    bits: int = 2                     # phase resolution; 0 = continuous
    kappa_db: float = 10.0            # Rician factor of the RIS links
    exp: float = 2.0                  # path-loss exponent of the RIS links
    payload_factor: float = 1.2       # aerial only: propulsion multiplier for carrying the panel
    phase_err_deg: float = 0.0        # std of random phase error added after quantization
    # Element model. "pattern": far-field model of Tang et al. (IEEE TWC 2021) with a
    # normalised element power pattern cos^q(theta) and element gain G = 2(q+1); q = 0.57
    # gives G = pi, i.e. effective area = physical area of a lambda/2 x lambda/2 element
    # (Ellingson, IEEE OJ-COMS 2021). "point": isotropic point scatterers (product of two
    # free-space losses), the model used by v8 / Jiang et al.
    elem_model: str = "pattern"
    elem_q: float = 0.57
    halfspace: bool = True            # reflect only into the front half-space of the panel

    @property
    def L(self) -> int:
        return self.ny * self.nz


# Urban block in the south-east of the area: a 140 m x 30 m courtyard enclosed by
# 40-45 m buildings. Ground devices inside it are shadowed from UAVs unless a UAV is
# nearly overhead. The building RIS hangs at the top of the south row's north face,
# looking into the courtyard and over the north row toward the open field.
URBAN_BLOCK = (
    Box(300.0, 60.0, 440.0, 95.0, 45.0),    # south row
    Box(300.0, 125.0, 440.0, 160.0, 45.0),  # north row
    Box(280.0, 60.0, 300.0, 160.0, 40.0),   # west cap
    Box(440.0, 60.0, 460.0, 160.0, 40.0),   # east cap
)


@dataclass(frozen=True)
class ScenarioConfig:
    name: str = "building"

    # Geometry and mission
    area: float = 500.0
    M: int = 40                       # slots
    dt: float = 2.5                   # seconds per slot
    K: int = 3                        # collector UAVs
    uav_alt: float = 60.0
    v_max: float = 20.0
    a_max: float = 5.0
    d_min: float = 20.0
    start_xy: tuple = (20.0, 20.0)
    start_spacing: float = 25.0
    final_xy: tuple = (480.0, 480.0)
    r_final: float = 40.0
    buildings: tuple = URBAN_BLOCK

    # Devices
    n_open: int = 18
    n_shadow: int = 6
    shadow_region: tuple = (310.0, 100.0, 430.0, 120.0)  # x0, y0, x1, y1 inside the courtyard
    fixed_devices: tuple = ()         # if non-empty, overrides random placement (legacy scene)
    data_mbit: tuple = (10.0, 30.0)   # 1.25-3.75 MB per device (image / log batch)
    window_s: float = 30.0
    always_on: bool = False           # True = no duty cycle (legacy scene)

    # Radio
    fc: float = 2.4e9
    p_tx_dbm: float = 10.0
    bw_hz: float = 180e3
    n0_dbm_hz: float = -174.0
    nf_db: float = 7.0
    n_sub: int = 1                    # devices served per UAV per slot (one NB-IoT carrier)

    # Direct device-UAV channel
    clutter: str = "alhourani"        # "alhourani" (elevation-dependent LoS) | "fixed"
    alh_a: float = 9.61               # Al-Hourani urban parameters
    alh_b: float = 0.16
    eta_los_db: float = 1.0
    eta_nlos_db: float = 20.0
    building_loss_db: float = 20.0    # extra loss when an explicit building blocks the link
    kappa_direct_db: float = 5.0
    exp_direct_fixed: float = 3.5     # only for clutter="fixed"

    # RIS
    ris: RISConfig = field(default_factory=RISConfig)
    arbitration: str = "timeshare"    # "timeshare" | "common" | "random" | "none"

    # Propulsion (Zeng, Xu, Zhang 2019 rotary-wing model)
    P_B: float = 79.8563
    P_I: float = 88.6279
    U_tip: float = 120.0
    v0: float = 4.03
    d0: float = 0.6
    rho: float = 1.225
    s: float = 0.05
    A: float = 0.503

    # Observation
    agent_id_obs: bool = True         # one-hot agent index (symmetry breaking for shared policies)
    top_k: int = 8                    # devices described in each UAV's observation

    # Return-to-home autopilot: when the remaining time is just enough to reach the final
    # point, the autopilot overrides the agent and flies home (data collection continues).
    # Overridden slots are counted and excluded from the policy-gradient loss.
    rth_shield: bool = True
    rth_frac: float = 0.9             # fraction of v_max the autopilot plans with (0.9 = least conservative value with 100% completion)

    # Reward
    lam_init: float = 6.0e3           # Dinkelbach price (bits/J); ~ straight-line + RIS baseline EE
    w_final: float = 5.0
    w_collision: float = 1.0
    final_slack: float = 0.8          # fraction of v_max assumed when checking reachability

    @property
    def T(self) -> float:
        return self.M * self.dt

    @property
    def n_devices(self) -> int:
        return len(self.fixed_devices) if self.fixed_devices else self.n_open + self.n_shadow

    @property
    def has_aerial_ris(self) -> bool:
        return self.arbitration != "none" and self.ris.kind == "aerial"

    @property
    def n_agents(self) -> int:
        return self.K + (1 if self.has_aerial_ris else 0)


def aim_normal(center, ground_target, sky_target) -> tuple:
    """Panel normal bisecting the directions to the shadowed ground and to the flight zone,
    as an installer would aim a fixed RIS."""
    import numpy as np

    c = np.asarray(center, float)
    u1 = np.asarray(ground_target, float) - c
    u2 = np.asarray(sky_target, float) - c
    n = u1 / np.linalg.norm(u1) + u2 / np.linalg.norm(u2)
    return tuple(float(v) for v in n / np.linalg.norm(n))


def _aimed(cfg: ScenarioConfig) -> ScenarioConfig:
    x0, y0, x1, y1 = cfg.shadow_region
    n = aim_normal(cfg.ris.center, ((x0 + x1) / 2, (y0 + y1) / 2, 0.0),
                   (cfg.area / 2, cfg.area / 2, cfg.uav_alt))
    return replace(cfg, ris=replace(cfg.ris, normal=n))


def building() -> ScenarioConfig:
    return _aimed(ScenarioConfig())


def aerial() -> ScenarioConfig:
    return ScenarioConfig(
        name="aerial",
        # 70 m: lowest altitude that keeps the downward-facing panel above the 60 m collectors;
        # static analysis showed lower carrier altitude gives the strongest cascaded link
        ris=RISConfig(kind="aerial", center=(40.0, 60.0, 70.0), normal=(0.0, 0.0, -1.0)),
    )


V8_DEVICES = (
    (3.0, 11.0), (6.0, 13.0), (9.0, 4.0), (12.0, 6.0), (2.0, 5.0), (15.0, 12.0),
    (7.0, 17.0), (14.0, 3.0), (4.0, 8.0), (11.0, 15.0), (17.0, 8.0), (1.0, 15.0),
)


def legacy_small() -> ScenarioConfig:
    """The v8 / Jiang et al. 20 m scene, re-expressed with real radio units."""
    return ScenarioConfig(
        name="legacy_small",
        area=20.0, M=20, dt=1.0, K=3, uav_alt=5.0, v_max=15.0, a_max=100.0, d_min=3.0,
        start_xy=(0.0, 0.0), start_spacing=3.0, final_xy=(16.0, 16.0), r_final=2.0,
        buildings=(), fixed_devices=V8_DEVICES, data_mbit=(1e6, 1e6), always_on=True,
        p_tx_dbm=30.0, n_sub=12, clutter="fixed",
        ris=RISConfig(kind="building", center=(5.0, 9.0, 0.0), normal=(0.0, 0.0, 1.0),
                      ny=10, nz=5, elem_model="point", halfspace=False),
    )


def random_area(layout_seed: int, base: ScenarioConfig | None = None) -> ScenarioConfig:
    """A procedurally generated city area (for multi-area federated learning).

    One urban block per area: a courtyard enclosed by two long building rows and two end
    caps, placed off the start->final diagonal at a random position, orientation and size.
    The RIS hangs 3 m above the top of the row that lies farther from the diagonal, on its
    courtyard face, so it sees the shadowed courtyard devices and, over the lower opposite
    row, the open field where the UAVs fly.
    """
    import numpy as np

    base = base or ScenarioConfig()
    rng = np.random.default_rng(10_000 + layout_seed)
    A = base.area
    length = rng.uniform(100.0, 160.0)          # courtyard length
    width = rng.uniform(25.0, 35.0)             # courtyard width
    depth = rng.uniform(30.0, 40.0)             # building depth
    h_ris_row = rng.uniform(40.0, 50.0)
    h_opp_row = rng.uniform(35.0, h_ris_row)    # opposite row no taller than the RIS row
    h_caps = rng.uniform(35.0, 45.0)
    along_x = bool(rng.integers(2))             # courtyard long axis along x or y
    side = 1.0 if rng.integers(2) else -1.0     # +1: block south-east of the diagonal
    # block centre: 120-180 m off the diagonal, within the area
    while True:
        t = rng.uniform(0.35, 0.65) * A
        off = rng.uniform(120.0, 180.0) / np.sqrt(2.0)
        cx, cy = t + side * off, t - side * off
        half_x = (length / 2 + 20) if along_x else (width / 2 + depth + 20)
        half_y = (width / 2 + depth + 20) if along_x else (length / 2 + 20)
        if half_x + 10 < cx < A - half_x - 10 and half_y + 10 < cy < A - half_y - 10:
            break
    w2, l2 = width / 2, length / 2
    if along_x:
        lo_row = Box(cx - l2, cy - w2 - depth, cx + l2, cy - w2, 0.0)   # south row
        hi_row = Box(cx - l2, cy + w2, cx + l2, cy + w2 + depth, 0.0)   # north row
        caps = (Box(cx - l2 - 20, cy - w2 - depth, cx - l2, cy + w2 + depth, h_caps),
                Box(cx + l2, cy - w2 - depth, cx + l2 + 20, cy + w2 + depth, h_caps))
        # diagonal lies north-west of a south-east block -> RIS on the south row, facing north
        ris_on_lo = side > 0
        normal = (0.0, 1.0, 0.0) if ris_on_lo else (0.0, -1.0, 0.0)
        y_wall = (cy - w2 + 0.5) if ris_on_lo else (cy + w2 - 0.5)
        ris_xy = (cx + rng.uniform(-20.0, 20.0), y_wall)
        shadow = (cx - l2 + 10, cy - w2 + 5, cx + l2 - 10, cy + w2 - 5)
    else:
        lo_row = Box(cx - w2 - depth, cy - l2, cx - w2, cy + l2, 0.0)   # west row
        hi_row = Box(cx + w2, cy - l2, cx + w2 + depth, cy + l2, 0.0)   # east row
        caps = (Box(cx - w2 - depth, cy - l2 - 20, cx + w2 + depth, cy - l2, h_caps),
                Box(cx - w2 - depth, cy + l2, cx + w2 + depth, cy + l2 + 20, h_caps))
        # diagonal lies west of a south-east block -> RIS on the east row, facing west
        ris_on_lo = side < 0
        normal = (1.0, 0.0, 0.0) if ris_on_lo else (-1.0, 0.0, 0.0)
        x_wall = (cx - w2 + 0.5) if ris_on_lo else (cx + w2 - 0.5)
        ris_xy = (x_wall, cy + rng.uniform(-20.0, 20.0))
        shadow = (cx - w2 + 5, cy - l2 + 10, cx + w2 - 5, cy + l2 - 10)
    h_lo, h_hi = (h_ris_row, h_opp_row) if ris_on_lo else (h_opp_row, h_ris_row)
    rows = (replace(lo_row, h=h_lo), replace(hi_row, h=h_hi))
    ris = replace(base.ris, kind="building", center=(ris_xy[0], ris_xy[1], h_ris_row + 3.0),
                  normal=normal)
    return _aimed(replace(base, name=f"area{layout_seed}", buildings=rows + caps,
                          shadow_region=shadow, ris=ris))


# Layout seeds for the multi-area study: disjoint training / validation / test areas.
# Each set covers the four layout types (block north-west / south-east of the flight
# diagonal x courtyard along x / along y); each training area is one distinct type.
TRAIN_AREAS = (1, 2, 9, 5)                     # NW-x, NW-y, SE-x, SE-y
VAL_AREAS = (105, 101, 102, 103)               # NW-x, NW-y, SE-x, SE-y
TEST_AREAS = (227, 237, 204, 206, 201, 211, 203, 207)   # 2 of each type


def building_noris() -> ScenarioConfig:
    """Same urban scene with the RIS switched off (isolates the RIS gain for learned flight)."""
    return replace(building(), name="building_noris", arbitration="none")


PRESETS = {"building": building, "building_noris": building_noris, "aerial": aerial,
           "legacy_small": legacy_small}


def get_preset(name: str, **overrides) -> ScenarioConfig:
    return replace(PRESETS[name](), **overrides)
