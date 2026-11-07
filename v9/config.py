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


def building() -> ScenarioConfig:
    return ScenarioConfig()


def aerial() -> ScenarioConfig:
    return ScenarioConfig(
        name="aerial",
        ris=RISConfig(kind="aerial", center=(40.0, 60.0, 100.0), normal=(0.0, 0.0, -1.0)),
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
                      ny=10, nz=5),
    )


PRESETS = {"building": building, "aerial": aerial, "legacy_small": legacy_small}


def get_preset(name: str, **overrides) -> ScenarioConfig:
    return replace(PRESETS[name](), **overrides)
