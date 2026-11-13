"""Regression tests for the findings of docs/analysis/06-v9-audit.md (v9.1 fixes)."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from v9 import physics as ph
from v9.baselines import Greedy, random_policy, run_episode, straight_policy
from v9.config import building, get_preset, legacy_small
from v9.env import RISUAVEnv
from v9.trainers import Learner, TrainConfig


def test_slot_energy_matches_fine_integral():
    """M1: composite Simpson on the linear velocity path instead of the trapezoid of powers."""
    cfg = building()
    cases = [((10.0, 0.0), (10.0, 0.0)), ((0.0, 0.0), (12.5, 0.0)), ((20.0, 0.0), (7.5, 0.0)),
             ((10.0, 0.0), (-2.5, 0.0)), ((8.0, -6.0), (0.0, 12.0))]   # last two pass through/near hover
    for v0, v1 in cases:
        v0, v1 = np.array(v0), np.array(v1)
        e = ph.slot_energy(v0[None], v1[None], cfg.dt, cfg)[0]
        t = np.linspace(0.0, 1.0, 200001)
        sp = np.linalg.norm(v0 + (v1 - v0) * t[:, None], axis=1)
        fine = np.trapezoid(ph.propulsion_power(sp, cfg), dx=cfg.dt / 200000)
        assert e == pytest.approx(fine, rel=2e-3)
    # constant speed is exact
    assert ph.slot_energy(np.array([[6.0, 8.0]]), np.array([[6.0, 8.0]]), cfg.dt, cfg)[0] == \
        pytest.approx(cfg.dt * ph.propulsion_power(10.0, cfg))


def test_legacy_scene_ris_is_active():
    """C3: the old 20 m scene's RIS must actually reflect (it was switched off in v9.0)."""
    env = RISUAVEnv(legacy_small())
    env.reset(seed=1)
    assert env.elig_static.all() and np.abs(env.h_ir_static).max() > 0


def test_phase_error_draws_do_not_shift_fading():
    """F5: a negligible phase error must not change results (separate random stream)."""
    cfg = building()
    a = run_episode(RISUAVEnv(cfg), Greedy(), seed=5_000_003)["ee"]
    b = run_episode(RISUAVEnv(replace(cfg, ris=replace(cfg.ris, phase_err_deg=1e-9))), Greedy(), seed=5_000_003)["ee"]
    assert a == pytest.approx(b, rel=1e-6)


def test_open_devices_are_outside_the_building_block():
    env = RISUAVEnv(building())
    x0, y0, x1, y1 = env.block_hull
    for s in range(50):
        env.reset(seed=s)
        o = env.dev[~env.shadow]
        inside = (o[:, 0] >= x0) & (o[:, 0] <= x1) & (o[:, 1] >= y0) & (o[:, 1] <= y1)
        assert not inside.any()


def test_random_policy_is_reproducible():
    cfg = building()
    a = run_episode(RISUAVEnv(cfg), random_policy, seed=7)["ee"]
    b = run_episode(RISUAVEnv(cfg), random_policy, seed=7)["ee"]
    assert a == b


def test_landing_slots_inside_area_aerial():
    env = RISUAVEnv(get_preset("aerial"))
    assert np.all((env.final_targets >= 0) & (env.final_targets <= env.cfg.area))


def test_collision_push_restores_3d_separation():
    env = RISUAVEnv(get_preset("aerial"))
    env.reset(seed=0)
    env.pos[0] = env.pos[env.K] - np.array([1.0, 0.0, 10.0])   # collector just below the carrier
    env.vel[:] = 0.0
    env.step(np.zeros((env.n_agents, 2)))
    assert np.linalg.norm(env.pos[0] - env.pos[env.K]) >= env.cfg.d_min - 1e-6


def test_boundary_stops_the_uav():
    env = RISUAVEnv(building())
    env.reset(seed=0)
    env.pos[0, :2] = [2.0, 250.0]
    env.vel[0] = [-10.0, 0.0]
    acts = np.zeros((env.n_agents, 2))
    acts[0] = [-1.0, 0.0]
    env.step(acts)
    assert env.pos[0, 0] == 0.0 and env.vel[0, 0] == 0.0


def test_fedavg_eval_view_is_the_averaged_model():
    """C2: FL methods are evaluated on the federated (averaged) model."""
    env = RISUAVEnv(building())
    lr = Learner(env, TrainConfig(algo="fedavg"))
    with torch.no_grad():
        for a, actor in lr.actors.items():
            for p in actor.parameters():
                p.fill_(float(a))
    v = lr.eval_view()
    for a in lr.clients:
        for p in v.actors[a].parameters():
            assert torch.allclose(p, torch.full_like(p, np.mean(lr.clients)))
    # the training copies are untouched
    assert torch.allclose(next(lr.actors[2].parameters()), torch.full_like(next(lr.actors[2].parameters()), 2.0))


def test_fedper_eval_view_keeps_personal_heads():
    env = RISUAVEnv(building())
    lr = Learner(env, TrainConfig(algo="fedper"))
    with torch.no_grad():
        for a, actor in lr.actors.items():
            for p in actor.parameters():
                p.fill_(float(a))
    v = lr.eval_view()
    sd = v.actors[2].state_dict()
    assert torch.allclose(sd["mu.0.weight"], torch.full_like(sd["mu.0.weight"], 1.0))
    assert torch.allclose(sd["mu.4.weight"], torch.full_like(sd["mu.4.weight"], 2.0))


def test_element_pattern_matches_tang_model():
    """M2: per-hop factor sqrt(pi*G)*cos^q, so the cascade is pi*G*F_in*F_out (Tang et al. 2021)."""
    ris = building().ris
    c = np.zeros(3)
    n = np.array([0.0, 1.0, 0.0])
    pts = np.array([[0.0, 10.0, 0.0], [10.0, 10.0, 0.0]])
    f = ph.element_power_factor(pts, c, n, ris)
    G = 2 * (ris.elem_q + 1)
    assert f[0] == pytest.approx(np.sqrt(np.pi * G))
    assert f[1] == pytest.approx(np.sqrt(np.pi * G) * np.cos(np.pi / 4) ** ris.elem_q)
    assert np.allclose(ph.element_power_factor(pts, c, n, replace(ris, elem_model="point")), 1.0)


def test_aimed_building_ris_faces_courtyard_and_flight_zone():
    cfg = building()
    c = np.asarray(cfg.ris.center)
    x0, y0, x1, y1 = cfg.shadow_region
    for p in ([(x0 + x1) / 2, (y0 + y1) / 2, 0.0], [250.0, 250.0, cfg.uav_alt]):
        v = np.asarray(p) - c
        ang = np.degrees(np.arccos(v @ np.asarray(cfg.ris.normal) / np.linalg.norm(v)))
        assert ang < 50.0


def test_decentralized_greedy_runs_and_differs():
    cfg = building()
    m1 = run_episode(RISUAVEnv(cfg), Greedy(centralized=True), seed=11)
    m2 = run_episode(RISUAVEnv(cfg), Greedy(centralized=False), seed=11)
    assert m1["reached_final"] == 1.0 and m2["reached_final"] == 1.0
    assert m1["ee"] != m2["ee"]


def test_straight_line_still_collision_free():
    m = run_episode(RISUAVEnv(building()), straight_policy, seed=1)
    assert m["reached_final"] == 1.0 and m["collisions"] == 0


@pytest.mark.parametrize("preset", ["building", "aerial", "legacy_small"])
def test_landing_slots_strictly_inside_final_zone(preset):
    env = RISUAVEnv(get_preset(preset))
    d = np.linalg.norm(env.final_targets - np.asarray(env.cfg.final_xy), axis=1)
    assert np.all(d < env.cfg.r_final)


def test_building_landing_slots_unchanged():
    env = RISUAVEnv(building())
    assert np.allclose(np.linalg.norm(env.final_targets - np.asarray(env.cfg.final_xy), axis=1), [24, 0, 24])


def test_separation_layer_keeps_uavs_apart():
    from v9.env import separate
    env = RISUAVEnv(building())
    env.reset(seed=0)
    env.pos[0, :2], env.pos[1, :2] = [200.0, 200.0], [240.0, 200.0]
    env.vel[:] = 0.0
    env.vel[0], env.vel[1] = [10.0, 0.0], [-10.0, 0.0]
    acts = np.zeros((env.n_agents, 2))
    acts[0], acts[1] = [1.0, 0.0], [-1.0, 0.0]          # flying head-on
    env.step(separate(env, acts))
    assert np.linalg.norm(env.pos[0] - env.pos[1]) >= env.cfg.d_min


def test_autopilot_does_not_cause_separation_violations():
    m = run_episode(RISUAVEnv(building()), Greedy(), seed=2_000_000)   # leaves late -> autopilot flies home
    assert m["rth_frac"] > 0 and m["collisions"] <= 1


def test_opt_shares_maximise_deliverable_bits():
    env = RISUAVEnv(replace(building(), arbitration="opt"))
    env.reset(seed=0)
    served = np.full(env.I, -1)
    served[0], served[1] = 0, 1
    env.remaining[:2] = [1e9, 1.0]                     # device 1 needs almost nothing
    g = np.zeros((2, env.I, env.K))
    g[0, 0, 0], g[1, 1, 1] = 1e-9, 1e-9               # config c serves device c only
    tau = env._opt_shares(g, served)
    assert tau.sum() == pytest.approx(1.0) and tau[0] >= 0.95


def test_imperfect_csi_lowers_ee_and_perfect_is_default():
    cfg = building()
    assert cfg.csi_nmse_db is None
    good = run_episode(RISUAVEnv(cfg), straight_policy, seed=5_000_001)["ee"]
    bad = run_episode(RISUAVEnv(replace(cfg, csi_nmse_db=0.0)), straight_policy, seed=5_000_001)["ee"]
    assert bad < good
