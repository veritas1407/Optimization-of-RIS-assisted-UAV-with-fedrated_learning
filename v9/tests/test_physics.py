import numpy as np
import pytest

from v9 import physics as ph
from v9.config import Box, building, legacy_small


def test_hover_power_matches_zeng():
    cfg = building()
    assert ph.propulsion_power(0.0, cfg) == pytest.approx(168.48, abs=0.1)


def test_min_power_speed_is_cruise_not_hover():
    cfg = building()
    v, p = ph.min_power_speed(cfg)
    assert 8.0 < v < 14.0
    assert p < ph.propulsion_power(0.0, cfg)


def _rand_links(rng, L=64):
    h_d = ph.cn(rng, ())
    h_ru = ph.cn(rng, (L,))
    h_ir = ph.cn(rng, (L,))
    return h_d, h_ru, h_ir


def test_aligned_phases_reach_analytic_maximum():
    """Regression test for the v8 sign bug (audit F1)."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        h_d, h_ru, h_ir = _rand_links(rng)
        theta = ph.aligned_phases(h_d, h_ru, h_ir, bits=0)
        g = ph.composite(np.array([[h_d]]), h_ru[None], h_ir[None], theta[None])[0, 0, 0]
        g_max = (abs(h_d) + np.sum(np.abs(h_ru) * np.abs(h_ir))) ** 2
        assert g == pytest.approx(g_max, rel=1e-9)


def test_quantized_phases_within_quantization_bound():
    rng = np.random.default_rng(1)
    L = 256
    ratios = []
    for _ in range(50):
        h_d, h_ru, h_ir = _rand_links(rng, L)
        h_d = h_d * 1e-3   # reflected path dominates
        theta = ph.aligned_phases(h_d, h_ru, h_ir, bits=2)
        g = ph.composite(np.array([[h_d]]), h_ru[None], h_ir[None], theta[None])[0, 0, 0]
        g_max = (abs(h_d) + np.sum(np.abs(h_ru) * np.abs(h_ir))) ** 2
        ratios.append(g / g_max)
    # 2-bit loss is sinc^2(pi/4) ~= 0.81 (-0.9 dB) in expectation
    assert 0.75 < np.mean(ratios) < 0.88


def test_v8_formula_is_worse_than_random():
    """Documents audit F1: the v8 conj() formula misaligns the reflected paths."""
    rng = np.random.default_rng(2)
    h_d, h_ru, h_ir = _rand_links(rng, 50)
    th_v8 = np.angle(h_d) - np.angle(np.conj(h_ru)) - np.angle(h_ir)
    g_v8 = ph.composite(np.array([[h_d]]), h_ru[None], h_ir[None], th_v8[None])[0, 0, 0]
    th_ok = ph.aligned_phases(h_d, h_ru, h_ir, bits=0)
    g_ok = ph.composite(np.array([[h_d]]), h_ru[None], h_ir[None], th_ok[None])[0, 0, 0]
    assert g_ok > 10 * g_v8


def test_best_aligned_gain_matches_composite():
    rng = np.random.default_rng(3)
    I, K, L = 4, 3, 32
    h_d = ph.cn(rng, (I, K))
    h_ru = ph.cn(rng, (K, L))
    h_ir = ph.cn(rng, (I, L))
    ub = ph.best_aligned_gain(h_d, h_ru, h_ir)
    for i in range(I):
        for k in range(K):
            th = ph.aligned_phases(h_d[i, k], h_ru[k], h_ir[i], bits=0)
            assert ph.composite(h_d, h_ru, h_ir, th[None])[0, i, k] == pytest.approx(ub[i, k])


def test_rician_second_moment():
    rng = np.random.default_rng(4)
    n = 200_000
    h = ph.rician(np.full(n, 0.0), np.full(n, 10.0), rng.uniform(0, 6.28, n), ph.cn(rng, n))
    assert np.mean(np.abs(h) ** 2) == pytest.approx(1.0, rel=0.02)


def test_segment_blocking():
    boxes = ph.boxes_array([Box(0, 0, 10, 10, 20)])
    p = np.array([[-5.0, 5.0, 1.0], [-5.0, 5.0, 1.0], [-5.0, 5.0, 30.0], [5.0, 5.0, 25.0]])
    q = np.array([[15.0, 5.0, 1.0], [-5.0, 15.0, 1.0], [15.0, 5.0, 30.0], [5.0, 5.0, 40.0]])
    assert list(ph.segments_blocked(p, q, boxes)) == [True, False, False, False]


def test_courtyard_shadowed_unless_overhead():
    cfg = building()
    boxes = ph.boxes_array(cfg.buildings)
    dev = np.array([[370.0, 110.0, 0.0]])
    far = np.array([[200.0, 300.0, 60.0]])
    above = np.array([[370.0, 110.0, 60.0]])
    assert ph.segments_blocked(dev, far, boxes)[0]
    assert not ph.segments_blocked(dev, above, boxes)[0]


def test_building_ris_sees_courtyard_and_sky():
    cfg = building()
    boxes = ph.boxes_array(cfg.buildings)
    ris = np.asarray(cfg.ris.center)
    dev = np.array([[340.0, 110.0, 0.0]])
    uav = np.array([[200.0, 300.0, 60.0]])
    assert ph.in_front(dev, ris, cfg.ris.normal)[0] and ph.in_front(uav, ris, cfg.ris.normal)[0]
    assert not ph.segments_blocked(dev, ris[None], boxes)[0]
    assert not ph.segments_blocked(uav, ris[None], boxes)[0]


def test_ris_elements_geometry():
    el = ph.ris_elements((0, 0, 0), (0, 1, 0), 4, 3, 0.5)
    assert el.shape == (12, 3)
    assert np.allclose(el[:, 1], 0.0)                  # lies in the wall plane
    assert np.allclose(el.mean(axis=0), 0.0)


def test_noise_power_value():
    cfg = building()
    dbm = 10 * np.log10(ph.noise_power_w(cfg)) + 30
    assert dbm == pytest.approx(-174 + 10 * np.log10(180e3) + 7, abs=1e-6)
