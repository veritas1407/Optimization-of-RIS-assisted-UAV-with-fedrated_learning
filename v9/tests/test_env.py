import numpy as np
import pytest

from v9.baselines import greedy_policy, run_episode, straight_policy
from v9.config import get_preset
from v9.env import RISUAVEnv


@pytest.mark.parametrize("preset", ["building", "aerial", "legacy_small"])
def test_shapes_and_finiteness(preset):
    env = RISUAVEnv(get_preset(preset))
    obs, state = env.reset(seed=3)
    assert obs.shape == (env.n_agents, env.obs_dim)
    assert state.shape == (env.state_dim,)
    rng = np.random.default_rng(0)
    for _ in range(env.cfg.M):
        obs, state, r, done, info = env.step(rng.uniform(-1, 1, (env.n_agents, 2)))
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
    assert done and "episode" in info


def test_same_seed_same_episode():
    cfg = get_preset("building")
    m1 = run_episode(RISUAVEnv(cfg), greedy_policy, seed=11)
    m2 = run_episode(RISUAVEnv(cfg), greedy_policy, seed=11)
    assert m1 == m2


def test_each_device_served_once_and_capacity():
    cfg = get_preset("building")
    env = RISUAVEnv(cfg)
    env.reset(seed=5)
    done = False
    while not done:
        before = env.remaining.copy()
        _, _, _, done, info = env.step(greedy_policy(env, None))
        served = info["served"]
        for k in range(env.K):
            assert np.sum(served == k) <= cfg.n_sub
        delivered = before - env.remaining
        assert np.all(delivered >= -1e-9)
        assert np.all(delivered[served < 0] == 0)            # unserved devices deliver nothing
        assert np.all(env.remaining >= -1e-9)


def test_episode_bits_equal_data_drop():
    env = RISUAVEnv(get_preset("building"))
    env.reset(seed=7)
    data0 = env.data0.sum()
    m = run_episode(env, straight_policy, seed=7)
    assert m["bits"] == pytest.approx(data0 - env.remaining.sum(), rel=1e-9)


def test_straight_line_reaches_final_without_collisions():
    m = run_episode(RISUAVEnv(get_preset("building")), straight_policy, seed=1)
    assert m["reached_final"] == 1.0 and m["collisions"] == 0


def test_return_to_home_shield_brings_random_policy_home():
    from v9.baselines import random_policy
    cfg = get_preset("building")
    reached = [run_episode(RISUAVEnv(cfg), random_policy, seed=s)["reached_final"] for s in range(5)]
    assert np.mean(reached) == 1.0
    off = get_preset("building", rth_shield=False)
    m = run_episode(RISUAVEnv(off), random_policy, seed=0)
    assert m["rth_frac"] == 0.0


def test_no_ris_mode_ignores_ris():
    cfg = get_preset("building", arbitration="none")
    env = RISUAVEnv(cfg)
    env.reset(seed=2)
    _, _, _, _, info = env.step(np.zeros((env.n_agents, 2)))
    assert info["ris_targets"] == []
