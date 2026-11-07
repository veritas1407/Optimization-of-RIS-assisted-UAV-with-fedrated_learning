import numpy as np
import torch

from v9.config import get_preset
from v9.env import RISUAVEnv
from v9.ppo import Actor, n_params
from v9.trainers import Learner, TrainConfig


def test_ratio_is_one_before_any_update():
    torch.manual_seed(0)
    actor = Actor(10)
    obs = np.random.default_rng(0).standard_normal((64, 10)).astype(np.float32)
    _, u, logp = actor.act(obs)
    d = actor.dist(torch.as_tensor(obs))
    new = d.log_prob(torch.as_tensor(u)).sum(-1).detach().numpy()
    assert np.allclose(np.exp(new - logp), 1.0, atol=1e-5)


def test_actions_in_bounds_and_logp_summed():
    actor = Actor(10)
    obs = np.zeros((5, 10), np.float32)
    a, u, logp = actor.act(obs)
    assert np.all(np.abs(a) <= 1.0)
    d = actor.dist(torch.as_tensor(obs))
    per_dim = d.log_prob(torch.as_tensor(u)).detach().numpy()
    assert np.allclose(per_dim.sum(-1), logp, atol=1e-5)


def test_fedavg_of_identical_clients_is_noop():
    env = RISUAVEnv(get_preset("building"))
    lr = Learner(env, TrainConfig(algo="fedavg"))
    before = {k: v.clone() for k, v in lr.actors[0].state_dict().items()}
    lr.aggregate()
    after = lr.actors[0].state_dict()
    assert all(torch.allclose(before[k], after[k]) for k in before)


def test_fedavg_averages_and_meters_bytes():
    env = RISUAVEnv(get_preset("building"))
    lr = Learner(env, TrainConfig(algo="fedavg"))
    with torch.no_grad():
        for a, actor in lr.actors.items():
            for p in actor.parameters():
                p.fill_(float(a))
    lr.aggregate()
    expected = np.mean(lr.clients)
    for a in lr.clients:
        for p in lr.actors[a].parameters():
            assert torch.allclose(p, torch.full_like(p, expected))
    per = (n_params(lr.actors[0]) + n_params(lr.critics[0])) * 4.0
    assert lr.comm_bytes == 2 * per * len(lr.clients)


def test_aerial_ris_agent_not_averaged_with_collectors():
    env = RISUAVEnv(get_preset("aerial"))
    lr = Learner(env, TrainConfig(algo="fedavg"))
    assert lr.clients == [0, 1, 2]
    assert env.n_agents == 4


def test_ippo_and_ctde_exchange_as_specified():
    env = RISUAVEnv(get_preset("building"))
    assert Learner(env, TrainConfig(algo="ippo")).bytes_per_step() == 0.0
    ctde = Learner(env, TrainConfig(algo="ctde"))
    assert ctde.bytes_per_step() == env.n_agents * (env.obs_dim + 4) * 4.0
