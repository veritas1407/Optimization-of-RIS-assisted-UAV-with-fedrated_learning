"""PPO building blocks for v9 (fixes audit finding F7).

* The policy is a tanh-squashed diagonal Gaussian. The buffer stores the pre-squash
  sample u; the action sent to the environment is tanh(u). Because the Jacobian term
  log(1 - tanh(u)^2) depends only on u, it cancels in the PPO ratio, so the ratio
  computed from the base Gaussian over stored u is exact. No post-hoc clamping.
* Log-probabilities are SUMMED over action dimensions.
* Value clipping uses the value predicted at collection time (stored), not a
  reconstruction from normalized advantages.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(inp: int, hidden: int, out: int, out_gain: float = 1.0) -> nn.Sequential:
    net = nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                        nn.Linear(hidden, out))
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)
    nn.init.orthogonal_(net[-1].weight, gain=out_gain)
    return net


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 2, hidden: int = 128, log_std0: float = -0.5):
        super().__init__()
        self.mu = mlp(obs_dim, hidden, act_dim, out_gain=0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std0))

    def dist(self, obs: torch.Tensor) -> Normal:
        std = torch.exp(self.log_std.clamp(-4.0, 1.0)).expand_as(self.mu(obs))
        return Normal(self.mu(obs), std)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        o = torch.as_tensor(obs, dtype=torch.float32)
        d = self.dist(o)
        u = d.mean if deterministic else d.sample()
        logp = d.log_prob(u).sum(-1)
        return torch.tanh(u).numpy(), u.numpy(), logp.numpy()


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.v = mlp(in_dim, hidden, 1, out_gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v(x).squeeze(-1)

    @torch.no_grad()
    def value(self, x: np.ndarray) -> np.ndarray:
        return self(torch.as_tensor(x, dtype=torch.float32)).numpy()


class RunningScale:
    """Scales rewards by the running std of the discounted return (standard PPO trick)."""

    def __init__(self, gamma: float):
        self.gamma = gamma
        self.ret = 0.0
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def __call__(self, r: float, done: bool) -> float:
        self.ret = self.ret * self.gamma + r
        self.n += 1
        d = self.ret - self.mean
        self.mean += d / self.n
        self.m2 += d * (self.ret - self.mean)
        if done:
            self.ret = 0.0
        std = np.sqrt(self.m2 / max(self.n - 1, 1)) if self.n > 1 else 1.0
        return r / max(std, 1e-3)


def gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_value: float,
        gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    g = 0.0
    for t in reversed(range(T)):
        nxt = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * nxt * nonterm - values[t]
        g = delta + gamma * lam * nonterm * g
        adv[t] = g
    return adv, adv + values


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 5
    minibatch: int = 256
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    ent_coef: float = 0.003
    vf_coef: float = 0.5
    max_grad: float = 0.5
    episodes_per_update: int = 8


def ppo_update(actor: Actor, critic: Critic, opt_a, opt_c, batch: dict, pc: PPOConfig,
               prox: dict | None = None, mu: float = 0.0, update_critic: bool = True) -> dict:
    """One PPO update. ``batch`` holds tensors obs, u, logp, adv, ret, v_old, cin.

    If ``prox`` (name -> tensor of global weights) and ``mu`` > 0 are given, a FedProx
    proximal term mu/2 * ||w - w_global||^2 is added to the actor and critic losses.
    """
    n = batch["obs"].shape[0]
    adv = batch["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "clipfrac": 0.0, "approx_kl": 0.0}
    count = 0
    for _ in range(pc.epochs):
        perm = torch.randperm(n)
        for s in range(0, n, pc.minibatch):
            idx = perm[s:s + pc.minibatch]
            d = actor.dist(batch["obs"][idx])
            logp = d.log_prob(batch["u"][idx]).sum(-1)
            log_ratio = logp - batch["logp"][idx]
            ratio = torch.exp(log_ratio)
            a = adv[idx]
            surr = torch.min(ratio * a, ratio.clamp(1 - pc.clip, 1 + pc.clip) * a)
            if "mask" in batch:   # steps where the autopilot overrode the policy carry no policy gradient
                m = batch["mask"][idx]
                pi_loss = -(surr * m).sum() / m.sum().clamp(min=1.0)
            else:
                pi_loss = -surr.mean()
            ent = d.entropy().sum(-1).mean()
            v = critic(batch["cin"][idx])
            v_old = batch["v_old"][idx]
            v_clip = v_old + (v - v_old).clamp(-pc.clip * 10, pc.clip * 10)
            ret = batch["ret"][idx]
            v_loss = 0.5 * torch.max((v - ret) ** 2, (v_clip - ret) ** 2).mean()
            loss_a = pi_loss - pc.ent_coef * ent
            loss_c = pc.vf_coef * v_loss
            if prox is not None and mu > 0.0:
                loss_a = loss_a + 0.5 * mu * sum(((p - prox["actor." + k]) ** 2).sum()
                                                 for k, p in actor.named_parameters() if "actor." + k in prox)
                loss_c = loss_c + 0.5 * mu * sum(((p - prox["critic." + k]) ** 2).sum()
                                                 for k, p in critic.named_parameters() if "critic." + k in prox)
            opt_a.zero_grad()
            opt_c.zero_grad()
            (loss_a + loss_c if update_critic else loss_a).backward()
            nn.utils.clip_grad_norm_(actor.parameters(), pc.max_grad)
            opt_a.step()
            if update_critic:
                nn.utils.clip_grad_norm_(critic.parameters(), pc.max_grad)
                opt_c.step()
            with torch.no_grad():
                stats["pi_loss"] += pi_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += ent.item()
                stats["clipfrac"] += ((ratio - 1).abs() > pc.clip).float().mean().item()
                stats["approx_kl"] += ((ratio - 1) - log_ratio).mean().item()
            count += 1
    return {k: v / max(count, 1) for k, v in stats.items()}


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
