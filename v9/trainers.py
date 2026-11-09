"""Training loops: centralized MAPPO (CTDE), independent PPO, FedAvg and FedProx.

All four share one rollout loop on the same environment; they differ only in who owns
which network and what is exchanged:

* ctde    - one actor per role (collectors share parameters) + one centralized critic
            on the global state. Every step, agents upload their transitions to a central
            learner; after every update the learner sends the actors back. (upper bound)
* ippo    - every agent has its own actor and a local critic on its own observation.
            Nothing is exchanged. (lower bound)
* fedavg  - like ippo, but every E local PPO updates the collectors upload their actor
            and critic weights to the aggregator, which averages and broadcasts them.
* fedprox - fedavg plus a proximal term mu/2 * ||w - w_global||^2 in each local loss.

The RIS-carrying UAV of the aerial preset has a different role and is never averaged
with the collectors.
"""
from __future__ import annotations

import copy
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .baselines import N_TEST, TEST_SEED_BASE, run_episode
from .config import ScenarioConfig
from .env import RISUAVEnv
from .ppo import Actor, Critic, PPOConfig, RunningScale, gae, n_params, ppo_update

VAL_SEED_BASE = 2_000_000
TRAIN_SEED_BASE = 10_000_000


@dataclass
class TrainConfig:
    algo: str = "ctde"            # ctde | ippo | fedavg | fedprox
    episodes: int = 2000
    E: int = 1                    # local PPO updates between aggregations (fedavg / fedprox)
    mu: float = 0.01              # FedProx proximal weight
    seed: int = 0
    eval_every: int = 100
    n_val: int = 20
    n_test: int = N_TEST
    hidden: int = 128
    critic_hidden_ctde: int = 256
    lam_update: bool = True       # Dinkelbach: move the energy price toward the achieved EE
    lam_rate: float = 0.1
    ppo: PPOConfig = field(default_factory=PPOConfig)


class Learner:
    """Owns the networks for one algorithm and exposes act / update / aggregate."""

    def __init__(self, env: RISUAVEnv, tc: TrainConfig):
        self.env = env
        self.tc = tc
        n, od = env.n_agents, env.obs_dim
        self.roles = [0 if a < env.K else 1 for a in range(n)]
        torch.manual_seed(tc.seed)
        pc = tc.ppo
        if tc.algo == "ctde":
            self.actors = {r: Actor(od, hidden=tc.hidden) for r in set(self.roles)}
            self.critic = Critic(env.state_dim, hidden=tc.critic_hidden_ctde)
            self.opt_a = {r: torch.optim.Adam(a.parameters(), lr=pc.lr_actor) for r, a in self.actors.items()}
            self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=pc.lr_critic)
        else:
            base_a, base_c = Actor(od, hidden=tc.hidden), Critic(od, hidden=tc.hidden)
            # Common initialisation (the server broadcasts the initial model).
            self.actors = {a: copy.deepcopy(base_a) for a in range(n)}
            self.critics = {a: copy.deepcopy(base_c) for a in range(n)}
            self.opt_a = {a: torch.optim.Adam(self.actors[a].parameters(), lr=pc.lr_actor) for a in range(n)}
            self.opt_c = {a: torch.optim.Adam(self.critics[a].parameters(), lr=pc.lr_critic) for a in range(n)}
            self.clients = [a for a in range(n) if self.roles[a] == 0]
            self.global_w = self._client_weights(self.clients[0])
        self.comm_bytes = 0.0
        self.n_updates = 0

    # ---------------------------------------------------------------- acting
    def actor_of(self, a: int) -> Actor:
        return self.actors[self.roles[a]] if self.tc.algo == "ctde" else self.actors[a]

    def act(self, obs: np.ndarray, deterministic: bool = False):
        acts, us, lps = [], [], []
        for a in range(self.env.n_agents):
            x, u, lp = self.actor_of(a).act(obs[a], deterministic)
            acts.append(x), us.append(u), lps.append(lp)
        return np.array(acts), np.array(us), np.array(lps)

    def values(self, obs: np.ndarray, state: np.ndarray) -> np.ndarray:
        if self.tc.algo == "ctde":
            return np.full(self.env.n_agents, float(self.critic.value(state[None])[0]))
        return np.array([float(self.critics[a].value(obs[a][None])[0]) for a in range(self.env.n_agents)])

    def policy(self, env, obs):
        return self.act(obs, deterministic=True)[0]

    def eval_view(self) -> "Learner":
        """The model that is actually deployed. For FedAvg/FedProx this is the federated
        (averaged) model, for FedPer the averaged shared layers with each client's own
        heads -- not the client copies that drifted since the last aggregation."""
        if self.tc.algo not in ("fedavg", "fedprox", "fedper"):
            return self
        view = copy.copy(self)
        view.actors = {a: copy.deepcopy(m) for a, m in self.actors.items()}
        sds = [self.actors[a].state_dict() for a in self.clients]
        avg = {k: torch.stack([sd[k] for sd in sds]).mean(0) for k in sds[0]
               if not (self.tc.algo == "fedper" and self._is_head("actor." + k))}
        for a in self.clients:
            view.actors[a].load_state_dict(avg, strict=False)
        return view

    def set_lr_frac(self, frac: float):
        """Scale the learning rates to frac x their initial values (linear annealing)."""
        pc = self.tc.ppo
        opts_a = self.opt_a.values()
        opts_c = [self.opt_c] if self.tc.algo == "ctde" else list(self.opt_c.values())
        for opt, lr in [(o, pc.lr_actor) for o in opts_a] + [(o, pc.lr_critic) for o in opts_c]:
            for g in opt.param_groups:
                g["lr"] = lr * frac

    # ---------------------------------------------------------------- communication
    def bytes_per_step(self) -> float:
        if self.tc.algo != "ctde":
            return 0.0
        # each agent uploads (obs, pre-squash action, log-prob, reward) as float32
        return self.env.n_agents * (self.env.obs_dim + 2 + 1 + 1) * 4.0

    def bytes_per_update(self) -> float:
        if self.tc.algo != "ctde":
            return 0.0
        # learner sends the updated actor to every agent
        return sum(n_params(self.actor_of(a)) for a in range(self.env.n_agents)) * 4.0

    # ---------------------------------------------------------------- federated helpers
    def _client_weights(self, a: int) -> dict:
        w = {"actor." + k: v.detach().clone() for k, v in self.actors[a].state_dict().items()}
        w.update({"critic." + k: v.detach().clone() for k, v in self.critics[a].state_dict().items()})
        return w

    @staticmethod
    def _is_head(key: str) -> bool:
        """Output layers kept local by personalized FL (FedPer): actor mean head, log-std, value head."""
        return key.startswith(("actor.mu.4.", "actor.log_std", "critic.v.4."))

    def aggregate(self):
        """FedAvg over the collector clients (equal sample counts -> equal weights).

        fedper averages only the shared hidden layers; each client keeps its own output heads.
        """
        ws = [self._client_weights(a) for a in self.clients]
        keys = [k for k in ws[0] if not (self.tc.algo == "fedper" and self._is_head(k))]
        avg = {k: torch.stack([w[k] for w in ws]).mean(0) for k in keys}
        for a in self.clients:
            self.actors[a].load_state_dict({k[6:]: v for k, v in avg.items() if k.startswith("actor.")},
                                           strict=False)
            self.critics[a].load_state_dict({k[7:]: v for k, v in avg.items() if k.startswith("critic.")},
                                            strict=False)
        self.global_w = avg
        per_client = sum(v.numel() for v in avg.values()) * 4.0
        self.comm_bytes += 2.0 * per_client * len(self.clients)     # upload + download

    # ---------------------------------------------------------------- update
    def update(self, traj: dict) -> dict:
        tc, pc = self.tc, self.tc.ppo
        n = self.env.n_agents
        T = len(traj["rew"])
        rew, done = np.array(traj["rew"], np.float32), np.array(traj["done"], np.float32)
        obs = np.array(traj["obs"], np.float32)       # (T, n, od)
        us = np.array(traj["u"], np.float32)          # (T, n, 2)
        lps = np.array(traj["logp"], np.float32)      # (T, n)
        vals = np.array(traj["val"], np.float32)      # (T, n)
        states = np.array(traj["state"], np.float32)  # (T, sd)
        mask = np.array(traj["mask"], np.float32)     # (T, n): 0 where the autopilot overrode
        stats = {}
        if tc.algo == "ctde":
            adv, ret = gae(rew, vals[:, 0], done, 0.0, pc.gamma, pc.lam)
            for r, actor in self.actors.items():
                agents = [a for a in range(n) if self.roles[a] == r]
                b = {
                    "obs": torch.as_tensor(obs[:, agents].reshape(-1, obs.shape[-1])),
                    "u": torch.as_tensor(us[:, agents].reshape(-1, 2)),
                    "logp": torch.as_tensor(lps[:, agents].reshape(-1)),
                    "adv": torch.as_tensor(np.repeat(adv, len(agents))),
                    "ret": torch.as_tensor(np.repeat(ret, len(agents))),
                    "v_old": torch.as_tensor(np.repeat(vals[:, 0], len(agents))),
                    "cin": torch.as_tensor(np.repeat(states, len(agents), axis=0)),
                    "mask": torch.as_tensor(mask[:, agents].reshape(-1)),
                }
                stats[f"role{r}"] = ppo_update(actor, self.critic, self.opt_a[r], self.opt_c, b, pc,
                                               update_critic=(r == 0))
            self.comm_bytes += self.bytes_per_update()
        else:
            prox = self.global_w if tc.algo == "fedprox" else None
            for a in range(n):
                adv, ret = gae(rew, vals[:, a], done, 0.0, pc.gamma, pc.lam)
                b = {
                    "obs": torch.as_tensor(obs[:, a]), "u": torch.as_tensor(us[:, a]),
                    "logp": torch.as_tensor(lps[:, a]), "adv": torch.as_tensor(adv),
                    "ret": torch.as_tensor(ret), "v_old": torch.as_tensor(vals[:, a]),
                    "cin": torch.as_tensor(obs[:, a]), "mask": torch.as_tensor(mask[:, a]),
                }
                use_prox = prox is not None and self.roles[a] == 0
                stats[f"agent{a}"] = ppo_update(self.actors[a], self.critics[a], self.opt_a[a], self.opt_c[a],
                                                b, pc, prox=prox if use_prox else None, mu=tc.mu)
        self.n_updates += 1
        if tc.algo in ("fedavg", "fedprox", "fedper") and self.n_updates % tc.E == 0:
            self.aggregate()
        return stats

    # ---------------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        d = {"actors": {str(k): v.state_dict() for k, v in self.actors.items()}}
        if self.tc.algo == "ctde":
            d["critic"] = self.critic.state_dict()
        else:
            d["critics"] = {str(k): v.state_dict() for k, v in self.critics.items()}
        return d

    def load_state_dict(self, d: dict):
        for k, v in d["actors"].items():
            self.actors[int(k)].load_state_dict(v)


def evaluate_learner(learner: Learner, cfg: ScenarioConfig, n: int, seed_base: int, lam: float) -> dict:
    env = RISUAVEnv(cfg)
    env.lam = lam
    rows = [run_episode(env, learner.policy, seed_base + j) for j in range(n)]
    out = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    out |= {k + "_std": float(np.nanstd([r[k] for r in rows], ddof=1)) for k in ("ee", "delivered_frac")}
    out["ee_per_episode"] = [float(r["ee"]) for r in rows]
    return out


def train(cfg: ScenarioConfig, tc: TrainConfig, out_dir: Path, verbose: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)   # small nets: one thread per run, several runs in parallel
    np.random.seed(tc.seed)
    env = RISUAVEnv(cfg)
    learner = Learner(env, tc)
    scale = RunningScale(tc.ppo.gamma)
    log_rows, eval_rows = [], []
    best_key, best_state = None, None
    traj = {k: [] for k in ("obs", "state", "u", "logp", "val", "rew", "done", "mask")}
    recent = []
    t0 = time.time()
    for ep in range(1, tc.episodes + 1):
        obs, state = env.reset(seed=TRAIN_SEED_BASE * (tc.seed + 1) + ep)
        done = False
        while not done:
            act, u, lp = learner.act(obs)
            val = learner.values(obs, state)
            nobs, nstate, r, done, info = env.step(act)
            learner.comm_bytes += learner.bytes_per_step()
            traj["obs"].append(obs), traj["state"].append(state), traj["u"].append(u)
            traj["logp"].append(lp), traj["val"].append(val)
            traj["rew"].append(scale(r, done)), traj["done"].append(float(done))
            traj["mask"].append(1.0 - info["overridden"].astype(np.float32))
            obs, state = nobs, nstate
        m = info["episode"]
        recent.append(m)
        if ep % tc.ppo.episodes_per_update == 0:
            if tc.ppo.lr_anneal:
                learner.set_lr_frac(max(0.1, 1.0 - (ep - 1) / tc.episodes))
            stats = learner.update(traj)
            traj = {k: [] for k in traj}
            row = {"episode": ep, "time_s": time.time() - t0, "lam": env.lam,
                   "comm_MB": learner.comm_bytes / 1e6}
            for k in ("ee", "delivered_frac", "reached_final", "collisions", "rth_frac"):
                row[k] = float(np.mean([x[k] for x in recent]))
            first = next(iter(stats.values()))
            row.update({k: first[k] for k in ("entropy", "approx_kl", "clipfrac")})
            log_rows.append(row)
            if tc.lam_update:
                env.lam = (1 - tc.lam_rate) * env.lam + tc.lam_rate * row["ee"]
            recent = []
        if ep % tc.eval_every == 0 or ep == tc.episodes:
            view = learner.eval_view()
            ev = evaluate_learner(view, cfg, tc.n_val, VAL_SEED_BASE, env.lam)
            ev.pop("ee_per_episode", None)
            ev.update({"episode": ep, "comm_MB": learner.comm_bytes / 1e6, "lam": env.lam})
            eval_rows.append(ev)
            key = (ev["reached_final"] >= 0.9, ev["ee"])
            if best_key is None or key > best_key:
                best_key, best_state = key, copy.deepcopy(view.state_dict())
                best_state["episode"] = ep
            if verbose:
                print(f"  [{tc.algo} s{tc.seed}] ep {ep:5d}  val EE {ev['ee']:7.0f}  deliv {ev['delivered_frac']:.2f}"
                      f"  final {ev['reached_final']:.2f}  rth {ev['rth_frac']:.2f}  coll {ev['collisions']:.1f}"
                      f"  comm {ev['comm_MB']:8.1f} MB"
                      f"  ({time.time() - t0:5.0f}s)", flush=True)
    # final test on the same held-out seeds the baselines use
    learner.load_state_dict(best_state)
    test = evaluate_learner(learner, cfg, tc.n_test, TEST_SEED_BASE, env.lam)
    test.update({"best_episode": best_state["episode"], "comm_MB": learner.comm_bytes / 1e6,
                 "train_time_s": time.time() - t0, "algo": tc.algo, "seed": tc.seed, "E": tc.E,
                 "mu": tc.mu, "preset": cfg.name})
    torch.save(best_state, out_dir / "best.pt")
    _write(out_dir / "train_log.csv", log_rows)
    _write(out_dir / "eval_log.csv", eval_rows)
    (out_dir / "test.json").write_text(json.dumps(test, indent=2))
    (out_dir / "config.json").write_text(json.dumps({"scenario": asdict(cfg), "train": asdict(tc)},
                                                    indent=2, default=str))
    if verbose:
        print(f"  [{tc.algo} s{tc.seed}] TEST EE {test['ee']:.0f}  delivered {test['delivered_frac']:.2f}"
              f"  final {test['reached_final']:.2f}  (best ep {test['best_episode']})")
    return test


def _write(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
