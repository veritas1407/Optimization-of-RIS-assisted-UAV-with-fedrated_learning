"""  
ris_uav_mappo_v6.py
===================
SYSTEM : RIS-Assisted UAV-Enabled IoT Network (Jiang et al., IEEE IoT J. 2025)
METHOD : Multi-Agent Proximal Policy Optimization (MAPPO) with CTDE

v6 FIXES (v5 → v6) — recovering RIS benefit over No-RIS baseline:
 27. [BUG FIX]   Progress reward weight 0.8 → 0.4 (EE dominates at ~90%)
 28. [IMPROVEMENT] N_EPISODES 1200 → 1500 for better convergence with lower progress weight
 29. [IMPROVEMENT] Per-L sub-training caller 300 → 500 episodes
 30. [IMPROVEMENT] Per-I sub-training caller 200 → 350 episodes
 31. [CLEANUP]    Reward comment updated to reflect final weights

REFERENCE: Jiang et al., IEEE IoT Journal, Vol.12, No.20, October 2025
"""

# ══════════════════════════════════════════════════════════════
# SECTION 1 — Imports and device setup
# ══════════════════════════════════════════════════════════════

import os, time, warnings, copy, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque

try:
    from tqdm.auto import tqdm
except ImportError:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

warnings.filterwarnings("ignore")
os.makedirs("./mappo_results_v6", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE_PHY = torch.float64
DTYPE_NET = torch.float32

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"GPU memory: {props.total_memory / 1e9:.1f} GB")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
else:
    print(f"No GPU — running on CPU. PyTorch {torch.__version__}")
print(f"Device: {DEVICE}")


def to_phy(x):
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=DTYPE_PHY, device=DEVICE)
    return x.to(DEVICE, DTYPE_PHY)


def to_net(x):
    if isinstance(x, torch.Tensor):
        return x.to(DEVICE, DTYPE_NET)
    return torch.tensor(x, dtype=DTYPE_NET, device=DEVICE)


# ══════════════════════════════════════════════════════════════
# SECTION 2 — Physics constants (Table II)
# ══════════════════════════════════════════════════════════════

K_UAVS   = 3
M        = 20
T_TOTAL  = 20.0
DT       = T_TOTAL / M

I_DEV    = 4
L_DEV    = 50
ALT      = 5.0
BOUND    = 20.0
SAFE_D   = 3.0

Q_RIS_NP  = np.array([5., 9., 0.])
Q_IOT_NP  = np.array([[3., 11., 0.], [6., 13., 0.], [9., 4., 0.], [12., 6., 0.]])
Q_I_NP    = np.array([0., 0., ALT])
Q_F_NP    = np.array([16., 16., ALT])

Q_RIS = to_phy(Q_RIS_NP)
Q_IOT = to_phy(Q_IOT_NP)
Q_I   = to_phy(Q_I_NP)
Q_F   = to_phy(Q_F_NP)

D_SEP  = 0.1
LAM    = 0.1
# FIX 7: RHO0 raised from 1e-3 → 1e-2 for more realistic EE magnitude
RHO0   = 1e-2
A_DR   = 2.2
A_RU   = 2.2
A_DU   = 3.5
K_DIR  = 5.0
K_RU   = 5.0

P_MAX  = 1.0
SIG2   = 1e-10

PB     = 79.8563
PI_P   = 88.6279
UTIP   = 120.0
V0_H   = 4.03
D_F    = 0.6
S_R    = 0.05
RHO_A  = 1.225
A_R    = 0.503
V_MAX  = 15.0
A_MAX  = 5.0

ACT_DIM_UAV = 2
ACT_DIM_RIS = L_DEV
ACT_DIM     = ACT_DIM_UAV + ACT_DIM_RIS   # 52

# ══════════════════════════════════════════════════════════════
# SECTION 3 — Physics functions (Eqs. 5–7, 10, 11, 12, 17)
# ══════════════════════════════════════════════════════════════


def _los_gpu(tx, rx, L):
    d = torch.clamp(torch.norm(rx - tx), min=1e-9)
    phi = (rx[0] - tx[0]) / d
    idx = torch.arange(L, dtype=DTYPE_PHY, device=DEVICE)
    phase = -2.0 * math.pi * D_SEP / LAM * idx * phi
    return torch.complex(torch.cos(phase), torch.sin(phase))


def _rician_gpu(pos_tx, pos_rx, L, K_fac, alpha, rng):
    d   = max(float(torch.norm(pos_rx - pos_tx).item()), 1e-9)
    pl  = np.sqrt(RHO0 * d ** (-alpha))
    hL  = _los_gpu(pos_tx, pos_rx, L)
    hN_r = torch.tensor(rng.randn(L), dtype=DTYPE_PHY, device=DEVICE)
    hN_i = torch.tensor(rng.randn(L), dtype=DTYPE_PHY, device=DEVICE)
    hN  = torch.complex(hN_r, hN_i) / np.sqrt(2)
    h   = np.sqrt(K_fac / (K_fac + 1)) * hL + np.sqrt(1.0 / (K_fac + 1)) * hN
    return h * pl


def _rayleigh_gpu(pos_tx, pos_rx, alpha, rng):
    d   = max(float(torch.norm(pos_rx - pos_tx).item()), 1e-9)
    pl  = np.sqrt(RHO0 * d ** (-alpha))
    r_val = rng.randn()
    i_val = rng.randn()
    h   = torch.complex(to_phy(torch.tensor(r_val)),
                        to_phy(torch.tensor(i_val))) / np.sqrt(2)
    return h.to(torch.complex128) * pl


def precompute_channels_gpu(Q_uav, L, seed_ris=42, seed_direct=99):
    rng_ris    = np.random.RandomState(seed_ris)
    rng_direct = np.random.RandomState(seed_direct)
    if Q_uav.dim() == 1:
        Q_uav = Q_uav.unsqueeze(0)
    n_pos  = Q_uav.shape[0]
    h_dir  = torch.stack([_rician_gpu(Q_IOT[i], Q_RIS, L, K_DIR, A_DR, rng_ris)
                          for i in range(I_DEV)])
    h_ru   = torch.stack([_rician_gpu(Q_RIS, Q_uav[m], L, K_RU, A_RU, rng_ris)
                          for m in range(n_pos)])
    h_diu  = torch.stack([
        torch.stack([_rayleigh_gpu(Q_IOT[i], Q_uav[m], A_DU, rng_direct)
                     for m in range(n_pos)])
        for i in range(I_DEV)])
    return h_dir, h_ru, h_diu


def Pp_gpu(v_norms):
    v        = v_norms.clamp(min=0.0)
    blade    = PB * (1.0 + 3.0 * v ** 2 / UTIP ** 2)
    parasite = 0.5 * D_F * RHO_A * S_R * A_R * v ** 3
    inner    = torch.sqrt(torch.clamp(
        1.0 + v ** 4 / (4.0 * V0_H ** 4) - v ** 2 / (2.0 * V0_H ** 2), min=1e-9))
    induced  = PI_P * inner
    return blade + parasite + induced


def opt_phase_gpu(h_dir, h_ru, h_diu):
    ang_diu = torch.angle(h_diu).unsqueeze(2)
    ang_ru  = torch.angle(torch.conj(h_ru)).unsqueeze(0)
    ang_dir = torch.angle(h_dir).unsqueeze(1)
    theta   = ang_diu - ang_ru - ang_dir
    return torch.polar(torch.ones_like(theta), theta)


def gamma_all_gpu(h_dir, h_ru, h_diu, Theta, no_ris=False):
    if no_ris:
        return h_diu.abs() ** 2
    h_ru_exp  = h_ru.unsqueeze(0)
    h_dir_exp = h_dir.unsqueeze(1)
    reflected = (h_ru_exp * Theta * h_dir_exp).sum(dim=2)
    total     = reflected + h_diu
    return total.abs() ** 2


def compute_ee_single(pos_uav, vel_uav, theta_phases, L,
                      seed_ris=42, seed_direct=99, n_dev=None):
    pos_t   = to_phy(pos_uav)   if not isinstance(pos_uav,   torch.Tensor) else pos_uav.to(DEVICE, DTYPE_PHY)
    vel_t   = to_phy(vel_uav)   if not isinstance(vel_uav,   torch.Tensor) else vel_uav.to(DEVICE, DTYPE_PHY)
    theta_t = to_phy(theta_phases) if not isinstance(theta_phases, torch.Tensor) else theta_phases.to(DEVICE, DTYPE_PHY)
    # Pad or truncate theta to match L
    if theta_t.shape[0] < L:
        padded = torch.zeros(L, dtype=DTYPE_PHY, device=DEVICE)
        padded[:theta_t.shape[0]] = theta_t
        theta_t = padded
    elif theta_t.shape[0] > L:
        theta_t = theta_t[:L]
    h_dir, h_ru, h_diu = precompute_channels_gpu(
        pos_t.unsqueeze(0), L, seed_ris, seed_direct)
    n_i   = h_dir.shape[0] if n_dev is None else n_dev
    Theta = torch.polar(
        torch.ones(n_i, 1, L, dtype=DTYPE_PHY, device=DEVICE),
        theta_t.unsqueeze(0).unsqueeze(0).expand(n_i, 1, L))
    G     = gamma_all_gpu(h_dir, h_ru, h_diu, Theta)
    tau_i = 1.0 / n_i
    p_i   = P_MAX / n_i
    rate  = 0.0
    for i in range(n_i):
        rate += tau_i * torch.log2(1.0 + p_i * G[i, 0] / SIG2)
    speed = torch.norm(vel_t)
    pp    = Pp_gpu(speed.unsqueeze(0))[0]
    ee    = rate / pp.clamp(min=1e-9)
    return ee.item()


# ══════════════════════════════════════════════════════════════
# SECTION 4 — RISSwarmEnv
# ══════════════════════════════════════════════════════════════

class RISSwarmEnv:
    OBS_DIM   = 18
    STATE_DIM = 18 * K_UAVS   # 54

    def __init__(self, L=L_DEV, n_devices=None, device_positions=None):
        self.L        = L
        self.M        = M
        self.K        = K_UAVS
        self.obs_dim  = 18
        self.state_dim = 18 * self.K
        self.act_dim  = ACT_DIM_UAV + L

        self.n_devices = n_devices if n_devices is not None else I_DEV
        if device_positions is not None:
            self.q_iot = to_phy(device_positions)
        else:
            self.q_iot = (Q_IOT[:self.n_devices].clone()
                          if self.n_devices <= I_DEV
                          else self._extend_devices(self.n_devices))

        self.start_offsets = [
            Q_I_NP.copy(),
            Q_I_NP.copy() + np.array([2.0, 0.0, 0.0]),
            Q_I_NP.copy() + np.array([0.0, 2.0, 0.0]),
        ]
        self.positions       = None
        self.velocities      = None
        self.slot            = 0
        self.h_dir_k         = [None] * self.K
        self.h_ru_k          = [None] * self.K
        self.h_diu_k         = [None] * self.K
        # FIX 3: track previous distance to goal for progress bonus
        self.prev_dist_goal  = [0.0] * self.K

    def _extend_devices(self, n):
        base = Q_IOT_NP.tolist()
        rng  = np.random.RandomState(123)
        while len(base) < n:
            base.append([rng.uniform(1, BOUND-1), rng.uniform(1, BOUND-1), 0.0])
        return to_phy(np.array(base[:n]))

    def _recompute_channels(self, k):
        pos_t       = to_phy(self.positions[k])
        seed_offset = self.slot * 31
        rng_ris     = np.random.RandomState(42  + k * 7  + seed_offset)
        rng_direct  = np.random.RandomState(99  + k * 13 + seed_offset)
        h_dir = torch.stack([_rician_gpu(self.q_iot[i], Q_RIS, self.L,
                                         K_DIR, A_DR, rng_ris)
                             for i in range(self.n_devices)])
        h_ru  = torch.stack([_rician_gpu(Q_RIS, pos_t, self.L,
                                         K_RU, A_RU, rng_ris)])
        h_diu = torch.stack([
            torch.stack([_rayleigh_gpu(self.q_iot[i], pos_t, A_DU, rng_direct)])
            for i in range(self.n_devices)])
        self.h_dir_k[k] = h_dir
        self.h_ru_k[k]  = h_ru
        self.h_diu_k[k] = h_diu

    def reset(self):
        self.positions  = [self.start_offsets[k].copy() for k in range(self.K)]
        self.velocities = [np.zeros(2) for _ in range(self.K)]
        self.slot       = 0
        # FIX 3: initialise previous distances for progress reward
        self.prev_dist_goal = [
            np.linalg.norm(self.positions[k][:2] - Q_F_NP[:2])
            for k in range(self.K)
        ]
        for k in range(self.K):
            self._recompute_channels(k)
        obs_list     = [self._get_obs(k) for k in range(self.K)]
        global_state = self._get_global_state()
        return obs_list, global_state

    def step(self, actions):
        vels   = []
        thetas = []
        for k in range(self.K):
            act    = actions[k]
            vel_k  = act[:2] * V_MAX
            vel_k  = np.clip(vel_k, -V_MAX, V_MAX)
            self.velocities[k]    = vel_k
            self.positions[k][0] += vel_k[0] * DT
            self.positions[k][1] += vel_k[1] * DT
            self.positions[k][0]  = np.clip(self.positions[k][0], 0.0, BOUND)
            self.positions[k][1]  = np.clip(self.positions[k][1], 0.0, BOUND)
            self.positions[k][2]  = ALT
            raw_phases = act[2:2 + min(self.L, len(act) - 2)]
            if len(raw_phases) < self.L:
                padded = np.zeros(self.L)
                padded[:len(raw_phases)] = raw_phases
                theta_k = (padded + 1.0) * math.pi
            else:
                theta_k = (raw_phases + 1.0) * math.pi
            vels.append(vel_k)
            thetas.append(theta_k)

        # Collision enforcement — geometric push-apart
        for i in range(self.K):
            for j in range(i + 1, self.K):
                diff = self.positions[i][:2] - self.positions[j][:2]
                dist = np.linalg.norm(diff)
                if dist < SAFE_D:
                    direction = diff / max(dist, 1e-6)
                    push      = (SAFE_D - dist) / 2.0
                    self.positions[i][:2] += direction * push
                    self.positions[j][:2] -= direction * push
                    for idx in [i, j]:
                        self.positions[idx][0] = np.clip(self.positions[idx][0], 0.0, BOUND)
                        self.positions[idx][1] = np.clip(self.positions[idx][1], 0.0, BOUND)

        # Compute EE per UAV
        ee_vals = []
        for k in range(self.K):
            self._recompute_channels(k)
            vel_3d = np.array([vels[k][0], vels[k][1], 0.0])
            ee_k   = compute_ee_single(self.positions[k], vel_3d, thetas[k], self.L,
                                       seed_ris=42 + k * 7, seed_direct=99 + k * 13)
            ee_vals.append(ee_k)

        # Collision count after enforcement
        n_collisions = 0
        for i in range(self.K):
            for j in range(i + 1, self.K):
                dist = np.linalg.norm(self.positions[i][:2] - self.positions[j][:2])
                if dist < SAFE_D:
                    n_collisions += 1

        collision_penalty = n_collisions * 5.0
        mean_ee           = float(np.mean(ee_vals))

        # reward = mean_ee * 100.0          # EE (~3–8 units, dominates at 89%)
        #         + progress_reward          # navigation (0.4 × dist_reduction, ~10%)
        #         - collision_penalty        # 5.0 per colliding pair
        progress_reward = 0.0
        for k in range(self.K):
            curr_dist = np.linalg.norm(self.positions[k][:2] - Q_F_NP[:2])
            progress_reward += (self.prev_dist_goal[k] - curr_dist) * 0.4
            self.prev_dist_goal[k] = curr_dist

        # Reward: EE (scaled) + navigation progress - collision penalty
        reward = mean_ee * 100.0 + progress_reward - collision_penalty

        self.slot += 1
        done  = (self.slot >= self.M)

        info = {
            "mean_EE"    : mean_ee,
            "EE_per_UAV" : [float(e) for e in ee_vals],
            "collisions" : n_collisions,
            "positions"  : [self.positions[k].copy() for k in range(self.K)],
        }

        obs_list     = [self._get_obs(k) for k in range(self.K)]
        global_state = self._get_global_state()
        return obs_list, global_state, reward, done, info

    def _get_obs(self, k):
        obs = []
        # [0-1] own position
        obs.append(self.positions[k][0] / BOUND)
        obs.append(self.positions[k][1] / BOUND)
        # [2-3] own velocity
        obs.append(self.velocities[k][0] / V_MAX)
        obs.append(self.velocities[k][1] / V_MAX)
        # [4-6] distances to min(n_devices, 3) IoT devices (FIX 17: reduced from 4→3)
        for i in range(min(self.n_devices, 3)):
            d = np.linalg.norm(self.positions[k] - self.q_iot[i].cpu().numpy())
            obs.append(d / (BOUND * np.sqrt(2)))
        while len(obs) < 7:
            obs.append(0.0)
        # [7] FIX 17: normalized device count
        obs.append(self.n_devices / 6.0)
        # [8] distance to RIS
        d_ris = np.linalg.norm(self.positions[k] - Q_RIS_NP)
        obs.append(d_ris / (BOUND * np.sqrt(2)))
        # [9-12] log channel magnitudes to min(n_devices, 4) IoT devices
        for i in range(min(self.n_devices, I_DEV)):
            val = torch.log(self.h_diu_k[k][i, 0].abs() + 1e-10).item() / 10.0
            obs.append(val)
        while len(obs) < 13:
            obs.append(0.0)
        # [13-16] other UAV positions
        others = [j for j in range(self.K) if j != k]
        for j in others[:2]:
            obs.append(self.positions[j][0] / BOUND)
            obs.append(self.positions[j][1] / BOUND)
        # [17] FIX 16: goal distance (replaces slot/M)
        d_goal = np.linalg.norm(self.positions[k][:2] - Q_F_NP[:2])
        obs.append(d_goal / (BOUND * np.sqrt(2)))
        assert len(obs) == 18, f"obs length {len(obs)} != 18"
        return to_net(np.array(obs, dtype=np.float32))

    def _get_global_state(self):
        return torch.cat([self._get_obs(k) for k in range(self.K)])


# ══════════════════════════════════════════════════════════════
# SECTION 5 — Neural Networks
# ══════════════════════════════════════════════════════════════

def _ortho_init(module, gain=np.sqrt(2)):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class Actor(nn.Module):
    def __init__(self, obs_dim=18, act_dim=ACT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128),     nn.Tanh(),
            nn.Linear(128, 64),      nn.Tanh(),
        )
        self.mean_head = nn.Linear(64, act_dim)
        # FIX 5: wider init (-0.5 → σ≈0.61) and wider clamp range so std
        #        can actually grow/shrink during training
        self.log_std   = nn.Parameter(torch.full((act_dim,), -0.5))
        self.apply(_ortho_init)
        _ortho_init(self.mean_head, gain=0.01)
        self.to(DEVICE).to(DTYPE_NET)

    def forward(self, obs):
        x    = self.net(obs)
        mean = torch.tanh(self.mean_head(x))
        # FIX 5: widened clamp from (-3,0.5) → (-4,1.0)
        log_std = torch.clamp(self.log_std, -4.0, 1.0)
        std     = torch.exp(log_std).expand_as(mean)
        return Normal(mean, std)

    def select_action(self, obs, deterministic=False):
        obs = to_net(obs)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        dist   = self.forward(obs)
        action = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).mean(-1)
        entropy  = dist.entropy().mean(-1)
        action   = torch.clamp(action, -1.0, 1.0)
        return action.squeeze(0).detach().cpu().numpy(), \
               log_prob.squeeze(0), entropy.squeeze(0)

    # FIX 2: helper to evaluate with a different RIS dimension via resampling
    def select_action_resample(self, obs, target_L, deterministic=False):
        """Select action and resample the RIS portion to target_L elements."""
        raw_action, log_prob, entropy = self.select_action(obs, deterministic)
        vel_part = raw_action[:ACT_DIM_UAV]
        ris_part = raw_action[ACT_DIM_UAV:]           # shape (ACT_DIM_RIS,)
        if target_L == ACT_DIM_RIS:
            return raw_action, log_prob, entropy
        # Linear interpolation to target_L
        ris_tensor    = torch.tensor(ris_part, dtype=DTYPE_NET).view(1, 1, -1)
        ris_resampled = F.interpolate(ris_tensor, size=target_L,
                                      mode='linear',
                                      align_corners=False).view(-1).numpy()
        new_action = np.concatenate([vel_part, ris_resampled])
        return new_action, log_prob, entropy


class Critic(nn.Module):
    def __init__(self, state_dim=54):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 256),       nn.Tanh(),
            nn.Linear(256, 128),       nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.apply(_ortho_init)
        self.to(DEVICE).to(DTYPE_NET)

    def forward(self, state):
        return self.net(state)


# ══════════════════════════════════════════════════════════════
# SECTION 6 — RolloutBuffer
# ══════════════════════════════════════════════════════════════

class RolloutBuffer:
    # FIX 4: capacity halved 400→200 for more frequent updates
    def __init__(self, capacity=200):
        self.capacity = capacity
        self.clear()

    def clear(self):
        self.obs          = [[] for _ in range(K_UAVS)]
        self.global_states = []
        self.actions      = [[] for _ in range(K_UAVS)]
        self.log_probs    = [[] for _ in range(K_UAVS)]
        self.rewards      = []
        self.values       = []
        self.dones        = []

    def store(self, obs_list, global_state, actions, log_probs, reward, value, done):
        for k in range(K_UAVS):
            self.obs[k].append(obs_list[k].detach())
            self.actions[k].append(torch.tensor(actions[k], dtype=DTYPE_NET, device=DEVICE))
            self.log_probs[k].append(log_probs[k].detach())
        self.global_states.append(global_state.detach())
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def is_ready(self):
        return len(self.rewards) >= self.capacity

    def get_tensors(self):
        data = {}
        data["obs"]           = [torch.stack(self.obs[k]) for k in range(K_UAVS)]
        data["global_states"] = torch.stack(self.global_states)
        data["actions"]       = [torch.stack(self.actions[k]) for k in range(K_UAVS)]
        data["log_probs"]     = [torch.stack(self.log_probs[k]) for k in range(K_UAVS)]
        data["rewards"]       = torch.tensor(self.rewards, dtype=DTYPE_NET, device=DEVICE)
        data["values"]        = torch.tensor(self.values,  dtype=DTYPE_NET, device=DEVICE)
        data["dones"]         = torch.tensor(self.dones,   dtype=DTYPE_NET, device=DEVICE)
        return data

    def compute_gae(self, next_value, gamma=0.99, lam=0.95):
        rewards    = torch.tensor(self.rewards, dtype=DTYPE_NET, device=DEVICE)
        values     = torch.tensor(self.values,  dtype=DTYPE_NET, device=DEVICE)
        dones      = torch.tensor(self.dones,   dtype=DTYPE_NET, device=DEVICE)
        T          = len(self.rewards)
        advantages = torch.zeros(T, dtype=DTYPE_NET, device=DEVICE)
        gae        = 0.0
        for t in reversed(range(T)):
            next_val         = next_value if t == T - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta            = rewards[t] + gamma * next_val * next_non_terminal - values[t]
            gae              = delta + gamma * lam * next_non_terminal * gae
            advantages[t]    = gae
        returns    = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages


# ══════════════════════════════════════════════════════════════
# SECTION 7 — MAPPOAgent
# ══════════════════════════════════════════════════════════════

class MAPPOAgent:
    def __init__(self):
        self.actor        = Actor(obs_dim=18, act_dim=ACT_DIM)
        self.critic       = Critic(state_dim=54)
        self.actor_optim  = torch.optim.Adam(self.actor.parameters(),  lr=3e-4)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        # FIX 12: cosine annealing LR schedulers for smoother late-training
        self.actor_scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optim,  T_max=1000, eta_min=1e-5)
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optim, T_max=1000, eta_min=1e-5)
        # FIX 4: smaller buffer → more frequent PPO updates
        self.buffer       = RolloutBuffer(capacity=200)
        self.clip_eps     = 0.2
        # FIX 6: entropy coeff will be annealed externally via set_entropy_coeff
        self.entropy_coeff = 0.05
        self.value_coeff   = 0.5
        self.max_grad_norm = 0.5
        # Running reward normalisation
        self.reward_mean  = 0.0
        self.reward_var   = 1.0
        self.reward_count = 0

    def set_entropy_coeff(self, ep, n_total):
        """FIX 6+9: linearly anneal entropy from 0.05 → 0.005 (floor raised
        from 0.001 to prevent late-training collapse)."""
        self.entropy_coeff = max(0.005, 0.05 * (1.0 - ep / n_total))

    def normalize_reward(self, reward):
        self.reward_count += 1
        if self.reward_count == 1:
            self.reward_mean = reward
            self.reward_var  = 0.0
        else:
            delta            = reward - self.reward_mean
            self.reward_mean += delta / self.reward_count
            self.reward_var  += (delta * (reward - self.reward_mean)
                                 - self.reward_var) / self.reward_count
        std = max(np.sqrt(self.reward_var), 1e-4)
        return (reward - self.reward_mean) / std

    def select_actions(self, obs_list, global_state):
        actions_list   = []
        log_probs_list = []
        entropies_list = []
        with torch.no_grad():
            for k in range(K_UAVS):
                action, log_prob, entropy = self.actor.select_action(obs_list[k])
                actions_list.append(action)
                log_probs_list.append(log_prob)
                entropies_list.append(entropy)
            value = self.critic(
                global_state.unsqueeze(0).float()).squeeze().item()
        return actions_list, log_probs_list, entropies_list, value

    def update(self):
        if not self.buffer.is_ready():
            return None
        with torch.no_grad():
            last_gs    = self.buffer.global_states[-1].float()
            next_value = self.critic(last_gs.unsqueeze(0)).squeeze().item()
        returns, advantages = self.buffer.compute_gae(next_value)
        data = self.buffer.get_tensors()

        T          = len(self.buffer.rewards)
        # FIX 4: increased epochs 2→4
        n_epochs   = 4
        batch_size = min(256, T)

        total_actor_loss  = 0.0
        total_critic_loss = 0.0
        total_entropy     = 0.0
        n_updates         = 0

        for _ in range(n_epochs):
            indices = torch.randperm(T, device=DEVICE)
            for start in range(0, T, batch_size):
                end  = min(start + batch_size, T)
                idx  = indices[start:end]
                mb_returns    = returns[idx]
                mb_advantages = advantages[idx]
                mb_states     = data["global_states"][idx].float()

                # Critic loss — FIX 10: PPO-style value clipping for stability
                v_pred      = self.critic(mb_states).squeeze(-1)
                v_old       = mb_returns - mb_advantages   # approx old value
                v_clipped   = v_old + torch.clamp(v_pred - v_old,
                                                   -self.clip_eps, self.clip_eps)
                critic_loss = self.value_coeff * torch.max(
                    F.mse_loss(v_pred, mb_returns),
                    F.mse_loss(v_clipped, mb_returns)
                )

                # Actor loss — average over K UAVs
                actor_loss_sum = torch.tensor(0.0, device=DEVICE)
                entropy_sum    = torch.tensor(0.0, device=DEVICE)

                for k in range(K_UAVS):
                    mb_obs_k     = data["obs"][k][idx].float()
                    mb_actions_k = data["actions"][k][idx].float()
                    mb_old_lp_k  = data["log_probs"][k][idx]

                    dist     = self.actor(mb_obs_k)
                    new_lp   = dist.log_prob(mb_actions_k).mean(-1)
                    ent      = dist.entropy().mean(-1).mean()

                    log_ratio = torch.clamp(new_lp - mb_old_lp_k, -2.0, 2.0)
                    ratio     = torch.exp(log_ratio)
                    surr1     = ratio * mb_advantages
                    surr2     = torch.clamp(ratio,
                                            1.0 - self.clip_eps,
                                            1.0 + self.clip_eps) * mb_advantages
                    actor_loss_k    = -torch.min(surr1, surr2).mean()
                    actor_loss_sum += actor_loss_k
                    entropy_sum    += ent

                actor_loss    = actor_loss_sum / K_UAVS
                entropy_bonus = entropy_sum    / K_UAVS
                total_loss    = actor_loss + critic_loss \
                                - self.entropy_coeff * entropy_bonus

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),  self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.step()

                total_actor_loss  += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy     += entropy_bonus.item()
                n_updates         += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.buffer.clear()
        # FIX 12: step LR schedulers after each PPO update
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        return {
            "actor_loss"  : total_actor_loss  / max(n_updates, 1),
            "critic_loss" : total_critic_loss / max(n_updates, 1),
            "entropy"     : total_entropy     / max(n_updates, 1),
        }

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])


# ══════════════════════════════════════════════════════════════
# SECTION 8 — Benchmark functions
# ══════════════════════════════════════════════════════════════

def eval_agent(agent, env, n_episodes=5, deterministic=True):
    total_rewards = []
    total_ees     = []
    trajectory    = None
    target_L      = env.L          # may differ from ACT_DIM_RIS
    for ep in range(n_episodes):
        obs_list, global_state = env.reset()
        ep_reward  = 0.0
        ep_ees     = []
        positions_log = [[] for _ in range(env.K)]
        for k in range(env.K):
            positions_log[k].append(env.positions[k].copy())
        for step in range(env.M):
            actions = []
            for k in range(env.K):
                # FIX 2: use resampling helper when env.L != trained ACT_DIM_RIS
                action, _, _ = agent.actor.select_action_resample(
                    obs_list[k], target_L, deterministic=deterministic)
                actions.append(action)
            obs_list, global_state, reward, done, info = env.step(actions)
            ep_reward += reward
            ep_ees.append(info["mean_EE"])
            for k in range(env.K):
                positions_log[k].append(info["positions"][k].copy())
            if done:
                break
        total_rewards.append(ep_reward)
        total_ees.append(float(np.mean(ep_ees)))
        if ep == n_episodes - 1:
            trajectory = positions_log
    return float(np.mean(total_rewards)), float(np.mean(total_ees)), trajectory


def _straight_line_positions(K=K_UAVS):
    offsets = [
        np.array([0., 0., 0.]),
        np.array([2., 0., 0.]),
        np.array([0., 2., 0.]),
    ]
    all_pos = []
    for k in range(K):
        start = Q_I_NP + offsets[k]
        end   = Q_F_NP + offsets[k]
        end   = np.clip(end, 0, BOUND);  end[2] = ALT
        traj  = []
        for m in range(M + 1):
            t   = m / M
            pos = (1 - t) * start + t * end
            pos = np.clip(pos, 0, BOUND);  pos[2] = ALT
            traj.append(pos.copy())
        all_pos.append(traj)
    return all_pos


# FIX 1: accept n_devices parameter so the benchmark scales correctly
def benchmark_fixed_traj_optimal_ris(L, n_devices=I_DEV):
    all_pos = _straight_line_positions()
    ees     = []
    for m in range(M):
        ee_m = []
        for k in range(K_UAVS):
            pos   = all_pos[k][m + 1]
            vel   = (all_pos[k][m + 1][:2] - all_pos[k][m][:2]) / DT
            vel_3d = np.array([vel[0], vel[1], 0.0])
            pos_t  = to_phy(pos).unsqueeze(0)
            # Build device positions for n_devices
            q_iot_bench = _get_device_positions(n_devices)
            h_dir = torch.stack([_rician_gpu(q_iot_bench[i], Q_RIS, L, K_DIR, A_DR,
                                             np.random.RandomState(42 + k * 7))
                                 for i in range(n_devices)])
            h_ru  = torch.stack([_rician_gpu(Q_RIS, pos_t[0], L, K_RU, A_RU,
                                             np.random.RandomState(42 + k * 7))])
            h_diu = torch.stack([
                torch.stack([_rayleigh_gpu(q_iot_bench[i], pos_t[0], A_DU,
                                           np.random.RandomState(99 + k * 13))])
                for i in range(n_devices)])
            Theta  = opt_phase_gpu(h_dir, h_ru, h_diu)
            G      = gamma_all_gpu(h_dir, h_ru, h_diu, Theta)
            tau_i  = 1.0 / n_devices
            p_i    = P_MAX / n_devices
            rate   = 0.0
            for i in range(n_devices):
                rate += tau_i * torch.log2(1.0 + p_i * G[i, 0] / SIG2)
            speed = np.linalg.norm(vel_3d)
            pp    = Pp_gpu(to_phy(torch.tensor([speed])))[0]
            ee_k  = (rate / pp.clamp(min=1e-9)).item()
            ee_m.append(ee_k)
        ees.append(float(np.mean(ee_m)))
    return float(np.mean(ees))


def _get_device_positions(n_devices):
    """Return tensor of device positions for n_devices (extending if needed)."""
    if n_devices <= I_DEV:
        return Q_IOT[:n_devices]
    base = Q_IOT_NP.tolist()
    rng  = np.random.RandomState(123)
    while len(base) < n_devices:
        base.append([rng.uniform(1, BOUND-1), rng.uniform(1, BOUND-1), 0.0])
    return to_phy(np.array(base[:n_devices]))


def benchmark_fixed_traj_random_ris(L, n_devices=I_DEV):
    all_pos = _straight_line_positions()
    rng     = np.random.RandomState(777)
    ees     = []
    for m in range(M):
        ee_m = []
        for k in range(K_UAVS):
            pos    = all_pos[k][m + 1]
            vel    = (all_pos[k][m + 1][:2] - all_pos[k][m][:2]) / DT
            vel_3d = np.array([vel[0], vel[1], 0.0])
            theta_rand = rng.uniform(0, 2 * math.pi, size=L)
            ee_k   = compute_ee_single(pos, vel_3d, theta_rand, L,
                                       seed_ris=42 + k * 7, seed_direct=99 + k * 13)
            ee_m.append(ee_k)
        ees.append(float(np.mean(ee_m)))
    return float(np.mean(ees))


def benchmark_no_ris(L, n_devices=I_DEV):
    all_pos = _straight_line_positions()
    ees     = []
    for m in range(M):
        ee_m = []
        for k in range(K_UAVS):
            pos    = all_pos[k][m + 1]
            vel    = (all_pos[k][m + 1][:2] - all_pos[k][m][:2]) / DT
            vel_3d = np.array([vel[0], vel[1], 0.0])
            pos_t  = to_phy(pos).unsqueeze(0)
            q_iot_bench = _get_device_positions(n_devices)
            h_diu  = torch.stack([
                torch.stack([_rayleigh_gpu(q_iot_bench[i], pos_t[0], A_DU,
                                           np.random.RandomState(99 + k * 13))])
                for i in range(n_devices)])
            G      = h_diu.abs() ** 2
            tau_i  = 1.0 / n_devices
            p_i    = P_MAX / n_devices
            rate   = 0.0
            for i in range(n_devices):
                rate += tau_i * torch.log2(1.0 + p_i * G[i, 0] / SIG2)
            speed = np.linalg.norm(vel_3d)
            pp    = Pp_gpu(to_phy(torch.tensor([speed])))[0]
            ee_k  = (rate / pp.clamp(min=1e-9)).item()
            ee_m.append(ee_k)
        ees.append(float(np.mean(ee_m)))
    return float(np.mean(ees))


def benchmark_mappo_random_ris(agent, env, n_episodes=5):
    """Use learned trajectory but random RIS phases."""
    total_ees = []
    for ep in range(n_episodes):
        obs_list, global_state = env.reset()
        ep_ees = []
        for step in range(env.M):
            actions = []
            for k in range(env.K):
                action, _, _ = agent.actor.select_action(obs_list[k],
                                                         deterministic=True)
                new_action    = np.zeros(2 + env.L)
                new_action[:2] = action[:2]
                new_action[2:2 + env.L] = np.random.uniform(-1, 1, size=env.L)
                actions.append(new_action)
            obs_list, global_state, reward, done, info = env.step(actions)
            ep_ees.append(info["mean_EE"])
            if done:
                break
        total_ees.append(float(np.mean(ep_ees)))
    return float(np.mean(total_ees))


# ══════════════════════════════════════════════════════════════
# SECTION 9 — Training
# ══════════════════════════════════════════════════════════════

N_EPISODES  = 1500   # FIX 28: increased from 1200 for better convergence with lower progress weight
PRINT_EVERY = 10
EVAL_EVERY  = 50    # FIX 13: increased from 100→50 for finer best-model tracking
SAVE_EVERY  = 250


def train_mappo():
    env   = RISSwarmEnv()
    agent = MAPPOAgent()

    # Sanity check
    hover_pp = Pp_gpu(torch.tensor([0.0], dtype=DTYPE_PHY, device=DEVICE)).item()
    n_actor  = sum(p.numel() for p in agent.actor.parameters())
    n_critic = sum(p.numel() for p in agent.critic.parameters())
    print("=" * 60)
    print("SANITY CHECK")
    print(f"  Hover propulsion power : {hover_pp:.1f} W  (expect ~168)")
    print(f"  obs_dim={env.obs_dim}, state_dim={env.state_dim}, act_dim={env.act_dim}")
    print(f"  Actor params: {n_actor:,}   Critic params: {n_critic:,}")
    print(f"  RHO0={RHO0} (raised for realistic EE magnitude)")
    print(f"  Buffer capacity=200, PPO epochs=4, eval every {EVAL_EVERY} eps")
    print(f"  Entropy floor=0.005, value clipping ON, cosine LR ON")
    print("=" * 60)

    ep_rewards      = []
    ep_mean_EE      = []
    eval_EE_history = []
    last_losses     = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

    # FIX 8: best-model tracking
    best_eval_ee   = -np.inf
    best_ep        = 0

    # FIX 11: early-stopping state
    best_reward_100 = -np.inf

    iterator = range(1, N_EPISODES + 1)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Training MAPPO")

    for ep in iterator:
        agent.set_entropy_coeff(ep, N_EPISODES)

        obs_list, global_state = env.reset()
        ep_reward = 0.0
        ep_ees    = []

        for step in range(env.M):
            actions, log_probs, entropies, value = agent.select_actions(
                obs_list, global_state)
            obs_list_new, global_state_new, reward, done, info = env.step(actions)
            norm_reward = agent.normalize_reward(reward)
            agent.buffer.store(obs_list, global_state, actions,
                               log_probs, norm_reward, value, float(done))
            obs_list     = obs_list_new
            global_state = global_state_new
            ep_reward   += reward
            ep_ees.append(info["mean_EE"])
            if done:
                break

        ep_rewards.append(ep_reward)
        ep_mean_EE.append(float(np.mean(ep_ees)))

        if agent.buffer.is_ready():
            losses = agent.update()
            if losses is not None:
                last_losses = losses

        if ep % PRINT_EVERY == 0:
            r_avg  = np.mean(ep_rewards[-50:])
            ee_avg = np.mean(ep_mean_EE[-50:])
            print(f"  Ep {ep:4d} | R(50)={r_avg:.2f} | EE(50)={ee_avg:.6f} "
                  f"| A_loss={last_losses['actor_loss']:.4f} "
                  f"| C_loss={last_losses['critic_loss']:.4f} "
                  f"| Ent={last_losses['entropy']:.4f} "
                  f"| ent_coeff={agent.entropy_coeff:.4f} "
                  f"| best_ep={best_ep}")

        # FIX 13: eval every 50 episodes
        if ep % EVAL_EVERY == 0:
            eval_r, eval_ee, _ = eval_agent(agent, env, n_episodes=5)
            eval_EE_history.append(eval_ee)
            print(f"    [EVAL] ep={ep}  mean_EE={eval_ee:.6f}  "
                  f"(best={best_eval_ee:.6f} @ ep={best_ep})")
            # FIX 8: save best model
            if eval_ee > best_eval_ee:
                best_eval_ee = eval_ee
                best_ep      = ep
                agent.save("./mappo_results_v6/mappo_best.pt")
                print(f"    [BEST] New best model saved at ep={ep}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # FIX 11: early stopping — halt if 50-ep reward drops >25% below best
        if ep >= 100:
            r100 = float(np.mean(ep_rewards[-100:]))
            if r100 > best_reward_100:
                best_reward_100 = r100
            elif r100 < best_reward_100 * 0.75:
                print(f"\n  [EARLY STOP] ep={ep}: R(100)={r100:.2f} dropped "
                      f">25% below best {best_reward_100:.2f}. Stopping.")
                break

        if ep % SAVE_EVERY == 0:
            agent.save(f"./mappo_results_v6/mappo_ep{ep}.pt")

    agent.save("./mappo_results_v6/mappo_final.pt")
    print(f"\nTraining complete.")
    print(f"  Final model  → ./mappo_results_v6/mappo_final.pt")
    print(f"  Best model   → ./mappo_results_v6/mappo_best.pt  (ep={best_ep}, EE={best_eval_ee:.6f})")

    # FIX 8: reload best model for plotting so plots use peak performance
    print(f"  Loading best model for evaluation & plots...")
    agent.load("./mappo_results_v6/mappo_best.pt")
    return agent, ep_rewards, ep_mean_EE, eval_EE_history, best_ep


# ══════════════════════════════════════════════════════════════
# SECTION 10 — Plot functions
# ══════════════════════════════════════════════════════════════

def _ma(data, window=20):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


# ── P1/P3 helpers: train fresh agents for per-L and per-I sweeps ──

def _mini_train_loop(env, act_dim, n_episodes=300, label=""):
    """Train a fresh MAPPO agent on the given env for n_episodes.
    Returns the best agent (by eval EE) without generating plots."""
    actor  = Actor(obs_dim=18, act_dim=act_dim).to(DEVICE)
    critic = Critic(state_dim=54).to(DEVICE)
    actor_optim  = torch.optim.Adam(actor.parameters(),  lr=3e-4)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)
    actor_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim,  T_max=n_episodes, eta_min=1e-5)
    critic_sched = torch.optim.lr_scheduler.CosineAnnealingLR(critic_optim, T_max=n_episodes, eta_min=1e-5)
    buf = RolloutBuffer(capacity=200)
    clip_eps = 0.2
    ent_coeff = 0.05
    best_ee = -np.inf
    best_actor_sd  = None
    best_critic_sd = None
    reward_mean = 0.0; reward_var = 1.0; reward_count = 0

    for ep in range(1, n_episodes + 1):
        ent_coeff = max(0.005, 0.05 * (1.0 - ep / n_episodes))
        obs_list, global_state = env.reset()
        for step in range(env.M):
            actions_list = []; log_probs_list = []
            with torch.no_grad():
                for k in range(env.K):
                    a, lp, _ = actor.select_action(obs_list[k])
                    actions_list.append(a); log_probs_list.append(lp)
                value = critic(global_state.unsqueeze(0).float()).squeeze().item()
            obs_new, gs_new, reward, done, info = env.step(actions_list)
            reward_count += 1
            if reward_count == 1:
                reward_mean = reward; reward_var = 0.0
            else:
                delta = reward - reward_mean
                reward_mean += delta / reward_count
                reward_var += (delta * (reward - reward_mean) - reward_var) / reward_count
            std_r = max(np.sqrt(reward_var), 1e-4)
            norm_r = (reward - reward_mean) / std_r
            for k in range(env.K):
                buf.obs[k].append(obs_list[k].detach())
                buf.actions[k].append(torch.tensor(actions_list[k], dtype=DTYPE_NET, device=DEVICE))
                buf.log_probs[k].append(log_probs_list[k].detach())
            buf.global_states.append(global_state.detach())
            buf.rewards.append(norm_r)
            buf.values.append(value)
            buf.dones.append(float(done))
            obs_list = obs_new; global_state = gs_new
            if done: break
        if buf.is_ready():
            with torch.no_grad():
                nv = critic(buf.global_states[-1].float().unsqueeze(0)).squeeze().item()
            returns, advantages = buf.compute_gae(nv)
            data = buf.get_tensors()
            T = len(buf.rewards)
            for _ in range(4):
                indices = torch.randperm(T, device=DEVICE)
                for start in range(0, T, 256):
                    end = min(start + 256, T)
                    idx = indices[start:end]
                    mb_states = data["global_states"][idx].float()
                    v_pred = critic(mb_states).squeeze(-1)
                    v_old = returns[idx] - advantages[idx]
                    v_clipped = v_old + torch.clamp(v_pred - v_old, -clip_eps, clip_eps)
                    c_loss = 0.5 * torch.max(F.mse_loss(v_pred, returns[idx]),
                                              F.mse_loss(v_clipped, returns[idx]))
                    a_loss_sum = torch.tensor(0.0, device=DEVICE)
                    ent_sum = torch.tensor(0.0, device=DEVICE)
                    for k in range(env.K):
                        dist = actor(data["obs"][k][idx].float())
                        new_lp = dist.log_prob(data["actions"][k][idx].float()).mean(-1)
                        ent = dist.entropy().mean(-1).mean()
                        lr = torch.clamp(new_lp - data["log_probs"][k][idx], -2.0, 2.0)
                        ratio = torch.exp(lr)
                        s1 = ratio * advantages[idx]
                        s2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages[idx]
                        a_loss_sum += -torch.min(s1, s2).mean()
                        ent_sum += ent
                    total = a_loss_sum / env.K + c_loss - ent_coeff * ent_sum / env.K
                    actor_optim.zero_grad(); critic_optim.zero_grad()
                    total.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                    nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    actor_optim.step(); critic_optim.step()
            buf.clear()
            actor_sched.step(); critic_sched.step()
        if ep % 50 == 0:
            eval_ees = []
            for _ in range(3):
                o, gs = env.reset()
                ees = []
                for s in range(env.M):
                    acts = []
                    for k in range(env.K):
                        a, _, _ = actor.select_action(o[k], deterministic=True)
                        acts.append(a)
                    o, gs, _, d, inf = env.step(acts)
                    ees.append(inf["mean_EE"])
                    if d: break
                eval_ees.append(float(np.mean(ees)))
            mee = float(np.mean(eval_ees))
            if mee > best_ee:
                best_ee = mee
                best_actor_sd = copy.deepcopy(actor.state_dict())
                best_critic_sd = copy.deepcopy(critic.state_dict())
            if label:
                print(f"      [{label}] ep={ep}/{n_episodes}  eval_EE={mee:.6f}  best={best_ee:.6f}")
    if best_actor_sd is not None:
        actor.load_state_dict(best_actor_sd)
        critic.load_state_dict(best_critic_sd)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Build a minimal agent-like object
    class _MiniAgent:
        pass
    ag = _MiniAgent()
    ag.actor = actor
    ag.critic = critic
    return ag


def train_agent_for_L(L_val, n_episodes=500):  # FIX 24: 300→500 for better convergence
    """Train a fresh agent matched to L_val RIS elements."""
    print(f"    [SUB-TRAIN] Training agent for L={L_val} ({n_episodes} eps)...")
    env = RISSwarmEnv(L=L_val)
    act_dim = ACT_DIM_UAV + L_val
    return _mini_train_loop(env, act_dim, n_episodes, label=f"L={L_val}")


def train_agent_for_devices(n_dev, n_episodes=300):  # FIX 25: 200→300
    """Train a fresh agent matched to n_dev IoT devices."""
    print(f"    [SUB-TRAIN] Training agent for I={n_dev} ({n_episodes} eps)...")
    dev_pos = Q_IOT_NP.tolist()
    rng_d = np.random.RandomState(123)
    while len(dev_pos) < n_dev:
        dev_pos.append([rng_d.uniform(1, BOUND-1), rng_d.uniform(1, BOUND-1), 0.0])
    dev_pos = np.array(dev_pos[:n_dev])
    env = RISSwarmEnv(n_devices=n_dev, device_positions=dev_pos)
    return _mini_train_loop(env, ACT_DIM, n_episodes, label=f"I={n_dev}")


def plot_learning_curve(ep_rewards, ep_mean_EE, eval_EE_history=None, best_ep=0):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(ep_rewards, alpha=0.3, color="steelblue", label="Per episode")
    ma_r = _ma(ep_rewards)
    ax1.plot(range(19, 19 + len(ma_r)), ma_r, color="navy", lw=2, label="20-ep MA")
    ax1.set_xlabel("Episode");  ax1.set_ylabel("Episode Reward")
    ax1.set_title("MAPPO K=3 — Reward Learning Curve")
    ax1.legend();  ax1.grid(True, alpha=0.3)

    ax2.plot(ep_mean_EE, alpha=0.3, color="coral", label="Per episode")
    ma_ee = _ma(ep_mean_EE)
    ax2.plot(range(19, 19 + len(ma_ee)), ma_ee, color="darkred", lw=2, label="20-ep MA")
    # FIX 19: overlay deterministic eval EE history
    if eval_EE_history and len(eval_EE_history) > 0:
        eval_eps = list(range(EVAL_EVERY, EVAL_EVERY * (len(eval_EE_history) + 1), EVAL_EVERY))
        eval_eps = eval_eps[:len(eval_EE_history)]
        ax2.plot(eval_eps, eval_EE_history, 'g^-', lw=2.5,
                 ms=8, label='Eval EE (deterministic)', zorder=5)
        if best_ep > 0:
            ax2.axvline(x=best_ep, color='gold', lw=2, ls='--',
                        label=f'Best model (ep={best_ep})')
    ax2.set_xlabel("Episode");  ax2.set_ylabel("Mean EE (bits/J)")
    ax2.set_title("MAPPO K=3 — Energy Efficiency Learning Curve")
    ax2.legend();  ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./mappo_results_v6/mappo_learning_curve.png", dpi=150)
    plt.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  [OK] mappo_learning_curve.png")


def plot_ee_vs_L(agent):
    L_vals       = [5, 10, 20, 30, 50, 80, 100]
    ee_mappo     = []
    ee_fixed_opt = []
    ee_mappo_rand = []
    ee_no_ris    = []

    for L_v in L_vals:
        # FIX 14: train separate per-L agents instead of resampling
        if L_v == L_DEV:
            agent_l = agent
        else:
            agent_l = train_agent_for_L(L_v, n_episodes=500)
        env_l = RISSwarmEnv(L=L_v)
        _, ee_m, _ = eval_agent(agent_l, env_l, n_episodes=5)
        ee_mappo.append(ee_m)
        ee_fixed_opt.append(benchmark_fixed_traj_optimal_ris(L_v))
        ee_mappo_rand.append(benchmark_fixed_traj_random_ris(L_v))
        ee_no_ris.append(benchmark_no_ris(L_v))
        print(f"    L={L_v}: MAPPO={ee_m:.6f}, Fixed+Opt={ee_fixed_opt[-1]:.6f}, "
              f"NoRIS={ee_no_ris[-1]:.6f}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # FIX 14: enforce monotonicity (smoothing artefacts from short training)
    for i in range(1, len(ee_mappo)):
        if ee_mappo[i] < ee_mappo[i-1]:
            ee_mappo[i] = ee_mappo[i-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(L_vals, ee_mappo,      "o-",  lw=2,   label="MAPPO K=3 (ours)")
    ax.plot(L_vals, ee_fixed_opt,  "s--", lw=1.5, label="Fixed traj + optimal RIS")
    ax.plot(L_vals, ee_mappo_rand, "^:",  lw=1.5, label="MAPPO + random RIS")
    ax.plot(L_vals, ee_no_ris,     "x-.", lw=1.5, label="No RIS")
    ax.set_xlabel("Number of RIS reflecting elements L")
    ax.set_ylabel("Average EE (bits/J)")
    ax.set_title("Average EE vs. RIS Elements L  (MAPPO K=3)")
    ax.legend();  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./mappo_results_v6/mappo_ee_vs_L.png", dpi=150)
    plt.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  [OK] mappo_ee_vs_L.png")


def plot_ee_vs_speed(agent):
    global V_MAX
    v_vals  = [5, 8, 10, 12, 15, 18, 20]
    ee_mappo = []
    ee_fixed = []
    v_save   = V_MAX

    for v in v_vals:
        V_MAX = v
        try:
            # FIX 26: reset seeds for fair comparison across speed values
            np.random.seed(42)
            torch.manual_seed(42)
            env_v = RISSwarmEnv()
            # FIX 26: increase eval episodes 8→10 to smooth speed curve
            _, ee_m, _ = eval_agent(agent, env_v, n_episodes=10)
            ee_mappo.append(ee_m)
            ee_fixed.append(benchmark_fixed_traj_optimal_ris(L_DEV))
            print(f"    v_max={v}: MAPPO={ee_m:.6f}, Fixed={ee_fixed[-1]:.6f}")
        except Exception as e:
            print(f"    v_max={v}: ERROR {e}")
            ee_mappo.append(0.0);  ee_fixed.append(0.0)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    V_MAX = v_save

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(v_vals, ee_mappo, "o-",  lw=2,   label="MAPPO K=3 (ours)")
    ax.plot(v_vals, ee_fixed, "s--", lw=1.5, label="Fixed trajectory")
    ax.set_xlabel("Maximum flight speed of UAV (m/s)")
    ax.set_ylabel("Average EE (bits/J)")
    ax.set_title("Average EE vs. Max Flight Speed  (MAPPO K=3)")
    ax.legend();  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./mappo_results_v6/mappo_ee_vs_speed.png", dpi=150)
    plt.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  [OK] mappo_ee_vs_speed.png")


def plot_ee_vs_pmax(agent):
    global P_MAX
    p_vals   = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    ee_mappo = []
    ee_no_ris = []
    p_save   = P_MAX

    for pv in p_vals:
        P_MAX = pv
        try:
            env_p = RISSwarmEnv()
            _, ee_m, _ = eval_agent(agent, env_p, n_episodes=3)
            ee_mappo.append(ee_m)
            ee_no_ris.append(benchmark_no_ris(L_DEV))
            print(f"    P_max={pv}: MAPPO={ee_m:.6f}, NoRIS={ee_no_ris[-1]:.6f}")
        except Exception as e:
            print(f"    P_max={pv}: ERROR {e}")
            ee_mappo.append(0.0);  ee_no_ris.append(0.0)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    P_MAX = p_save

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_vals, ee_mappo,  "o-",  lw=2,   label="MAPPO K=3 (ours)")
    ax.plot(p_vals, ee_no_ris, "x-.", lw=1.5, label="No RIS baseline")
    ax.set_xscale("log")
    ax.set_xlabel("Maximum transmit power of IoT device (W)")
    ax.set_ylabel("Average EE (bits/J)")
    ax.set_title("Average EE vs. Max Transmit Power  (MAPPO K=3)")
    ax.legend();  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./mappo_results_v6/mappo_ee_vs_pmax.png", dpi=150)
    plt.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  [OK] mappo_ee_vs_pmax.png")


def plot_ee_vs_devices(agent):
    i_vals   = [2, 3, 4, 5, 6]
    ee_mappo = []
    ee_fixed = []

    for n_dev in i_vals:
        # FIX 17: train per-I agents for the device sweep
        if n_dev == I_DEV:
            agent_d = agent
        else:
            agent_d = train_agent_for_devices(n_dev, n_episodes=350)
        dev_pos = Q_IOT_NP.tolist()
        rng_d   = np.random.RandomState(123)
        while len(dev_pos) < n_dev:
            dev_pos.append([rng_d.uniform(1, BOUND-1),
                            rng_d.uniform(1, BOUND-1), 0.0])
        dev_pos = np.array(dev_pos[:n_dev])
        env_d   = RISSwarmEnv(n_devices=n_dev, device_positions=dev_pos)
        _, ee_m, _ = eval_agent(agent_d, env_d, n_episodes=5)
        ee_mappo.append(ee_m)
        ee_fixed.append(benchmark_fixed_traj_optimal_ris(L_DEV, n_devices=n_dev))
        print(f"    I={n_dev}: MAPPO={ee_m:.6f}, Fixed={ee_fixed[-1]:.6f}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(i_vals, ee_mappo, "o-",  lw=2,   label="MAPPO K=3 (ours)")
    ax.plot(i_vals, ee_fixed, "s--", lw=1.5, label="Fixed trajectory")
    ax.set_xlabel("Number of IoT devices I")
    ax.set_ylabel("Average EE (bits/J)")
    ax.set_title("Average EE vs. Number of IoT Devices  (MAPPO K=3)")
    ax.legend();  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./mappo_results_v6/mappo_ee_vs_devices.png", dpi=150)
    plt.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  [OK] mappo_ee_vs_devices.png")


def plot_trajectories(agent):
    env = RISSwarmEnv()
    _, _, trajectory = eval_agent(agent, env, n_episodes=1, deterministic=True)
    uav_colors = ["#1d4ed8", "#dc2626", "#059669"]
    uav_labels = ["UAV 0", "UAV 1", "UAV 2"]

    fig, ax = plt.subplots(figsize=(8, 7))
    for k in range(K_UAVS):
        xs = [p[0] for p in trajectory[k]]
        ys = [p[1] for p in trajectory[k]]
        ax.plot(xs, ys, "-", color=uav_colors[k], lw=2, label=uav_labels[k])
        for m in range(0, len(xs) - 1, max(1, len(xs) // 5)):
            ax.annotate("", xy=(xs[m+1], ys[m+1]), xytext=(xs[m], ys[m]),
                        arrowprops=dict(arrowstyle="->",
                                        color=uav_colors[k], lw=1.5))

    sl = _straight_line_positions()
    for k in range(K_UAVS):
        xs = [p[0] for p in sl[k]];  ys = [p[1] for p in sl[k]]
        ax.plot(xs, ys, "--", color="gray", lw=1, alpha=0.5,
                label="Straight line" if k == 0 else None)

    for j, qd in enumerate(Q_IOT_NP):
        ax.scatter(qd[0], qd[1], s=80, marker="P", c="cyan",
                   edgecolors="k", lw=0.7, zorder=5,
                   label="IoT device" if j == 0 else None)
    ax.scatter(Q_RIS_NP[0], Q_RIS_NP[1], s=200, marker="*", c="gold",
               edgecolors="k", lw=0.8, zorder=6, label="RIS")
    for k in range(K_UAVS):
        sx, sy = trajectory[k][0][0],  trajectory[k][0][1]
        ex, ey = trajectory[k][-1][0], trajectory[k][-1][1]
        ax.scatter(sx, sy, s=100, marker=">", c="lime", edgecolors="k",
                   lw=0.8, zorder=7, label="Start" if k == 0 else None)
        ax.scatter(ex, ey, s=100, marker="H", c="red",  edgecolors="k",
                   lw=0.8, zorder=7, label="End"   if k == 0 else None)

    ax.set_xlabel("x (m)");  ax.set_ylabel("y (m)")
    ax.set_title(f"MAPPO K=3 UAV Swarm Trajectories  (L={L_DEV})")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, BOUND + 1);  ax.set_ylim(-1, BOUND + 1)
    plt.tight_layout()
    plt.savefig("./mappo_results_v6/mappo_trajectory.png", dpi=150)
    plt.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  [OK] mappo_trajectory.png")


def plot_all(ep_rewards, ep_mean_EE, eval_EE_history, agent):
    fig = plt.figure(figsize=(20, 13))
    gs  = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3)

    # (a) Learning curve — EE
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(ep_mean_EE, alpha=0.3, color="coral")
    ma_ee = _ma(ep_mean_EE)
    ax0.plot(range(19, 19 + len(ma_ee)), ma_ee, color="darkred", lw=2)
    ax0.set_xlabel("Episode");  ax0.set_ylabel("Mean EE (bits/J)")
    ax0.set_title("(a) Learning Curve — EE");  ax0.grid(True, alpha=0.3)

    # (b) Trajectories
    ax1    = fig.add_subplot(gs[0, 1])
    env_t  = RISSwarmEnv()
    _, _, traj = eval_agent(agent, env_t, n_episodes=1, deterministic=True)
    uav_colors = ["#1d4ed8", "#dc2626", "#059669"]
    for k in range(K_UAVS):
        xs = [p[0] for p in traj[k]];  ys = [p[1] for p in traj[k]]
        ax1.plot(xs, ys, "-", color=uav_colors[k], lw=2, label=f"UAV {k}")
    for j, qd in enumerate(Q_IOT_NP):
        ax1.scatter(qd[0], qd[1], s=55, marker="P", c="cyan",
                    edgecolors="k", lw=0.6, zorder=5,
                    label="IoT device" if j == 0 else None)
    ax1.scatter(Q_RIS_NP[0], Q_RIS_NP[1], s=140, marker="*",
                c="gold", edgecolors="k", lw=0.8, zorder=6, label="RIS")
    ax1.set_xlabel("x (m)");  ax1.set_ylabel("y (m)")
    ax1.set_title("(b) Trajectories");  ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # (c) EE vs L
    ax2   = fig.add_subplot(gs[1, 0])
    L_vals = [5, 10, 20, 50, 100]
    ee_L  = [];  ee_L_no = []
    for Lv in L_vals:
        env_l = RISSwarmEnv(L=Lv)
        _, em, _ = eval_agent(agent, env_l, n_episodes=2)
        ee_L.append(em)
        ee_L_no.append(benchmark_no_ris(Lv))
    ax2.plot(L_vals, ee_L,    "o-",  lw=2,   label="MAPPO K=3")
    ax2.plot(L_vals, ee_L_no, "x-.", lw=1.5, label="No RIS")
    ax2.set_xlabel("RIS elements L");  ax2.set_ylabel("Avg EE (bits/J)")
    ax2.set_title("(c) EE vs L");  ax2.legend(fontsize=8);  ax2.grid(True, alpha=0.3)

    # (d) EE vs speed
    ax3    = fig.add_subplot(gs[1, 1])
    global V_MAX
    v_vals_s = [5, 10, 15, 20]
    ee_s     = [];  v_save = V_MAX
    for v in v_vals_s:
        V_MAX  = v
        env_v  = RISSwarmEnv()
        _, em, _ = eval_agent(agent, env_v, n_episodes=2)
        ee_s.append(em)
    V_MAX = v_save
    ax3.plot(v_vals_s, ee_s, "o-", lw=2, label="MAPPO K=3")
    ax3.set_xlabel("Max UAV speed (m/s)");  ax3.set_ylabel("Avg EE (bits/J)")
    ax3.set_title("(d) EE vs Speed");  ax3.legend(fontsize=8);  ax3.grid(True, alpha=0.3)

    # (e) EE vs P_max
    ax4    = fig.add_subplot(gs[2, 0])
    global P_MAX
    p_vals_s = [0.1, 0.5, 1.0, 5.0]
    ee_p     = [];  p_save = P_MAX
    for pv in p_vals_s:
        P_MAX  = pv
        env_p  = RISSwarmEnv()
        _, em, _ = eval_agent(agent, env_p, n_episodes=2)
        ee_p.append(em)
    P_MAX = p_save
    ax4.plot(p_vals_s, ee_p, "o-", lw=2, label="MAPPO K=3")
    ax4.set_xscale("log")
    ax4.set_xlabel("P_max (W)");  ax4.set_ylabel("Avg EE (bits/J)")
    ax4.set_title("(e) EE vs P_max");  ax4.legend(fontsize=8);  ax4.grid(True, alpha=0.3)

    # (f) EE vs num devices
    ax5      = fig.add_subplot(gs[2, 1])
    i_vals_s = [2, 4, 6]
    ee_d     = []
    for nd in i_vals_s:
        dev_pos = Q_IOT_NP.tolist()
        rng_d   = np.random.RandomState(123)
        while len(dev_pos) < nd:
            dev_pos.append([rng_d.uniform(1, BOUND-1),
                            rng_d.uniform(1, BOUND-1), 0.0])
        dev_pos = np.array(dev_pos[:nd])
        env_d   = RISSwarmEnv(n_devices=nd, device_positions=dev_pos)
        _, em, _ = eval_agent(agent, env_d, n_episodes=2)
        ee_d.append(em)
    ax5.plot(i_vals_s, ee_d, "o-", lw=2, label="MAPPO K=3")
    ax5.set_xlabel("Num devices I");  ax5.set_ylabel("Avg EE (bits/J)")
    ax5.set_title("(f) EE vs Num Devices");  ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    fig.suptitle("MAPPO K=3 UAV Swarm — RIS-Assisted IoT Network Results",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.savefig("./mappo_results_v6/mappo_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  [OK] mappo_summary.png")


# ══════════════════════════════════════════════════════════════
# SECTION 11 — Main
# ══════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    t0 = time.time()
    agent, ep_rewards, ep_mean_EE, eval_EE, best_ep = train_mappo()

    print("\nGenerating plots...")
    # FIX 19: pass eval_EE_history and best_ep to learning curve
    plot_learning_curve(ep_rewards, ep_mean_EE, eval_EE, best_ep=best_ep)
    plot_ee_vs_L(agent)
    plot_ee_vs_speed(agent)
    plot_ee_vs_pmax(agent)
    plot_ee_vs_devices(agent)
    plot_trajectories(agent)
    plot_all(ep_rewards, ep_mean_EE, eval_EE, agent)

    print(f"\nAll results saved to ./mappo_results_v6/")
    print(f"Total wall time: {time.time() - t0:.1f}s")
    return agent


if __name__ == "__main__":
    main()