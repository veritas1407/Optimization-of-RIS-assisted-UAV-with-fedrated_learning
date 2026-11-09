"""RIS-assisted multi-UAV IoT data-collection environment (v9.1).

One episode = one mission of M slots. Agents are the K collector UAVs plus, in the
aerial preset, one UAV carrying the RIS. Each agent outputs a 2-D velocity command in
[-1, 1]^2. The environment owns everything else: kinematics, energy, channels,
device scheduling, and the closed-form RIS configuration (one physical RIS, shared).

v9.1 changes (audit, docs/analysis/06-v9-audit.md): composite-Simpson propulsion energy,
links evaluated at the mid-slot position, velocity zeroed at the area boundary, 3-D
collision push, landing slots inside the area, open devices kept out of the building
block, a separate random stream for RIS phase errors / random phases, aimed aerial panel,
configurable number of observed devices and served-last-slot flags in the observation.
"""
from __future__ import annotations

import numpy as np

from . import physics as ph
from .config import ScenarioConfig

DEV_FEATS = 10
MAX_RIS_CONFIGS = 8


def separate(env, acts, adjust=None, see_velocities: bool = True, margin: float = 1.25,
             iters: int = 3) -> np.ndarray:
    """Adjust velocity commands (in [-1, 1]^2) so the UAVs keep the minimum 3-D separation at
    the end of the slot: predicted end positions closer than margin x the required horizontal
    gap are pushed apart. Only agents with adjust[k] change their command. With
    see_velocities=False an agent only knows where the others are now (what a learned UAV
    observes), not where they are flying. Used by the return-to-home autopilot and by the
    heuristic baselines."""
    cfg = env.cfg
    n = env.n_agents
    adjust = np.ones(n, bool) if adjust is None else np.asarray(adjust, bool)
    v = np.asarray(acts, float) * cfg.v_max
    dv_max = cfg.a_max * cfg.dt

    def end_pos(k, vk):
        dv = vk - env.vel[k]
        nd = np.linalg.norm(dv)
        v1 = env.vel[k] + (dv * dv_max / nd if nd > dv_max else dv)
        return env.pos[k, :2] + 0.5 * (env.vel[k] + v1) * cfg.dt

    for _ in range(iters):
        ends = [end_pos(k, v[k]) for k in range(n)]
        moved = False
        for a in np.flatnonzero(adjust):
            for b in range(n):
                if a == b:
                    continue
                dz = abs(env.pos[a, 2] - env.pos[b, 2])
                h_req = np.sqrt(max(cfg.d_min ** 2 - dz ** 2, 0.0))
                if h_req <= 0.0:
                    continue
                pb = ends[b] if see_velocities else env.pos[b, :2]
                rel = ends[a] - pb
                dist = np.linalg.norm(rel)
                gap = margin * h_req
                if dist < gap:
                    u = rel / dist if dist > 1e-6 else np.array([np.cos(a), np.sin(a)])
                    share = 0.5 if adjust[b] else 1.0     # a non-adjusting neighbour will not move
                    v[a] = v[a] + u * share * 2.0 * (gap - dist) / cfg.dt
                    moved = True
        if not moved:
            break
    sp = np.linalg.norm(v, axis=1, keepdims=True)
    v = np.where(sp > cfg.v_max, v * cfg.v_max / np.maximum(sp, 1e-9), v)
    return v / cfg.v_max


class RISUAVEnv:
    def __init__(self, cfg: ScenarioConfig, seed: int = 0):
        self.set_scenario(cfg)
        self.lam = float(cfg.lam_init)
        self.agent_id = cfg.agent_id_obs
        self.top_k = cfg.top_k
        self.obs_dim = (5 + 3 + 2 * (self.n_agents - 1) + 2 + self.top_k * DEV_FEATS + 2
                        + (self.n_agents if self.agent_id else 0))
        self.state_dim = self.obs_dim * self.n_agents
        self.act_dim = 2
        self.rng = np.random.default_rng(seed)
        self.rng_ris = np.random.default_rng(seed + 7)
        self.episode_seed = seed

    def set_scenario(self, cfg: ScenarioConfig):
        """(Re)load the scenario geometry. Used to switch city areas between episodes;
        the observation/action sizes must stay the same."""
        if hasattr(self, "obs_dim"):
            assert cfg.n_agents == self.n_agents and cfg.n_devices == self.I, "area must keep sizes"
        self.cfg = cfg
        self.boxes = ph.boxes_array(cfg.buildings)
        self.p_w = ph.tx_power_w(cfg)
        self.noise_w = ph.noise_power_w(cfg)
        self.K = cfg.K
        self.n_agents = cfg.n_agents
        self.aerial = cfg.has_aerial_ris
        self.use_ris = cfg.arbitration != "none"
        self.I = cfg.n_devices
        self.L = cfg.ris.L
        self.v_pmin, self.p_min = ph.min_power_speed(cfg)
        self.e_ref_slot = self.n_agents * self.p_min * cfg.dt
        self.spacing = cfg.ris.spacing_wl * ph.wavelength(cfg.fc)
        if self.use_ris and not self.aerial:
            self.ris_center = np.asarray(cfg.ris.center, dtype=float)
            self.ris_elems = ph.ris_elements(self.ris_center, cfg.ris.normal,
                                             cfg.ris.ny, cfg.ris.nz, self.spacing)
        # one landing slot per agent strictly inside the final zone (and inside the area):
        # a line across the diagonal if it fits, otherwise a ring (tiny legacy final zone)
        n = self.n_agents
        offs = (np.arange(n) - (n - 1) / 2.0) * cfg.d_min * 1.2
        if np.abs(offs).max() <= 0.9 * cfg.r_final:
            perp = np.array([1.0, -1.0]) / np.sqrt(2.0)
            rel = offs[:, None] * perp[None, :]
        else:
            ring = min(0.9 * cfg.r_final, 1.3 * cfg.d_min / (2.0 * np.sin(np.pi / n))) if n > 1 else 0.0
            ang = np.pi / 4 + 2 * np.pi * np.arange(n) / n
            rel = ring * np.column_stack([np.cos(ang), np.sin(ang)])
        ft = np.asarray(cfg.final_xy, float)[None, :] + rel
        self.final_targets = np.clip(ft, 2.0, cfg.area - 2.0)
        # building block hull: open-field devices are placed outside it
        if len(self.boxes):
            self.block_hull = np.array([self.boxes[:, 0].min(), self.boxes[:, 1].min(),
                                        self.boxes[:, 3].max(), self.boxes[:, 4].max()])
        else:
            self.block_hull = None

    # ------------------------------------------------------------ episode setup

    def _place_devices(self):
        cfg = self.cfg
        if cfg.fixed_devices:
            xy = np.asarray(cfg.fixed_devices, dtype=float)
            shadow = np.zeros(len(xy), dtype=bool)
        else:
            pts = []
            while len(pts) < cfg.n_open:
                p = self.rng.uniform(10.0, cfg.area - 10.0, size=2)
                if self.block_hull is not None:
                    x0, y0, x1, y1 = self.block_hull
                    if x0 - 5.0 <= p[0] <= x1 + 5.0 and y0 - 5.0 <= p[1] <= y1 + 5.0:
                        continue
                pts.append(p)
            x0, y0, x1, y1 = cfg.shadow_region
            sh = np.column_stack([self.rng.uniform(x0, x1, cfg.n_shadow),
                                  self.rng.uniform(y0, y1, cfg.n_shadow)])
            xy = np.vstack([np.array(pts).reshape(-1, 2), sh])
            shadow = np.r_[np.zeros(cfg.n_open, bool), np.ones(cfg.n_shadow, bool)]
        self.dev = np.column_stack([xy, np.zeros(len(xy))])
        self.shadow = shadow

    def reset(self, seed: int | None = None):
        cfg = self.cfg
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.episode_seed = seed
        # RIS phase errors / random phases get their own stream so that they never shift
        # the fading draws (keeps comparisons between settings paired)
        self.rng_ris = np.random.default_rng(int(self.rng.integers(2 ** 62)))
        self._place_devices()
        lo, hi = cfg.data_mbit
        self.data0 = self.rng.uniform(lo, hi, self.I) * 1e6
        self.remaining = self.data0.copy()
        if cfg.always_on:
            self.win_start = np.zeros(self.I)
            self.win_end = np.full(self.I, np.inf)
        else:
            self.win_start = self.rng.uniform(0.0, cfg.T - cfg.window_s, self.I)
            self.win_end = self.win_start + cfg.window_s

        n = self.n_agents
        self.pos = np.zeros((n, 3))
        sx, sy = cfg.start_xy
        for k in range(self.K):
            self.pos[k] = [sx + k * cfg.start_spacing, sy, cfg.uav_alt]
        if self.aerial:
            self.pos[self.K] = np.asarray(cfg.ris.center, dtype=float)
        self.pos[:, :2] = np.clip(self.pos[:, :2], 0.0, cfg.area)
        self.vel = np.zeros((n, 2))
        self.slot = 0

        if self.use_ris and not self.aerial:
            self.h_ir_static, self.elig_static = ph.ris_side_channel(
                self.dev, self.ris_elems, self.ris_center, cfg.ris.normal, cfg, self.boxes,
                self.rng, model="los")

        self.last_snr_db = np.zeros((self.I, self.K))
        self.last_served = np.full(self.I, -1)
        self.blocked = ph.segments_blocked(self.dev[:, None, :], self.pos[None, :self.K, :], self.boxes)
        self.ris_elig = self._ris_eligibility(self.pos)
        self.phi = self._potentials()
        self.ep = {"bits": 0.0, "energy": 0.0, "collisions": 0, "bits_shadow": 0.0, "rth_slots": 0}
        obs = self._obs()
        return obs, obs.ravel().copy()

    # ------------------------------------------------------------ helpers

    def _ris_center(self, pos):
        if not self.use_ris:
            return None
        return pos[self.K].copy() if self.aerial else self.ris_center

    def _aerial_normal(self, pos):
        """Aerial panel on a gimbal: aimed between straight down and the collectors."""
        c = pos[self.K]
        to_uav = pos[: self.K].mean(axis=0) - c
        n = np.array([0.0, 0.0, -1.0]) + to_uav / max(np.linalg.norm(to_uav), 1e-6)
        return n / np.linalg.norm(n)

    def _ris_eligibility(self, pos):
        if not self.use_ris:
            return np.zeros(self.I, bool)
        if not self.aerial:
            return self.elig_static.copy()
        return ~ph.segments_blocked(self.dev, pos[self.K][None, :], self.boxes)

    def _time_left(self, slot=None):
        slot = self.slot if slot is None else slot
        return max(self.cfg.T - slot * self.cfg.dt, 0.0)

    def _potentials(self):
        cfg = self.cfg
        d = np.linalg.norm(self.pos[:, :2] - np.asarray(cfg.final_xy), axis=1)
        reach = cfg.final_slack * cfg.v_max * self._time_left()
        return -np.maximum(0.0, d - cfg.r_final - reach) / cfg.area

    # ------------------------------------------------------------ step

    def step(self, actions):
        cfg = self.cfg
        a = np.clip(np.asarray(actions, dtype=float).reshape(self.n_agents, 2), -1.0, 1.0)

        # return-to-home autopilot (safety shield)
        overridden = np.zeros(self.n_agents, dtype=bool)
        if cfg.rth_shield:
            tl_now = self._time_left()
            tl_after = self._time_left(self.slot + 1)
            for k in range(self.n_agents):
                to = self.final_targets[k] - self.pos[k, :2]
                d = np.linalg.norm(to)
                if d < 5.0:
                    continue
                v_home = cfg.v_max * to / d
                turn_t = np.linalg.norm(v_home - self.vel[k]) / cfg.a_max   # time to swing round
                budget = cfg.rth_frac * cfg.v_max * max(tl_after - turn_t, 0.0)
                if d >= budget:
                    # head home at the speed that arrives exactly at the end (capped at v_max)
                    a[k] = to / d * min(1.0, d / (cfg.v_max * max(tl_now, cfg.dt)))
                    overridden[k] = True
            if overridden.any():   # the autopilot also keeps the minimum separation
                a = separate(self, a, adjust=overridden)
            self.ep["rth_slots"] += int(overridden.sum())

        # kinematics: velocity changes linearly within the slot (acceleration-limited)
        v0 = self.vel
        v_cmd = a * cfg.v_max
        dv = v_cmd - v0
        dv_n = np.linalg.norm(dv, axis=1, keepdims=True)
        dv = dv * np.minimum(1.0, cfg.a_max * cfg.dt / np.maximum(dv_n, 1e-9))
        v1 = v0 + dv
        sp = np.linalg.norm(v1, axis=1, keepdims=True)
        v1 = v1 * np.minimum(1.0, cfg.v_max / np.maximum(sp, 1e-9))
        p0 = self.pos[:, :2].copy()
        p_mid = p0 + (3.0 * v0 + v1) / 8.0 * cfg.dt          # position at t0 + dt/2
        p_end = p0 + 0.5 * (v0 + v1) * cfg.dt
        hit = (p_end < 0.0) | (p_end > cfg.area)
        p_mid = np.clip(p_mid, 0.0, cfg.area)
        p_end = np.clip(p_end, 0.0, cfg.area)
        v1 = np.where(hit, 0.0, v1)                              # stopped by the boundary
        # propulsion energy over the slot (composite Simpson on the linear velocity path)
        energy = ph.slot_energy(v0, v1, cfg.dt, cfg)
        if self.aerial:
            energy[self.K] *= cfg.ris.payload_factor
        self.vel = v1

        # links over the slot are evaluated at the mid-slot positions
        pos_mid = self.pos.copy()
        pos_mid[:, :2] = p_mid
        self.pos[:, :2] = p_end

        # collisions (3-D distance) counted before the safety projection
        n_coll = 0
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                diff = self.pos[i] - self.pos[j]
                dist = np.linalg.norm(diff)
                if dist < cfg.d_min:
                    n_coll += 1
                    dz = abs(diff[2])
                    h_req = np.sqrt(max(cfg.d_min ** 2 - dz ** 2, 0.0))   # horizontal gap needed
                    dxy = diff[:2]
                    nxy = np.linalg.norm(dxy)
                    u = dxy / nxy if nxy > 1e-6 else np.array([1.0, 0.0])
                    push = 0.5 * (h_req - nxy)
                    self.pos[i, :2] = np.clip(self.pos[i, :2] + u * push, 0.0, cfg.area)
                    self.pos[j, :2] = np.clip(self.pos[j, :2] - u * push, 0.0, cfg.area)

        bits, served, ris_targets, eff_gain = self._communicate(pos_mid)
        self.last_served = served
        # geometric blockage seen from the new positions (for the next observation)
        self.blocked = ph.segments_blocked(self.dev[:, None, :], self.pos[None, :self.K, :], self.boxes)
        self.ris_elig = self._ris_eligibility(self.pos)

        self.slot += 1
        done = self.slot >= cfg.M
        phi_new = self._potentials()
        shaping = cfg.w_final * float(np.sum(phi_new - self.phi))
        self.phi = phi_new

        bits_tot = float(bits.sum())
        e_tot = float(energy.sum())
        reward = (bits_tot - self.lam * e_tot) / (self.lam * self.e_ref_slot)
        reward += shaping - cfg.w_collision * n_coll

        self.ep["bits"] += bits_tot
        self.ep["energy"] += e_tot
        self.ep["collisions"] += n_coll
        self.ep["bits_shadow"] += float(bits[self.shadow].sum())

        info = {
            "bits": bits_tot, "energy": e_tot, "collisions": n_coll, "overridden": overridden,
            "served": served, "ris_targets": ris_targets,
            "pos": self.pos.copy(), "remaining": self.remaining.copy(),
        }
        if done:
            info["episode"] = self.episode_metrics()
        obs = self._obs()
        return obs, obs.ravel().copy(), reward, done, info

    def episode_metrics(self) -> dict:
        cfg = self.cfg
        d_final = np.linalg.norm(self.pos[:, :2] - np.asarray(cfg.final_xy), axis=1)
        total = self.data0.sum()
        sh = self.shadow
        return {
            "ee": self.ep["bits"] / max(self.ep["energy"], 1e-9),
            "bits": self.ep["bits"],
            "energy": self.ep["energy"],
            "delivered_frac": self.ep["bits"] / total,
            "completed_frac": float(np.mean(self.remaining <= 1.0)),
            "shadow_delivered_frac": (self.ep["bits_shadow"] / self.data0[sh].sum()) if sh.any() else np.nan,
            "collisions": self.ep["collisions"],
            "reached_final": float(np.mean(d_final <= cfg.r_final)),
            "rth_frac": self.ep["rth_slots"] / (cfg.M * self.n_agents),
        }

    # ------------------------------------------------------------ communication

    def _communicate(self, pos):
        cfg = self.cfg
        t0 = (self.slot) * cfg.dt
        tmid = t0 + 0.5 * cfg.dt
        uav = pos[: self.K]
        h_d, _ = ph.direct_channel(self.dev, uav, cfg, self.boxes, self.rng)

        if self.use_ris:
            center = self._ris_center(pos)
            if self.aerial:
                normal = self._aerial_normal(pos)
                elems = ph.ris_elements(center, normal, cfg.ris.ny, cfg.ris.nz, self.spacing)
                h_ir, _ = ph.ris_side_channel(self.dev, elems, center, normal, cfg,
                                              self.boxes, self.rng, model="a2g")
            else:
                normal = cfg.ris.normal
                elems = self.ris_elems
                h_ir = self.h_ir_static
            h_ru, _ = ph.ris_side_channel(uav, elems, center, normal, cfg,
                                          self.boxes, self.rng, model="los")
        else:
            h_ir = np.zeros((self.I, 1), complex)
            h_ru = np.zeros((self.K, 1), complex)

        g_dir = np.abs(h_d) ** 2
        if cfg.arbitration in ("timeshare", "common"):
            g_best = np.maximum(g_dir, ph.best_aligned_gain(h_d, h_ru, h_ir))
        else:
            g_best = g_dir
        r_est = ph.rate_bps(g_best, cfg, self.p_w, self.noise_w)

        # deadline-aware greedy scheduling: most urgent device first, to its best UAV
        active = (self.win_start <= tmid) & (tmid < self.win_end) & (self.remaining > 1.0)
        served = np.full(self.I, -1)
        free = np.full(self.K, cfg.n_sub)
        if active.any():
            left = np.maximum(self.win_end - t0, cfg.dt)
            urgency = np.where(active, self.remaining / left, -np.inf)
            for i in np.argsort(-urgency):
                if not active[i]:
                    break
                cand = np.where(free > 0, r_est[i], -1.0)
                k = int(np.argmax(cand))
                if cand[k] <= 1e3:
                    continue
                served[i] = k
                free[k] -= 1
                if not free.any():
                    break

        # RIS configuration(s) for this slot
        thetas = None
        ris_targets = []
        if self.use_ris and cfg.arbitration != "none":
            idx = np.where(served >= 0)[0]
            if cfg.arbitration == "random":
                thetas = self.rng_ris.uniform(0.0, 2 * np.pi, (1, h_ir.shape[1]))
            else:
                k_of = served[idx]
                dep = idx[g_best[idx, k_of] > 2.0 * g_dir[idx, k_of]]
                left = np.maximum(self.win_end - t0, cfg.dt)
                order = dep[np.argsort(-(self.remaining[dep] / left[dep]))]
                if cfg.arbitration == "common":
                    order = order[:1]
                order = order[:MAX_RIS_CONFIGS]
                if len(order) == 0:
                    elig_srv = idx[np.abs(h_ir[idx]).sum(1) > 0] if len(idx) else idx
                    order = elig_srv[:1]
                ris_targets = [(int(i), int(served[i])) for i in order]
                if ris_targets:
                    thetas = np.stack([
                        ph.aligned_phases(h_d[i, k], h_ru[k], h_ir[i], cfg.ris.bits,
                                          cfg.ris.phase_err_deg, self.rng_ris)
                        for i, k in ris_targets])
        if thetas is None:
            gains = g_dir[None]
        else:
            gains = ph.composite(h_d, h_ru, h_ir, thetas)
        rates = ph.rate_bps(gains, cfg, self.p_w, self.noise_w).mean(axis=0)   # (I, K)
        eff_gain = gains.mean(axis=0)
        self.last_snr_db = 10.0 * np.log10(np.maximum(self.p_w * eff_gain / self.noise_w, 1e-3))

        bits = np.zeros(self.I)
        idx = np.where(served >= 0)[0]
        if len(idx):
            bits[idx] = np.minimum(self.remaining[idx], rates[idx, served[idx]] * cfg.dt)
            self.remaining[idx] -= bits[idx]
        return bits, served, ris_targets, eff_gain

    # ------------------------------------------------------------ observations

    def _obs(self) -> np.ndarray:
        cfg = self.cfg
        A = cfg.area
        tl = self._time_left()
        t0 = self.slot * cfg.dt
        final = np.asarray(cfg.final_xy)
        ris_c = self._ris_center(self.pos)
        alive = (self.remaining > 1.0) & (self.win_end > t0)
        active = alive & (self.win_start <= t0)
        dmax = cfg.data_mbit[1] * 1e6
        tnorm = np.where(active, (self.win_end - t0) / max(cfg.window_s, cfg.dt),
                         np.where(alive, -(self.win_start - t0) / cfg.T, 0.0))
        tnorm = np.clip(np.nan_to_num(tnorm, posinf=1.0), -1.0, 1.0)
        K = self.top_k
        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        for a in range(self.n_agents):
            p = self.pos[a, :2]
            f = [p[0] / A, p[1] / A, self.vel[a, 0] / cfg.v_max, self.vel[a, 1] / cfg.v_max, tl / cfg.T]
            to_f = final - p
            dist_f = np.linalg.norm(to_f)
            f += [to_f[0] / A, to_f[1] / A,
                  np.clip((cfg.v_max * tl - dist_f) / A, -1.0, 1.0)]
            for b in range(self.n_agents):
                if b != a:
                    f += list((self.pos[b, :2] - p) / A)
            f += list((ris_c[:2] - p) / A) if ris_c is not None else [0.0, 0.0]
            rel = (self.dev[:, :2] - p) / A
            dist = np.linalg.norm(rel, axis=1)
            is_ris_agent = a >= self.K
            if is_ris_agent:
                blocked_me = ph.segments_blocked(self.dev, self.pos[a][None, :], self.boxes)
                snr = self.last_snr_db.max(axis=1)
                key = dist - 0.5 * self.shadow + 0.3 * (~active)
            else:
                blocked_me = self.blocked[:, a]
                snr = self.last_snr_db[:, a]
                key = dist + 0.3 * (~active)
            key = np.where(alive, key, np.inf)
            top = np.argsort(key)[:K]
            dev_f = np.zeros((K, DEV_FEATS))
            for j, i in enumerate(top):
                if not alive[i]:
                    continue
                by_me = float(self.last_served[i] == a)
                by_other = float(self.last_served[i] >= 0 and self.last_served[i] != a)
                dev_f[j] = [rel[i, 0], rel[i, 1], self.remaining[i] / dmax, float(active[i]),
                            tnorm[i], float(blocked_me[i]), float(self.ris_elig[i]),
                            np.clip(snr[i] / 50.0, -1.0, 1.0), by_me, by_other]
            f += list(dev_f.ravel())
            f += [self.remaining.sum() / max(self.data0.sum(), 1.0), float(is_ris_agent)]
            if self.agent_id:   # lets a shared/averaged policy give each UAV its own role
                f += list(np.eye(self.n_agents)[a])
            obs[a] = f
        return obs
