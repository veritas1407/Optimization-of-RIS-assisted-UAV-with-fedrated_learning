import numpy as np
import pytest
import torch

from v9 import physics as ph
from v9.config import TEST_AREAS, TRAIN_AREAS, VAL_AREAS, random_area
from v9.env import RISUAVEnv
from v9.multiarea import _avg_into
from v9.trainers import Learner, TrainConfig

ALL = TRAIN_AREAS + VAL_AREAS + TEST_AREAS


def test_area_sets_are_disjoint():
    assert not (set(TRAIN_AREAS) & set(VAL_AREAS) or set(TRAIN_AREAS) & set(TEST_AREAS)
                or set(VAL_AREAS) & set(TEST_AREAS))


@pytest.mark.parametrize("seed", ALL)
def test_generated_area_is_physically_valid(seed):
    cfg = random_area(seed)
    boxes = ph.boxes_array(cfg.buildings)
    ris = np.asarray(cfg.ris.center)
    x0, y0, x1, y1 = cfg.shadow_region
    dev = np.array([[(x0 + x1) / 2, (y0 + y1) / 2, 0.0]])
    assert not ph.inside_any_box(ris[None, :2], boxes)[0]
    assert not ph.segments_blocked(dev, ris[None], boxes)[0]          # RIS sees the courtyard
    assert ph.in_front(dev, ris, cfg.ris.normal)[0]
    far = np.array([[cfg.area / 2, cfg.area / 2, cfg.uav_alt]])        # UAV over the flight diagonal
    assert ph.segments_blocked(dev, far, boxes)[0]                     # direct link is shadowed
    assert all(0 <= v <= cfg.area for b in cfg.buildings for v in (b.x0, b.x1, b.y0, b.y1))


def test_training_areas_cover_all_layout_types():
    types = set()
    for s in TRAIN_AREAS:
        c = random_area(s)
        x, y, _ = c.ris.center
        x0, y0, x1, y1 = c.shadow_region
        types.add(("NW" if y > x else "SE", "x" if (x1 - x0) > (y1 - y0) else "y"))
    assert len(types) == 4


def test_env_can_switch_areas():
    env = RISUAVEnv(random_area(TRAIN_AREAS[0]))
    env.reset(seed=0)
    env.set_scenario(random_area(TRAIN_AREAS[1]))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (env.n_agents, env.obs_dim)
    assert np.allclose(env.ris_center, random_area(TRAIN_AREAS[1]).ris.center)


def test_actor_only_aggregation_keeps_critics_local():
    learners = [Learner(RISUAVEnv(random_area(s)), TrainConfig(algo="ctde")) for s in TRAIN_AREAS[:2]]
    with torch.no_grad():
        for i, lr in enumerate(learners):
            for p in list(lr.actors[0].parameters()) + list(lr.critic.parameters()):
                p.fill_(float(i))
    _avg_into(learners, "actor")
    for i, lr in enumerate(learners):
        assert all(torch.allclose(p, torch.full_like(p, 0.5)) for p in lr.actors[0].parameters())
        assert all(torch.allclose(p, torch.full_like(p, float(i))) for p in lr.critic.parameters())
