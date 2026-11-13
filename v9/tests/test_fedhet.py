import numpy as np
import torch

from v9.config import HET_TEST, HET_TRAIN, HET_VAL, area_is_valid, hetero_area, random_area
from v9.env import RISUAVEnv
from v9.fedhet import compress_update


def test_het_area_sets_disjoint_valid_and_heterogeneous():
    assert not (set(HET_TRAIN) & set(HET_VAL) or set(HET_TRAIN) & set(HET_TEST) or set(HET_VAL) & set(HET_TEST))
    cfgs = [hetero_area(s) for s in HET_TRAIN]
    assert all(area_is_valid(random_area(s)) for s in HET_TRAIN + HET_VAL + HET_TEST)
    assert len({c.n_devices for c in cfgs}) > 3 and len({c.ris.ny for c in cfgs}) > 2
    dims = {RISUAVEnv(c).obs_dim for c in cfgs}
    assert len(dims) == 1          # one network fits every area


def test_compression_bytes_and_error():
    torch.manual_seed(0)
    d = {"w": torch.randn(200, 50), "b": torch.randn(50)}
    n = sum(v.numel() for v in d.values())
    full, nb, _ = compress_update(d, "none", None)
    assert nb == 4 * n and torch.equal(full["w"], d["w"])
    q, nbq, _ = compress_update(d, "q8", None)
    assert nbq == n + 8 and (q["w"] - d["w"]).abs().max() <= d["w"].abs().max() / 127 * 0.51 + 1e-6
    s, nbs, res = compress_update(d, "top10", None)
    assert nbs < 0.25 * nb and (s["w"] != 0).sum() == int(0.1 * d["w"].numel())
    assert torch.allclose(s["w"] + res["w"], d["w"])           # nothing is lost: kept in the residual


def test_top10_error_feedback_transmits_everything_eventually():
    d = {"w": torch.ones(100)}
    res, sent = None, torch.zeros(100)
    for _ in range(10):
        s, _, res = compress_update({"w": torch.zeros(100)} if sent.sum() > 0 else d, "top10", res)
        sent += s["w"]
    assert torch.allclose(sent, torch.ones(100))
