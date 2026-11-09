"""Small statistics helpers (no scipy): sample std with ddof=1, Student-t 95 % confidence
intervals, paired differences and a percentile bootstrap."""
from __future__ import annotations

import numpy as np

# two-sided 97.5 % quantiles of Student's t for df = 1..30; larger df -> normal-ish table below
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
         2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
         2.052, 2.048, 2.045, 2.042]


def t975(df: int) -> float:
    if df < 1:
        return float("nan")
    if df <= 30:
        return _T975[df - 1]
    if df <= 60:
        return 2.042 + (2.000 - 2.042) * (df - 30) / 30
    if df <= 120:
        return 2.000 + (1.980 - 2.000) * (df - 60) / 60
    return 1.960


def mean_ci(x) -> dict:
    """Mean, sample std (ddof=1), n and the half-width of the 95 % t confidence interval."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    return {"mean": float(np.mean(x)) if n else float("nan"), "std": sd, "n": n,
            "ci95": t975(n - 1) * sd / np.sqrt(n) if n > 1 else float("nan")}


def paired(a, b) -> dict:
    """Paired comparison of a and b on the same episodes / seeds: mean of (a - b) with its 95 % CI,
    and the relative change of the means (a/b - 1) in %."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = mean_ci(a - b)
    d["rel_pct"] = 100.0 * (np.mean(a) / np.mean(b) - 1.0)
    d["frac_a_better"] = float(np.mean(a > b))
    return d


def bootstrap_ci(x, stat=np.mean, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    s = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))
