"""PIV — Predict · Intervene · Validate.

Statistical utilities for validating mechanistic claims in neural networks.

Paper: "A Statistical Framework for Mechanistic Claims in Neural Networks:
The Predict--Intervene--Validate Pipeline", ICML 2026 Workshop on Hypothesis
Testing. https://openreview.net/forum?id=o8aRR9YuxE

Pure-stdlib implementation (no scipy/numpy required).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from typing import Mapping, Sequence

__version__ = "0.1.0"
__all__ = [
    "wilson_ci",
    "sample_size_correlation",
    "sample_size_two_proportions",
    "sample_size_cohens_d",
    "binomial_tail_p",
    "beta_binomial_tail_p",
    "cluster_bootstrap_p",
    "freeze",
    "checklist",
]


# ----------------------------------------------------------------- normal dist

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# ------------------------------------------------------------------- intervals

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score 95% (default) confidence interval for a binomial rate."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must be in [0, n]")
    z = _norm_ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# -------------------------------------------------------------- power analysis

def sample_size_correlation(r: float, power: float = 0.80,
                            alpha: float = 0.05) -> int:
    """Minimum n to detect Pearson correlation r (two-sided, Fisher z)."""
    if not 0 < abs(r) < 1:
        raise ValueError("r must be in (0, 1)")
    z_a = _norm_ppf(1 - alpha / 2)
    z_b = _norm_ppf(power)
    fz = 0.5 * math.log((1 + abs(r)) / (1 - abs(r)))
    return math.ceil(((z_a + z_b) / fz) ** 2 + 3)


def sample_size_two_proportions(p1: float, p2: float, power: float = 0.80,
                                alpha: float = 0.05) -> int:
    """Minimum n per group to detect p1 vs p2 (two-sided, normal approx)."""
    if not (0 <= p1 <= 1 and 0 <= p2 <= 1) or p1 == p2:
        raise ValueError("p1, p2 must be distinct rates in [0, 1]")
    z_a = _norm_ppf(1 - alpha / 2)
    z_b = _norm_ppf(power)
    pbar = (p1 + p2) / 2
    num = (z_a * math.sqrt(2 * pbar * (1 - pbar)) +
           z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)


def sample_size_cohens_d(d: float, power: float = 0.80,
                         alpha: float = 0.05) -> int:
    """Minimum n per group to detect effect size d (two-sample t, two-sided).

    Normal approximation with Guenther's (1981) small-sample correction
    (+ z_alpha^2 / 4), matching exact t-test tables (e.g. d=0.8 -> 26).
    """
    if d == 0:
        raise ValueError("d must be nonzero")
    z_a = _norm_ppf(1 - alpha / 2)
    z_b = _norm_ppf(power)
    return math.ceil(2 * ((z_a + z_b) / abs(d)) ** 2 + z_a ** 2 / 4)


# ----------------------------------------------------------------- null models

def binomial_tail_p(k: int, n: int, p0: float) -> float:
    """Exact binomial upper-tail P(X >= k | n, p0) via log-space summation."""
    if not 0 <= k <= n:
        raise ValueError("k must be in [0, n]")
    if p0 <= 0:
        return 0.0 if k > 0 else 1.0
    if p0 >= 1:
        return 1.0
    total = 0.0
    for i in range(k, n + 1):
        log_term = (math.lgamma(n + 1) - math.lgamma(i + 1) -
                    math.lgamma(n - i + 1) +
                    i * math.log(p0) + (n - i) * math.log(1 - p0))
        total += math.exp(log_term)
    return min(1.0, total)


def beta_binomial_tail_p(k: int, n: int, a: float, b: float) -> float:
    """Upper-tail P(X >= k) under Beta-Binomial(n, a, b) (over-dispersed null)."""
    if a <= 0 or b <= 0:
        raise ValueError("a, b must be positive")
    total = 0.0
    log_beta_ab = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    for i in range(k, n + 1):
        log_term = (math.lgamma(n + 1) - math.lgamma(i + 1) -
                    math.lgamma(n - i + 1) +
                    math.lgamma(i + a) + math.lgamma(n - i + b) -
                    math.lgamma(n + a + b) - log_beta_ab)
        total += math.exp(log_term)
    return min(1.0, total)


def cluster_bootstrap_p(successes_by_model: Sequence[Sequence[int]],
                        null_rate: float, n_boot: int = 10_000,
                        seed: int = 0) -> float:
    """Cluster bootstrap (resampling models) for H0: success rate <= null_rate.

    successes_by_model: per-model sequences of 0/1 outcomes (clusters).
    Returns the fraction of bootstrap resamples whose pooled success rate
    fails to exceed null_rate (one-sided), floored at 1/n_boot.
    """
    if not successes_by_model:
        raise ValueError("need at least one model cluster")
    rng = random.Random(seed)
    m = len(successes_by_model)
    hits = 0
    for _ in range(n_boot):
        tot = n = 0
        for _ in range(m):
            cluster = successes_by_model[rng.randrange(m)]
            tot += sum(cluster)
            n += len(cluster)
        if n == 0 or tot / n <= null_rate:
            hits += 1
    return max(hits, 1) / n_boot


# ------------------------------------------------------------ pre-registration

def freeze(obj: Mapping, path: str | None = None) -> dict:
    """Pre-register a predictor/certificate: canonical-JSON sha256 + timestamp.

    Returns {"sha256", "frozen_at_unix", "frozen_at_iso"}; optionally writes
    the record to `path`. Commit the record (or git-tag it) before generating
    any test data.
    """
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    record = {
        "sha256": hashlib.sha256(blob).hexdigest(),
        "frozen_at_unix": int(time.time()),
        "frozen_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"certificate": dict(obj), **record}, f, indent=2)
    return record


# --------------------------------------------------------------------- report

_CHECKLIST_ITEMS = (
    "Quantitative prediction pre-registered before test data",
    "Held-out test cohort with precision/recall/CI",
    ">= 5 alternative hypotheses tested and rejected",
    "Causal intervention with effect size (d) and p-value",
    "Confounds enumerated and mitigated",
    "OOD generalisation of the claim (not just the model)",
    "Sample-size justification via power analysis",
    "95% CIs on all reported rates",
)


def checklist(answers: Sequence[bool] | None = None) -> str:
    """Render the PIV reporting checklist as markdown."""
    answers = answers or [False] * len(_CHECKLIST_ITEMS)
    if len(answers) != len(_CHECKLIST_ITEMS):
        raise ValueError(f"expected {len(_CHECKLIST_ITEMS)} answers")
    lines = ["# PIV reporting checklist", ""]
    for done, item in zip(answers, _CHECKLIST_ITEMS):
        lines.append(f"- [{'x' if done else ' '}] {item}?")
    return "\n".join(lines)
