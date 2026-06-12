"""Minimal test suite for piv (stdlib-only parts run without torch)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import piv


def test_wilson():
    lo, hi = piv.wilson_ci(265, 271)
    assert 0.95 < lo < hi <= 1.0


def test_power():
    assert piv.sample_size_correlation(0.3) == 85
    assert piv.sample_size_cohens_d(0.8) == 26
    assert piv.sample_size_two_proportions(0.5, 0.0001) <= 12


def test_nulls():
    assert piv.binomial_tail_p(160, 160, 0.119) < 1e-90
    assert 0 < piv.beta_binomial_tail_p(160, 160, 1.2, 8.9) < 1
    assert piv.cluster_bootstrap_p([[1] * 4] * 50, 0.5, n_boot=500) <= 0.002


def test_freeze():
    r = piv.freeze({"tau": 0.8})
    assert len(r["sha256"]) == 64 and r["frozen_at_unix"] > 0


def test_casestudy_smoke():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch not installed; skipping case-study smoke test")
        return
    from piv.casestudy import (certify, labels_for, persistence,
                               train_population)
    import torch
    x = torch.tensor([[[1., 1., 0., 1.]]])
    assert labels_for(x, 3).tolist() == [[[0., 1., 1., 1.]]]
    pop, _ = train_population(range(3), k=3, H=8, epochs=300, batch=64)
    pers = persistence(pop, trials=100, horizon=100)
    cert = certify(pop)
    assert pers.shape == (3,) and cert["score"].shape == (3,)


if __name__ == "__main__":
    for fn in [test_wilson, test_power, test_nulls, test_freeze,
               test_casestudy_smoke]:
        fn()
        print(f"ok {fn.__name__}")
    print("all tests pass")
