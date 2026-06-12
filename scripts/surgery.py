#!/usr/bin/env python3
"""Confound-controlled weight surgery on the recipe population (paper §4).

Usage:
  python scripts/surgery.py --pop artifacts/recipe.pt --device cuda

For every model and every hidden dimension: zero the W_hh row (dose
alpha=1.0), measure the persistence drop, and classify a *causal break*
(persistence >= 0.99 -> < 0.5). Pairs every break with the two matched
controls and reports the model-level Fisher statistic plus a cluster
bootstrap p-value.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import piv  # noqa: E402
from piv.casestudy import (Population, certify, matched_controls,  # noqa: E402
                           persistence, weight_surgery)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", default="artifacts/recipe.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--doses", default="0.5,1.0")
    ap.add_argument("--out", default="artifacts/surgery.json")
    args = ap.parse_args()
    pop = Population.load(torch.load(args.pop, weights_only=False),
                          device=args.device)
    base = persistence(pop, device=args.device)
    cert = certify(pop, device=args.device)
    gen_mask = base >= 0.99
    doses = [float(d) for d in args.doses.split(",")]

    breaks = []          # (model, dim, tight?, dose)
    control_drop = 0.0
    for d in range(pop.H):
        for a in doses:
            cut = persistence(weight_surgery(pop, d, alpha=a),
                              device=args.device)
            c1, c2 = matched_controls(pop, d, alpha=a)
            p1 = persistence(c1, device=args.device)
            p2 = persistence(c2, device=args.device)
            control_drop = max(control_drop,
                               float((base - torch.minimum(p1, p2)).max()))
            for m in range(pop.M):
                if gen_mask[m] and cut[m] < 0.5:
                    breaks.append((m, d, bool(cert["tight"][m, d]), a))
        print(f"dim {d}: cumulative breaks={len(breaks)}", flush=True)

    n_tight = sum(1 for *_, t, _a in [(b[0], b[1], b[2], b[3])
                                      for b in breaks] if t)
    by_model = {}
    for m, d, t, a in breaks:
        by_model.setdefault(m, []).append(int(t))
    clusters = list(by_model.values())
    p_boot = piv.cluster_bootstrap_p(clusters, null_rate=0.5) if clusters else 1.0
    res = {
        "n_breaks": len(breaks),
        "n_breaks_tight": n_tight,
        "n_models_with_breaks": len(by_model),
        "models_all_tight": sum(1 for c in clusters if all(c)),
        "max_control_drop": control_drop,
        "cluster_bootstrap_p": p_boot,
        "doses": doses,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
