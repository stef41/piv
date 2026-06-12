#!/usr/bin/env python3
"""Train every population for the PIV case study and dump artifacts.

Usage:
  python scripts/run_all.py --device cuda --out artifacts/

Populations (paper §2):
  dev       100 baseline Elman RNNs, k=3 H=8, seeds 42..141
  recipe    100 recipe-trained models (saturation penalty), same seeds
  blind     300 held-out baseline models, seeds 200..499
  ood       160 recipe models, k=12..25, H=60..100 (hyperparameter-frozen)

Tiny models; the whole thing fits on one GPU via model-batched training.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from piv.casestudy import (manifest_rows, save_artifacts,  # noqa: E402
                           train_population)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--epochs", type=int, default=10_000)
    ap.add_argument("--quick", action="store_true",
                    help="pilot: 10x fewer epochs, 10x fewer models")
    ap.add_argument("--only", default="",
                    help="comma list of cohorts to run (dev,recipe,blind,ood or ood_k12...)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    epochs = args.epochs // 10 if args.quick else args.epochs
    div = 10 if args.quick else 1
    only = [s.strip() for s in args.only.split(",") if s.strip()]
    want = lambda name: (not only) or any(
        name == o or (o == "ood" and name.startswith("ood_")) for o in only)
    t0 = time.time()

    def run(name, seeds, k, H, sat):
        if not want(name):
            return []
        print(f"[{time.time()-t0:7.0f}s] training {name} "
              f"(M={len(seeds)}, k={k}, H={H}, epochs={epochs})", flush=True)
        pop, _ = train_population(seeds, k=k, H=H, epochs=epochs,
                                  sat_lambda=sat, device=args.device,
                                  log_every=max(1, epochs // 4))
        rows = manifest_rows(pop, name, device=args.device)
        save_artifacts(pop, rows, os.path.join(args.out, name))
        ng = sum(r["is_generalizer"] for r in rows)
        nc = sum(r["certified"] for r in rows)
        print(f"  -> {ng}/{len(rows)} generalise, {nc}/{len(rows)} certified",
              flush=True)
        return rows

    run("dev", range(42, 142)[: 100 // div], k=3, H=8, sat=0.0)
    run("recipe", range(42, 142)[: 100 // div], k=3, H=8, sat=0.1)
    run("blind", range(200, 500)[: 300 // div], k=3, H=8, sat=0.0)

    # OOD: k=12..25 with H~5k (frozen hyperparameters, recipe on)
    ood_rows = []
    ks = list(range(12, 26))
    per_k = max(1, (160 // div) // len(ks))
    seed0 = 1000
    for k in ks:
        H = min(100, max(60, 5 * k))
        seeds = range(seed0, seed0 + per_k)
        seed0 += per_k
        rows = run(f"ood_k{k}", seeds, k=k, H=H, sat=0.1)
        ood_rows += rows
    if ood_rows:
        with open(os.path.join(args.out, "ood_manifest.jsonl"), "w") as f:
            for r in ood_rows:
                f.write(json.dumps(r) + "\n")
    print(f"[{time.time()-t0:7.0f}s] done -> {args.out}/", flush=True)


if __name__ == "__main__":
    main()
