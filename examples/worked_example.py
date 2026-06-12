"""Worked example: PIV on a synthetic population (~30 lines).

Demonstrates the API end-to-end on synthetic data. The RNN case-study
reproduction (real models + data) ships with the artifact release; see the
"Artifact status" table in the README.
"""
import random

import piv

rng = random.Random(42)

# --- Predict: pre-register a certificate, then evaluate blind ---------------
certificate = {"score": "max_input_invariance", "threshold": 0.8}
print("pre-registration:", piv.freeze(certificate))

n = 300
labels = [rng.random() < 0.96 for _ in range(n)]                  # generaliser?
scores = [0.9 if y and rng.random() < 0.92 else 0.3 for y in labels]
preds = [s >= certificate["threshold"] for s in scores]
tp = sum(p and y for p, y in zip(preds, labels))
print("precision CI:", piv.wilson_ci(tp, sum(preds)))
print("n for r>=0.3:", piv.sample_size_correlation(0.3))           # 85

# --- Intervene: cluster-bootstrap null over models ---------------------------
breaks_by_model = [[1] * 4 for _ in range(100)]                    # 393-ish breaks
print("surgery p (cluster bootstrap):",
      piv.cluster_bootstrap_p(breaks_by_model, null_rate=0.5))

# --- Validate: over-dispersed OOD null ---------------------------------------
print("OOD p (beta-binomial):", piv.beta_binomial_tail_p(160, 160, a=1.2, b=8.9))
print()
print(piv.checklist([True] * 8))
