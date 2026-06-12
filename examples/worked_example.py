"""Worked example: PIV on the Elman-RNN case study (~30 lines).

Trains a small population, applies the pre-registered certificate blind,
and runs one targeted surgery with matched controls.
GPU: pass device="cuda". Full populations: scripts/run_all.py / `make all`.
"""
import piv
from piv.casestudy import (CERTIFICATE, certify, matched_controls,
                           persistence, train_population, weight_surgery)

device = "cpu"

# --- Predict: pre-register, then evaluate blind ------------------------------
print("frozen certificate:", piv.freeze(CERTIFICATE))
dev_pop, _ = train_population(range(42, 52), k=3, H=8, epochs=2000,
                              device=device)
blind_pop, _ = train_population(range(200, 210), k=3, H=8, epochs=2000,
                                device=device)
pers = persistence(blind_pop, device=device)
cert = certify(blind_pop, device=device)
y = (pers >= 0.99)
tp = int((cert["certified"] & y).sum())
print(f"blind: {int(cert['certified'].sum())} certified, "
      f"{tp} true positives, precision CI "
      f"{piv.wilson_ci(tp, max(int(cert['certified'].sum()), 1))}")

# --- Intervene: targeted surgery vs matched controls -------------------------
m = int(cert["certified"].nonzero()[0]) if cert["certified"].any() else 0
d = int(cert["iv"][m].argmax())
cut = persistence(weight_surgery(blind_pop, d), device=device)
c1, c2 = matched_controls(blind_pop, d)
print(f"surgery on tight dim {d} of model {m}: "
      f"persistence {float(pers[m]):.3f} -> {float(cut[m]):.3f}; "
      f"controls -> {float(persistence(c1, device=device)[m]):.3f}, "
      f"{float(persistence(c2, device=device)[m]):.3f}")

# --- Validate: power analysis for the OOD stage ------------------------------
print("models needed to detect 80% vs 0% OOD success:",
      piv.sample_size_two_proportions(0.8, 0.0001))
