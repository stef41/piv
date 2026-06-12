# PIV

**Predict. Intervene. Validate.**

[![tests](https://github.com/stef41/piv/actions/workflows/test.yml/badge.svg)](https://github.com/stef41/piv/actions/workflows/test.yml)
[![python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/github/license/stef41/piv)](LICENSE)

Your mechanism explains the model after the fact. Would it survive a
prediction it had to make in advance?

PIV is a protocol (and a small, dependency-free Python toolkit) for testing
"feature X causes behaviour Y" like the statistical claim it is:

1. **Predict.** Freeze a quantitative predictor *before* the test population
   exists. Score it blind: AUROC, precision, recall, Wilson CIs, against
   dumb baselines.
2. **Intervene.** Remove the mechanism with matched controls (norm-matched
   random direction, matched-rank noise) over a dose grid. Report effect
   sizes, not just p-values.
3. **Validate.** Re-test the claim, not the model, in a regime never used
   during discovery: frozen hyperparameters, scaled task.

From the ICML 2026 Hypothesis Testing Workshop paper
[*A Statistical Framework for Mechanistic Claims in Neural Networks*](https://openreview.net/forum?id=o8aRR9YuxE).

## Install

```bash
pip install git+https://github.com/stef41/piv
```

The core is pure stdlib: no numpy, no scipy. Only the optional
`piv.casestudy` module (batched RNN populations) needs `torch`.

## Sixty seconds

```python
import piv

piv.sample_size_correlation(0.3)   # 85 models to detect r >= 0.3 at 80% power
piv.wilson_ci(265, 271)            # (0.952, 0.990)

# Pre-register your predictor before generating any test data
piv.freeze({"score": "max_iv", "tau": 0.8}, path="certificate.json")
# {'sha256': 'e099c617...', 'frozen_at_iso': '2026-06-12T18:17:22Z'}
# commit it, tag it, then generate the test set

# Nulls that respect how interp data actually clusters
piv.cluster_bootstrap_p(outcomes_by_model, null_rate=0.5)  # resample models, not sequences
piv.beta_binomial_tail_p(160, 160, a=1.2, b=8.9)           # over-dispersed OOD null
```

## How many models do you need?

The first question every population-level interp study has to answer, and
the one almost none do. At 80% power, alpha = 0.05:

| You want to detect | Test | Models |
|---|---|---:|
| Correlation r >= 0.3 | Pearson via Fisher z | **85** |
| Precision >= 90% vs chance | exact binomial | **50** |
| Precision pinned to within 5 points | Wilson CI width | **141** |
| Effect size d >= 0.8 | two-sample t | **26** |
| Break rate 50% vs 0% | Fisher exact | **12** |
| OOD success 80% vs 0% | Fisher exact | **8** |

Every number is computed by this library and checked in CI
([tests/test_piv.py](tests/test_piv.py)).

## The checklist

Eight boxes for authors and reviewers. Full version with per-stage detail in
[CHECKLIST.md](CHECKLIST.md).

- [ ] Prediction pre-registered before test data?
- [ ] Held-out cohort with precision/recall/CI?
- [ ] 5+ alternative hypotheses tested and rejected?
- [ ] Intervention with effect size and p-value?
- [ ] Confounds enumerated and mitigated?
- [ ] OOD generalisation of the claim, not just the model?
- [ ] Sample size justified by power analysis?
- [ ] 95% CIs on every reported rate?

## Case study: latch dimensions in Elman RNNs

A complete instantiation ships with the repo. Populations of small RNNs
train as one batched tensor program (the model index is just a leading
dimension), so hundreds of models fit on a single GPU. On top of that:
a frozen input-invariance certificate, weight surgery with two matched
controls, and OOD scaling at frozen hyperparameters.

```bash
pip install torch
make pilot              # 10x-reduced run, fine on a laptop CPU
make all DEVICE=cuda    # full populations from raw seeds
```

Or open [notebooks/demo_blind_prediction.ipynb](notebooks/demo_blind_prediction.ipynb)
to run the Predict stage end to end in under four minutes on CPU.

## Provenance

This repository is a from-scratch reimplementation of the paper's case
study, released at camera-ready time. Everything regenerates
deterministically from the seed ranges in
[scripts/run_all.py](scripts/run_all.py). The certificate was fixed
([certificate.json](certificate.json), git tag `certificate-freeze`) before
any released artifact was generated. Rerun results ship with the artifacts
release and are reported as measured.

## Cite

```bibtex
@inproceedings{bugaud2026piv,
  title     = {A Statistical Framework for Mechanistic Claims in Neural
               Networks: The Predict--Intervene--Validate Pipeline},
  author    = {Bugaud, Zacharie},
  booktitle = {ICML 2026 Workshop on Hypothesis Testing},
  year      = {2026},
  url       = {https://openreview.net/forum?id=o8aRR9YuxE}
}
```

MIT.
