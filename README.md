# PIV — Predict · Intervene · Validate

A statistical framework for validating mechanistic claims in neural networks.

**Paper:** *A Statistical Framework for Mechanistic Claims in Neural Networks:
The Predict–Intervene–Validate Pipeline.* ICML 2026 Workshop on Hypothesis
Testing (Submission 11, accepted).
[OpenReview](https://openreview.net/forum?id=o8aRR9YuxE)

## The pipeline

Any claim of the form "mechanism X causes behaviour Y" decomposes into three
falsifiable stages:

1. **Predict** — derive a quantitative prediction from the mechanism, *freeze
   it before generating test data* (pre-registration), and evaluate blind on a
   held-out population. Report precision/recall/specificity/balanced
   accuracy/AUROC with Wilson 95% CIs, against simple baselines.
2. **Intervene** — perform targeted interventions (weight surgery, activation
   ablation, rank-1 LoRA counter-finetune) with matched controls and
   dose-response analysis. Report effect sizes and confounds.
3. **Validate** — test OOD generalisation of the *claim* (not just the model)
   with hyperparameter-frozen transfer.

## Install

```bash
pip install git+https://github.com/stef41/piv        # stdlib core
pip install torch                                     # for piv.casestudy
```

```python
import piv

piv.wilson_ci(265, 271)                    # (0.952, 0.990)
piv.sample_size_correlation(r=0.3)         # 85
piv.freeze(my_certificate_dict)            # sha256 + timestamp pre-registration
```

## Reproduce the case study

```bash
make pilot               # 10x-reduced smoke run, CPU
make all DEVICE=cuda     # full populations from raw seeds (GPU)
```

`scripts/run_all.py` trains the dev (100), recipe (100), blind (300) and
OOD k=12-25 (160) populations as batched tensor programs;
`scripts/blind_baselines.py` produces the baselines table + ROC;
`scripts/surgery.py` runs confound-controlled weight surgery with matched
controls and the cluster-bootstrap null. The demo notebook
(`notebooks/demo_blind_prediction.ipynb`) re-runs blind prediction on a
single CPU in under 4 minutes.

## PIV reporting checklist

For mechanistic interpretability submissions (see `CHECKLIST.md`):

- [ ] Quantitative prediction pre-registered before test data?
- [ ] Held-out test cohort with precision/recall/CI?
- [ ] ≥ 5 alternative hypotheses tested and rejected?
- [ ] Causal intervention with effect size (d) and p-value?
- [ ] Confounds enumerated and mitigated?
- [ ] OOD generalisation of the *claim* (not just the model)?
- [ ] Sample-size justification via power analysis?
- [ ] 95% CIs on all reported rates?

## Artifact status

| Artifact | Status |
|---|---|
| `piv` package (power analysis, CIs, pre-registration, nulls) | released |
| `piv.casestudy` (Elman-RNN populations, certificate, surgery) | released |
| PIV checklist | released (`CHECKLIST.md`) |
| Demo notebook (blind prediction, <4 min CPU) | released (`notebooks/`) |
| `make all` reproduction from raw seeds | released |
| Frozen certificate | `certificate.json`, git tag `certificate-freeze` |
| Blind-prediction models (300) + OOD models (160) + manifests | GitHub release `artifacts-v1` (regenerable via `make all`) |

**Provenance note.** This repository is a from-scratch reimplementation of
the paper's case study, released at camera-ready time; the git history
(including the `certificate-freeze` tag) is dated accordingly. The
certificate parameters are fixed by `certificate.json` before any artifact
in the release was generated, and every population regenerates
deterministically from the seed ranges in `scripts/run_all.py`.

## Citation

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

## License

MIT
