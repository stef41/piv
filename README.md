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
pip install git+https://github.com/stef41/piv
```

```python
import piv

piv.wilson_ci(265, 271)                    # (0.952, 0.990)
piv.sample_size_correlation(r=0.3)         # 85
piv.freeze(my_certificate_dict)            # sha256 + timestamp pre-registration
```

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
| `piv` package (power analysis, CIs, pre-registration, nulls) | released (this repo) |
| PIV checklist | released (`CHECKLIST.md`) |
| 300 held-out blind-prediction models | uploading (GitHub release) |
| 160 OOD transfer models + JSON manifest | uploading (GitHub release) |
| Demo notebook (blind prediction, <4 min CPU) | uploading |
| `make all` reproduction target | uploading |

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
