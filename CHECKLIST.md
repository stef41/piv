# The PIV Checklist for Mechanistic Interpretability Submissions

Use this checklist when reporting a population-level mechanistic claim
("mechanism X causes behaviour Y"). Each item maps to a section of the PIV
paper (ICML 2026 Hypothesis Testing Workshop).

## Predict — predictive validity

- [ ] **Pre-registration.** The quantitative prediction (certificate, score,
  threshold) was frozen *before* the test population was generated. The
  freezing event is verifiable (git tag, hash + timestamp; `piv.freeze`).
- [ ] **Held-out cohort.** Evaluation on models never used during mechanism
  discovery or predictor development.
- [ ] **Full operating characteristics.** Precision, recall, specificity,
  balanced accuracy, AUROC — not precision alone (prevalence can make
  precision uninformative).
- [ ] **Baselines.** Always-positive, prevalence-matched random, and at least
  one simple single-feature baseline.
- [ ] **Wilson 95% CIs** on every reported rate.
- [ ] **Alternative hypotheses.** ≥ 5 plausible alternative predictors tested
  and their failure documented, with multiple-comparison correction.

## Intervene — causal necessity

- [ ] **Matched controls.** Every targeted intervention paired with (i) a
  random direction matched in norm and (ii) a matched-rank random
  perturbation.
- [ ] **Dose-response.** Intervention swept over a dose grid; monotonicity
  reported.
- [ ] **Effect sizes.** Cohen's d / odds ratios alongside p-values.
- [ ] **Null model stated.** The exact null (e.g. Bernoulli at
  control-inflated error rate, Bonferroni over doses) is spelled out; a
  cluster bootstrap over models relaxes i.i.d. assumptions.
- [ ] **Confounds enumerated.** Off-target effects quantified
  (matched-control effect, targeted/off-target ratio) and their scaling
  behaviour reported.
- [ ] **≥ 2 operators.** Agreement across weight surgery / activation
  ablation / LoRA counter-finetune is evidence against off-target confounds.

## Validate — generality

- [ ] **OOD transfer of the claim.** The mechanism, identified in one regime,
  predicts behaviour in a regime never used during discovery
  (hyperparameter-frozen).
- [ ] **Null model stated.** E.g. beta-binomial with model-level
  over-dispersion, fit on the training population.
- [ ] **Negative controls.** Populations where the mechanism predicts failure.
- [ ] **Independent replication** on at least one held-out population.

## Design

- [ ] **Power analysis.** Sample sizes justified at 80% power, α = 0.05
  (`piv.sample_size_*`). Reference minima: r ≥ 0.3 → n ≥ 85; precision CI
  width < 0.10 → n ≥ 141; Fisher 50% vs 0% break rate → n ≥ 12; OOD 80% vs
  0% → n ≥ 8 per condition.
