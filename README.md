# LAB-24
# Causal ML — DML and Causal Forests for Policy Evaluation

## Objective
Estimate the causal effect of 401(k) eligibility on household net financial
assets using Double Machine Learning and Causal Forest methods, and
characterize treatment effect heterogeneity across the income distribution.

## Methodology
- **Diagnosed and repaired a broken DML implementation** with three bugs:
  in-sample prediction (data leakage in cross-fitting), missing treatment
  residualization, and an incorrect scalar mean used in place of the
  IV-style ratio estimator — verified the fix recovers the true ATE of 5.0
  on a simulated DGP with known ground truth.
- **Estimated the ATE of 401(k) eligibility** on net total financial assets
  using `DoubleMLPLR` with Random Forest nuisance learners (`n_estimators=200`,
  `max_depth=5`) and 5-fold cross-fitting, recovering an ATE of approximately
  \$8,000–\$10,000 (statistically significant, p < 0.05).
- **Assessed robustness to unmeasured confounders** via sensitivity analysis
  (`cf_y=0.03`, `cf_d=0.03`); the positive estimate survived the assumed
  confounding bounds with the full confidence interval remaining above zero.
- **Fit a `CausalForestDML`** (EconML) with 500 causal trees and 5-fold
  cross-fitting to generate individual-level CATE predictions, recovering
  a mean CATE consistent with the DML ATE and a standard deviation
  revealing substantial treatment effect heterogeneity.
- **Compared subgroup DML (quartile-level) to the Causal Forest
  (individual-level)**: variance decomposition showed that within-quartile
  CATE variation exceeded between-quartile variation, meaning income
  quartile labels alone are an insufficient summary of who benefits most
  from 401(k) access.

## Key Findings
Double Machine Learning estimates that 401(k) eligibility raises net
financial assets by approximately \$8,000–\$10,000 on average, a result
that is statistically significant and robust to moderate unmeasured
confounding. The Causal Forest reveals that this average masks
substantial individual-level heterogeneity: the distribution of CATEs
spans a wide range, and a meaningful share of high-response individuals
are not concentrated in the top income quartile — they are distributed
across the income spectrum in ways that quartile-level subgroup analysis
cannot detect. This finding has direct policy relevance: a targeting
strategy based solely on income quartile would misallocate outreach
resources relative to one that leverages the full CATE surface estimated
by the Causal Forest.

## Stack
Python · DoubleML · EconML · scikit-learn · pandas · matplotlib · numpy

## Reproducibility
All random states fixed at 42. Cross-fitting uses 5 folds throughout.
Nuisance models: `RandomForestRegressor(n_estimators=200, max_depth=5)`.
