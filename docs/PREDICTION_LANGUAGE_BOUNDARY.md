# Prediction Language Boundary

## Decision

The manuscript will use association, molecular stratification, and retrospective clinical endpoint relevance language. It will not use predictor, prediction model, clinical test, diagnostic, or treatment-selection wording as the main claim.

## Reason

The current models are retrospective cohort-specific associations. They report odds ratios per 1 SD barrier-axis score and confidence intervals, but they do not yet include prespecified training/test splits, calibration, Brier score, decision-curve analysis, or prospective validation.

## Consequence for Outputs

`results/benchmarks/prediction_eval.tsv` is not generated in this change. If future writing uses predictive language, that table must be generated and must include discrimination, calibration, Brier score or equivalent, cohort/split, and sample size.

## Allowed Wording

- associated with mucosal healing
- associated with week-8 endoscopic and histologic healing
- age-stratified molecular stratification
- retrospective clinical endpoint association
- cohort-level endpoint relevance

## Forbidden Wording

- predicts response in clinical practice
- treatment-selection biomarker
- diagnostic test
- prospective clinical utility
- deployable assay
