#!/usr/bin/env python3
"""Fit endpoint models for the barrier-axis score."""

from __future__ import annotations

import csv
import math
import os

import pandas as pd
import statsmodels.formula.api as smf


INPUT = "data/processed/GSE73661/baseline_endpoint.tsv"
OUTPUT = "results/clinical/endpoint_models.tsv"


def fit_logistic(data: pd.DataFrame, formula: str, model_name: str) -> dict[str, str]:
    model = smf.logit(formula, data=data).fit(disp=False)
    term = "axis_score_z"
    coef = model.params[term]
    se = model.bse[term]
    pvalue = model.pvalues[term]
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se
    return {
        "dataset_id": "GSE73661",
        "endpoint": "mucosal_healing",
        "model_formula": formula,
        "covariates": "none" if model_name == "unadjusted" else "treatment_family",
        "effect_type": "odds_ratio_per_1sd_axis_score",
        "effect": f"{math.exp(coef):.6g}",
        "ci_lower": f"{math.exp(ci_low):.6g}",
        "ci_upper": f"{math.exp(ci_high):.6g}",
        "pvalue": f"{pvalue:.6g}",
        "fdr": f"{pvalue:.6g}",
        "n": str(int(model.nobs)),
        "interpretation_boundary": "retrospective_baseline_expression_association; not prospective clinical utility",
    }


def main() -> int:
    data = pd.read_csv(INPUT, sep="\t")
    data["axis_score"] = pd.to_numeric(data["axis_score"], errors="coerce")
    data["mucosal_healing"] = pd.to_numeric(data["mucosal_healing"], errors="coerce")
    data = data.dropna(subset=["axis_score", "mucosal_healing"]).copy()
    sd = data["axis_score"].std(ddof=1)
    data["axis_score_z"] = (data["axis_score"] - data["axis_score"].mean()) / sd

    rows = [
        fit_logistic(data, "mucosal_healing ~ axis_score_z", "unadjusted"),
    ]
    if data["treatment_family"].nunique() > 1:
        rows.append(fit_logistic(data, "mucosal_healing ~ axis_score_z + C(treatment_family)", "treatment_adjusted"))

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", newline="") as handle:
        fields = [
            "dataset_id",
            "endpoint",
            "model_formula",
            "covariates",
            "effect_type",
            "effect",
            "ci_lower",
            "ci_upper",
            "pvalue",
            "fdr",
            "n",
            "interpretation_boundary",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={OUTPUT} models={len(rows)} n={len(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
