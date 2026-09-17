#!/usr/bin/env python3
"""Compare barrier-axis models against established IBD response markers."""

from __future__ import annotations

import csv
import math
import os

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


ENDPOINT = "data/processed/GSE73661/baseline_endpoint.tsv"
COMPARATOR_EXPR = "data/processed/GSE73661/comparator_gene_expression.tsv"
OUTPUT = "results/clinical/comparator_models.tsv"


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=1)


def read_comparator_score() -> pd.DataFrame:
    expr = pd.read_csv(COMPARATOR_EXPR, sep="\t")
    expr = expr.set_index("Gene Symbol")
    z = expr.apply(lambda row: zscore(row.astype(float)), axis=1)
    score = z.mean(axis=0)
    return pd.DataFrame({"baseline_sample_id": score.index, "comparator_score": score.values})


def main() -> int:
    endpoint = pd.read_csv(ENDPOINT, sep="\t")
    comp = read_comparator_score()
    data = endpoint.merge(comp, on="baseline_sample_id", how="inner")
    for col in ["axis_score", "comparator_score", "mucosal_healing"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["axis_score", "comparator_score", "mucosal_healing"]).copy()
    data["axis_score_z"] = zscore(data["axis_score"])
    data["comparator_score_z"] = zscore(data["comparator_score"])

    base = smf.logit("mucosal_healing ~ comparator_score_z + C(treatment_family)", data=data).fit(disp=False)
    axis = smf.logit("mucosal_healing ~ comparator_score_z + axis_score_z + C(treatment_family)", data=data).fit(disp=False)
    lr_stat = 2 * (axis.llf - base.llf)
    lr_p = stats.chi2.sf(lr_stat, df=1)
    coef = axis.params["axis_score_z"]
    ci = axis.conf_int().loc["axis_score_z"]

    rows = [
        {
            "dataset_id": "GSE73661",
            "comparator_set": "OSM_TREM1_TNF_IL13RA2_TNFRSF1B_mean_z",
            "base_model": "mucosal_healing ~ comparator_score_z + C(treatment_family)",
            "axis_model": "mucosal_healing ~ comparator_score_z + axis_score_z + C(treatment_family)",
            "delta_metric": "likelihood_ratio_p_for_axis_increment",
            "delta_value": f"{lr_p:.6g}",
            "axis_effect_type": "odds_ratio_per_1sd_axis_score_adjusted_for_comparator",
            "axis_effect": f"{math.exp(coef):.6g}",
            "ci_lower": f"{math.exp(ci[0]):.6g}",
            "ci_upper": f"{math.exp(ci[1]):.6g}",
            "pvalue": f"{axis.pvalues['axis_score_z']:.6g}",
            "n": str(int(axis.nobs)),
            "conclusion": "axis_increment_tested_retrospectively; clinical utility not established",
        }
    ]

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", newline="") as handle:
        fields = [
            "dataset_id",
            "comparator_set",
            "base_model",
            "axis_model",
            "delta_metric",
            "delta_value",
            "axis_effect_type",
            "axis_effect",
            "ci_lower",
            "ci_upper",
            "pvalue",
            "n",
            "conclusion",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={OUTPUT} n={len(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
