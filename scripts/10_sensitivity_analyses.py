#!/usr/bin/env python3
"""Run predefined sensitivity analyses for the barrier-axis project."""

from __future__ import annotations

import csv
import math
import os

import pandas as pd
import statsmodels.formula.api as smf


GSE73661 = "data/processed/GSE73661/baseline_endpoint.tsv"
GSE57945 = "data/processed/GSE57945/metadata.tsv"
AXIS = "results/axis/barrier_axis_scores.tsv"
OUTPUT = "results/clinical/sensitivity_models.tsv"


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=1)


def model_row(dataset_id: str, endpoint: str, sensitivity: str, formula: str, data: pd.DataFrame, term: str):
    try:
        model = smf.logit(formula, data=data).fit(disp=False)
        coef = model.params[term]
        ci = model.conf_int().loc[term]
        return {
            "dataset_id": dataset_id,
            "endpoint": endpoint,
            "sensitivity": sensitivity,
            "model": formula,
            "effect_type": f"odds_ratio_for_{term}",
            "effect": f"{math.exp(coef):.6g}",
            "ci_lower": f"{math.exp(ci[0]):.6g}",
            "ci_upper": f"{math.exp(ci[1]):.6g}",
            "pvalue": f"{model.pvalues[term]:.6g}",
            "n": str(int(model.nobs)),
            "status": "ok",
        }
    except Exception as exc:
        return {
            "dataset_id": dataset_id,
            "endpoint": endpoint,
            "sensitivity": sensitivity,
            "model": formula,
            "effect_type": f"odds_ratio_for_{term}",
            "effect": "NA",
            "ci_lower": "NA",
            "ci_upper": "NA",
            "pvalue": "NA",
            "n": str(len(data)),
            "status": f"failed:{type(exc).__name__}",
        }


def gse73661_rows() -> list[dict[str, str]]:
    data = pd.read_csv(GSE73661, sep="\t")
    for col in ["axis_score", "upstream_score", "mmp_score", "junction_score", "baseline_mayo", "mucosal_healing"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["axis_score", "mucosal_healing"]).copy()
    for col in ["axis_score", "upstream_score", "mmp_score", "junction_score"]:
        data[col + "_z"] = zscore(data[col])

    rows = [
        model_row("GSE73661", "mucosal_healing", "adjust_baseline_mayo_and_treatment", "mucosal_healing ~ axis_score_z + baseline_mayo + C(treatment_family)", data, "axis_score_z"),
        model_row("GSE73661", "mucosal_healing", "mmp_component", "mucosal_healing ~ mmp_score_z + C(treatment_family)", data, "mmp_score_z"),
        model_row("GSE73661", "mucosal_healing", "upstream_component", "mucosal_healing ~ upstream_score_z + C(treatment_family)", data, "upstream_score_z"),
        model_row("GSE73661", "mucosal_healing", "junction_component", "mucosal_healing ~ junction_score_z + C(treatment_family)", data, "junction_score_z"),
    ]
    for treatment in sorted(data["treatment_family"].unique()):
        subset = data[data["treatment_family"] == treatment].copy()
        if subset["mucosal_healing"].nunique() == 2 and len(subset) >= 10:
            rows.append(model_row("GSE73661", "mucosal_healing", f"treatment_stratum_{treatment}", "mucosal_healing ~ axis_score_z", subset, "axis_score_z"))
    return rows


def main() -> int:
    rows = gse73661_rows()
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    fields = [
        "dataset_id",
        "endpoint",
        "sensitivity",
        "model",
        "effect_type",
        "effect",
        "ci_lower",
        "ci_upper",
        "pvalue",
        "n",
        "status",
    ]
    with open(OUTPUT, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={OUTPUT} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
