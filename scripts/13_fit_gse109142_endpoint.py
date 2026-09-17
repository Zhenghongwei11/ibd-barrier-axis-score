#!/usr/bin/env python3
"""Fit PROTECT/GSE109142 week-4 remission endpoint models."""

from __future__ import annotations

import csv
import math
import os

import pandas as pd
import statsmodels.formula.api as smf


AXIS = "results/axis/barrier_axis_scores.tsv"
METADATA = "data/processed/GSE109142/metadata.tsv"
ENDPOINT_OUT = "data/processed/GSE109142/baseline_endpoint.tsv"
MODEL_OUT = "results/clinical/endpoint_models.tsv"


def zscore(series: pd.Series) -> pd.Series:
    sd = series.std(ddof=1)
    return (series - series.mean()) / sd if sd else series * 0


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace({"NA": None, "unknown": None, "": None}), errors="coerce")


def model_row(data: pd.DataFrame, formula: str, covariates: str) -> dict[str, str]:
    model = smf.logit(formula, data=data).fit(disp=False)
    coef = model.params["axis_score_z"]
    ci = model.conf_int().loc["axis_score_z"]
    pvalue = model.pvalues["axis_score_z"]
    return {
        "dataset_id": "GSE109142",
        "endpoint": "week_4_remission",
        "model_formula": formula,
        "covariates": covariates,
        "effect_type": "odds_ratio_per_1sd_axis_score",
        "effect": f"{math.exp(coef):.6g}",
        "ci_lower": f"{math.exp(ci[0]):.6g}",
        "ci_upper": f"{math.exp(ci[1]):.6g}",
        "pvalue": f"{pvalue:.6g}",
        "fdr": f"{pvalue:.6g}",
        "n": str(int(model.nobs)),
        "interpretation_boundary": "retrospective_baseline_expression_association; pediatric_UC_week4_remission; not prospective clinical utility",
    }


def build_endpoint() -> pd.DataFrame:
    axis = pd.read_csv(AXIS, sep="\t")
    axis = axis[axis["dataset_id"] == "GSE109142"].copy()
    axis["axis_score"] = pd.to_numeric(axis["axis_score"], errors="coerce")
    meta = pd.read_csv(METADATA, sep="\t")
    data = meta.merge(axis, on="sample_id", how="inner")
    data = data[(data["diagnosis"] == "Ulcerative Colitis") & data["week_4_remission"].isin(["Yes", "No"])].copy()
    data["week4_remission"] = data["week_4_remission"].eq("Yes").astype(int)
    for col in ["age_at_diagnosis", "pucai", "total_mayo_score", "histology_severity_score", "baseline_calprotectin"]:
        data[col + "_num"] = numeric(data[col])
    for col in ["axis_score", "pucai_num", "total_mayo_score_num", "histology_severity_score_num"]:
        data[col + "_z"] = zscore(data[col])
    fields = [
        "sample_id",
        "week4_remission",
        "week_4_remission",
        "axis_score",
        "axis_score_z",
        "upstream_score",
        "mmp_score",
        "junction_score",
        "sex",
        "age_at_diagnosis_num",
        "pucai_num",
        "total_mayo_score_num",
        "histology_severity_score_num",
        "initial_treatment",
        "baseline_calprotectin_num",
        "week_4_calprotectin",
    ]
    os.makedirs(os.path.dirname(ENDPOINT_OUT), exist_ok=True)
    data[fields].to_csv(ENDPOINT_OUT, sep="\t", index=False)
    return data


def main() -> int:
    data = build_endpoint()
    rows = [
        model_row(data.dropna(subset=["axis_score_z", "week4_remission"]), "week4_remission ~ axis_score_z", "none"),
        model_row(data.dropna(subset=["axis_score_z", "week4_remission", "total_mayo_score_num_z"]), "week4_remission ~ axis_score_z + total_mayo_score_num_z", "baseline_total_mayo"),
        model_row(
            data.dropna(subset=["axis_score_z", "week4_remission", "total_mayo_score_num_z", "initial_treatment"]),
            "week4_remission ~ axis_score_z + total_mayo_score_num_z + C(initial_treatment)",
            "baseline_total_mayo;initial_treatment",
        ),
    ]
    existing = []
    if os.path.exists(MODEL_OUT):
        with open(MODEL_OUT, newline="") as handle:
            existing = [row for row in csv.DictReader(handle, delimiter="\t") if row["dataset_id"] != "GSE109142"]
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
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    with open(MODEL_OUT, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(existing + rows)
    print(f"wrote={ENDPOINT_OUT} n={len(data)}; appended_models={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
