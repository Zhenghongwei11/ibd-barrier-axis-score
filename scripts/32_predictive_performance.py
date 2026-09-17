#!/usr/bin/env python3
"""Apparent predictive-performance summaries for retrospective endpoint models."""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import brier_score_loss, roc_auc_score


OUT = "results/clinical/apparent_predictive_performance.tsv"


@dataclass(frozen=True)
class EndpointSpec:
    dataset_id: str
    endpoint_path: str
    sample_col: str
    endpoint_col: str
    covariates: tuple[str, ...] = ()
    role: str = ""


SPECS = [
    EndpointSpec("GSE73661", "data/processed/GSE73661/baseline_endpoint.tsv", "baseline_sample_id", "mucosal_healing", ("C(treatment_family)",), "adult_mucosal_healing"),
    EndpointSpec("GSE12251", "data/processed/GSE12251/baseline_endpoint.tsv", "sample_id", "wk8_endoscopic_histologic_healing", ("C(dose)",), "adult_healing_like"),
    EndpointSpec("GSE92415", "data/processed/GSE92415/baseline_endpoint.tsv", "sample_id", "week6_clinical_response", ("C(treatment)", "baseline_mayo_score_z"), "adult_response"),
    EndpointSpec("GSE206285", "data/processed/GSE206285/baseline_endpoint.tsv", "sample_id", "week8_mucosal_healing", ("C(treatment_family)",), "adult_mucosal_healing"),
    EndpointSpec("GSE206285", "data/processed/GSE206285/baseline_endpoint.tsv", "sample_id", "week8_clinical_remission", ("C(treatment_family)",), "adult_remission"),
    EndpointSpec("GSE23597", "data/processed/GSE23597/baseline_endpoint.tsv", "sample_id", "week8_response", ("C(dose)",), "adult_response"),
    EndpointSpec("GSE23597", "data/processed/GSE23597/baseline_endpoint.tsv", "sample_id", "week30_response", ("C(dose)",), "adult_response"),
    EndpointSpec("GSE16879", "data/processed/GSE16879/baseline_endpoint.tsv", "sample_id", "infliximab_response", ("C(disease)",), "adult_or_mixed_response"),
    EndpointSpec("GSE109142", "data/processed/GSE109142/baseline_endpoint.tsv", "sample_id", "week4_remission", ("total_mayo_score_num_z", "C(initial_treatment)"), "pediatric_remission"),
]


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=1)
    if sd == 0 or math.isnan(sd):
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - s.mean()) / sd


def prepare(spec: EndpointSpec) -> pd.DataFrame:
    data = pd.read_csv(spec.endpoint_path, sep="\t")
    if spec.dataset_id == "GSE206285" and "treatment_family" not in data.columns:
        data["treatment_family"] = data["treatment"].astype(str).str.extract(r"^(Ustekinumab|Placebo)", expand=False).fillna("Other")
    if spec.dataset_id == "GSE92415" and "baseline_mayo_score_z" not in data.columns:
        data["baseline_mayo_score_z"] = zscore(data["baseline_mayo_score"])
    if spec.dataset_id == "GSE109142" and "total_mayo_score_num_z" not in data.columns:
        data["total_mayo_score_num_z"] = zscore(data["total_mayo_score_num"])
    data[spec.endpoint_col] = pd.to_numeric(data[spec.endpoint_col], errors="coerce")
    data["axis_score_z"] = zscore(data["axis_score"])
    return data.dropna(subset=[spec.endpoint_col, "axis_score_z"]).copy()


def calibration_slope(y: pd.Series, p: np.ndarray) -> float:
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    logits = np.log(p / (1 - p))
    tmp = pd.DataFrame({"y": y.astype(int).values, "lp": logits})
    try:
        model = smf.logit("y ~ lp", data=tmp).fit(disp=False, maxiter=200)
        return float(model.params["lp"])
    except Exception:
        return float("nan")


def fit_row(spec: EndpointSpec, adjusted: bool) -> dict[str, str]:
    data = prepare(spec)
    covar = " + ".join(spec.covariates) if adjusted else ""
    formula = f"{spec.endpoint_col} ~ axis_score_z" + (f" + {covar}" if covar else "")
    base = {
        "dataset_id": spec.dataset_id,
        "endpoint": spec.endpoint_col,
        "role": spec.role,
        "model_type": "covariate_adjusted" if adjusted and spec.covariates else "axis_only",
        "n": str(len(data)),
        "events": str(int(data[spec.endpoint_col].sum())),
        "formula": formula,
    }
    if data[spec.endpoint_col].nunique() < 2 or len(data) < 8:
        return {**base, "auc": "NA", "brier": "NA", "calibration_slope": "NA", "status": "insufficient_endpoint_variation"}
    try:
        model = smf.logit(formula, data=data).fit(disp=False, maxiter=200)
        p = model.predict(data)
        auc = roc_auc_score(data[spec.endpoint_col], p)
        brier = brier_score_loss(data[spec.endpoint_col], p)
        slope = calibration_slope(data[spec.endpoint_col], p)
        return {**base, "auc": f"{auc:.6g}", "brier": f"{brier:.6g}", "calibration_slope": f"{slope:.6g}" if not math.isnan(slope) else "NA", "status": "apparent_performance_no_external_validation"}
    except Exception as exc:
        return {**base, "auc": "NA", "brier": "NA", "calibration_slope": "NA", "status": f"failed:{type(exc).__name__}"}


def main() -> int:
    rows = []
    for spec in SPECS:
        rows.append(fit_row(spec, adjusted=False))
        if spec.covariates:
            rows.append(fit_row(spec, adjusted=True))
    os.makedirs("results/clinical", exist_ok=True)
    with open(OUT, "w", newline="") as handle:
        fields = ["dataset_id", "endpoint", "role", "model_type", "n", "events", "formula", "auc", "brier", "calibration_slope", "status"]
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={OUT} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
