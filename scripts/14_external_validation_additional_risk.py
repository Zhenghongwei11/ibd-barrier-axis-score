#!/usr/bin/env python3
"""External phenotype validation in GSE117993 and GSE101794."""

from __future__ import annotations

import csv
import math
import os

import pandas as pd
import statsmodels.formula.api as smf


AXIS = "results/axis/barrier_axis_scores.tsv"
SUMMARY_OUT = "results/replication/combined_summary.tsv"


def zscore(series: pd.Series) -> pd.Series:
    sd = series.std(ddof=1)
    return (series - series.mean()) / sd if sd else series * 0


def fit(dataset_id: str, data: pd.DataFrame, endpoint: str, formula: str, boundary: str) -> dict[str, str]:
    model = smf.logit(formula, data=data).fit(disp=False)
    coef = model.params["axis_score_z"]
    ci = model.conf_int().loc["axis_score_z"]
    return {
        "dataset_id": dataset_id,
        "endpoint": endpoint,
        "model": formula,
        "effect_type": "odds_ratio_per_1sd_axis_score",
        "effect": f"{math.exp(coef):.6g}",
        "ci_lower": f"{math.exp(ci[0]):.6g}",
        "ci_upper": f"{math.exp(ci[1]):.6g}",
        "pvalue": f"{model.pvalues['axis_score_z']:.6g}",
        "fdr": f"{model.pvalues['axis_score_z']:.6g}",
        "n": str(int(model.nobs)),
        "interpretation_boundary": boundary,
    }


def axis_for(dataset_id: str) -> pd.DataFrame:
    axis = pd.read_csv(AXIS, sep="\t")
    axis = axis[axis["dataset_id"] == dataset_id].copy()
    axis["axis_score"] = pd.to_numeric(axis["axis_score"], errors="coerce")
    return axis


def gse117993_rows() -> list[dict[str, str]]:
    meta = pd.read_csv("data/processed/GSE117993/metadata.tsv", sep="\t")
    data = meta.merge(axis_for("GSE117993"), on="sample_id", how="inner").dropna(subset=["axis_score"]).copy()
    data["axis_score_z"] = zscore(data["axis_score"])
    rows = []
    ibd = data[data["diagnosis"].isin(["CD", "UC", "Control"])].copy()
    ibd["ibd_status"] = ibd["diagnosis"].isin(["CD", "UC"]).astype(int)
    rows.append(fit("GSE117993", ibd, "IBD_vs_control_rectum", "ibd_status ~ axis_score_z", "independent_rectal_disease_phenotype_validation; not treatment_response"))
    for disease in ["CD", "UC"]:
        subset = data[data["diagnosis"].isin([disease, "Control"])].copy()
        subset["disease_status"] = (subset["diagnosis"] == disease).astype(int)
        rows.append(fit("GSE117993", subset, f"{disease}_vs_control_rectum", "disease_status ~ axis_score_z", "independent_rectal_disease_phenotype_validation; not treatment_response"))
    return rows


def gse101794_rows() -> list[dict[str, str]]:
    meta = pd.read_csv("data/processed/GSE101794/metadata.tsv", sep="\t")
    data = meta.merge(axis_for("GSE101794"), on="sample_id", how="inner").dropna(subset=["axis_score"]).copy()
    data["axis_score_z"] = zscore(data["axis_score"])
    data["age_z"] = zscore(pd.to_numeric(data["age_at_diagnosis_in_years"], errors="coerce"))
    data["cd_status"] = data["diagnosis"].eq("CD").astype(int)
    rows = [
        fit("GSE101794", data, "CD_vs_nonIBD_ileum", "cd_status ~ axis_score_z", "independent_ileal_disease_phenotype_validation; not treatment_response"),
        fit(
            "GSE101794",
            data.dropna(subset=["age_z"]),
            "CD_vs_nonIBD_ileum_age_sex_adjusted",
            "cd_status ~ axis_score_z + age_z + C(sex)",
            "independent_ileal_disease_phenotype_validation; not treatment_response",
        ),
    ]
    return rows


def write_rows(path: str, rows: list[dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "dataset_id",
        "endpoint",
        "model",
        "effect_type",
        "effect",
        "ci_lower",
        "ci_upper",
        "pvalue",
        "fdr",
        "n",
        "interpretation_boundary",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rows_117993 = gse117993_rows()
    rows_101794 = gse101794_rows()
    write_rows("results/replication/GSE117993_effects.tsv", rows_117993)
    write_rows("results/replication/GSE101794_effects.tsv", rows_101794)
    existing = []
    if os.path.exists(SUMMARY_OUT):
        with open(SUMMARY_OUT, newline="") as handle:
            existing = [row for row in csv.DictReader(handle, delimiter="\t") if row["dataset_id"] not in {"GSE117993", "GSE101794"}]
    write_rows(SUMMARY_OUT, existing + rows_117993 + rows_101794)
    print(f"wrote additional validation rows={len(rows_117993) + len(rows_101794)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
