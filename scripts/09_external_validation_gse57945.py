#!/usr/bin/env python3
"""External phenotype validation of the barrier-axis score in GSE57945."""

from __future__ import annotations

import csv
import math
import os
import re

import pandas as pd
import statsmodels.formula.api as smf


AXIS = "results/axis/barrier_axis_scores.tsv"
METADATA = "data/processed/GSE57945/metadata.tsv"
OUT_EFFECTS = "results/replication/GSE57945_effects.tsv"
OUT_SUMMARY = "results/replication/combined_summary.tsv"


def risk_id(title: str) -> str | None:
    match = re.search(r"(CCFA_Risk_\d+)", title)
    return match.group(1) if match else None


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=1)


def fit(data: pd.DataFrame, endpoint: str, formula: str, boundary: str) -> dict[str, str]:
    model = smf.logit(formula, data=data).fit(disp=False)
    coef = model.params["axis_score_z"]
    ci = model.conf_int().loc["axis_score_z"]
    return {
        "dataset_id": "GSE57945",
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


def main() -> int:
    axis = pd.read_csv(AXIS, sep="\t")
    axis = axis[axis["dataset_id"] == "GSE57945"].copy()
    axis["axis_score"] = pd.to_numeric(axis["axis_score"], errors="coerce")
    meta = pd.read_csv(METADATA, sep="\t")
    meta["risk_id"] = meta["title"].map(risk_id)
    data = meta.merge(axis, left_on="risk_id", right_on="sample_id", how="inner")
    data = data.dropna(subset=["axis_score"]).copy()
    data["axis_score_z"] = zscore(data["axis_score"])

    rows = []
    inflammation = data[data["histopathology"].isin(["Normal", "Microscopic inflammation", "Macroscopic inflammation"])].copy()
    inflammation["macroscopic_inflammation"] = (inflammation["histopathology"] == "Macroscopic inflammation").astype(int)
    rows.append(
        fit(
            inflammation,
            "macroscopic_inflammation_vs_normal_or_microscopic",
            "macroscopic_inflammation ~ axis_score_z",
            "independent_ileal_phenotype_validation; not treatment_response",
        )
    )

    ulcer = data[data["deep_ulcer"].isin(["Yes", "No", "no"])].copy()
    ulcer["deep_ulcer_binary"] = ulcer["deep_ulcer"].str.lower().eq("yes").astype(int)
    rows.append(
        fit(
            ulcer,
            "deep_ulcer",
            "deep_ulcer_binary ~ axis_score_z",
            "independent_ileal_phenotype_validation; not treatment_response",
        )
    )

    os.makedirs(os.path.dirname(OUT_EFFECTS), exist_ok=True)
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
    with open(OUT_EFFECTS, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with open(OUT_SUMMARY, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={OUT_EFFECTS} rows={len(rows)} merged_n={len(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
