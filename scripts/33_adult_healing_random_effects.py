#!/usr/bin/env python3
"""Exploratory random-effects synthesis for adult healing-like endpoints."""

from __future__ import annotations

import csv
import math
import os

import pandas as pd
from scipy import stats


INFILE = "results/clinical/age_stratified_endpoint_models.tsv"
OUT = "results/meta/adult_healing_random_effects.tsv"


PRIMARY = [
    ("GSE73661", "mucosal_healing"),
    ("GSE206285", "week8_mucosal_healing"),
]

SENSITIVITY = [
    ("GSE73661", "mucosal_healing"),
    ("GSE12251", "week8_endoscopic_histologic_healing"),
    ("GSE206285", "week8_mucosal_healing"),
]


def select_unadjusted(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for dataset_id, endpoint in pairs:
        sub = df[
            df["dataset_id"].eq(dataset_id)
            & df["endpoint_name"].eq(endpoint)
            & df["covariates"].eq("none")
        ]
        if not sub.empty:
            rows.append(sub.iloc[0])
    return pd.DataFrame(rows)


def random_effects(sub: pd.DataFrame, label: str) -> dict[str, str]:
    if len(sub) < 2:
        return {
            "analysis": label,
            "cohorts_included": ";".join(sub["dataset_id"].astype(str)) if not sub.empty else "",
            "k": str(len(sub)),
            "pooled_or": "NA",
            "ci_lower": "NA",
            "ci_upper": "NA",
            "pvalue": "NA",
            "tau2": "NA",
            "i2": "NA",
            "q": "NA",
            "status": "not_run_minimum_two_cohorts_required",
        }
    yi = sub["effect"].astype(float).map(math.log).to_numpy()
    lo = sub["ci_lower"].astype(float).map(math.log).to_numpy()
    hi = sub["ci_upper"].astype(float).map(math.log).to_numpy()
    sei = (hi - lo) / (2 * 1.959963984540054)
    vi = sei**2
    wi = 1 / vi
    fixed = (wi * yi).sum() / wi.sum()
    q = (wi * (yi - fixed) ** 2).sum()
    df_q = len(yi) - 1
    c = wi.sum() - (wi**2).sum() / wi.sum()
    tau2 = max(0.0, (q - df_q) / c) if c > 0 else 0.0
    wre = 1 / (vi + tau2)
    pooled = (wre * yi).sum() / wre.sum()
    se = math.sqrt(1 / wre.sum())
    z = pooled / se
    p = 2 * stats.norm.sf(abs(z))
    i2 = max(0.0, (q - df_q) / q) * 100 if q > 0 else 0.0
    return {
        "analysis": label,
        "cohorts_included": ";".join(f"{r.dataset_id}:{r.endpoint_name}" for r in sub.itertuples()),
        "k": str(len(sub)),
        "pooled_or": f"{math.exp(pooled):.6g}",
        "ci_lower": f"{math.exp(pooled - 1.959963984540054 * se):.6g}",
        "ci_upper": f"{math.exp(pooled + 1.959963984540054 * se):.6g}",
        "pvalue": f"{p:.6g}",
        "tau2": f"{tau2:.6g}",
        "i2": f"{i2:.6g}",
        "q": f"{q:.6g}",
        "status": "exploratory_random_effects_not_primary_claim",
    }


def main() -> int:
    df = pd.read_csv(INFILE, sep="\t")
    rows = [
        random_effects(select_unadjusted(df, PRIMARY), "adult_mucosal_healing_primary_exploratory"),
        random_effects(select_unadjusted(df, SENSITIVITY), "adult_healing_like_sensitivity_including_GSE12251"),
    ]
    os.makedirs("results/meta", exist_ok=True)
    with open(OUT, "w", newline="") as handle:
        fields = ["analysis", "cohorts_included", "k", "pooled_or", "ci_lower", "ci_upper", "pvalue", "tau2", "i2", "q", "status"]
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={OUT} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
