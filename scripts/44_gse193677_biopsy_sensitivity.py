#!/usr/bin/env python3
"""GSE193677 biopsy-selection sensitivity analyses.

The output table records supported and unsupported subset analyses so reviewers
can see which biopsy/site/disease-stratum checks were feasible from public
metadata.
"""

from __future__ import annotations

import gzip
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent
AXIS = ROOT / "results/axis/barrier_axis_scores.tsv"
OUT = ROOT / "results/clinical/GSE193677_biopsy_selection_sensitivity.tsv"


def import_gse193677_module():
    spec = importlib.util.spec_from_file_location("r39", ROOT / "scripts/39_gse193677_replication.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=1)
    return (s - s.mean()) / sd if sd and not math.isnan(sd) else s * 0


def read_joined() -> pd.DataFrame:
    r39 = import_gse193677_module()
    meta = r39.parse_series_matrix()
    meta["disease_group"] = meta["ibd_disease"].replace({"CD": "IBD", "UC": "IBD", "Control": "Control"})
    meta["ibd_vs_control"] = (meta["disease_group"] == "IBD").astype(int)
    meta["ibd_endoseverity_num"] = meta["ibd_endoseverity_4levels"].map(
        {"Inactive": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
    )
    for col in [
        "ibd_endoseverity_num",
        "nancyindex",
        "ghas_sum7",
        "ibdmesuc_mayo_score",
        "ibdsescd_totalsescd",
        "study_eligibility_age_at_endo",
    ]:
        meta[col] = pd.to_numeric(meta[col], errors="coerce")
    meta["age_z"] = zscore(meta["study_eligibility_age_at_endo"])
    meta["sex"] = meta["demographics_gender"].replace({"": np.nan})
    meta["site"] = meta["regionre"].replace({"": np.nan})
    with gzip.open(r39.ADJCOUNTS, "rt", errors="replace") as handle:
        header = [t.strip('"') for t in handle.readline().split()]
    order = {sid: i for i, sid in enumerate(header)}
    meta["_order"] = meta["sample_id"].map(order)
    axis = pd.read_csv(AXIS, sep="\t")
    axis = axis[axis["dataset_id"].eq("GSE193677")].copy()
    return meta.merge(axis, on="sample_id", how="inner")


def summarize_subset(data: pd.DataFrame) -> dict:
    return {
        "n_samples": int(len(data)),
        "n_participants": int(data["participant_id"].nunique()),
        "n_ibd": int((data["disease_group"] == "IBD").sum()),
        "n_control": int((data["disease_group"] == "Control").sum()),
        "n_uc": int((data["ibd_disease"] == "UC").sum()),
        "n_cd": int((data["ibd_disease"] == "CD").sum()),
        "n_sites": int(data["site"].nunique(dropna=True)),
    }


def select_one(data: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "first_expression_order":
        sort_cols = ["_order"]
        asc = [True]
    elif rule == "most_inflamed":
        sort_cols = ["title_inflamed", "ibd_endoseverity_num", "_order"]
        asc = [False, False, True]
    elif rule == "least_inflamed":
        sort_cols = ["title_inflamed", "ibd_endoseverity_num", "_order"]
        asc = [True, True, True]
    else:
        raise ValueError(rule)
    return data.sort_values(sort_cols, ascending=asc).groupby("participant_id", as_index=False).first()


def model_row(scope: str, rule: str, model_type: str, data: pd.DataFrame, formula: str, gee: bool = False) -> dict:
    base = {
        "analysis_scope": scope,
        "selection_rule": rule,
        "endpoint": "IBD versus control",
        "model_type": model_type,
        "effect_measure": "odds_ratio_per_1sd",
        "term": "axis_score_z",
        **summarize_subset(data),
    }
    needed = ["axis_score", "ibd_vs_control"]
    if "age_z" in formula:
        needed.extend(["age_z", "sex", "site"])
    model_data = data.dropna(subset=needed).copy()
    if model_data["ibd_vs_control"].nunique() < 2:
        return {**base, "n_model": int(len(model_data)), "effect": "", "ci_lower": "", "ci_upper": "", "pvalue": "",
                "status": "not_modeled_single_outcome_class", "model_formula": formula}
    if len(model_data) < 30:
        return {**base, "n_model": int(len(model_data)), "effect": "", "ci_lower": "", "ci_upper": "", "pvalue": "",
                "status": "not_modeled_low_n", "model_formula": formula}
    model_data["axis_score_z"] = zscore(model_data["axis_score"])
    try:
        if gee:
            model = smf.gee(
                formula,
                groups="participant_id",
                data=model_data,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable(),
            ).fit()
        else:
            model = smf.logit(formula, data=model_data).fit(disp=False)
        ci = model.conf_int().loc["axis_score_z"]
        return {
            **base,
            "n_model": int(model.nobs),
            "effect": float(math.exp(model.params["axis_score_z"])),
            "ci_lower": float(math.exp(ci[0])),
            "ci_upper": float(math.exp(ci[1])),
            "pvalue": float(model.pvalues["axis_score_z"]),
            "status": "modeled",
            "model_formula": formula,
        }
    except Exception as exc:
        return {**base, "n_model": int(len(model_data)), "effect": "", "ci_lower": "", "ci_upper": "",
                "pvalue": "", "status": f"model_failed:{exc}", "model_formula": formula}


def correlation_rows(scope: str, rule: str, data: pd.DataFrame) -> list[dict]:
    rows = []
    measures = [
        ("endoscopic_severity_4levels", "ibd_endoseverity_num"),
        ("nancy_histology_index", "nancyindex"),
        ("ghas_histology_sum7", "ghas_sum7"),
        ("mayo_score_uc", "ibdmesuc_mayo_score"),
        ("sescd_total_cd", "ibdsescd_totalsescd"),
    ]
    ibd = data[data["disease_group"].eq("IBD")].copy()
    for label, col in measures:
        model_data = ibd.dropna(subset=["axis_score", col])
        base = {
            "analysis_scope": scope,
            "selection_rule": rule,
            "endpoint": label,
            "model_type": "spearman_correlation",
            "effect_measure": "spearman_rho",
            "term": "axis_score",
            **summarize_subset(model_data),
            "model_formula": f"spearman(axis_score, {col})",
        }
        if len(model_data) < 20:
            rows.append({**base, "n_model": int(len(model_data)), "effect": "", "ci_lower": "", "ci_upper": "",
                         "pvalue": "", "status": "not_modeled_low_n"})
            continue
        rho, p = stats.spearmanr(model_data["axis_score"], model_data[col])
        if abs(rho) < 1 and len(model_data) > 3:
            z = math.atanh(rho)
            se = 1 / math.sqrt(len(model_data) - 3)
            lo, hi = math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)
        else:
            lo = hi = float("nan")
        rows.append({**base, "n_model": int(len(model_data)), "effect": float(rho), "ci_lower": float(lo),
                     "ci_upper": float(hi), "pvalue": float(p), "status": "modeled"})
    return rows


def main() -> int:
    os.makedirs(OUT.parent, exist_ok=True)
    data = read_joined()
    colon_sites = {"Rectum", "LeftColon", "RightColon", "Transverse", "Sigmoid"}
    subsets: list[tuple[str, str, pd.DataFrame, bool]] = [
        ("all_biopsies", "all public biopsies with participant-level clustering", data, True),
        ("one_biopsy_per_participant", "first_expression_order", select_one(data, "first_expression_order"), False),
        ("one_biopsy_per_participant", "most_inflamed", select_one(data, "most_inflamed"), False),
        ("one_biopsy_per_participant", "least_inflamed", select_one(data, "least_inflamed"), False),
        (
            "colorectal_sites",
            "first_expression_order_within_colorectal_sites",
            select_one(data[data["site"].isin(colon_sites)], "first_expression_order"),
            False,
        ),
        (
            "ileal_site",
            "first_expression_order_within_ileum",
            select_one(data[data["site"].eq("Ileum")], "first_expression_order"),
            False,
        ),
        (
            "uc_or_control",
            "first_expression_order_within_uc_or_control",
            select_one(data[data["ibd_disease"].isin(["UC", "Control"])], "first_expression_order"),
            False,
        ),
        (
            "cd_or_control",
            "first_expression_order_within_cd_or_control",
            select_one(data[data["ibd_disease"].isin(["CD", "Control"])], "first_expression_order"),
            False,
        ),
    ]

    rows = []
    for scope, rule, subset, use_gee in subsets:
        rows.append(model_row(scope, rule, "unadjusted_gee" if use_gee else "unadjusted_logistic",
                              subset, "ibd_vs_control ~ axis_score_z", gee=use_gee))
        rows.append(model_row(scope, rule, "age_sex_site_adjusted_gee" if use_gee else "age_sex_site_adjusted_logistic",
                              subset, "ibd_vs_control ~ axis_score_z + age_z + C(sex) + C(site)", gee=use_gee))
        rows.extend(correlation_rows(scope, rule, subset))
    pd.DataFrame(rows).to_csv(OUT, sep="\t", index=False)
    print(f"wrote={OUT} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
