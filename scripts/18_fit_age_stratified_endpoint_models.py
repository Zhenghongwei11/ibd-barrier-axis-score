#!/usr/bin/env python3
"""Fit and harmonize age-stratified endpoint models."""

from __future__ import annotations

import csv
import math
import os

import pandas as pd
import statsmodels.formula.api as smf


OUT = "results/clinical/age_stratified_endpoint_models.tsv"
SYNTHESIS_OUT = "results/meta/age_endpoint_synthesis.tsv"
GSE12251_MODEL_OUT = "results/clinical/GSE12251_endpoint_models.tsv"
GSE92415_MODEL_OUT = "results/clinical/GSE92415_endpoint_models.tsv"
ADULT_EXPANSION_MODEL_OUT = "results/clinical/adult_expansion_endpoint_models.tsv"


def zscore(series: pd.Series) -> pd.Series:
    sd = series.std(ddof=1)
    return (series - series.mean()) / sd if sd else series * 0


def bh_fdr(pvalues: list[float]) -> list[float]:
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [1.0] * n
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        original_rank = n - rank + 1
        value = min(prev, pvalues[idx] * n / original_rank)
        adjusted[idx] = min(value, 1.0)
        prev = adjusted[idx]
    return adjusted


def fit_logit_gse12251(data: pd.DataFrame, formula: str, covariates: str) -> dict[str, str]:
    model = smf.logit(formula, data=data).fit(disp=False)
    coef = model.params["axis_score_z"]
    ci = model.conf_int().loc["axis_score_z"]
    pvalue = float(model.pvalues["axis_score_z"])
    return {
        "dataset_id": "GSE12251",
        "age_stratum": "adult",
        "endpoint_family": "mucosal_healing_or_endoscopic_histologic_healing",
        "endpoint_name": "week8_endoscopic_histologic_healing",
        "tissue_site": "colonic mucosal biopsy",
        "treatment_context": "pre-infliximab ACT1 biopsy",
        "model_formula": formula,
        "covariates": covariates,
        "effect_type": "odds_ratio_per_1sd_axis_score",
        "effect": f"{math.exp(coef):.6g}",
        "ci_lower": f"{math.exp(ci[0]):.6g}",
        "ci_upper": f"{math.exp(ci[1]):.6g}",
        "pvalue": f"{pvalue:.6g}",
        "fdr": f"{pvalue:.6g}",
        "n": str(int(model.nobs)),
        "interpretation_boundary": "retrospective_baseline_expression_association; adult_ACT1_week8_endoscopic_histologic_healing; not prospective clinical utility",
        "clinical_role": "adult_direct_endpoint_validation",
    }


def gse12251_rows() -> list[dict[str, str]]:
    data = pd.read_csv("data/processed/GSE12251/baseline_endpoint.tsv", sep="\t")
    data["axis_score"] = pd.to_numeric(data["axis_score"], errors="coerce")
    data["wk8_endoscopic_histologic_healing"] = pd.to_numeric(data["wk8_endoscopic_histologic_healing"], errors="coerce")
    data = data.dropna(subset=["axis_score", "wk8_endoscopic_histologic_healing"]).copy()
    data["axis_score_z"] = zscore(data["axis_score"])
    rows = [
        fit_logit_gse12251(data, "wk8_endoscopic_histologic_healing ~ axis_score_z", "none"),
    ]
    if data["dose"].nunique() > 1:
        rows.append(fit_logit_gse12251(data, "wk8_endoscopic_histologic_healing ~ axis_score_z + C(dose)", "infliximab_dose"))
    return rows


def fit_logit_gse92415(data: pd.DataFrame, formula: str, covariates: str, subset_label: str = "all_baseline_uc") -> dict[str, str]:
    model = smf.logit(formula, data=data).fit(disp=False)
    coef = model.params["axis_score_z"]
    ci = model.conf_int().loc["axis_score_z"]
    pvalue = float(model.pvalues["axis_score_z"])
    return {
        "dataset_id": "GSE92415",
        "age_stratum": "adult",
        "endpoint_family": "treatment_response_or_remission",
        "endpoint_name": "week6_clinical_response",
        "tissue_site": "colon mucosal biopsy",
        "treatment_context": f"PURSUIT-SC baseline biopsy; {subset_label}",
        "model_formula": formula,
        "covariates": covariates,
        "effect_type": "odds_ratio_per_1sd_axis_score",
        "effect": f"{math.exp(coef):.6g}",
        "ci_lower": f"{math.exp(ci[0]):.6g}",
        "ci_upper": f"{math.exp(ci[1]):.6g}",
        "pvalue": f"{pvalue:.6g}",
        "fdr": f"{pvalue:.6g}",
        "n": str(int(model.nobs)),
        "interpretation_boundary": "retrospective_baseline_expression_association; adult_PURSUIT_SC_week6_response; treatment-response label from GEO metadata; not prospective clinical utility",
        "clinical_role": "adult_direct_endpoint_validation",
    }


def gse92415_rows() -> list[dict[str, str]]:
    path = "data/processed/GSE92415/baseline_endpoint.tsv"
    if not os.path.exists(path):
        return []
    data = pd.read_csv(path, sep="\t")
    data["axis_score"] = pd.to_numeric(data["axis_score"], errors="coerce")
    data["week6_clinical_response"] = pd.to_numeric(data["week6_clinical_response"], errors="coerce")
    data["baseline_mayo_score"] = pd.to_numeric(data["baseline_mayo_score"], errors="coerce")
    data = data.dropna(subset=["axis_score", "week6_clinical_response"]).copy()
    data["axis_score_z"] = zscore(data["axis_score"])
    data["baseline_mayo_score_z"] = zscore(data["baseline_mayo_score"])
    rows = [
        fit_logit_gse92415(data, "week6_clinical_response ~ axis_score_z", "none"),
    ]
    if data["treatment"].nunique() > 1:
        rows.append(fit_logit_gse92415(data, "week6_clinical_response ~ axis_score_z + C(treatment)", "treatment"))
        rows.append(fit_logit_gse92415(data, "week6_clinical_response ~ axis_score_z + C(treatment) + baseline_mayo_score_z", "treatment_baseline_mayo"))
    golimumab = data[data["treatment"].str.lower() == "golimumab"].copy()
    if len(golimumab) >= 20 and golimumab["week6_clinical_response"].nunique() == 2:
        golimumab["axis_score_z"] = zscore(golimumab["axis_score"])
        rows.append(fit_logit_gse92415(golimumab, "week6_clinical_response ~ axis_score_z", "golimumab_only", "golimumab_only"))
    return rows


def fit_logit_adult(
    dataset_id: str,
    data: pd.DataFrame,
    endpoint: str,
    formula: str,
    covariates: str,
    endpoint_family: str,
    tissue_site: str,
    treatment_context: str,
    clinical_role: str = "adult_direct_endpoint_validation",
) -> dict[str, str]:
    model = smf.logit(formula, data=data).fit(disp=False)
    coef = model.params["axis_score_z"]
    ci = model.conf_int().loc["axis_score_z"]
    pvalue = float(model.pvalues["axis_score_z"])
    return {
        "dataset_id": dataset_id,
        "age_stratum": "adult",
        "endpoint_family": endpoint_family,
        "endpoint_name": endpoint,
        "tissue_site": tissue_site,
        "treatment_context": treatment_context,
        "model_formula": formula,
        "covariates": covariates,
        "effect_type": "odds_ratio_per_1sd_axis_score",
        "effect": f"{math.exp(coef):.6g}",
        "ci_lower": f"{math.exp(ci[0]):.6g}",
        "ci_upper": f"{math.exp(ci[1]):.6g}",
        "pvalue": f"{pvalue:.6g}",
        "fdr": f"{pvalue:.6g}",
        "n": str(int(model.nobs)),
        "interpretation_boundary": f"retrospective_baseline_expression_association; {dataset_id}_{endpoint}; not prospective clinical utility",
        "clinical_role": clinical_role,
    }


def adult_expansion_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    gse206 = "data/processed/GSE206285/baseline_endpoint.tsv"
    if os.path.exists(gse206):
        data = pd.read_csv(gse206, sep="\t")
        data["axis_score"] = pd.to_numeric(data["axis_score"], errors="coerce")
        data["axis_score_z"] = zscore(data["axis_score"])
        data["treatment_family"] = data["treatment"].fillna("").str.replace(r" \\(.*\\)", "", regex=True)
        for endpoint, family in [
            ("week8_mucosal_healing", "mucosal_healing"),
            ("week8_clinical_remission", "clinical_remission"),
        ]:
            d = data.copy()
            d[endpoint] = pd.to_numeric(d[endpoint], errors="coerce")
            d = d.dropna(subset=["axis_score_z", endpoint]).copy()
            if len(d) and d[endpoint].nunique() == 2:
                rows.append(
                    fit_logit_adult(
                        "GSE206285",
                        d,
                        endpoint,
                        f"{endpoint} ~ axis_score_z",
                        "none",
                        family,
                        "sigmoid colon biopsy",
                        "UNIFI baseline biopsy",
                        "adult_direct_endpoint_expansion",
                    )
                )
                rows.append(
                    fit_logit_adult(
                        "GSE206285",
                        d,
                        endpoint,
                        f"{endpoint} ~ axis_score_z + C(treatment_family)",
                        "treatment_family",
                        family,
                        "sigmoid colon biopsy",
                        "UNIFI baseline biopsy",
                        "adult_direct_endpoint_expansion",
                    )
                )

    gse23597 = "data/processed/GSE23597/baseline_endpoint.tsv"
    if os.path.exists(gse23597):
        data = pd.read_csv(gse23597, sep="\t")
        data["axis_score"] = pd.to_numeric(data["axis_score"], errors="coerce")
        data["axis_score_z"] = zscore(data["axis_score"])
        for endpoint in ["week8_response", "week30_response"]:
            d = data.copy()
            d[endpoint] = pd.to_numeric(d[endpoint], errors="coerce")
            d = d.dropna(subset=["axis_score_z", endpoint]).copy()
            if len(d) and d[endpoint].nunique() == 2:
                rows.append(
                    fit_logit_adult(
                        "GSE23597",
                        d,
                        endpoint,
                        f"{endpoint} ~ axis_score_z",
                        "none",
                        "treatment_response_or_remission",
                        "colonic mucosal biopsy",
                        "ACT1 baseline biopsy",
                    )
                )
                if d["dose"].nunique() > 1:
                    rows.append(
                        fit_logit_adult(
                            "GSE23597",
                            d,
                            endpoint,
                            f"{endpoint} ~ axis_score_z + C(dose)",
                            "dose",
                            "treatment_response_or_remission",
                            "colonic mucosal biopsy",
                            "ACT1 baseline biopsy",
                        )
                    )

    gse16879 = "data/processed/GSE16879/baseline_endpoint.tsv"
    if os.path.exists(gse16879):
        data = pd.read_csv(gse16879, sep="\t")
        data["axis_score"] = pd.to_numeric(data["axis_score"], errors="coerce")
        data["axis_score_z"] = zscore(data["axis_score"])
        data["infliximab_response"] = pd.to_numeric(data["infliximab_response"], errors="coerce")
        data = data.dropna(subset=["axis_score_z", "infliximab_response"]).copy()
        if len(data) and data["infliximab_response"].nunique() == 2:
            rows.append(
                fit_logit_adult(
                    "GSE16879",
                    data,
                    "infliximab_response",
                    "infliximab_response ~ axis_score_z",
                    "none",
                    "treatment_response_or_remission",
                    "mucosal biopsy",
                    "pretreatment first-infliximab biopsy",
                )
            )
            if data["disease"].nunique() > 1:
                rows.append(
                    fit_logit_adult(
                        "GSE16879",
                        data,
                        "infliximab_response",
                        "infliximab_response ~ axis_score_z + C(disease)",
                        "disease_subtype",
                        "treatment_response_or_remission",
                        "mucosal biopsy",
                        "pretreatment first-infliximab biopsy",
                    )
                )
    return rows


def row_from_endpoint_model(row: dict[str, str]) -> dict[str, str]:
    dataset_id = row["dataset_id"]
    if dataset_id == "GSE73661":
        age = "adult"
        family = "mucosal_healing"
        tissue = "colonic mucosal biopsy"
        treatment = "vedolizumab_or_infliximab"
        role = "adult_direct_endpoint_anchor"
    elif dataset_id == "GSE109142":
        age = "pediatric_early_onset"
        family = "treatment_response_or_remission"
        tissue = "rectal mucosal biopsy"
        treatment = "pediatric_UC_induction_context"
        role = "pediatric_remission_context"
    else:
        age = "unresolved"
        family = "unresolved"
        tissue = "unresolved"
        treatment = "unresolved"
        role = "supportive"
    return {
        "dataset_id": dataset_id,
        "age_stratum": age,
        "endpoint_family": family,
        "endpoint_name": row["endpoint"],
        "tissue_site": tissue,
        "treatment_context": treatment,
        "model_formula": row["model_formula"],
        "covariates": row["covariates"],
        "effect_type": row["effect_type"],
        "effect": row["effect"],
        "ci_lower": row["ci_lower"],
        "ci_upper": row["ci_upper"],
        "pvalue": row["pvalue"],
        "fdr": row["fdr"],
        "n": row["n"],
        "interpretation_boundary": row["interpretation_boundary"],
        "clinical_role": role,
    }


def row_from_replication(row: dict[str, str]) -> dict[str, str]:
    family = "phenotype_validation"
    if row["endpoint"] in {"macroscopic_inflammation_vs_normal_or_microscopic", "deep_ulcer"}:
        family = "ulceration_or_inflammatory_severity"
    dataset_info = {
        "GSE57945": ("pediatric_early_onset", "ileal mucosal biopsy", "treatment-naive pediatric IBD phenotype", "pediatric_mucosal_injury_support"),
        "GSE101794": ("pediatric_early_onset", "ileal mucosal biopsy", "treatment-naive pediatric CD phenotype", "pediatric_phenotype_support"),
        "GSE117993": ("pediatric_early_onset", "rectal mucosal biopsy", "pediatric IBD phenotype", "pediatric_phenotype_support"),
    }
    age, tissue, treatment, role = dataset_info.get(row["dataset_id"], ("unresolved", "unresolved", "unresolved", "supportive"))
    return {
        "dataset_id": row["dataset_id"],
        "age_stratum": age,
        "endpoint_family": family,
        "endpoint_name": row["endpoint"],
        "tissue_site": tissue,
        "treatment_context": treatment,
        "model_formula": row["model"],
        "covariates": "none" if " + " not in row["model"] else "available_covariates",
        "effect_type": row["effect_type"],
        "effect": row["effect"],
        "ci_lower": row["ci_lower"],
        "ci_upper": row["ci_upper"],
        "pvalue": row["pvalue"],
        "fdr": row["fdr"],
        "n": row["n"],
        "interpretation_boundary": row["interpretation_boundary"],
        "clinical_role": role,
    }


def load_existing_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open("results/clinical/endpoint_models.tsv", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append(row_from_endpoint_model(row))
    with open("results/replication/combined_summary.tsv", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append(row_from_replication(row))
    return rows


def write_table(path: str, rows: list[dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "dataset_id",
        "age_stratum",
        "endpoint_family",
        "endpoint_name",
        "tissue_site",
        "treatment_context",
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
        "clinical_role",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def synthesis_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        effect = float(row["effect"])
        endpoint_group = row["endpoint_family"]
        pooling = "no"
        reason = "single_or_endpoint_incompatible"
        if endpoint_group in {"mucosal_healing", "mucosal_healing_or_endoscopic_histologic_healing"} and row["age_stratum"] == "adult":
            reason = "adult direct endpoint family; pooling requires a second compatible adult direct endpoint and aligned direction"
        out.append(
            {
                "dataset_id": row["dataset_id"],
                "age_stratum": row["age_stratum"],
                "endpoint_family": row["endpoint_family"],
                "endpoint_name": row["endpoint_name"],
                "endpoint_compatibility_group": endpoint_group,
                "effect_direction": "inverse_or_lower_score_favors_endpoint" if effect < 1 else "positive_or_higher_score_associates_with_endpoint",
                "pooling_eligible": pooling,
                "non_pooling_reason": reason,
                "clinical_role": row["clinical_role"],
                "n": row["n"],
            }
        )
    return out


def write_synthesis(path: str, rows: list[dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "dataset_id",
        "age_stratum",
        "endpoint_family",
        "endpoint_name",
        "endpoint_compatibility_group",
        "effect_direction",
        "pooling_eligible",
        "non_pooling_reason",
        "clinical_role",
        "n",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    gse12251 = gse12251_rows()
    write_table(GSE12251_MODEL_OUT, gse12251)
    gse92415 = gse92415_rows()
    if gse92415:
        write_table(GSE92415_MODEL_OUT, gse92415)
    adult_expansion = adult_expansion_rows()
    if adult_expansion:
        write_table(ADULT_EXPANSION_MODEL_OUT, adult_expansion)
    rows = load_existing_rows() + gse12251 + gse92415 + adult_expansion
    pvalues = [float(row["pvalue"]) for row in rows]
    for row, fdr in zip(rows, bh_fdr(pvalues)):
        row["fdr"] = f"{fdr:.6g}"
    write_table(OUT, rows)
    write_synthesis(SYNTHESIS_OUT, synthesis_rows(rows))
    print(f"wrote={OUT} rows={len(rows)}; wrote={SYNTHESIS_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
