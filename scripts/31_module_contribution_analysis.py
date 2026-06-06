#!/usr/bin/env python3
"""Module contribution and score-variant sensitivity analyses."""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


OUT_MODELS = "results/clinical/module_contribution_models.tsv"
OUT_CORR = "results/clinical/module_contribution_correlations.tsv"

MODULES = {
    "regulatory_module": {"IPMK": -1, "IPPK": -1, "HDAC3": -1, "NCOR1": -1, "NCOR2": -1},
    "mmp_injury_module": {"MMP1": 1, "MMP3": 1, "MMP10": 1, "MMP12": 1, "MMP13": 1},
    "junctional_module": {"TJP1": -1, "OCLN": -1, "CLDN2": 1},
    "junctional_no_cldn2": {"TJP1": -1, "OCLN": -1},
    "cldn2_only": {"CLDN2": 1},
}

VARIANTS = {
    "original_axis": ("regulatory_module", "mmp_injury_module", "junctional_module"),
    "mmp_only": ("mmp_injury_module",),
    "regulatory_only": ("regulatory_module",),
    "junctional_only": ("junctional_module",),
    "axis_without_mmp": ("regulatory_module", "junctional_module"),
    "axis_without_regulatory": ("mmp_injury_module", "junctional_module"),
    "axis_without_junctional": ("regulatory_module", "mmp_injury_module"),
    "axis_without_cldn2": ("regulatory_module", "mmp_injury_module", "junctional_no_cldn2"),
    "cldn2_only": ("cldn2_only",),
}


@dataclass(frozen=True)
class EndpointSpec:
    dataset_id: str
    endpoint_path: str
    expression_path: str
    sample_col: str
    endpoint_col: str
    covariates: tuple[str, ...] = ()
    role: str = ""


SPECS = [
    EndpointSpec("GSE73661", "data/processed/GSE73661/baseline_endpoint.tsv", "data/processed/GSE73661/axis_gene_expression.tsv", "baseline_sample_id", "mucosal_healing", ("C(treatment_family)",), "adult_mucosal_healing"),
    EndpointSpec("GSE12251", "data/processed/GSE12251/baseline_endpoint.tsv", "data/processed/GSE12251/axis_gene_expression.tsv", "sample_id", "wk8_endoscopic_histologic_healing", ("C(dose)",), "adult_healing_like"),
    EndpointSpec("GSE92415", "data/processed/GSE92415/baseline_endpoint.tsv", "data/processed/GSE92415/axis_gene_expression.tsv", "sample_id", "week6_clinical_response", ("C(treatment)", "baseline_mayo_score_z"), "adult_response"),
    EndpointSpec("GSE206285", "data/processed/GSE206285/baseline_endpoint.tsv", "data/processed/GSE206285/axis_gene_expression.tsv", "sample_id", "week8_mucosal_healing", ("C(treatment_family)",), "adult_mucosal_healing"),
    EndpointSpec("GSE206285", "data/processed/GSE206285/baseline_endpoint.tsv", "data/processed/GSE206285/axis_gene_expression.tsv", "sample_id", "week8_clinical_remission", ("C(treatment_family)",), "adult_remission"),
    EndpointSpec("GSE23597", "data/processed/GSE23597/baseline_endpoint.tsv", "data/processed/GSE23597/axis_gene_expression.tsv", "sample_id", "week8_response", ("C(dose)",), "adult_response"),
    EndpointSpec("GSE23597", "data/processed/GSE23597/baseline_endpoint.tsv", "data/processed/GSE23597/axis_gene_expression.tsv", "sample_id", "week30_response", ("C(dose)",), "adult_response"),
    EndpointSpec("GSE16879", "data/processed/GSE16879/baseline_endpoint.tsv", "data/processed/GSE16879/axis_gene_expression.tsv", "sample_id", "infliximab_response", ("C(disease)",), "adult_or_mixed_response"),
    EndpointSpec("GSE109142", "data/processed/GSE109142/baseline_endpoint.tsv", "data/processed/GSE109142/axis_gene_expression.tsv", "sample_id", "week4_remission", ("total_mayo_score_num_z", "C(initial_treatment)"), "pediatric_remission"),
]


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=1)
    if sd == 0 or math.isnan(sd):
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - s.mean()) / sd


def read_expression_scores(path: str) -> pd.DataFrame:
    expr = pd.read_csv(path, sep="\t").set_index("Gene Symbol")
    expr.index = expr.index.astype(str).str.upper()
    out = pd.DataFrame({"sample_id_for_expr": expr.columns})
    module_scores: dict[str, pd.Series] = {}
    for module_name, genes in MODULES.items():
        signed = []
        for gene, direction in genes.items():
            if gene in expr.index:
                signed.append(zscore(expr.loc[gene].astype(float)) * direction)
        if len(signed) >= max(1, math.ceil(len(genes) * 0.5)):
            module_scores[module_name] = pd.concat(signed, axis=1).mean(axis=1)
        else:
            module_scores[module_name] = pd.Series(np.nan, index=expr.columns)
    for module_name, scores in module_scores.items():
        out[module_name] = scores.reindex(expr.columns).values
    for variant, modules in VARIANTS.items():
        out[variant] = out[list(modules)].mean(axis=1)
    return out


def prepare_endpoint(spec: EndpointSpec) -> pd.DataFrame:
    endpoint = pd.read_csv(spec.endpoint_path, sep="\t")
    if spec.dataset_id == "GSE206285" and "treatment_family" not in endpoint.columns:
        endpoint["treatment_family"] = endpoint["treatment"].astype(str).str.extract(r"^(Ustekinumab|Placebo)", expand=False).fillna("Other")
    if spec.dataset_id == "GSE92415" and "baseline_mayo_score_z" not in endpoint.columns:
        endpoint["baseline_mayo_score_z"] = zscore(endpoint["baseline_mayo_score"])
    if spec.dataset_id == "GSE109142" and "total_mayo_score_num_z" not in endpoint.columns:
        endpoint["total_mayo_score_num_z"] = zscore(endpoint["total_mayo_score_num"])
    scores = read_expression_scores(spec.expression_path)
    data = endpoint.merge(scores, left_on=spec.sample_col, right_on="sample_id_for_expr", how="inner")
    for col in [spec.endpoint_col, *VARIANTS.keys()]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=[spec.endpoint_col, "original_axis"]).copy()
    for variant in VARIANTS:
        data[f"{variant}_z"] = zscore(data[variant])
    return data


def fit_logit(data: pd.DataFrame, endpoint: str, term: str, covariates: tuple[str, ...]) -> dict[str, str]:
    covar = " + ".join(covariates)
    formula = f"{endpoint} ~ {term}" + (f" + {covar}" if covar else "")
    try:
        model = smf.logit(formula, data=data).fit(disp=False, maxiter=200)
        coef = model.params[term]
        ci = model.conf_int().loc[term]
        return {
            "model_formula": formula,
            "effect": f"{math.exp(coef):.6g}",
            "ci_lower": f"{math.exp(ci[0]):.6g}",
            "ci_upper": f"{math.exp(ci[1]):.6g}",
            "pvalue": f"{model.pvalues[term]:.6g}",
            "aic": f"{model.aic:.6g}",
            "mcfadden_r2": f"{1 - model.llf / model.llnull:.6g}" if model.llnull != 0 else "NA",
            "status": "ok",
        }
    except Exception as exc:
        return {
            "model_formula": formula,
            "effect": "NA",
            "ci_lower": "NA",
            "ci_upper": "NA",
            "pvalue": "NA",
            "aic": "NA",
            "mcfadden_r2": "NA",
            "status": f"failed:{type(exc).__name__}",
        }


def main() -> int:
    rows: list[dict[str, str]] = []
    corr_rows: list[dict[str, str]] = []
    for spec in SPECS:
        data = prepare_endpoint(spec)
        if data.empty or data[spec.endpoint_col].nunique() < 2:
            continue
        for variant in VARIANTS:
            term = f"{variant}_z"
            result = fit_logit(data.dropna(subset=[term]), spec.endpoint_col, term, spec.covariates)
            rows.append(
                {
                    "dataset_id": spec.dataset_id,
                    "endpoint": spec.endpoint_col,
                    "role": spec.role,
                    "variant": variant,
                    "n": str(len(data.dropna(subset=[term]))),
                    "covariates": ";".join(spec.covariates) if spec.covariates else "none",
                    **result,
                }
            )
        for module in ["regulatory_module", "mmp_injury_module", "junctional_module", "junctional_no_cldn2", "cldn2_only"]:
            if module == "mmp_injury_module":
                tmp = data[[module]].dropna()
                rho, pval = (1.0, 0.0)
            else:
                tmp = data[[module, "mmp_injury_module"]].dropna()
                rho, pval = (stats.spearmanr(tmp[module], tmp["mmp_injury_module"]) if len(tmp) >= 8 else (np.nan, np.nan))
            corr_rows.append(
                {
                    "dataset_id": spec.dataset_id,
                    "endpoint": spec.endpoint_col,
                    "module": module,
                    "n": str(len(tmp)),
                    "spearman_r_with_mmp_module": f"{rho:.6g}" if not np.isnan(rho) else "NA",
                    "spearman_p": f"{pval:.6g}" if not np.isnan(pval) else "NA",
                }
            )
    os.makedirs("results/clinical", exist_ok=True)
    with open(OUT_MODELS, "w", newline="") as handle:
        fields = ["dataset_id", "endpoint", "role", "variant", "n", "covariates", "model_formula", "effect", "ci_lower", "ci_upper", "pvalue", "aic", "mcfadden_r2", "status"]
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(OUT_CORR, "w", newline="") as handle:
        fields = ["dataset_id", "endpoint", "module", "n", "spearman_r_with_mmp_module", "spearman_p"]
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(corr_rows)
    print(f"wrote={OUT_MODELS} rows={len(rows)}")
    print(f"wrote={OUT_CORR} rows={len(corr_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
