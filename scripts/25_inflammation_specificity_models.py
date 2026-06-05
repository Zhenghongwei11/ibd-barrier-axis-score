#!/usr/bin/env python3
"""Assess whether barrier-axis associations persist after generic inflammation adjustment."""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


COMPARATOR_GENES = ["OSM", "TREM1", "TNF", "TNFRSF1B", "IL6", "IL1B", "CXCL8", "IL13RA2"]
OUT_MODELS = "results/clinical/inflammation_specificity_models.tsv"
OUT_CORR = "results/clinical/inflammation_specificity_correlations.tsv"
AXIS_SCORES = "results/axis/barrier_axis_scores.tsv"


@dataclass(frozen=True)
class EndpointSpec:
    dataset_id: str
    endpoint_path: str
    expression_path: str
    sample_col: str
    endpoint_col: str
    covariates: tuple[str, ...] = ()


SPECS = [
    EndpointSpec("GSE73661", "data/processed/GSE73661/baseline_endpoint.tsv", "data/processed/GSE73661/comparator_gene_expression.tsv", "baseline_sample_id", "mucosal_healing", ("C(treatment_family)",)),
    EndpointSpec("GSE12251", "data/processed/GSE12251/baseline_endpoint.tsv", "data/processed/GSE12251/axis_gene_expression.tsv", "sample_id", "wk8_endoscopic_histologic_healing"),
    EndpointSpec("GSE92415", "data/processed/GSE92415/baseline_endpoint.tsv", "data/processed/GSE92415/axis_gene_expression.tsv", "sample_id", "week6_clinical_response", ("C(treatment)", "baseline_mayo_score_z")),
    EndpointSpec("GSE206285", "data/processed/GSE206285/baseline_endpoint.tsv", "data/processed/GSE206285/axis_gene_expression.tsv", "sample_id", "week8_mucosal_healing", ("C(treatment_family)",)),
    EndpointSpec("GSE206285", "data/processed/GSE206285/baseline_endpoint.tsv", "data/processed/GSE206285/axis_gene_expression.tsv", "sample_id", "week8_clinical_remission", ("C(treatment_family)",)),
    EndpointSpec("GSE23597", "data/processed/GSE23597/baseline_endpoint.tsv", "data/processed/GSE23597/axis_gene_expression.tsv", "sample_id", "week8_response", ("C(dose)",)),
    EndpointSpec("GSE23597", "data/processed/GSE23597/baseline_endpoint.tsv", "data/processed/GSE23597/axis_gene_expression.tsv", "sample_id", "week30_response", ("C(dose)",)),
    EndpointSpec("GSE16879", "data/processed/GSE16879/baseline_endpoint.tsv", "data/processed/GSE16879/axis_gene_expression.tsv", "sample_id", "infliximab_response", ("C(disease)",)),
    EndpointSpec("GSE109142", "data/processed/GSE109142/baseline_endpoint.tsv", "data/processed/GSE109142/axis_gene_expression.tsv", "sample_id", "week4_remission", ("total_mayo_score_num_z", "C(initial_treatment)")),
    EndpointSpec("GSE101794", "data/processed/GSE101794/metadata.tsv", "data/processed/GSE101794/axis_gene_expression.tsv", "sample_id", "cd_status", ("age_z", "C(sex)",)),
    EndpointSpec("GSE117993", "data/processed/GSE117993/metadata.tsv", "data/processed/GSE117993/axis_gene_expression.tsv", "sample_id", "ibd_status"),
]


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=1)
    if sd == 0 or math.isnan(sd):
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - s.mean()) / sd


def read_inflammation_score(path: str) -> tuple[pd.DataFrame, str]:
    expr = pd.read_csv(path, sep="\t").set_index("Gene Symbol")
    available = [g for g in COMPARATOR_GENES if g in expr.index]
    if len(available) < 2:
        return pd.DataFrame(), ";".join(available)
    z = expr.loc[available].apply(lambda row: zscore(row.astype(float)), axis=1)
    score = z.mean(axis=0)
    return pd.DataFrame({"sample_id_for_expr": score.index, "inflammation_score": score.values}), ";".join(available)


def prepare_endpoint(spec: EndpointSpec) -> tuple[pd.DataFrame, str]:
    endpoint = pd.read_csv(spec.endpoint_path, sep="\t")
    if "axis_score" not in endpoint.columns:
        axis = pd.read_csv(AXIS_SCORES, sep="\t")
        axis = axis.loc[axis["dataset_id"].eq(spec.dataset_id), ["sample_id", "axis_score", "upstream_score", "mmp_score", "junction_score"]]
        endpoint = endpoint.merge(axis, left_on=spec.sample_col, right_on="sample_id", how="left", suffixes=("", "_axis"))
        if spec.sample_col != "sample_id" and "sample_id_axis" in endpoint.columns:
            endpoint = endpoint.drop(columns=["sample_id_axis"])
    if spec.dataset_id == "GSE206285" and "treatment_family" not in endpoint.columns:
        endpoint["treatment_family"] = endpoint["treatment"].astype(str).str.extract(r"^(Ustekinumab|Placebo)", expand=False).fillna("Other")
    if spec.dataset_id == "GSE101794":
        endpoint["cd_status"] = endpoint["diagnosis"].astype(str).str.upper().eq("CD").astype(int)
        endpoint["age_z"] = zscore(endpoint["age_at_diagnosis_in_years"])
    if spec.dataset_id == "GSE117993":
        endpoint["ibd_status"] = endpoint["diagnosis"].astype(str).str.upper().ne("CONTROL").astype(int)
    if spec.dataset_id == "GSE92415":
        endpoint["baseline_mayo_score_z"] = zscore(endpoint["baseline_mayo_score"])
    if spec.dataset_id == "GSE109142":
        endpoint["total_mayo_score_num_z"] = zscore(endpoint["total_mayo_score_num"])

    inflammation, available_genes = read_inflammation_score(spec.expression_path)
    if inflammation.empty:
        return pd.DataFrame(), available_genes
    data = endpoint.merge(inflammation, left_on=spec.sample_col, right_on="sample_id_for_expr", how="inner")
    for col in ["axis_score", "inflammation_score", spec.endpoint_col]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["axis_score", "inflammation_score", spec.endpoint_col]).copy()
    data["axis_score_z"] = zscore(data["axis_score"])
    data["inflammation_score_z"] = zscore(data["inflammation_score"])
    return data, available_genes


def fit_model(formula: str, data: pd.DataFrame, term: str) -> tuple[str, str, str, str, str, object | None]:
    try:
        model = smf.logit(formula, data=data).fit(disp=False, maxiter=200)
        coef = model.params[term]
        ci = model.conf_int().loc[term]
        return (
            f"{math.exp(coef):.6g}",
            f"{math.exp(ci[0]):.6g}",
            f"{math.exp(ci[1]):.6g}",
            f"{model.pvalues[term]:.6g}",
            "ok",
            model,
        )
    except Exception as exc:
        return "NA", "NA", "NA", "NA", f"failed:{type(exc).__name__}", None


def model_rows(spec: EndpointSpec) -> tuple[list[dict[str, str]], dict[str, str]]:
    data, available_genes = prepare_endpoint(spec)
    corr_row = {
        "dataset_id": spec.dataset_id,
        "endpoint": spec.endpoint_col,
        "n": str(len(data)),
        "available_inflammation_genes": available_genes,
        "axis_inflammation_spearman_r": "NA",
        "axis_inflammation_spearman_p": "NA",
        "status": "ok" if not data.empty else "no_aligned_data_or_insufficient_genes",
    }
    if len(data) >= 8:
        rho, pval = stats.spearmanr(data["axis_score"], data["inflammation_score"], nan_policy="omit")
        corr_row["axis_inflammation_spearman_r"] = f"{rho:.6g}"
        corr_row["axis_inflammation_spearman_p"] = f"{pval:.6g}"
    if data.empty or data[spec.endpoint_col].nunique() < 2 or len(data) < 8:
        return [], corr_row

    covar = " + ".join(spec.covariates)
    covar_suffix = f" + {covar}" if covar else ""
    formula_axis = f"{spec.endpoint_col} ~ axis_score_z{covar_suffix}"
    formula_infl = f"{spec.endpoint_col} ~ inflammation_score_z{covar_suffix}"
    formula_joint = f"{spec.endpoint_col} ~ axis_score_z + inflammation_score_z{covar_suffix}"

    axis_eff, axis_lo, axis_hi, axis_p, axis_status, axis_model = fit_model(formula_axis, data, "axis_score_z")
    infl_eff, infl_lo, infl_hi, infl_p, infl_status, infl_model = fit_model(formula_infl, data, "inflammation_score_z")
    joint_axis_eff, joint_axis_lo, joint_axis_hi, joint_axis_p, joint_status, joint_model = fit_model(formula_joint, data, "axis_score_z")
    joint_infl_eff, joint_infl_lo, joint_infl_hi, joint_infl_p, _, _ = fit_model(formula_joint, data, "inflammation_score_z")

    lr_p = "NA"
    if infl_model is not None and joint_model is not None:
        lr_stat = 2 * (joint_model.llf - infl_model.llf)
        lr_p = f"{stats.chi2.sf(lr_stat, df=1):.6g}"

    base = {
        "dataset_id": spec.dataset_id,
        "endpoint": spec.endpoint_col,
        "n": str(int(joint_model.nobs) if joint_model is not None else len(data)),
        "available_inflammation_genes": available_genes,
        "covariates": ";".join(spec.covariates) if spec.covariates else "none",
    }
    rows = [
        {**base, "model_type": "axis_only", "model": formula_axis, "term": "axis_score_z", "effect": axis_eff, "ci_lower": axis_lo, "ci_upper": axis_hi, "pvalue": axis_p, "lr_p_axis_increment_over_inflammation": "NA", "status": axis_status},
        {**base, "model_type": "inflammation_only", "model": formula_infl, "term": "inflammation_score_z", "effect": infl_eff, "ci_lower": infl_lo, "ci_upper": infl_hi, "pvalue": infl_p, "lr_p_axis_increment_over_inflammation": "NA", "status": infl_status},
        {**base, "model_type": "joint_adjusted", "model": formula_joint, "term": "axis_score_z", "effect": joint_axis_eff, "ci_lower": joint_axis_lo, "ci_upper": joint_axis_hi, "pvalue": joint_axis_p, "lr_p_axis_increment_over_inflammation": lr_p, "status": joint_status},
        {**base, "model_type": "joint_adjusted", "model": formula_joint, "term": "inflammation_score_z", "effect": joint_infl_eff, "ci_lower": joint_infl_lo, "ci_upper": joint_infl_hi, "pvalue": joint_infl_p, "lr_p_axis_increment_over_inflammation": "NA", "status": joint_status},
    ]
    return rows, corr_row


def write_tsv(path: str, rows: list[dict[str, str]], fields: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    model_out: list[dict[str, str]] = []
    corr_out: list[dict[str, str]] = []
    for spec in SPECS:
        rows, corr = model_rows(spec)
        model_out.extend(rows)
        corr_out.append(corr)
    write_tsv(
        OUT_MODELS,
        model_out,
        ["dataset_id", "endpoint", "n", "available_inflammation_genes", "covariates", "model_type", "model", "term", "effect", "ci_lower", "ci_upper", "pvalue", "lr_p_axis_increment_over_inflammation", "status"],
    )
    write_tsv(
        OUT_CORR,
        corr_out,
        ["dataset_id", "endpoint", "n", "available_inflammation_genes", "axis_inflammation_spearman_r", "axis_inflammation_spearman_p", "status"],
    )
    print(f"wrote={OUT_MODELS} rows={len(model_out)}")
    print(f"wrote={OUT_CORR} rows={len(corr_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
