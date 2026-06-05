#!/usr/bin/env python3
"""Export compact source-data tables for each submission figure."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT = Path("results/figures/source_data")


def read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def write_tsv(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, sep="\t", index=False)


def existing_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def figure1() -> None:
    cohorts = read_tsv("results/dataset_summary.tsv")
    models = read_tsv("results/clinical/age_stratified_endpoint_models.tsv")
    model_counts = (
        models.groupby(["dataset_id", "age_stratum", "clinical_role"], dropna=False)
        .size()
        .reset_index(name="modeled_endpoint_rows")
    )
    out = cohorts.merge(model_counts, on="dataset_id", how="left", suffixes=("", "_model"))
    write_tsv(out, "Figure1_study_architecture_source_data.tsv")


def figure2() -> None:
    strata = read_tsv("results/clinical/clinical_score_strata_summary.tsv")
    models = read_tsv("results/clinical/age_stratified_endpoint_models.tsv")
    adult_strata = strata.loc[strata["age_stratum"].str.contains("adult", na=False)].copy()
    adult_models = models.loc[models["age_stratum"].eq("adult")].copy()
    adult_strata["source_panel"] = "score_tertile_event_rates"
    adult_models["source_panel"] = "adult_endpoint_effect_estimates"
    write_tsv(adult_strata, "Figure2_adult_score_strata_source_data.tsv")
    write_tsv(adult_models, "Figure2_adult_endpoint_models_source_data.tsv")


def figure3() -> None:
    models = read_tsv("results/clinical/age_stratified_endpoint_models.tsv")
    cols = existing_cols(
        models,
        [
            "dataset_id",
            "age_stratum",
            "endpoint_family",
            "endpoint_name",
            "tissue_site",
            "treatment_context",
            "clinical_role",
            "effect",
            "ci_lower",
            "ci_upper",
            "pvalue",
            "fdr",
            "n",
            "interpretation_boundary",
        ],
    )
    write_tsv(models[cols], "Figure3_age_stratified_atlas_source_data.tsv")


def figure4() -> None:
    evidence = read_tsv("results/validation/evidence_grading.tsv")
    non_pool = read_tsv("results/meta/non_poolable_endpoints.tsv")
    specificity = read_tsv("results/clinical/inflammation_specificity_correlations.tsv")
    evidence["source_panel"] = "evidence_grading"
    non_pool["source_panel"] = "non_pooling_boundary"
    specificity["source_panel"] = "inflammation_specificity_feasibility"
    write_tsv(evidence, "Figure4_evidence_grading_source_data.tsv")
    write_tsv(non_pool, "Figure4_non_pooling_source_data.tsv")
    write_tsv(specificity, "Figure4_specificity_feasibility_source_data.tsv")


def supplementary() -> None:
    write_tsv(read_tsv("results/clinical/sensitivity_models.tsv"), "FigureS1_sensitivity_models_source_data.tsv")

    module_frames: list[pd.DataFrame] = []
    figure_s2_specs = [
        ("GSE73661", "GSE73661 healing", "mucosal_healing"),
        ("GSE206285", "GSE206285 healing", "week8_mucosal_healing"),
        ("GSE206285", "GSE206285 remission", "week8_clinical_remission"),
    ]
    module_specs = [
        ("upstream_score", "Upstream"),
        ("mmp_score", "MMP injury"),
        ("junction_score", "Junction"),
    ]
    for dataset_id, endpoint_label, endpoint_col in figure_s2_specs:
        path = Path("data/processed") / dataset_id / "baseline_endpoint.tsv"
        if not path.exists():
            continue
        df = read_tsv(str(path))
        if endpoint_col not in df.columns:
            continue
        id_candidates = existing_cols(df, ["sample_id", "baseline_sample_id", "donor_id", "patient_id"])
        record_id = df[id_candidates[0]].astype(str) if id_candidates else df.index.astype(str)
        endpoint_values = pd.to_numeric(df[endpoint_col], errors="coerce")
        for module_col, module_name in module_specs:
            if module_col not in df.columns:
                continue
            module_values = pd.to_numeric(df[module_col], errors="coerce")
            long_df = pd.DataFrame(
                {
                    "dataset_id": dataset_id,
                    "endpoint_label": endpoint_label,
                    "endpoint_column": endpoint_col,
                    "record_id": record_id,
                    "module": module_name,
                    "module_score": module_values,
                    "favorable_endpoint": endpoint_values,
                }
            ).dropna(subset=["module_score", "favorable_endpoint"])
            long_df["favorable_endpoint"] = long_df["favorable_endpoint"].astype(int)
            module_frames.append(long_df)
    if module_frames:
        write_tsv(pd.concat(module_frames, ignore_index=True, sort=False), "FigureS2_module_detail_source_data.tsv")

    gse206285 = read_tsv("data/processed/GSE206285/baseline_endpoint.tsv")
    cols = existing_cols(
        gse206285,
        [
            "donor_id",
            "sample_id",
            "treatment",
            "treatment_family",
            "week8_mucosal_healing",
            "week8_clinical_remission",
            "axis_score",
            "upstream_score",
            "mmp_score",
            "junction_score",
        ],
    )
    write_tsv(gse206285[cols], "FigureS3_gse206285_remission_source_data.tsv")


def main() -> int:
    figure1()
    figure2()
    figure3()
    figure4()
    supplementary()
    print(f"wrote_source_data_dir={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
