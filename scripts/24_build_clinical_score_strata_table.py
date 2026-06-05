#!/usr/bin/env python3
"""Build clinically interpretable endpoint rates by barrier-axis score strata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DETAIL_OUT = Path("results/clinical/clinical_score_strata.tsv")
SUMMARY_OUT = Path("results/clinical/clinical_score_strata_summary.tsv")


ENDPOINTS = [
    {
        "dataset_id": "GSE73661",
        "age_stratum": "adult",
        "path": "data/processed/GSE73661/baseline_endpoint.tsv",
        "endpoint_col": "mucosal_healing",
        "endpoint_label": "Mucosal healing",
        "clinical_role": "adult direct endpoint",
    },
    {
        "dataset_id": "GSE12251",
        "age_stratum": "adult",
        "path": "data/processed/GSE12251/baseline_endpoint.tsv",
        "endpoint_col": "wk8_endoscopic_histologic_healing",
        "endpoint_label": "Week 8 endoscopic and histologic healing",
        "clinical_role": "adult direct endpoint",
    },
    {
        "dataset_id": "GSE92415",
        "age_stratum": "adult",
        "path": "data/processed/GSE92415/baseline_endpoint.tsv",
        "endpoint_col": "week6_clinical_response",
        "endpoint_label": "Week 6 clinical response",
        "clinical_role": "adult direct endpoint",
    },
    {
        "dataset_id": "GSE206285",
        "age_stratum": "adult",
        "path": "data/processed/GSE206285/baseline_endpoint.tsv",
        "endpoint_col": "week8_mucosal_healing",
        "endpoint_label": "Week 8 mucosal healing",
        "clinical_role": "adult direct endpoint",
    },
    {
        "dataset_id": "GSE206285",
        "age_stratum": "adult",
        "path": "data/processed/GSE206285/baseline_endpoint.tsv",
        "endpoint_col": "week8_clinical_remission",
        "endpoint_label": "Week 8 clinical remission",
        "clinical_role": "adult direct endpoint",
    },
    {
        "dataset_id": "GSE23597",
        "age_stratum": "adult",
        "path": "data/processed/GSE23597/baseline_endpoint.tsv",
        "endpoint_col": "week8_response",
        "endpoint_label": "Week 8 response",
        "clinical_role": "adult direct endpoint",
    },
    {
        "dataset_id": "GSE23597",
        "age_stratum": "adult",
        "path": "data/processed/GSE23597/baseline_endpoint.tsv",
        "endpoint_col": "week30_response",
        "endpoint_label": "Week 30 response",
        "clinical_role": "adult direct endpoint",
    },
    {
        "dataset_id": "GSE16879",
        "age_stratum": "adult_or_mixed",
        "path": "data/processed/GSE16879/baseline_endpoint.tsv",
        "endpoint_col": "infliximab_response",
        "endpoint_label": "Infliximab response",
        "clinical_role": "adult or mixed direct endpoint",
    },
    {
        "dataset_id": "GSE109142",
        "age_stratum": "pediatric_or_early_onset",
        "path": "data/processed/GSE109142/baseline_endpoint.tsv",
        "endpoint_col": "week4_remission",
        "endpoint_label": "Week 4 remission",
        "clinical_role": "pediatric direct remission context",
    },
]


def zscore(values: pd.Series) -> pd.Series:
    sd = values.std(ddof=1)
    return (values - values.mean()) / sd if sd else values * 0


def rate_label(event_n: int, n: int) -> str:
    return f"{event_n}/{n} ({100 * event_n / n:.1f}%)"


def build_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for spec in ENDPOINTS:
        path = Path(spec["path"])
        if not path.exists():
            continue
        data = pd.read_csv(path, sep="\t")
        endpoint_col = spec["endpoint_col"]
        if endpoint_col not in data.columns:
            continue
        data["axis_score"] = pd.to_numeric(data["axis_score"], errors="coerce")
        data[endpoint_col] = pd.to_numeric(data[endpoint_col], errors="coerce")
        data = data.dropna(subset=["axis_score", endpoint_col]).copy()
        if data.empty:
            continue

        data["axis_score_z"] = zscore(data["axis_score"])
        data["score_stratum"] = pd.qcut(
            data["axis_score_z"].rank(method="first"),
            q=3,
            labels=["Low score", "Middle score", "High score"],
        )

        rates: dict[str, str] = {}
        event_rates: dict[str, float] = {}
        for stratum in ["Low score", "Middle score", "High score"]:
            subset = data.loc[data["score_stratum"].astype(str) == stratum].copy()
            n = int(len(subset))
            event_n = int(subset[endpoint_col].sum())
            event_rate = event_n / n if n else float("nan")
            rates[stratum] = rate_label(event_n, n)
            event_rates[stratum] = event_rate
            detail_rows.append(
                {
                    "dataset_id": spec["dataset_id"],
                    "age_stratum": spec["age_stratum"],
                    "endpoint": spec["endpoint_label"],
                    "clinical_role": spec["clinical_role"],
                    "score_stratum": stratum,
                    "score_stratum_definition": "within-cohort tertile of standardized barrier-axis score",
                    "n": n,
                    "event_n": event_n,
                    "event_rate": f"{event_rate:.6g}",
                    "mean_axis_score_z": f"{subset['axis_score_z'].mean():.6g}",
                    "median_axis_score_z": f"{subset['axis_score_z'].median():.6g}",
                    "interpretation": "For healing, remission, or response endpoints, a higher event rate means a more favorable clinical outcome.",
                }
            )

        low_minus_high = event_rates["Low score"] - event_rates["High score"]
        summary_rows.append(
            {
                "dataset_id": spec["dataset_id"],
                "age_stratum": spec["age_stratum"],
                "endpoint": spec["endpoint_label"],
                "clinical_role": spec["clinical_role"],
                "modeled_n": int(len(data)),
                "low_score_event_rate": rates["Low score"],
                "middle_score_event_rate": rates["Middle score"],
                "high_score_event_rate": rates["High score"],
                "low_minus_high_percentage_points": f"{100 * low_minus_high:.1f}",
                "clinical_readout": "Lower score shows higher favorable-endpoint rate"
                if low_minus_high > 0
                else "No monotonic lower-score advantage in tertile readout",
                "boundary": "Exploratory clinical interpretability table; thresholds are cohort-internal tertiles, not locked clinical cutoffs.",
            }
        )

    return detail_rows, summary_rows


def main() -> None:
    DETAIL_OUT.parent.mkdir(parents=True, exist_ok=True)
    detail_rows, summary_rows = build_rows()
    pd.DataFrame(detail_rows).to_csv(DETAIL_OUT, sep="\t", index=False)
    pd.DataFrame(summary_rows).to_csv(SUMMARY_OUT, sep="\t", index=False)
    print(f"wrote {DETAIL_OUT} rows={len(detail_rows)}")
    print(f"wrote {SUMMARY_OUT} rows={len(summary_rows)}")


if __name__ == "__main__":
    main()
