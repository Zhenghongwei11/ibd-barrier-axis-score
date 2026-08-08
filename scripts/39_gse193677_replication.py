#!/usr/bin/env python3
"""GSE193677 (MSCCR) independent replication: score, IBD vs control, activity correlations.

Score logic is imported from scripts/04_compute_barrier_axis_scores.py so the
prespecified module weights and signed within-dataset z-score method are reused exactly.
"""

from __future__ import annotations

import csv
import gzip
import importlib.util
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent
SERIES_MATRIX = ROOT / "data/raw/GSE193677/GSE193677_series_matrix.txt.gz"
ADJCOUNTS = ROOT / "data/raw/GSE193677/GSE193677_MSCCR_Biopsy_adjcounts.txt.gz"
COUNTS = ROOT / "data/raw/GSE193677/GSE193677_MSCCR_Biopsy_counts.txt.gz"

META_OUT = ROOT / "results/replication/GSE193677_sample_metadata.tsv"
FEASIBILITY_OUT = ROOT / "results/replication/GSE193677_feasibility_log.tsv"
SCORE_OUT = ROOT / "results/axis/barrier_axis_scores.tsv"
AVAIL_OUT = ROOT / "results/axis/gene_availability.tsv"
REPL_MODELS_OUT = ROOT / "results/replication/GSE193677_replication_models.tsv"
ACTIVITY_OUT = ROOT / "results/clinical/GSE193677_activity_correlations.tsv"

# Canonical Ensembl IDs for the 13 prespecified genes (GRCh38).
GENE_ENSEMBL = {
    "IPMK": "ENSG00000151150",
    "IPPK": "ENSG00000128191",
    "HDAC3": "ENSG00000171720",
    "NCOR1": "ENSG00000141027",
    "NCOR2": "ENSG00000196498",
    "MMP1": "ENSG00000196611",
    "MMP3": "ENSG00000149968",
    "MMP10": "ENSG00000166670",
    "MMP12": "ENSG00000110347",
    "MMP13": "ENSG00000137745",
    "TJP1": "ENSG00000104067",
    "OCLN": "ENSG00000197822",
    "CLDN2": "ENSG00000165376",
}

_spec = importlib.util.spec_from_file_location(
    "barrier_axis_scoring", ROOT / "scripts/04_compute_barrier_axis_scores.py"
)
scoring = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scoring)


def parse_series_matrix() -> pd.DataFrame:
    """Parse GSE193677 series matrix into a sample-level table."""
    gsms: list[str] = []
    title_values: list[str] = []
    rows: dict[str, dict[str, str]] = {}
    with gzip.open(SERIES_MATRIX, "rt", errors="replace") as handle:
        for line in handle:
            if not line.startswith("!"):
                continue
            key, _, rest = line.rstrip("\n").partition("\t")
            cols = [c.strip().strip('"') for c in rest.split("\t") if c.strip()]
            if key == "!Sample_geo_accession":
                gsms = cols
                for g in gsms:
                    rows.setdefault(g, {})
            elif key == "!Sample_title":
                title_values = cols
            elif key.startswith("!Sample_characteristics"):
                for g, v in zip(gsms, cols):
                    if ":" in v:
                        k, val = v.split(":", 1)
                        rows[g][k.strip()] = val.strip()
    for g, v in zip(gsms, title_values):
        rows[g]["title"] = v
    frame = pd.DataFrame.from_dict(rows, orient="index").reset_index(names="gsm")
    name = frame["title"].str.split(",").str[0].str.strip()
    frame["sample_id"] = name
    m = frame["sample_id"].str.extract(r"MSCCR_reGRID_(\d+)_Biopsy_(\d+)").astype(float)
    frame["participant_id"] = m[0].astype(int)
    frame["biopsy_number"] = m[1].astype(int)
    frame["title_disease"] = frame["title"].str.split(",").str[1].str.strip()
    frame["title_site"] = frame["title"].str.split(",").str[2].str.strip().str.split(" ").str[0]
    frame["title_inflamed"] = frame["title"].str.split(",").str[2].str.contains(r"\bI\b").astype(int)
    return frame


def read_adjcounts_gene_rows() -> dict[str, list[float]]:
    """Read the 13 score genes from the MSCCR adjcounts matrix (space-separated, quoted).

    Returns {GENE_SYMBOL: [values in column order]}.
    """
    wanted = {v: k for k, v in GENE_ENSEMBL.items()}
    aggregated: dict[str, list[float]] = {}
    with gzip.open(ADJCOUNTS, "rt", errors="replace") as handle:
        header = handle.readline().split()
        n_samples = len(header)
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            gene = parts[0].strip('"')
            symbol = wanted.get(gene)
            if symbol is None:
                continue
            values = []
            for tok in parts[1 : n_samples + 1]:
                try:
                    values.append(float(tok.strip('"')))
                except ValueError:
                    values.append(float("nan"))
            aggregated[symbol] = values
    return aggregated, n_samples


def zscore_series(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=1) if s.std(ddof=1) else s * 0


def odds_ratio_row(model, term: str, label: str, n: int, extra: dict | None = None) -> dict:
    ci = model.conf_int().loc[term]
    row = {
        "analysis": label,
        "n": n,
        "term": term,
        "or_per_1sd": float(np.exp(model.params[term])),
        "ci_lower": float(np.exp(ci[0])),
        "ci_upper": float(np.exp(ci[1])),
        "pvalue": float(model.pvalues[term]),
    }
    if extra:
        row.update(extra)
    return row


def spearman_row(series_x: pd.Series, series_y: pd.Series, label: str, n_label: int) -> dict:
    mask = series_x.notna() & series_y.notna()
    x, y = series_x[mask], series_y[mask]
    rho, p = stats.spearmanr(x, y)
    n = len(x)
    if n >= 4 and abs(rho) < 1:
        z = math.atanh(rho)
        se = 1.0 / math.sqrt(n - 3)
        lo, hi = math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)
    else:
        lo = hi = float("nan")
    return {
        "activity_measure": label,
        "n_pairs": n,
        "n_ibd_participants": n_label,
        "spearman_rho": float(rho),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "pvalue": float(p),
    }


def main() -> int:
    os.makedirs(ROOT / "results/replication", exist_ok=True)
    os.makedirs(ROOT / "results/clinical", exist_ok=True)

    # Idempotency: remove any prior GSE193677 rows from the canonical tables.
    for path, col in [(SCORE_OUT, 1), (AVAIL_OUT, 0)]:
        if path.exists():
            with open(path, "r", newline="") as handle:
                header = handle.readline()
                keep = [line for line in handle if not line.split("\t")[col] == "GSE193677"]
            with open(path, "w", newline="") as handle:
                handle.write(header)
                handle.writelines(keep)

    meta = parse_series_matrix()
    meta["disease_group"] = meta["ibd_disease"].replace({"CD": "IBD", "UC": "IBD", "Control": "Control"})
    meta["ibd_vs_control"] = (meta["disease_group"] == "IBD").astype(int)
    meta["ibd_endoseverity_num"] = meta["ibd_endoseverity_4levels"].map(
        {"Inactive": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
    )
    for col in [
        "ibd_endoseverity_num", "ibdmesuc_mayo_score", "ibdsescd_totalsescd",
        "nancyindex", "ghas_sum7", "colitisactivityindex_sccai",
        "harveybradshawindex_hbi_score", "crp_jjmgl_log2", "log2_fecalcalpro_mgperg",
        "study_eligibility_age_at_endo",
    ]:
        meta[col] = pd.to_numeric(meta[col], errors="coerce")

    aggregated, n_expr = read_adjcounts_gene_rows()
    # Read expression header to obtain column order.
    with gzip.open(ADJCOUNTS, "rt", errors="replace") as handle:
        header = [t.strip('"') for t in handle.readline().split()]
    expr_sample_ids = header

    joined = meta[meta["sample_id"].isin(expr_sample_ids)].copy()
    missing_meta = len(expr_sample_ids) - len(joined)
    missing_expr = len(meta) - len(joined)
    print(f"expression columns={len(expr_sample_ids)} metadata={len(meta)} joined={len(joined)} "
          f"missing_meta={missing_meta} missing_expr={missing_expr}")

    sample_id_to_idx = {sid: i for i, sid in enumerate(expr_sample_ids)}
    joined = joined.sort_values("sample_id").reset_index(drop=True)
    gene_by_sample: dict[str, list[float]] = {}
    for gene, values in aggregated.items():
        gene_by_sample[gene] = [values[sample_id_to_idx[sid]] for sid in joined["sample_id"]]

    rows = scoring.component_scores(gene_by_sample, len(joined))
    score_frame = pd.DataFrame(
        [
            {
                "sample_id": sid,
                "dataset_id": "GSE193677",
                "axis_score": r[0],
                "upstream_score": r[1]["upstream"],
                "mmp_score": r[1]["mmp"],
                "junction_score": r[1]["junction"],
                "scoring_method": "signed_within_dataset_zscore_mean",
                "qc_flags": "ok" if not math.isnan(r[0]) else "insufficient_axis_genes",
            }
            for sid, r in zip(joined["sample_id"], rows)
        ]
    )
    joined = joined.merge(score_frame, on="sample_id", how="left")

    # Append to canonical score tables.
    exists = SCORE_OUT.exists()
    score_frame.to_csv(SCORE_OUT, sep="\t", index=False, mode="a", header=not exists)
    avail = pd.DataFrame(
        [
            {"dataset_id": "GSE193677", "module": mod, "gene": gene, "available": "yes" if gene in aggregated else "no"}
            for mod, genes in scoring.MODULES.items()
            for gene in genes
        ]
    )
    exists = AVAIL_OUT.exists()
    avail.to_csv(AVAIL_OUT, sep="\t", index=False, mode="a", header=not exists)

    meta.to_csv(META_OUT, sep="\t", index=False)

    # Feasibility log.
    feas = [
        {"stratum": "total_samples", "n": int(len(joined))},
        {"stratum": "participants", "n": int(joined["participant_id"].nunique())},
        {"stratum": "ibd_samples", "n": int((joined["disease_group"] == "IBD").sum())},
        {"stratum": "control_samples", "n": int((joined["disease_group"] == "Control").sum())},
        {"stratum": "ibd_participants", "n": int(joined.loc[joined["disease_group"] == "IBD", "participant_id"].nunique())},
        {"stratum": "control_participants", "n": int(joined.loc[joined["disease_group"] == "Control", "participant_id"].nunique())},
        {"stratum": "inflamed_ibd_samples", "n": int(((joined["disease_group"] == "IBD") & (joined["title_inflamed"] == 1)).sum())},
        {"stratum": "scored_samples", "n": int(joined["axis_score"].notna().sum())},
    ]
    for gene, values in aggregated.items():
        feas.append({"stratum": f"gene_available:{gene}", "n": len(values)})
    pd.DataFrame(feas).to_csv(FEASIBILITY_OUT, sep="\t", index=False)

    # ---------- Analysis set: one biopsy per participant (first in expression order) ----------
    order = {sid: i for i, sid in enumerate(expr_sample_ids)}
    joined["_order"] = joined["sample_id"].map(order)
    one = joined.sort_values("_order").groupby("participant_id", as_index=False).first().reset_index(drop=True)
    one["axis_score_z"] = zscore_series(one["axis_score"])
    one["age_z"] = zscore_series(one["study_eligibility_age_at_endo"])
    one["sex"] = one["demographics_gender"].map({"Male": 0, "Female": 1})

    models: list[dict] = []
    for label, subset in [("one_per_participant", one)]:
        y = subset["ibd_vs_control"]
        x_un = sm.add_constant(subset["axis_score_z"])
        m_un = sm.Logit(y, x_un).fit(disp=False)
        models.append(odds_ratio_row(m_un, "axis_score_z", f"{label}_ibd_vs_control_unadjusted", int(subset.shape[0])))

        covars = subset[["axis_score_z", "age_z", "sex"]].copy()
        covars = pd.concat([covars, pd.get_dummies(subset["title_site"], prefix="region", drop_first=True)], axis=1)
        covars = covars.astype(float)
        covars = sm.add_constant(covars)
        m_adj = sm.Logit(subset["ibd_vs_control"], covars).fit(disp=False)
        models.append(odds_ratio_row(m_adj, "axis_score_z", f"{label}_ibd_vs_control_adjusted", int(subset.shape[0])))

    # All-biopsy sensitivity with GEE clustered by participant.
    all_ = joined.dropna(subset=["axis_score"]).copy()
    all_["axis_score_z"] = zscore_series(all_["axis_score"])
    all_["age_z"] = zscore_series(all_["study_eligibility_age_at_endo"])
    all_["sex"] = all_["demographics_gender"].map({"Male": 0, "Female": 1})
    covars = all_[["axis_score_z", "age_z", "sex"]].copy()
    covars = pd.concat([covars, pd.get_dummies(all_["title_site"], prefix="region", drop_first=True)], axis=1)
    covars = covars.astype(float)
    covars = sm.add_constant(covars)
    try:
        gee = sm.GEE(
            all_["ibd_vs_control"], covars,
            groups=all_["participant_id"], family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()
        ci = gee.conf_int().loc["axis_score_z"]
        models.append({
            "analysis": "all_biopsies_gee_adjusted",
            "n": int(all_.shape[0]),
            "term": "axis_score_z",
            "or_per_1sd": float(np.exp(gee.params["axis_score_z"])),
            "ci_lower": float(np.exp(ci[0])),
            "ci_upper": float(np.exp(ci[1])),
            "pvalue": float(gee.pvalues["axis_score_z"]),
        })
    except Exception as exc:  # pragma: no cover - log and continue
        models.append({"analysis": "all_biopsies_gee_adjusted", "n": int(all_.shape[0]),
                       "term": "axis_score_z", "or_per_1sd": float("nan"),
                       "ci_lower": float("nan"), "ci_upper": float("nan"),
                       "pvalue": float("nan"), "error": str(exc)})

    pd.DataFrame(models).to_csv(REPL_MODELS_OUT, sep="\t", index=False)

    # ---------- Activity correlations (IBD participants, one biopsy per participant) ----------
    ibd_one = one[one["disease_group"] == "IBD"].copy()
    correlations: list[dict] = []
    measures = [
        ("ibd_endoseverity_num", "endoscopic_severity_4levels"),
        ("ibdmesuc_mayo_score", "mayo_score_uc"),
        ("ibdsescd_totalsescd", "sescd_total_cd"),
        ("nancyindex", "nancy_histology_index"),
        ("ghas_sum7", "ghas_histology_sum7"),
        ("colitisactivityindex_sccai", "sccai_uc"),
        ("harveybradshawindex_hbi_score", "hbi_cd"),
        ("crp_jjmgl_log2", "crp_log2"),
        ("log2_fecalcalpro_mgperg", "fecal_calprotectin_log2"),
    ]
    for col, label in measures:
        correlations.append(spearman_row(ibd_one["axis_score"], ibd_one[col], label, int(ibd_one.shape[0])))
    pd.DataFrame(correlations).to_csv(ACTIVITY_OUT, sep="\t", index=False)

    # Endoscopic severity strata summary.
    strata = (
        ibd_one.dropna(subset=["ibd_endoseverity_num"])
        .groupby("ibd_endoseverity_num")["axis_score"]
        .agg(n="count", mean="mean", sd="std")
        .reset_index()
    )
    strata.to_csv(ROOT / "results/clinical/GSE193677_endoseverity_strata.tsv", sep="\t", index=False)

    print("replication models:")
    for r in models:
        print(" ", r.get("analysis"), "OR", round(r.get("or_per_1sd", float("nan")), 3),
              "p", round(r.get("pvalue", float("nan")), 4), "n", r.get("n"))
    print("activity correlations (top 5):")
    for r in correlations[:5]:
        print(" ", r["activity_measure"], "rho", round(r["spearman_rho"], 3), "p", round(r["pvalue"], 4), "n", r["n_pairs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
