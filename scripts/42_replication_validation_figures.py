#!/usr/bin/env python3
"""Supplementary figures for GSE193677 replication and validation benchmarks.

Outputs:
  FigureS6_gse193677_replication: GSE193677 IBD-vs-control ORs, endoscopic
    severity strata, and activity correlations.
  FigureS7_permutation_benchmark: observed OR vs null distribution per
    adult endpoint cohort with empirical P.
  FigureS8_celltype_localization: score-gene detection and expression by
    broad compartment in public single-cell IBD data.
Export: PNG (600 dpi) + PDF + SVG under plots/publication/submission_grade/.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "plots/publication/submission_grade")
os.makedirs(OUT_DIR, exist_ok=True)

REPL_MODELS = os.path.join(ROOT, "results/replication/GSE193677_replication_models.tsv")
ACTIVITY = os.path.join(ROOT, "results/clinical/GSE193677_activity_correlations.tsv")
STRATA = os.path.join(ROOT, "results/clinical/GSE193677_endoseverity_strata.tsv")
PERM = os.path.join(ROOT, "results/validation/permutation_benchmark.tsv")
LOCAL = os.path.join(ROOT, "results/validation/celltype_localization.tsv")
SRC_DIR = os.path.join(ROOT, "results/figures/source_data")
os.makedirs(SRC_DIR, exist_ok=True)

INK = "#252B35"
MUTED = "#6B7685"
ADULT_C = "#1A5C9A"
SIG_C = "#1A5C9A"
NULL_C = "#B8C4CF"
GRID = "#E2E8EF"


def p_text(p: float) -> str:
    if p < 0.001:
        return "P < 0.001"
    if p < 0.01:
        return f"P = {p:.4f}"
    return f"P = {p:.3f}"


def save(fig, name: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            os.path.join(OUT_DIR, f"{name}.{ext}"),
            dpi=600 if ext == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    print(f"saved {name}")


def figure_s6_replication() -> None:
    models = pd.read_csv(REPL_MODELS, sep="\t")
    activity = pd.read_csv(ACTIVITY, sep="\t")
    strata = pd.read_csv(STRATA, sep="\t")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))

    # Panel A: IBD vs control ORs.
    ax = axes[0]
    labels = {
        "one_per_participant_ibd_vs_control_unadjusted": "IBD vs control\n(unadjusted)",
        "one_per_participant_ibd_vs_control_adjusted": "IBD vs control\n(age, sex, site)",
        "all_biopsies_gee_adjusted": "All biopsies, GEE\n(age, sex, site)",
    }
    rows = [r for r in models.to_dict("records") if r["analysis"] in labels]
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        ax.errorbar(
            r["or_per_1sd"], yi,
            xerr=[[r["or_per_1sd"] - r["ci_lower"]], [r["ci_upper"] - r["or_per_1sd"]]],
            fmt="o", color=SIG_C, markersize=6, linewidth=1.4, capsize=3,
        )
        p = r["pvalue"]
        ax.text(r["ci_upper"] * 1.03, yi, f"OR {r['or_per_1sd']:.2f}\n{p_text(p)}",
                va="center", fontsize=7, color=INK)
    ax.axvline(1.0, color=MUTED, linestyle="--", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[r["analysis"]] for r in rows], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio per 1 SD score", fontsize=9)
    ax.set_title("GSE193677 (MSCCR) replication\nIBD vs control, n = 1,162 participants", fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_xlim(0.8, 3.2)

    # Panel B: endoscopic severity strata.
    ax = axes[1]
    strata = strata.sort_values("ibd_endoseverity_num")
    x = strata["ibd_endoseverity_num"].astype(int)
    ax.errorbar(x, strata["mean"], yerr=strata["sd"] / np.sqrt(strata["n"]),
                fmt="o-", color=ADULT_C, markersize=6, linewidth=1.4, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["Inactive", "Mild", "Moderate", "Severe"], fontsize=8)
    ax.set_ylabel("Mean barrier-injury score", fontsize=9)
    ax.set_title("Score by endoscopic severity\n(IBD participants)", fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    for xi, row in strata.iterrows():
        ax.text(xi, row["mean"] + 0.03, f"n={int(row['n'])}", ha="center", fontsize=7, color=MUTED)

    # Panel C: activity correlations.
    ax = axes[2]
    labels_c = {
        "endoscopic_severity_4levels": "Endoscopic severity (4 levels)",
        "mayo_score_uc": "Mayo score (UC)",
        "sescd_total_cd": "SES-CD total (CD)",
        "nancy_histology_index": "Nancy histology index",
        "ghas_histology_sum7": "GHAS histology (sum 7)",
        "sccai_uc": "SCCAI (UC)",
        "hbi_cd": "HBI (CD)",
        "crp_log2": "CRP (log2)",
        "fecal_calprotectin_log2": "Fecal calprotectin (log2)",
    }
    activity = activity[activity["activity_measure"].isin(labels_c)].copy()
    activity["label"] = activity["activity_measure"].map(labels_c)
    activity = activity.sort_values("spearman_rho")
    y = np.arange(len(activity))
    for yi, r in zip(y, activity.to_dict("records")):
        ax.errorbar(r["spearman_rho"], yi, xerr=[[r["spearman_rho"] - r["ci_lower"]], [r["ci_upper"] - r["spearman_rho"]]],
                    fmt="o", color=SIG_C, markersize=5, linewidth=1.2, capsize=3)
    ax.axvline(0, color=MUTED, linestyle="--", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(activity["label"], fontsize=7.5)
    ax.set_xlabel("Spearman rho (95% CI)", fontsize=9)
    ax.set_title("Score vs clinical activity indices\n(one biopsy per IBD participant)", fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.6)

    fig.suptitle("Independent prospective-cohort replication (GSE193677, MSCCR)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "FigureS6_gse193677_replication")


def figure_s7_permutation() -> None:
    perm = pd.read_csv(PERM, sep="\t")
    perm = perm.sort_values("empirical_p_twosided")
    fig, ax = plt.subplots(figsize=(8.2, 0.62 * len(perm) + 1.0))
    y = np.arange(len(perm))[::-1]
    for yi, r in zip(y, perm.to_dict("records")):
        lo, hi = r["random_or_p2_5"], r["random_or_p97_5"]
        med = r["random_or_median"]
        ax.plot([lo, hi], [yi, yi], color=NULL_C, linewidth=4, solid_capstyle="round", zorder=1)
        ax.plot(med, yi, "s", color=NULL_C, markersize=7, zorder=2, label="Random sets (median, 2.5-97.5%)" if yi == y[0] else None)
        ax.plot(r["observed_or"], yi, "o", color=SIG_C, markersize=9, zorder=3,
                label="Observed score" if yi == y[0] else None)
        p = r["empirical_p_twosided"]
        ptext = "P < 0.001" if p < 0.001 else f"P = {p:.3f}"
        ax.text(hi * 1.04, yi, ptext, va="center", fontsize=8, color=INK)
    ax.axvline(1.0, color=MUTED, linestyle="--", linewidth=0.9, zorder=0)
    ax.set_xscale("log")
    ax.set_yticks(y)
    labels = [f"{r['dataset_id']} · {r['endpoint']}" for r in perm.to_dict("records")]
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Odds ratio per 1 SD (log scale; random sets matched to 13-gene structure)", fontsize=9)
    ax.set_title("Permutation benchmark: observed score vs 1,000 random gene sets of equal size", fontsize=10)
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    fig.tight_layout()
    save(fig, "FigureS7_permutation_benchmark")


def figure_s8_localization() -> None:
    loc = pd.read_csv(LOCAL, sep="\t")
    loc = loc[loc["compartment"].isin(["epithelial", "immune", "stromal"])].copy()
    genes = [
        "MMP1", "MMP3", "MMP10", "MMP12", "MMP13",
        "CLDN2", "TJP1", "OCLN",
        "HDAC3", "IPMK", "IPPK", "NCOR1", "NCOR2",
    ]
    diseases = ["HC", "CD", "UC"]
    compartments = ["epithelial", "immune", "stromal"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 6.2), sharey=True)
    vmax = 0.45
    for ax, disease in zip(axes, diseases):
        mat = np.zeros((len(genes), len(compartments)))
        det = np.zeros((len(genes), len(compartments)))
        for i, g in enumerate(genes):
            for j, c in enumerate(compartments):
                row = loc[(loc["gene"] == g) & (loc["compartment"] == c) & (loc["disease"] == disease)]
                if len(row):
                    mat[i, j] = row["mean_log1p_cpm"].iloc[0]
                    det[i, j] = row["pct_detected"].iloc[0]
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(compartments)))
        ax.set_xticklabels(compartments, fontsize=8)
        ax.set_yticks(range(len(genes)))
        ax.set_yticklabels(genes if disease == diseases[0] else [], fontsize=8)
        ax.set_title({"HC": "Healthy control", "CD": "Crohn's disease", "UC": "Ulcerative colitis"}[disease],
                     fontsize=9)
        for i in range(len(genes)):
            for j in range(len(compartments)):
                ax.text(j, i, f"{det[i, j]:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if mat[i, j] > vmax * 0.6 else INK)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Mean log1p CPM")
    fig.suptitle("Cell-type localization of score genes (GSE214695, 46,702 cells; values = % cells detected)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "FigureS8_celltype_localization")


def main() -> int:
    figure_s6_replication()
    figure_s7_permutation()
    figure_s8_localization()
    # Source-data exports for replication and validation figures.
    models = pd.read_csv(REPL_MODELS, sep="\t")
    activity = pd.read_csv(ACTIVITY, sep="\t")
    strata = pd.read_csv(STRATA, sep="\t")
    perm = pd.read_csv(PERM, sep="\t")
    loc = pd.read_csv(LOCAL, sep="\t")
    s6 = pd.concat(
        [
            models.assign(panel="A_ibd_vs_control"),
            strata.assign(panel="B_endoseverity_strata"),
            activity.assign(panel="C_activity_correlations"),
        ],
        ignore_index=True,
    )
    s6.to_csv(os.path.join(SRC_DIR, "FigureS6_gse193677_replication_source_data.tsv"), sep="\t", index=False)
    perm.to_csv(os.path.join(SRC_DIR, "FigureS7_permutation_benchmark_source_data.tsv"), sep="\t", index=False)
    loc.to_csv(os.path.join(SRC_DIR, "FigureS8_celltype_localization_source_data.tsv"), sep="\t", index=False)
    print("wrote replication and validation source-data exports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
