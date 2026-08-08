#!/usr/bin/env python3
"""Supplementary figures for comparator and sensitivity analyses."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PLOT_DIR = ROOT / "plots/publication/submission_grade"
SOURCE_DIR = ROOT / "results/figures/source_data"


COLORS = {
    "primary": "#2f5f9f",
    "mmp": "#b85c38",
    "inflammatory": "#6d7f2a",
    "neutral": "#68707a",
    "light": "#d9e6f7",
    "line": "#222222",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
    }
)


def p_text(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "P < 0.001"
    return f"P = {p:.3f}"


def savefig(name: str) -> None:
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(PLOT_DIR / f"{name}.{ext}", dpi=450, bbox_inches="tight")


def comparator_figure() -> None:
    bench = pd.read_csv(ROOT / "results/benchmarks/comparator_signature_benchmark.tsv", sep="\t")
    data = bench[
        (bench["dataset_id"].eq("GSE193677"))
        & (bench["endpoint"].eq("IBD versus control"))
        & (bench["model_type"].eq("age_sex_site_adjusted"))
    ].copy()
    order = [
        "barrier_injury_score",
        "mmp_injury",
        "tnf_inflammatory",
        "osm_osmr",
        "hallmark_inflammatory_like",
        "neutrophil_myeloid",
        "stromal_remodeling",
        "junctional_complex",
        "upstream_regulatory",
    ]
    data["order"] = data["signature_id"].map({k: i for i, k in enumerate(order)})
    data = data.sort_values("order")
    data.to_csv(SOURCE_DIR / "FigureS9_comparator_benchmark_source_data.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    y = np.arange(len(data))
    colors = [
        COLORS["primary"] if s == "barrier_injury_score" else
        COLORS["mmp"] if s == "mmp_injury" else
        COLORS["inflammatory"] if "inflammatory" in s or s in {"osm_osmr", "neutrophil_myeloid"} else
        COLORS["neutral"]
        for s in data["signature_id"]
    ]
    ax.errorbar(
        data["effect"],
        y,
        xerr=[data["effect"] - data["ci_lower"], data["ci_upper"] - data["effect"]],
        fmt="none",
        ecolor=COLORS["line"],
        elinewidth=1.1,
        capsize=2.5,
        zorder=1,
    )
    ax.scatter(data["effect"], y, s=46, color=colors, edgecolor="white", linewidth=0.8, zorder=2)
    labels = data["display_label"].str.replace(" signature", "", regex=False).str.replace(" module", "", regex=False)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(1, color="#444444", lw=0.9, ls="--")
    ax.set_xlim(0.75, 3.25)
    ax.set_xticks([0.8, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_xlabel("Odds ratio per 1-SD higher score")
    ax.set_title("Comparator signatures in GSE193677", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#e5e7eb", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    for i, (_, row) in enumerate(data.iterrows()):
        ax.text(row["ci_upper"] + 0.06, i, p_text(row["pvalue"]), va="center", fontsize=7.5, color="#30343b")
    plt.tight_layout()
    savefig("FigureS9_comparator_benchmark")
    plt.close(fig)


def biopsy_sensitivity_figure() -> None:
    sens = pd.read_csv(ROOT / "results/clinical/GSE193677_biopsy_selection_sensitivity.tsv", sep="\t")
    data = sens[
        sens["endpoint"].eq("IBD versus control")
        & sens["model_type"].isin(["age_sex_site_adjusted_logistic", "age_sex_site_adjusted_gee"])
        & sens["status"].eq("modeled")
    ].copy()
    label_map = {
        "all_biopsies": "All biopsies, clustered",
        "one_biopsy_per_participant": {
            "first_expression_order": "One biopsy per participant",
            "most_inflamed": "Most inflamed biopsy",
            "least_inflamed": "Least inflamed biopsy",
        },
        "colorectal_sites": "Colorectal sites",
        "ileal_site": "Ileal site",
        "uc_or_control": "UC or control",
        "cd_or_control": "CD or control",
    }
    labels = []
    for _, row in data.iterrows():
        if row["analysis_scope"] == "one_biopsy_per_participant":
            labels.append(label_map["one_biopsy_per_participant"][row["selection_rule"]])
        else:
            labels.append(label_map[row["analysis_scope"]])
    data["plot_label"] = labels
    data.to_csv(SOURCE_DIR / "FigureS10_GSE193677_biopsy_sensitivity_source_data.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    y = np.arange(len(data))
    ax.errorbar(
        data["effect"],
        y,
        xerr=[data["effect"] - data["ci_lower"], data["ci_upper"] - data["effect"]],
        fmt="none",
        ecolor=COLORS["line"],
        elinewidth=1.1,
        capsize=2.5,
    )
    ax.scatter(data["effect"], y, s=48, color=COLORS["primary"], edgecolor="white", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(data["plot_label"])
    ax.invert_yaxis()
    ax.axvline(1, color="#444444", lw=0.9, ls="--")
    ax.set_xlim(0.85, 2.75)
    ax.set_xticks([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5])
    ax.set_xlabel("Odds ratio per 1-SD higher score")
    ax.set_title("GSE193677 biopsy-selection sensitivity", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#e5e7eb", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    for i, (_, row) in enumerate(data.iterrows()):
        ax.text(row["ci_upper"] + 0.05, i, f"n = {int(row['n_model'])}", va="center", fontsize=7.5)
    plt.tight_layout()
    savefig("FigureS10_GSE193677_biopsy_sensitivity")
    plt.close(fig)


def matched_null_figure() -> None:
    summ = pd.read_csv(ROOT / "results/validation/matched_null_benchmark.tsv", sep="\t")
    null = pd.read_csv(ROOT / "results/validation/matched_null_iterations.tsv", sep="\t")
    modeled = summ[summ["status"].eq("modeled")].iloc[0]
    source = null.copy()
    source["observed_or"] = modeled["observed_or"]
    source.to_csv(SOURCE_DIR / "FigureS11_matched_null_source_data.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.hist(null["random_or"], bins=36, color=COLORS["light"], edgecolor="#89a9d6", linewidth=0.8)
    ax.axvline(modeled["observed_or"], color=COLORS["mmp"], lw=2.0)
    ax.axvline(1, color="#444444", lw=0.9, ls="--")
    ax.set_xlabel("Odds ratio from matched random gene set")
    ax.set_ylabel("Number of iterations")
    ax.set_title("Expression- and variance-matched null distribution", loc="left", fontweight="bold")
    ax.text(
        0.98,
        0.92,
        f"Observed OR = {modeled['observed_or']:.2f}\nEmpirical P = {modeled['empirical_p_twosided']:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cbd5e1", "lw": 0.8},
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    savefig("FigureS11_matched_null")
    plt.close(fig)


def main() -> int:
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(SOURCE_DIR, exist_ok=True)
    comparator_figure()
    biopsy_sensitivity_figure()
    matched_null_figure()
    print("wrote FigureS9-FigureS11 supplementary benchmark and sensitivity figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
