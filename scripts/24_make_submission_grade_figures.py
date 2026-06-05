#!/usr/bin/env python3
"""Submission-grade figure system for the age-stratified IBD barrier-axis manuscript.

Script: 24_make_submission_grade_figures.py
Output: plots/publication/submission_grade/

Design principles
-----------------
- No PPT-style large colored rounded boxes or decorative text panels
- Information-dense, journal-appropriate visual hierarchy
- Adult direct endpoint evidence explicitly anchored and prominent
- Age stratification clearly separated (not mixed)
- Interpretation scope embedded in figure context, not floating text overlays
- 600 DPI PNG + PDF + SVG outputs
"""

from __future__ import annotations

import os
import math
import re
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

OUT_DIR = "plots/publication/submission_grade"

# ─────────────────────────────────────────────────────────────────────────────
# Restrained clinical colour palette
# ─────────────────────────────────────────────────────────────────────────────
INK     = "#252B35"    # near-black body text
MUTED   = "#6B7685"    # secondary / subdued text
GRID    = "#E2E8EF"    # light grid lines
RULE    = "#B8C4CF"    # section dividers and borders
PALE    = "#F7F9FB"    # very light background fill
WHITE   = "#FFFFFF"

# Stratum colours
ADULT_C = "#1A5C9A"    # adult deep blue
ADULT_M = "#4A87C2"    # adult medium
ADULT_L = "#D6E8F5"    # adult very pale fill
PRED_C  = "#B86200"    # pediatric remission: dark amber
PRED_L  = "#FDEBD0"    # pediatric remission fill
PINJ_C  = "#9E2A2B"    # pediatric injury: deep red
PINJ_L  = "#FAE0DF"    # pediatric injury fill
PHEN_C  = "#5B3EA0"    # pediatric phenotype: purple
PHEN_L  = "#EDE7F8"    # pediatric phenotype fill

GRAY_M  = "#7A8898"    # mid gray
GRAY_L  = "#EBF0F4"    # light gray
SIG_C   = "#1A5C9A"    # significant point fill (same as adult)
NSIG_C  = "#A0AFBE"    # non-significant (open / lighter)

# clinical_role → colour mapping
ROLE_COLOR = {
    "adult_direct_endpoint_anchor":     ADULT_C,
    "adult_direct_endpoint_validation": ADULT_C,
    "adult_direct_endpoint_expansion":  ADULT_C,
    "pediatric_remission_context":      PRED_C,
    "pediatric_mucosal_injury_support": PINJ_C,
    "pediatric_phenotype_support":      PHEN_C,
}

ROLE_LABEL = {
    "adult_direct_endpoint_anchor":     "Adult direct",
    "adult_direct_endpoint_validation": "Adult direct",
    "adult_direct_endpoint_expansion":  "Adult direct (expansion)",
    "pediatric_remission_context":      "Pediatric remission",
    "pediatric_mucosal_injury_support": "Pediatric injury",
    "pediatric_phenotype_support":      "Pediatric phenotype",
}

ENDPOINT_NICE = {
    "mucosal_healing":
        "Mucosal healing",
    "week8_endoscopic_histologic_healing":
        "Week-8 combined healing",
    "week6_clinical_response":
        "Week-6 clinical response",
    "week8_mucosal_healing":
        "Week-8 mucosal healing",
    "week8_clinical_remission":
        "Week-8 clinical remission",
    "week8_response":
        "Week-8 response (ACT1)",
    "week30_response":
        "Week-30 response (ACT1)",
    "infliximab_response":
        "Infliximab response",
    "week_4_remission":
        "Week-4 remission",
    "deep_ulcer":
        "Deep ulcer",
    "macroscopic_inflammation_vs_normal_or_microscopic":
        "Macroscopic inflammation",
    "CD_vs_nonIBD_ileum":
        "CD vs non-IBD (ileum)",
    "IBD_vs_control_rectum":
        "IBD vs control (rectum)",
    "CD_vs_control_rectum":
        "CD vs control (rectum)",
    "UC_vs_control_rectum":
        "UC vs control (rectum)",
    "CD_vs_nonIBD_ileum_age_sex_adjusted":
        "CD vs non-IBD – adjusted",
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def setup_style() -> None:
    mpl.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype":       42,
        "svg.fonttype":       "none",
        "font.size":          7.0,
        "axes.labelsize":     7.2,
        "axes.titlesize":     7.8,
        "xtick.labelsize":    6.8,
        "ytick.labelsize":    6.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.6,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.major.size":   3,
        "ytick.major.size":   3,
        "legend.frameon":     False,
        "legend.fontsize":    6.5,
    })


def ensure_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        path = os.path.join(OUT_DIR, f"{stem}.{ext}")
        kw: dict = {"bbox_inches": "tight", "facecolor": "white"}
        if ext == "png":
            kw["dpi"] = 600
        fig.savefig(path, **kw)
    plt.close(fig)


def panel_label(ax: plt.Axes, s: str, x: float = -0.06, y: float = 1.04) -> None:
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="top", color=INK)


def off_ax(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def sig_star(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def load_primary_models() -> pd.DataFrame:
    """Load primary (unadjusted) endpoint models."""
    df = pd.read_csv("results/clinical/age_stratified_endpoint_models.tsv", sep="\t")
    return df[df["covariates"] == "none"].copy()


def quasirandom_x(n: int, center: float, width: float = 0.18,
                  seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-width, width, n)
    return center + x


def parse_rate_cell(cell: str) -> tuple[int, int, float]:
    """Parse cells like '8/21 (38.1%)' into event, n, percent."""
    match = re.match(r"\s*(\d+)\s*/\s*(\d+)\s*\(([\d.]+)%\)", str(cell))
    if not match:
        return 0, 0, float("nan")
    event_n, total_n, pct = match.groups()
    return int(event_n), int(total_n), float(pct)


def p_text(pvalue: float) -> str:
    if pvalue < 0.001:
        return "p<0.001"
    return f"p={pvalue:.3f}"


def role_family(role: str) -> str:
    if role.startswith("adult"):
        return "adult"
    if "remission" in role:
        return "pediatric_remission"
    if "injury" in role:
        return "pediatric_injury"
    return "pediatric_phenotype"


def role_fill(role: str) -> str:
    family = role_family(role)
    return {
        "adult": ADULT_L,
        "pediatric_remission": PRED_L,
        "pediatric_injury": PINJ_L,
        "pediatric_phenotype": PHEN_L,
    }[family]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Study architecture and cohort registry
# ─────────────────────────────────────────────────────────────────────────────

def figure1_study_architecture() -> None:
    """
    Figure 1: Evidence landscape.

    Scientific purpose: make the paper's design legible at a glance:
    biology-inspired score, age-aware endpoint landscape, and claim hierarchy.
    """
    fig = plt.figure(figsize=(8.5, 5.2))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[0.42, 0.58],
                  height_ratios=[0.48, 0.52], hspace=0.24, wspace=0.22)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[1, 0])

    _fig1_score_capsule(ax_a)
    _fig1_evidence_landscape(ax_b)
    _fig1_claim_ladder(ax_c)

    fig.text(0.02, 0.99,
             "Figure 1  |  Biology-inspired score and age-aware clinical evidence landscape",
             fontsize=8.5, fontweight="bold", color=INK, va="top")
    save(fig, "Figure1_study_architecture")


def _fig1_score_capsule(ax: plt.Axes) -> None:
    """Compact rationale panel: biology informs the score but is not the claim."""
    off_ax(ax)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    panel_label(ax, "a", x=-0.02, y=1.02)

    ax.text(3, 88, "Biology-informed score", fontsize=7.8,
            fontweight="bold", color=INK)
    ax.text(3, 78, "InsP6-HDAC3-MMP barrier biology motivates a\n"
                   "prespecified mucosal transcriptomic score.",
            fontsize=6.0, color=MUTED, va="top")

    modules = [
        ("Upstream", ADULT_C, "IPMK · HDAC3 · NCOR"),
        ("MMP injury", PINJ_C, "MMP1/3/10/12/13"),
        ("Junction", PHEN_C, "TJP1 · OCLN · CLDN2"),
    ]
    for i, (name, color, genes) in enumerate(modules):
        y = 50 - i * 14
        ax.add_patch(Rectangle((4, y), 84, 10, fc=WHITE, ec=color, lw=0.9))
        ax.add_patch(Rectangle((4, y), 18, 10, fc=color, ec=color, lw=0.9, alpha=0.16))
        ax.text(7, y + 6.3, name, fontsize=6.0, fontweight="bold",
                color=color, va="center")
        ax.text(28, y + 5.0, genes, fontsize=5.7, color=INK, va="center")

    ax.annotate("", xy=(92, 36), xytext=(92, 57),
                arrowprops=dict(arrowstyle="-|>", color=GRAY_M, lw=1.0))
    ax.text(96, 46, "clinical\nassessment", fontsize=5.7, color=MUTED,
            va="center", ha="left")
    ax.text(3, 7, "Rationale only: no diet, supplement, or causal mechanism claim.",
            fontsize=5.4, color=PINJ_C, style="italic")


def _fig1_evidence_landscape(ax: plt.Axes) -> None:
    """Bubble landscape: age stratum, endpoint strength, and evaluable sample size."""
    panel_label(ax, "b", x=-0.06, y=1.04)
    cohorts = pd.read_csv("results/dataset_summary.tsv", sep="\t")
    coords = {
        "GSE73661":  ("Adult", 3.0, "Healing", ADULT_C, 64),
        "GSE12251":  ("Adult", 2.7, "Healing", ADULT_C, 22),
        "GSE92415":  ("Adult", 2.30, "Response", ADULT_M, 87),
        "GSE206285": ("Adult", 3.2, "Healing/remission", ADULT_C, 550),
        "GSE23597":  ("Adult", 2.03, "Response", ADULT_M, 45),
        "GSE16879":  ("Adult/mixed", 1.78, "Biologic response", ADULT_M, 61),
        "GSE109142": ("Pediatric", 1.9, "Remission", PRED_C, 206),
        "GSE57945":  ("Pediatric", 1.3, "Injury", PINJ_C, 174),
        "GSE101794": ("Pediatric", 0.9, "Phenotype", PHEN_C, 304),
        "GSE117993": ("Pediatric", 0.9, "Phenotype", PHEN_C, 190),
    }
    x_map = {"Adult": 0.0, "Adult/mixed": 0.15, "Pediatric": 1.0}
    jitter = {
        "GSE73661": -0.08, "GSE12251": 0.06, "GSE92415": -0.02,
        "GSE206285": 0.10, "GSE23597": 0.12, "GSE16879": -0.02,
        "GSE109142": -0.08, "GSE57945": 0.06, "GSE101794": -0.08,
        "GSE117993": 0.10,
    }
    label_offsets = {
        "GSE23597": (0.03, -0.06, "left", "center"),
        "GSE16879": (0.00, -0.24, "center", "top"),
    }

    ax.set_title("Endpoint evidence landscape", loc="left", fontsize=7.6,
                 fontweight="bold", pad=4)
    ax.axvspan(-0.22, 0.32, color=ADULT_L, alpha=0.35, zorder=0)
    ax.axvspan(0.78, 1.22, color=PRED_L, alpha=0.35, zorder=0)
    ax.text(0.05, 3.55, "Adult direct endpoint cohorts", ha="center",
            fontsize=6.4, color=ADULT_C, fontweight="bold")
    ax.text(1.0, 3.55, "Pediatric / early-onset cohorts", ha="center",
            fontsize=6.4, color=PRED_C, fontweight="bold")

    for _, row in cohorts.iterrows():
        ds = row["dataset_id"]
        if ds not in coords:
            continue
        age, y, endpoint, color, n = coords[ds]
        x = x_map[age] + jitter.get(ds, 0)
        size = 24 + math.sqrt(n) * 11
        ax.scatter(x, y, s=size, color=color, alpha=0.80, edgecolor=WHITE,
                   linewidth=0.8, zorder=4)
        dx, dy, ha, va = label_offsets.get(ds, (0.0, -0.21, "center", "top"))
        ax.text(x + dx, y + dy, ds.replace("GSE", ""), fontsize=5.2,
                ha=ha, va=va, color=INK)
        if ds in {"GSE206285", "GSE109142", "GSE101794"}:
            ax.text(x, y + 0.18, f"n={n}", fontsize=5.2, ha="center",
                    va="bottom", color=MUTED)

    ax.set_xlim(-0.28, 1.28)
    ax.set_ylim(0.55, 3.75)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["Adult", "Pediatric"], fontsize=6.8,
                       fontweight="bold")
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["Phenotype /\ninjury", "Response /\nremission",
                        "Mucosal\nhealing"], fontsize=6.0)
    ax.set_ylabel("Endpoint strength", fontsize=6.6)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.tick_params(axis="x", length=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    handles = [
        mpatches.Patch(color=ADULT_C, label="Adult healing/remission"),
        mpatches.Patch(color=ADULT_M, label="Adult response"),
        mpatches.Patch(color=PRED_C, label="Pediatric remission"),
        mpatches.Patch(color=PINJ_C, label="Pediatric injury"),
        mpatches.Patch(color=PHEN_C, label="Pediatric phenotype"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=5.2,
              handlelength=0.8, borderpad=0.2, labelspacing=0.25)


def _fig1_claim_ladder(ax: plt.Axes) -> None:
    """Compact interpretation hierarchy for the age-stratified analysis."""
    off_ax(ax)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    panel_label(ax, "c", x=-0.02, y=1.02)
    ax.text(4, 91, "Interpretation levels", fontsize=7.6, fontweight="bold",
            color=INK)
    rows = [
        (ADULT_C, "Observed", "adult retrospective\nmolecular stratification"),
        (PRED_C, "Separate", "pediatric / early-onset\ncontext"),
        (GRAY_M, "Not pooled", "endpoint & tissue\nheterogeneity"),
        (PINJ_C, "Not assessed", "diagnosis, treatment\nselection, InsP6 efficacy"),
    ]
    for i, (color, label, text) in enumerate(rows):
        y = 73 - i * 18
        ax.add_patch(Rectangle((5, y), 24, 10, fc=color, ec="none", alpha=0.16))
        ax.text(8, y + 5, label, fontsize=5.9, fontweight="bold",
                color=color, va="center")
        ax.text(35, y + 5, text, fontsize=5.7, color=INK, va="center")
        if i < len(rows) - 1:
            ax.plot([17, 17], [y - 2, y - 8], color=RULE, lw=0.8)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Adult direct endpoint evidence
# ─────────────────────────────────────────────────────────────────────────────

def _draw_distribution(
    ax: plt.Axes,
    grp0: pd.Series,
    grp1: pd.Series,
    title: str,
    subtitle: str,
    n0_label: str,
    n1_label: str,
    or_text: str,
    ylab: bool = True,
) -> None:
    """Compact quasirandom-jitter + box-and-median strip plot."""
    grp0 = grp0.dropna().astype(float).to_numpy()
    grp1 = grp1.dropna().astype(float).to_numpy()

    # Box (IQR + whisker)
    for pos, vals, c, fc in [(0, grp0, PINJ_C, "#FAEBD7"), (1, grp1, ADULT_C, ADULT_L)]:
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        lo_w = max(vals[vals >= q1 - 1.5 * iqr].min(), q1 - 1.5 * iqr)
        hi_w = min(vals[vals <= q3 + 1.5 * iqr].max(), q3 + 1.5 * iqr)
        # whisker
        ax.plot([pos, pos], [lo_w, q1], color=c, lw=0.8, zorder=2)
        ax.plot([pos, pos], [q3, hi_w], color=c, lw=0.8, zorder=2)
        # box
        ax.add_patch(Rectangle((pos - 0.16, q1), 0.32, iqr,
                               fc=fc, ec=c, lw=0.7, zorder=3))
        # median
        ax.plot([pos - 0.16, pos + 0.16], [med, med],
                color=c, lw=1.4, zorder=4)
        # jitter dots
        xs = quasirandom_x(len(vals), pos, width=0.21,
                            seed=hash(title + str(pos)) % 999)
        s_size = max(4, min(14, 400 / len(vals)))
        ax.scatter(xs, vals, s=s_size, color=c, alpha=0.55,
                    edgecolors="none", zorder=5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{n0_label}", f"{n1_label}"], fontsize=6.5)
    ax.set_title(title, loc="left", fontweight="bold", pad=3, fontsize=7.4)
    ax.text(0.02, 0.965, subtitle, transform=ax.transAxes,
            fontsize=6.5, color=MUTED, va="top")
    ax.text(0.97, 0.885, or_text, transform=ax.transAxes,
            fontsize=6.3, color=ADULT_C, va="top", ha="right", fontweight="bold")
    ax.axhline(0, color=GRID, lw=0.7, zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.4, zorder=0)
    y_min = min(grp0.min(), grp1.min())
    y_max = max(grp0.max(), grp1.max())
    y_span = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_span, y_max + 0.25 * y_span)
    if ylab:
        ax.set_ylabel("Barrier-axis score (z)", fontsize=6.8)
    ax.tick_params(axis="x", length=0)


def figure2_adult_endpoint_evidence() -> None:
    """
    Figure 2: Clinical stratification hero.

    Panel a  —  favorable-endpoint rates in low vs high score strata.
    Panel b  —  adult direct endpoint OR estimates.
    Panel c  —  representative raw score distributions in anchor cohorts.
    """
    g1 = pd.read_csv("data/processed/GSE73661/baseline_endpoint.tsv", sep="\t")
    g2 = pd.read_csv("data/processed/GSE206285/baseline_endpoint.tsv", sep="\t")
    models = load_primary_models()

    fig = plt.figure(figsize=(8.5, 6.2))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[0.60, 0.40],
                  height_ratios=[0.62, 0.38], hspace=0.32, wspace=0.30)

    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    sub = gs[1, 1].subgridspec(1, 2, wspace=0.28)
    ax_c1 = fig.add_subplot(sub[0, 0])
    ax_c2 = fig.add_subplot(sub[0, 1])

    _fig2_stratification_dumbbell(ax_a)
    _fig2_adult_or_forest(ax_b, models)

    panel_label(ax_c1, "c", x=-0.18, y=1.08)
    _draw_distribution(
        ax_c1,
        g1.loc[g1["mucosal_healing"] == 0, "axis_score"],
        g1.loc[g1["mucosal_healing"] == 1, "axis_score"],
        title="GSE73661",
        subtitle="mucosal healing",
        n0_label="No",
        n1_label="Yes",
        or_text="OR 0.52",
        ylab=True,
    )
    g2["axis_score"] = pd.to_numeric(g2["axis_score"], errors="coerce")
    _draw_distribution(
        ax_c2,
        g2.loc[g2["week8_mucosal_healing"] == 0, "axis_score"],
        g2.loc[g2["week8_mucosal_healing"] == 1, "axis_score"],
        title="GSE206285",
        subtitle="week-8 healing",
        n0_label="No",
        n1_label="Yes",
        or_text="OR 0.56",
        ylab=False,
    )

    fig.text(0.02, 0.99, "Figure 2  |  Clinical outcome stratification by barrier-axis score",
             fontsize=8.5, fontweight="bold", color=INK, va="top")
    save(fig, "Figure2_adult_endpoint_evidence")


def _fig2_stratification_dumbbell(ax: plt.Axes) -> None:
    """Hero panel: low-score versus high-score favorable endpoint rates."""
    panel_label(ax, "a", x=-0.07, y=1.03)
    rates = pd.read_csv("results/clinical/clinical_score_strata_summary.tsv", sep="\t")
    rows = []
    for row in rates.itertuples():
        low_event, low_n, low_pct = parse_rate_cell(row.low_score_event_rate)
        mid_event, mid_n, mid_pct = parse_rate_cell(row.middle_score_event_rate)
        high_event, high_n, high_pct = parse_rate_cell(row.high_score_event_rate)
        label = f"{row.dataset_id}  {row.endpoint}"
        rows.append({
            "label": label,
            "age": row.age_stratum,
            "clinical_role": row.clinical_role,
            "n": int(row.modeled_n),
            "low": low_pct,
            "mid": mid_pct,
            "high": high_pct,
            "delta": float(row.low_minus_high_percentage_points),
        })
    plot_df = pd.DataFrame(rows).sort_values("delta", ascending=True)
    y = np.arange(len(plot_df))

    ax.set_title("Favorable endpoint rate by within-cohort score tertile",
                 loc="left", fontsize=7.6, fontweight="bold", pad=5)
    for yi, row in zip(y, plot_df.itertuples()):
        strip_color = ADULT_C if str(row.age).startswith("adult") else PRED_C
        ax.axhspan(yi - 0.43, yi + 0.43, color=PALE if yi % 2 == 0 else WHITE,
                   zorder=0)
        ax.plot([row.high, row.low], [yi, yi], color=RULE, lw=1.8, zorder=1)
        ax.scatter(row.high, yi, s=34, color=GRAY_M, edgecolor=WHITE,
                   linewidth=0.7, zorder=4)
        ax.scatter(row.mid, yi, s=26, color=ADULT_M, edgecolor=WHITE,
                   linewidth=0.7, alpha=0.85, zorder=4)
        ax.scatter(row.low, yi, s=42, color=ADULT_C, edgecolor=WHITE,
                   linewidth=0.7, zorder=5)
        ax.add_patch(Rectangle((-0.055, yi - 0.34), 0.016, 0.68,
                               transform=ax.get_yaxis_transform(), fc=strip_color,
                               ec="none", alpha=0.85, clip_on=False))
        ax.text(-0.015, yi,
                row.label.replace("Week 8 ", "W8 ").replace("Week 30 ", "W30 "),
                fontsize=5.8, ha="right", va="center", color=INK,
                transform=ax.get_yaxis_transform(), clip_on=False)
        ax.text(max(row.low, row.high) + 3.0, yi, f"+{row.delta:.1f} pp",
                fontsize=5.8, color=strip_color, va="center", fontweight="bold")

    ax.set_xlim(0, 92)
    ax.set_ylim(-0.8, len(plot_df) - 0.2)
    ax.set_yticks([])
    ax.set_xlabel("Favorable endpoint rate (%)", fontsize=7.0)
    ax.grid(axis="x", color=GRID, lw=0.45)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ADULT_C,
                   markeredgecolor=WHITE, markersize=5, label="Low score"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ADULT_M,
                   markeredgecolor=WHITE, markersize=4.5, label="Middle"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=GRAY_M,
                   markeredgecolor=WHITE, markersize=4.5, label="High"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=5.6,
              ncol=3, handletextpad=0.3, columnspacing=0.8)


def _fig2_adult_or_forest(ax: plt.Axes, models: pd.DataFrame) -> None:
    """Compact adult direct endpoint forest plot."""
    panel_label(ax, "b", x=-0.08, y=1.04)
    adult_order = [
        ("GSE73661", "mucosal_healing"),
        ("GSE12251", "week8_endoscopic_histologic_healing"),
        ("GSE206285", "week8_mucosal_healing"),
        ("GSE206285", "week8_clinical_remission"),
        ("GSE92415", "week6_clinical_response"),
        ("GSE23597", "week8_response"),
        ("GSE23597", "week30_response"),
        ("GSE16879", "infliximab_response"),
    ]
    rows = []
    for ds, ep in adult_order:
        sub = models[(models["dataset_id"] == ds) & (models["endpoint_name"] == ep)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        rows.append({
            "ds": ds,
            "ep": ep,
            "effect": float(row.effect),
            "lo": float(row.ci_lower),
            "hi": float(row.ci_upper),
            "pvalue": float(row.pvalue),
        })
    y = np.arange(len(rows))[::-1]
    ax.set_title("Adult direct endpoint models", loc="left",
                 fontsize=7.4, fontweight="bold", pad=4)
    for i, (yi, row) in enumerate(zip(y, rows)):
        ax.axhspan(yi - 0.43, yi + 0.43, color=ADULT_L if i % 2 == 0 else WHITE,
                   alpha=0.38, zorder=0)
        p05 = row["pvalue"] < 0.05
        ax.errorbar(row["effect"], yi,
                    xerr=[[row["effect"] - row["lo"]], [row["hi"] - row["effect"]]],
                    fmt="o", ms=4.0, color=ADULT_C, mfc=ADULT_C if p05 else WHITE,
                    mec=ADULT_C, mew=0.9, capsize=2.2, lw=1.0, zorder=4)
        ax.text(0.045, yi + 0.15, f"{row['ds']} · {ENDPOINT_NICE.get(row['ep'], row['ep'])}",
                fontsize=5.8, color=INK, fontweight="bold",
                transform=ax.get_yaxis_transform())
        ax.text(0.045, yi - 0.24, f"OR {row['effect']:.2f} ({row['lo']:.2f}-{row['hi']:.2f}); {p_text(row['pvalue'])}",
                fontsize=5.2, color=MUTED, transform=ax.get_yaxis_transform())
    ax.axvline(1, color=GRAY_M, lw=0.8, ls="--", zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 2.2)
    ax.set_xticks([0.01, 0.1, 1.0])
    ax.set_xticklabels(["0.01", "0.1", "1.0"])
    ax.set_yticks([])
    ax.set_xlabel("OR per 1 SD score", fontsize=6.8)
    ax.grid(axis="x", color=GRID, lw=0.4, zorder=0)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Age-stratified clinical evidence atlas
# ─────────────────────────────────────────────────────────────────────────────

def figure3_age_stratified_atlas() -> None:
    """
    Figure 3: Age-stratified evidence matrix.
    """
    models = load_primary_models()
    fig = plt.figure(figsize=(8.5, 5.7))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.70, 0.30], wspace=0.23)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    _fig3_evidence_matrix(ax_a, models)
    _fig3_direction_summary(ax_b, models)

    fig.text(0.02, 0.99, "Figure 3  |  Age-stratified endpoint evidence map",
             fontsize=8.5, fontweight="bold", color=INK, va="top")
    save(fig, "Figure3_age_stratified_atlas")


def _endpoint_column(endpoint_family: str, endpoint_name: str) -> str:
    if "phenotype" in endpoint_family:
        return "Phenotype"
    if "ulceration" in endpoint_family:
        return "Injury"
    if "remission" in endpoint_family and endpoint_name != "week_4_remission":
        return "Remission"
    if "healing" in endpoint_family or "healing" in endpoint_name:
        return "Healing"
    if "response" in endpoint_family or "response" in endpoint_name or endpoint_name == "week_4_remission":
        return "Response /\nremission"
    return "Other"


def _fig3_evidence_matrix(ax: plt.Axes, models: pd.DataFrame) -> None:
    panel_label(ax, "a", x=-0.05, y=1.04)
    ax.set_title("Cohort-by-endpoint evidence matrix", loc="left",
                 fontsize=7.6, fontweight="bold", pad=4)
    cohort_order = [
        "GSE206285", "GSE73661", "GSE12251", "GSE92415", "GSE23597", "GSE16879",
        "GSE109142", "GSE57945", "GSE101794", "GSE117993",
    ]
    cols = ["Healing", "Remission", "Response /\nremission", "Injury", "Phenotype"]
    x_map = {c: i for i, c in enumerate(cols)}
    y_map = {c: len(cohort_order) - 1 - i for i, c in enumerate(cohort_order)}

    ax.axhspan(y_map["GSE16879"] - 0.5, y_map["GSE206285"] + 0.5,
               color=ADULT_L, alpha=0.20, zorder=0)
    ax.axhspan(y_map["GSE117993"] - 0.5, y_map["GSE109142"] + 0.5,
               color=PRED_L, alpha=0.25, zorder=0)
    for y in range(len(cohort_order)):
        ax.axhline(y - 0.5, color=GRID, lw=0.45, zorder=0)
    for x in range(len(cols)):
        ax.axvline(x - 0.5, color=GRID, lw=0.45, zorder=0)

    for row in models.itertuples():
        ds = row.dataset_id
        if ds not in y_map:
            continue
        col = _endpoint_column(row.endpoint_family, row.endpoint_name)
        if col not in x_map:
            continue
        # Multiple endpoints in one family are nudged vertically.
        nudge = {
            "week8_response": -0.12,
            "week30_response": 0.12,
            "IBD_vs_control_rectum": 0.00,
            "CD_vs_control_rectum": -0.15,
            "UC_vs_control_rectum": 0.15,
            "macroscopic_inflammation_vs_normal_or_microscopic": -0.12,
            "deep_ulcer": 0.12,
        }.get(row.endpoint_name, 0.0)
        x = x_map[col]
        y = y_map[ds] + nudge
        color = ROLE_COLOR.get(row.clinical_role, GRAY_M)
        p05 = float(row.pvalue) < 0.05
        size = 18 + math.sqrt(float(row.n)) * 7
        marker = "o" if "adult" in row.clinical_role else "s"
        ax.scatter(x, y, s=size, marker=marker, color=color if p05 else WHITE,
                   edgecolor=color, linewidth=1.0, zorder=4)
        ax.text(x, y, f"{float(row.effect):.1f}", fontsize=4.9,
                ha="center", va="center", color=WHITE if p05 else color,
                fontweight="bold", zorder=5)

    ax.set_xlim(-0.55, len(cols) - 0.45)
    ax.set_ylim(-0.55, len(cohort_order) - 0.25)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=6.2)
    ax.set_yticks([y_map[c] for c in cohort_order])
    ax.set_yticklabels([c.replace("GSE", "") for c in cohort_order], fontsize=6.3)
    ax.tick_params(axis="both", length=0)
    ax.text(-0.47, y_map["GSE206285"] + 0.42, "Adult", color=ADULT_C,
            fontsize=5.8, fontweight="bold", va="top")
    ax.text(-0.47, y_map["GSE109142"] + 0.42, "Pediatric", color=PRED_C,
            fontsize=5.8, fontweight="bold", va="top")
    ax.text(0.01, -0.12,
            "Circle: adult or adult/mixed; square: pediatric. Filled: p<0.05; open: p>=0.05. Number inside marker is OR.",
            transform=ax.transAxes, fontsize=5.4, color=MUTED, va="top")


def _fig3_direction_summary(ax: plt.Axes, models: pd.DataFrame) -> None:
    panel_label(ax, "b", x=-0.08, y=1.04)
    off_ax(ax)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Age-aware readout", loc="left", fontsize=7.4,
                 fontweight="bold", pad=4)
    adult = models[models["clinical_role"].str.startswith("adult")]
    ped_rem = models[models["clinical_role"].eq("pediatric_remission_context")]
    ped_inj = models[models["clinical_role"].eq("pediatric_mucosal_injury_support")]
    ped_ph = models[models["clinical_role"].eq("pediatric_phenotype_support")]
    rows = [
        (ADULT_C, "Adult direct endpoints",
         f"{len(adult)} models; all OR < 1",
         "lower score favors healing /\nremission / response"),
        (PRED_C, "Pediatric remission",
         f"{len(ped_rem)} model; OR < 1",
         "directionally consistent,\nnot adult validation"),
        (PINJ_C, "Pediatric injury",
         f"{len(ped_inj)} models; OR > 1",
         "higher score marks\nmucosal injury"),
        (PHEN_C, "Pediatric phenotype",
         f"{len(ped_ph)} models; OR > 1",
         "higher score marks\nIBD phenotype"),
    ]
    for i, (color, title, stat, note) in enumerate(rows):
        y = 83 - i * 22
        ax.add_patch(Rectangle((3, y - 8), 7, 16, fc=color, ec="none",
                               alpha=0.85))
        ax.text(15, y + 4.5, title, fontsize=6.5, color=color,
                fontweight="bold", va="center")
        ax.text(15, y - 1.5, stat, fontsize=5.8, color=INK, va="center")
        ax.text(15, y - 7.0, note, fontsize=5.4, color=MUTED, va="center")
        if i < len(rows) - 1:
            ax.plot([3, 96], [y - 12, y - 12], color=GRID, lw=0.6)
    ax.text(3, 4, "No single pooled pan-IBD estimate is displayed.",
            fontsize=5.6, color=PINJ_C, style="italic")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Interpretation scope and non-pooling rationale
# ─────────────────────────────────────────────────────────────────────────────

def figure4_evidence_architecture() -> None:
    """
    Figure 4: Interpretation scope and next-validation path.
    """
    fig = plt.figure(figsize=(7.6, 5.2))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.58, 0.42], wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    _fig4_translation_ladder(ax_a)
    _fig4_nonclaim_boundary(ax_b)

    fig.text(0.02, 0.99, "Figure 4  |  Interpretation scope and next-validation path",
             fontsize=8.5, fontweight="bold", color=INK, va="top")
    save(fig, "Figure4_evidence_architecture")


def _fig4_translation_ladder(ax: plt.Axes) -> None:
    panel_label(ax, "a", x=-0.05, y=1.03)
    off_ax(ax)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Current interpretation", loc="left",
                 fontsize=7.5, fontweight="bold", pad=4)
    stages = [
        (PINJ_C, "Not evaluated", "Clinical-test deployment",
         "no locked threshold, calibration,\nDCA, or prospective assay"),
        (GRAY_M, "Cohort-specific", "Endpoint-aware synthesis",
         "cohort-level interpretation;\nno forced pooled pan-IBD OR"),
        (PRED_C, "Separate context", "Pediatric / early-onset\nbiological relevance",
         "remission context plus injury /\nphenotype evidence; not adult\nvalidation"),
        (ADULT_C, "Observed", "Retrospective molecular\nstratification",
         "6 adult direct endpoint cohorts;\nlow-score strata show higher\nfavorable endpoint rates"),
    ]
    x0, box_w, box_h = 8, 78, 16
    for i, (color, tag, title, note) in enumerate(stages):
        y = 79 - i * 22
        ax.add_patch(Rectangle((x0, y), box_w, box_h, fc=WHITE, ec=color,
                               lw=1.0))
        ax.add_patch(Rectangle((x0, y), 18, box_h, fc=color, ec=color,
                               lw=0, alpha=0.14))
        ax.text(x0 + 2.2, y + 10.5, tag, fontsize=5.7, fontweight="bold",
                color=color, va="center")
        ax.text(x0 + 24, y + 10.5, title, fontsize=6.4, fontweight="bold",
                color=INK, va="center")
        ax.text(x0 + 24, y + 4.8, note, fontsize=5.5, color=MUTED,
                va="center")
        if i < len(stages) - 1:
            ax.annotate("", xy=(x0 + box_w / 2, y - 3),
                        xytext=(x0 + box_w / 2, y - 7),
                        arrowprops=dict(arrowstyle="-|>", color=RULE, lw=1.0))
    ax.text(x0, 4, "Interpretation: useful for retrospective stratification and trial-enrichment hypotheses,\nnot for individual treatment selection.",
            fontsize=5.6, color=MUTED, style="italic")


def _fig4_nonclaim_boundary(ax: plt.Axes) -> None:
    panel_label(ax, "b", x=-0.08, y=1.03)
    off_ax(ax)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Future validation", loc="left",
                 fontsize=7.5, fontweight="bold", pad=4)
    steps = [
        ("Assay lock", "fixed gene set, normalization,\nthreshold rule"),
        ("Calibration", "absolute risk estimates and\nexternal calibration"),
        ("Decision impact", "decision-curve analysis;\nendoscopy-follow-up utility"),
        ("Prospective validation", "predefined biopsy cohort;\nclinical workflow test"),
    ]
    for i, (title, note) in enumerate(steps):
        y = 82 - i * 20
        ax.scatter(13, y, s=130, color=ADULT_C if i < 2 else GRAY_M,
                   edgecolor=WHITE, linewidth=0.8, zorder=3)
        ax.text(13, y, str(i + 1), ha="center", va="center",
                fontsize=6.0, color=WHITE, fontweight="bold")
        ax.text(25, y + 4, title, fontsize=6.5, fontweight="bold",
                color=INK, va="center")
        ax.text(25, y - 3, note, fontsize=5.6, color=MUTED, va="center")
        if i < len(steps) - 1:
            ax.plot([13, 13], [y - 4, y - 16], color=RULE, lw=0.9, zorder=1)

    ax.add_patch(Rectangle((7, 1), 86, 12, fc=PINJ_L, ec=PINJ_C, lw=0.8,
                           alpha=0.45))
    ax.text(11, 9, "InsP6-HDAC3-MMP remains biological rationale only",
            fontsize=5.9, color=PINJ_C, fontweight="bold", va="center")
    ax.text(11, 4.5, "No dietary efficacy, supplementation efficacy, or causal mechanism claim.",
            fontsize=5.4, color=MUTED, va="center")


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary Figure 1: Sensitivity / adjusted models
# ─────────────────────────────────────────────────────────────────────────────

def figureS1_sensitivity_models() -> None:
    """Refactored S1: Visual forest for adjusted models."""
    df = pd.read_csv("results/clinical/age_stratified_endpoint_models.tsv", sep="\t")
    adj = df[df["covariates"] != "none"].copy()
    if adj.empty: return

    n_rows = len(adj)
    fig_h = max(4.5, 0.38 * n_rows + 1.2)
    fig = plt.figure(figsize=(7.5, fig_h))
    ax = fig.add_subplot(1, 1, 1)

    adj = adj.sort_values(["clinical_role", "dataset_id", "endpoint_name"])
    y_arr = np.arange(n_rows)[::-1].astype(float)

    for yi, row in zip(y_arr, adj.itertuples()):
        color = ROLE_COLOR.get(row.clinical_role, GRAY_M)
        eff, lo, hi = float(row.effect), float(row.ci_lower), float(row.ci_upper)
        p05 = float(row.pvalue) < 0.05
        mfc = color if p05 else WHITE
        ax.axhspan(yi-0.45, yi+0.45, color=PALE if int(yi)%2==0 else WHITE, alpha=0.4)
        ax.errorbar(eff, yi, xerr=[[eff - lo], [hi - eff]], fmt="o", ms=4.2, color=color, mfc=mfc, mec=color, mew=1, capsize=2, lw=1.1, zorder=4)
        ep_nice = ENDPOINT_NICE.get(row.endpoint_name, row.endpoint_name)
        cov_lbl = str(row.covariates).replace(";", ", ")
        ax.text(0.05, yi+0.15, f"{row.dataset_id} · {ep_nice}", fontsize=6.2, fontweight="bold", color=INK, transform=ax.get_yaxis_transform())
        ax.text(0.05, yi-0.25, f"Adjusted for: {cov_lbl}", fontsize=5.6, color=MUTED, transform=ax.get_yaxis_transform())

    ax.axvline(1, color=GRAY_M, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(0.01, 2.2)
    ax.set_xticks([0.01, 0.1, 1.0])
    ax.set_xticklabels(["0.01", "0.1", "1.0"])
    ax.set_yticks([])
    ax.set_xlabel("OR per 1 SD score (log scale)", fontsize=7.0)
    ax.set_title("Supplementary Figure 1 | Adjusted sensitivity models", loc="left", fontsize=7.5, fontweight="bold", pad=8)
    save(fig, "FigureS1_sensitivity_models")


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary Figure 2: Score gene module detail
# ─────────────────────────────────────────────────────────────────────────────

def figureS2_score_module_detail() -> None:
    """Supplementary Figure 2: module-level endpoint contrast heatmap."""
    files = [
        ("data/processed/GSE73661/baseline_endpoint.tsv", "GSE73661\nhealing", "mucosal_healing"),
        ("data/processed/GSE206285/baseline_endpoint.tsv", "GSE206285\nhealing", "week8_mucosal_healing"),
        ("data/processed/GSE206285/baseline_endpoint.tsv", "GSE206285\nremission", "week8_clinical_remission"),
    ]
    module_cols = [
        ("upstream_score", "Upstream"),
        ("mmp_score", "MMP injury"),
        ("junction_score", "Junction"),
    ]
    records = []
    for fpath, label, endpoint_col in files:
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath, sep="\t")
        if endpoint_col not in df.columns:
            continue
        for col, module in module_cols:
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            endpoint = pd.to_numeric(df[endpoint_col], errors="coerce")
            tmp = pd.DataFrame({"score": vals, "endpoint": endpoint}).dropna()
            if tmp.empty:
                continue
            favorable = tmp.loc[tmp["endpoint"] == 1, "score"]
            unfavorable = tmp.loc[tmp["endpoint"] == 0, "score"]
            records.append({
                "endpoint": label,
                "module": module,
                "delta": favorable.mean() - unfavorable.mean(),
            })
    if not records:
        return
    rdf = pd.DataFrame(records)
    endpoints = [x[1] for x in files]
    modules = [x[1] for x in module_cols]
    mat = np.full((len(modules), len(endpoints)), np.nan)
    for i, module in enumerate(modules):
        for j, endpoint in enumerate(endpoints):
            sub = rdf[(rdf["module"] == module) & (rdf["endpoint"] == endpoint)]
            if not sub.empty:
                mat[i, j] = float(sub["delta"].iloc[0])

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.45, vmax=0.45, aspect="auto")
    ax.set_xticks(range(len(endpoints)))
    ax.set_xticklabels(endpoints, fontsize=6.6)
    ax.set_yticks(range(len(modules)))
    ax.set_yticklabels(modules, fontsize=6.8, fontweight="bold")
    for i in range(len(modules)):
        for j in range(len(endpoints)):
            val = mat[i, j]
            if not np.isnan(val):
                text_color = WHITE if val < -0.3 or val > 0.3 else INK
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=6.0, color=text_color)
    ax.set_title("Supplementary Figure 2 | Module score difference in favorable vs unfavorable endpoints",
                 loc="left", fontsize=7.3, fontweight="bold", pad=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean difference", fontsize=6.2)
    cbar.ax.tick_params(labelsize=5.8)
    ax.text(0.0, -0.25, "Positive values indicate higher module score in favorable endpoint samples.",
            transform=ax.transAxes, fontsize=5.5, color=MUTED)
    plt.tight_layout()
    save(fig, "FigureS2_score_module_detail")


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary Figure 3: GSE206285 score by clinical remission
# ─────────────────────────────────────────────────────────────────────────────

def figureS3_gse206285_remission() -> None:
    """Supplementary Figure 3: GSE206285 clinical remission."""
    fpath = "data/processed/GSE206285/baseline_endpoint.tsv"
    if not os.path.exists(fpath): return
    g2 = pd.read_csv(fpath, sep="\t"); g2["axis_score"] = pd.to_numeric(g2["axis_score"], errors="coerce")
    if "week8_clinical_remission" not in g2.columns: return
    g2_0, g2_1 = g2.loc[g2["week8_clinical_remission"] == 0, "axis_score"], g2.loc[g2["week8_clinical_remission"] == 1, "axis_score"]
    fig, ax = plt.subplots(figsize=(3.2, 3.8))
    _draw_distribution(ax, g2_0, g2_1, title="GSE206285", subtitle="Adult UC UNIFI · clinical remission", n0_label=f"No remission\nn={len(g2_0)}", n1_label=f"Remission\nn={len(g2_1)}", or_text="OR 0.67 (0.52–0.85)", ylab=True)
    fig.suptitle("Supplementary Figure 3 | Clinical remission validation", fontsize=7.5, fontweight="bold", y=1.02)
    plt.tight_layout(); save(fig, "FigureS3_gse206285_remission")


def main() -> int:
    setup_style(); ensure_dir()
    print(f"[24] Generating submission-grade figures → {OUT_DIR}")
    figure1_study_architecture(); figure2_adult_endpoint_evidence(); figure3_age_stratified_atlas(); figure4_evidence_architecture()
    figureS1_sensitivity_models(); figureS2_score_module_detail(); figureS3_gse206285_remission()
    print(f"[24] Done. All figures saved to {OUT_DIR}"); return 0

if __name__ == "__main__":
    raise SystemExit(main())
