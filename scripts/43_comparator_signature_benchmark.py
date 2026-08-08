#!/usr/bin/env python3
"""Head-to-head comparator benchmarks for the barrier-injury score.

The script separates two evidence layers:

1. Module-level adult endpoint comparisons already supported by the processed
   endpoint tables (original score, MMP-only, upstream, junctional).
2. Full comparator signature comparisons in GSE193677, where a gene-level
   RNA-seq matrix allows transparent scoring of inflammatory, stromal, and
   myeloid comparator signatures.

This preserves scientific traceability and avoids remapping older probe-level
arrays into broad gene-set scores unless a validated platform-wide mapping is
available.
"""

from __future__ import annotations

import gzip
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent
SIGNATURES = ROOT / "docs/COMPARATOR_SIGNATURES.tsv"
MODULE_MODELS = ROOT / "results/clinical/module_contribution_models.tsv"
INFLAMMATION_MODELS = ROOT / "results/clinical/inflammation_specificity_models.tsv"
COMPARATOR_MODELS = ROOT / "results/clinical/comparator_models.tsv"
GSE73661_COMP = ROOT / "data/processed/GSE73661/comparator_gene_expression.tsv"
AXIS_SCORES = ROOT / "results/axis/barrier_axis_scores.tsv"

OUT_DIR = ROOT / "results/benchmarks"
BENCHMARK_OUT = OUT_DIR / "comparator_signature_benchmark.tsv"
INCREMENT_OUT = OUT_DIR / "comparator_incremental_models.tsv"
AVAIL_OUT = OUT_DIR / "signature_availability.tsv"


# Canonical Ensembl IDs for comparator genes used in GSE193677. Genes with
# uncertain mapping are intentionally omitted rather than guessed.
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
    "OSM": "ENSG00000099985",
    "OSMR": "ENSG00000145623",
    "IL6ST": "ENSG00000134352",
    "LIF": "ENSG00000128342",
    "LIFR": "ENSG00000113594",
    "TREM1": "ENSG00000124731",
    "TNF": "ENSG00000232810",
    "TNFRSF1B": "ENSG00000028137",
    "IL13RA2": "ENSG00000123496",
    "IL6": "ENSG00000136244",
    "IL1B": "ENSG00000125538",
    "CXCL8": "ENSG00000169429",
    "CXCL1": "ENSG00000163739",
    "CXCL2": "ENSG00000081041",
    "CXCL10": "ENSG00000169245",
    "CCL2": "ENSG00000108691",
    "ICAM1": "ENSG00000090339",
    "SELE": "ENSG00000007908",
    "NFKBIA": "ENSG00000100906",
    "TNFAIP3": "ENSG00000118503",
    "PTGS2": "ENSG00000073756",
    "PLAUR": "ENSG00000011422",
    "SOD2": "ENSG00000112096",
    "TLR2": "ENSG00000137462",
    "RELB": "ENSG00000104856",
    "NFKB1": "ENSG00000109320",
    "COL1A1": "ENSG00000108821",
    "COL1A2": "ENSG00000164692",
    "COL3A1": "ENSG00000168542",
    "FAP": "ENSG00000078098",
    "VIM": "ENSG00000026025",
    "ACTA2": "ENSG00000107796",
    "POSTN": "ENSG00000133110",
    "FN1": "ENSG00000115414",
    "SPARC": "ENSG00000113140",
    "TIMP1": "ENSG00000102265",
    "TGFB1": "ENSG00000105329",
    "PDGFRB": "ENSG00000113721",
    "S100A8": "ENSG00000143546",
    "S100A9": "ENSG00000163220",
    "FCGR3B": "ENSG00000162747",
    "CSF3R": "ENSG00000119535",
    "CXCR2": "ENSG00000180871",
    "MPO": "ENSG00000005381",
    "ELANE": "ENSG00000197561",
    "LCN2": "ENSG00000148346",
    "FCGR1A": "ENSG00000150337",
}


def parse_gene_list(text: str) -> list[tuple[str, float]]:
    genes = []
    for item in text.split(";"):
        gene, sign = item.split(":")
        genes.append((gene, float(sign)))
    return genes


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=1)
    return (s - s.mean()) / sd if sd and not math.isnan(sd) else s * 0


def fit_logit(data: pd.DataFrame, formula: str, term: str) -> dict:
    model = smf.logit(formula, data=data).fit(disp=False)
    ci = model.conf_int().loc[term]
    return {
        "n": int(model.nobs),
        "model_formula": formula,
        "term": term,
        "effect_type": "odds_ratio_per_1sd",
        "effect": float(math.exp(model.params[term])),
        "ci_lower": float(math.exp(ci[0])),
        "ci_upper": float(math.exp(ci[1])),
        "pvalue": float(model.pvalues[term]),
        "aic": float(model.aic),
        "llf": float(model.llf),
    }


def lrt_p(base_llf: float, joint_llf: float, df: int = 1) -> float:
    stat = max(0.0, 2.0 * (joint_llf - base_llf))
    return float(stats.chi2.sf(stat, df))


def load_signatures() -> pd.DataFrame:
    frame = pd.read_csv(SIGNATURES, sep="\t")
    frame["parsed_genes"] = frame["genes"].map(parse_gene_list)
    return frame


def import_gse193677_module():
    spec = importlib.util.spec_from_file_location("r39", ROOT / "scripts/39_gse193677_replication.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_gse193677_expression(signatures: pd.DataFrame):
    r39 = import_gse193677_module()
    wanted_symbols = sorted({gene for genes in signatures["parsed_genes"] for gene, _ in genes})
    wanted_ids = {GENE_ENSEMBL[g]: g for g in wanted_symbols if g in GENE_ENSEMBL}
    rows: dict[str, list[float]] = {}
    with gzip.open(r39.ADJCOUNTS, "rt", errors="replace") as handle:
        header = [t.strip('"') for t in handle.readline().split()]
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            ens = parts[0].strip('"')
            symbol = wanted_ids.get(ens)
            if symbol is not None:
                rows[symbol] = [float(t.strip('"')) for t in parts[1:]]
    expr = pd.DataFrame(rows, index=header)
    expr.index.name = "sample_id"
    return expr, header


def gse193677_analysis_frame(signatures: pd.DataFrame):
    r39 = import_gse193677_module()
    meta = r39.parse_series_matrix()
    meta["disease_group"] = meta["ibd_disease"].replace({"CD": "IBD", "UC": "IBD", "Control": "Control"})
    meta["ibd_vs_control"] = (meta["disease_group"] == "IBD").astype(int)
    meta["age_z"] = zscore(pd.to_numeric(meta["study_eligibility_age_at_endo"], errors="coerce"))
    meta["sex"] = meta["demographics_gender"].replace({"": np.nan})
    meta["site"] = meta["regionre"].replace({"": np.nan})
    expr, header = read_gse193677_expression(signatures)
    order = {sid: i for i, sid in enumerate(header)}
    meta["_order"] = meta["sample_id"].map(order)
    one = meta[meta["sample_id"].isin(expr.index)].sort_values("_order").groupby("participant_id", as_index=False).first()
    one = one.set_index("sample_id").join(expr, how="inner").reset_index()
    return one, expr


def score_signature(expr: pd.DataFrame, genes: list[tuple[str, float]]) -> tuple[pd.Series, list[str], list[str]]:
    available = [gene for gene, _ in genes if gene in expr.columns]
    missing = [gene for gene, _ in genes if gene not in expr.columns]
    if len(available) < 2:
        return pd.Series(np.nan, index=expr.index), available, missing
    signed = []
    sign_map = dict(genes)
    for gene in available:
        signed.append(zscore(expr[gene]) * sign_map[gene])
    score = pd.concat(signed, axis=1).mean(axis=1)
    return score, available, missing


def module_benchmark_rows() -> list[dict]:
    if not MODULE_MODELS.exists():
        return []
    module = pd.read_csv(MODULE_MODELS, sep="\t")
    label_map = pd.read_csv(SIGNATURES, sep="\t").set_index("signature_id")["display_label"].to_dict()
    keep_variants = {
        "original_axis": "barrier_injury_score",
        "mmp_only": "mmp_injury",
        "regulatory_only": "upstream_regulatory",
        "junctional_only": "junctional_complex",
    }
    adult_keep = module[module["dataset_id"].isin(["GSE73661", "GSE206285", "GSE16879", "GSE12251", "GSE92415", "GSE23597"])]
    adult_keep = adult_keep[adult_keep["variant"].isin(keep_variants)]
    rows = []
    for _, row in adult_keep.iterrows():
        rows.append(
            {
                "dataset_id": row["dataset_id"],
                "endpoint": row["endpoint"],
                "signature_id": keep_variants[row["variant"]],
                "display_label": label_map.get(keep_variants[row["variant"]], row["variant"].replace("_", " ")),
                "model_type": "cohort_specific_adjusted_module_model",
                "n": int(row["n"]),
                "model_formula": row["model_formula"],
                "term": row["variant"] + "_z",
                "effect_type": "odds_ratio_per_1sd",
                "effect": row["effect"],
                "ci_lower": row["ci_lower"],
                "ci_upper": row["ci_upper"],
                "pvalue": row["pvalue"],
                "aic": row["aic"],
                "availability_scope": "prespecified_score_components_available",
            }
        )
    return rows


def inflammation_rows() -> tuple[list[dict], list[dict], list[dict]]:
    rows, incr, avail = [], [], []
    if not INFLAMMATION_MODELS.exists():
        return rows, incr, avail
    inf = pd.read_csv(INFLAMMATION_MODELS, sep="\t")
    for _, row in inf.iterrows():
        if row["model_type"] not in {"inflammation_only", "joint_adjusted"}:
            continue
        sig = "tnf_inflammatory" if row["term"] == "inflammation_score_z" else "barrier_injury_score"
        rows.append(
            {
                "dataset_id": row["dataset_id"],
                "endpoint": row["endpoint"],
                "signature_id": sig,
                "display_label": sig.replace("_", " "),
                "model_type": row["model_type"],
                "n": int(row["n"]),
                "model_formula": row["model"],
                "term": row["term"],
                "effect_type": "odds_ratio_per_1sd",
                "effect": row["effect"],
                "ci_lower": row["ci_lower"],
                "ci_upper": row["ci_upper"],
                "pvalue": row["pvalue"],
                "aic": "",
                "availability_scope": "compact_inflammatory_comparator_available",
            }
        )
    # Availability for the existing compact comparator table.
    if GSE73661_COMP.exists():
        comp = pd.read_csv(GSE73661_COMP, sep="\t", index_col=0)
        avail.append(
            {
                "dataset_id": "GSE73661",
                "signature_id": "tnf_inflammatory",
                "available_genes": ";".join([g for g in comp.index if g in {"OSM", "TREM1", "TNF", "TNFRSF1B", "IL13RA2"}]),
                "missing_genes": "IL6;IL1B;CXCL8",
                "n_available_genes": int(len(comp.index)),
                "n_requested_genes": 8,
                "availability_status": "partial_compact_comparator_from_processed_matrix",
            }
        )
    if COMPARATOR_MODELS.exists():
        comp_models = pd.read_csv(COMPARATOR_MODELS, sep="\t")
        for _, row in comp_models.iterrows():
            incr.append(
                {
                    "dataset_id": row["dataset_id"],
                    "endpoint": "mucosal_healing",
                    "comparator_signature_id": "tnf_inflammatory",
                    "n": int(row["n"]),
                    "base_model": row["base_model"],
                    "joint_model": row["axis_model"],
                    "axis_effect": row["axis_effect"],
                    "axis_ci_lower": row["ci_lower"],
                    "axis_ci_upper": row["ci_upper"],
                    "axis_pvalue": row["pvalue"],
                    "comparator_effect_joint": "",
                    "comparator_pvalue_joint": "",
                    "likelihood_ratio_p_for_axis_increment": row["delta_value"],
                    "status": "modeled_existing_compact_inflammatory_comparator",
                }
            )
    return rows, incr, avail


def gse193677_rows(signatures: pd.DataFrame) -> tuple[list[dict], list[dict], list[dict]]:
    one, expr = gse193677_analysis_frame(signatures)
    axis = pd.read_csv(AXIS_SCORES, sep="\t")
    axis = axis[axis["dataset_id"].eq("GSE193677")].copy()
    axis = axis.rename(
        columns={
            "axis_score": "barrier_injury_score",
            "mmp_score": "mmp_injury",
            "upstream_score": "upstream_regulatory",
            "junction_score": "junctional_complex",
        }
    )
    score_frame = one[["sample_id", "ibd_vs_control", "age_z", "sex", "site"]].copy()
    score_frame = score_frame.merge(
        axis[["sample_id", "barrier_injury_score", "mmp_injury", "upstream_regulatory", "junctional_complex"]],
        on="sample_id",
        how="left",
    )
    availability = []
    for _, sig in signatures.iterrows():
        if sig["signature_id"] in {"barrier_injury_score", "mmp_injury", "upstream_regulatory", "junctional_complex"}:
            available = [gene for gene, _ in sig["parsed_genes"] if gene in GENE_ENSEMBL]
            missing = [gene for gene, _ in sig["parsed_genes"] if gene not in GENE_ENSEMBL]
        else:
            score, available, missing = score_signature(one.set_index("sample_id"), sig["parsed_genes"])
            score_frame[sig["signature_id"]] = score.reindex(score_frame["sample_id"]).to_numpy()
        availability.append(
            {
                "dataset_id": "GSE193677",
                "signature_id": sig["signature_id"],
                "available_genes": ";".join(available),
                "missing_genes": ";".join(missing),
                "n_available_genes": len(available),
                "n_requested_genes": len(sig["parsed_genes"]),
                "availability_status": "modeled" if len(available) >= 2 else "not_modeled_insufficient_genes",
            }
        )

    rows, incr = [], []
    for _, sig in signatures.iterrows():
        sid = sig["signature_id"]
        if score_frame[sid].notna().sum() < 30:
            continue
        model_data = score_frame.dropna(subset=[sid, "ibd_vs_control"]).copy()
        model_data[f"{sid}_z"] = zscore(model_data[sid])
        for model_type, formula in [
            ("unadjusted", f"ibd_vs_control ~ {sid}_z"),
            ("age_sex_site_adjusted", f"ibd_vs_control ~ {sid}_z + age_z + C(sex) + C(site)"),
        ]:
            try:
                fit = fit_logit(model_data.dropna(subset=["age_z", "sex", "site"]) if "age_z" in formula else model_data,
                                formula, f"{sid}_z")
            except Exception as exc:  # pragma: no cover - captured in output table
                rows.append(
                    {
                        "dataset_id": "GSE193677",
                        "endpoint": "IBD versus control",
                        "signature_id": sid,
                        "display_label": sig["display_label"],
                        "model_type": model_type,
                        "n": int(len(model_data)),
                        "model_formula": formula,
                        "term": f"{sid}_z",
                        "effect_type": "model_failed",
                        "effect": "",
                        "ci_lower": "",
                        "ci_upper": "",
                        "pvalue": "",
                        "aic": "",
                        "availability_scope": f"error:{exc}",
                    }
                )
                continue
            rows.append(
                {
                    "dataset_id": "GSE193677",
                    "endpoint": "IBD versus control",
                    "signature_id": sid,
                    "display_label": sig["display_label"],
                    "model_type": model_type,
                    "availability_scope": "gene_level_rnaseq_matrix",
                    **fit,
                }
            )

    # Incremental models: barrier score added to each non-score comparator.
    base_score = "barrier_injury_score"
    for _, sig in signatures.iterrows():
        sid = sig["signature_id"]
        if sid == base_score or score_frame[sid].notna().sum() < 30:
            continue
        model_data = score_frame.dropna(subset=[base_score, sid, "age_z", "sex", "site"]).copy()
        model_data[f"{base_score}_z"] = zscore(model_data[base_score])
        model_data[f"{sid}_z"] = zscore(model_data[sid])
        base_formula = f"ibd_vs_control ~ {sid}_z + age_z + C(sex) + C(site)"
        joint_formula = f"ibd_vs_control ~ {sid}_z + {base_score}_z + age_z + C(sex) + C(site)"
        try:
            base_fit = fit_logit(model_data, base_formula, f"{sid}_z")
            joint_axis = fit_logit(model_data, joint_formula, f"{base_score}_z")
            joint_comp = fit_logit(model_data, joint_formula, f"{sid}_z")
        except Exception as exc:  # pragma: no cover
            incr.append(
                {
                    "dataset_id": "GSE193677",
                    "endpoint": "IBD versus control",
                    "comparator_signature_id": sid,
                    "n": int(len(model_data)),
                    "base_model": base_formula,
                    "joint_model": joint_formula,
                    "axis_effect": "",
                    "axis_ci_lower": "",
                    "axis_ci_upper": "",
                    "axis_pvalue": "",
                    "comparator_effect_joint": "",
                    "comparator_pvalue_joint": "",
                    "likelihood_ratio_p_for_axis_increment": "",
                    "status": f"model_failed:{exc}",
                }
            )
            continue
        incr.append(
            {
                "dataset_id": "GSE193677",
                "endpoint": "IBD versus control",
                "comparator_signature_id": sid,
                "n": int(base_fit["n"]),
                "base_model": base_formula,
                "joint_model": joint_formula,
                "axis_effect": joint_axis["effect"],
                "axis_ci_lower": joint_axis["ci_lower"],
                "axis_ci_upper": joint_axis["ci_upper"],
                "axis_pvalue": joint_axis["pvalue"],
                "comparator_effect_joint": joint_comp["effect"],
                "comparator_pvalue_joint": joint_comp["pvalue"],
                "likelihood_ratio_p_for_axis_increment": lrt_p(base_fit["llf"], joint_axis["llf"]),
                "status": "modeled",
            }
        )
    return rows, incr, availability


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    signatures = load_signatures()
    benchmark_rows = module_benchmark_rows()
    inf_rows, inf_incr, availability = inflammation_rows()
    benchmark_rows.extend(inf_rows)
    increment_rows = inf_incr
    gse_rows, gse_incr, gse_avail = gse193677_rows(signatures)
    benchmark_rows.extend(gse_rows)
    increment_rows.extend(gse_incr)
    availability.extend(gse_avail)

    pd.DataFrame(benchmark_rows).to_csv(BENCHMARK_OUT, sep="\t", index=False)
    pd.DataFrame(increment_rows).to_csv(INCREMENT_OUT, sep="\t", index=False)
    pd.DataFrame(availability).to_csv(AVAIL_OUT, sep="\t", index=False)
    print(f"wrote={BENCHMARK_OUT} rows={len(benchmark_rows)}")
    print(f"wrote={INCREMENT_OUT} rows={len(increment_rows)}")
    print(f"wrote={AVAIL_OUT} rows={len(availability)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
