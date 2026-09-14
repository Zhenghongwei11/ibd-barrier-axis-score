#!/usr/bin/env python3
"""Random gene-set permutation benchmark for the barrier-injury score.

For each adult endpoint cohort, draws N_ITER random feature sets with the same
available module structure (upstream -1, MMP +1, junction with 2 -1 / 1 +1)
from the cohort's measured features, fits the same unadjusted logistic model,
and compares the observed OR to the null distribution (empirical two-sided P
and extreme-direction percentile).

Feature level is probe-level for Affymetrix cohorts and gene-level for
GSE193677 (RNA-seq); the observed ORs are taken from the published model tables
so the null comparison uses the same endpoint data and analysis set.
"""

from __future__ import annotations

import gzip
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/validation/permutation_benchmark.tsv"
N_ITER = 1000
SEED = 20260807

MODULE_SIZES = [("upstream", 5, -1), ("mmp", 5, 1), ("junction", 3, None)]


def read_tab(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def zscore_matrix(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def random_set_score(zmat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Score from a random feature set mirroring the module structure."""
    module_scores = []
    for _name, size, sign in MODULE_SIZES:
        idx = rng.choice(zmat.shape[0], size=size, replace=False)
        if sign is not None:
            module_scores.append(zmat[idx].mean(axis=0) * sign)
        else:
            # Mirror junction module: two barrier-supportive (-1), one injury-aligned (+1).
            signs = np.full(size, -1.0)
            signs[rng.integers(0, size)] = 1.0
            module_scores.append((zmat[idx] * signs[:, None]).mean(axis=0))
    score = np.mean(module_scores, axis=0)
    return (score - score.mean()) / (score.std(ddof=1) or 1.0)


def run_benchmark(
    dataset_id: str,
    endpoint: str,
    observed_or: float,
    zmat: np.ndarray,
    y: np.ndarray,
    feature_level: str,
    n_features: int,
    n_samples: int,
) -> dict:
    rng = np.random.default_rng(SEED)
    observed_logor = math.log(observed_or)
    random_ors = np.empty(N_ITER)
    for i in range(N_ITER):
        score_z = random_set_score(zmat, rng)
        X = sm.add_constant(score_z)
        model = sm.Logit(y, X).fit(disp=False)
        random_ors[i] = math.exp(model.params[1])
    abs_log = np.abs(np.log(random_ors))
    empirical_p = (1.0 + np.sum(abs_log >= abs(observed_logor))) / (1.0 + N_ITER)
    if observed_or < 1:
        percentile = float(np.mean(random_ors <= observed_or))
    else:
        percentile = float(np.mean(random_ors >= observed_or))
    return {
        "dataset_id": dataset_id,
        "endpoint": endpoint,
        "feature_level": feature_level,
        "n_features": n_features,
        "n_samples": n_samples,
        "n_iter": N_ITER,
        "seed": SEED,
        "observed_or": observed_or,
        "observed_abs_log_or": abs(observed_logor),
        "random_or_median": float(np.median(random_ors)),
        "random_or_p2_5": float(np.percentile(random_ors, 2.5)),
        "random_or_p97_5": float(np.percentile(random_ors, 97.5)),
        "empirical_p_twosided": empirical_p,
        "percentile_extreme_direction": percentile,
    }


def load_gse193677_one_per_participant():
    """Load GSE193677 gene-level matrix restricted to one biopsy per participant."""
    spec = importlib.util.spec_from_file_location(
        "r39", ROOT / "scripts/39_gse193677_replication.py"
    )
    r39 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r39)
    meta = r39.parse_series_matrix()
    meta["disease_group"] = meta["ibd_disease"].replace({"CD": "IBD", "UC": "IBD", "Control": "Control"})
    meta["ibd_vs_control"] = (meta["disease_group"] == "IBD").astype(int)
    with gzip.open(r39.ADJCOUNTS, "rt", errors="replace") as handle:
        header = [t.strip('"') for t in handle.readline().split()]
    order = {sid: i for i, sid in enumerate(header)}
    meta["_order"] = meta["sample_id"].map(order)
    one = meta.sort_values("_order").groupby("participant_id", as_index=False).first()
    keep = set(one["sample_id"])
    genes = []
    cols = []
    with gzip.open(r39.ADJCOUNTS, "rt", errors="replace") as handle:
        header = [t.strip('"') for t in handle.readline().split()]
        keep_idx = [i for i, sid in enumerate(header) if sid in keep]
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            genes.append(parts[0].strip('"'))
            cols.append([float(t) for t in parts[1:]])
    mat = np.asarray(cols, dtype=np.float32)[:, keep_idx]
    y = one.set_index("sample_id").loc[[header[i] for i in keep_idx], "ibd_vs_control"].to_numpy().astype(int)
    return mat, y, len(genes)


def observed_gse193677_or() -> float:
    models = pd.read_csv(ROOT / "results/replication/GSE193677_replication_models.tsv", sep="\t")
    row = models[models["analysis"].eq("one_per_participant_ibd_vs_control_unadjusted")]
    if row.empty:
        raise RuntimeError("Missing GSE193677 unadjusted replication model")
    return float(row["or_per_1sd"].iloc[0])


def main() -> int:
    os.makedirs(OUT.parent, exist_ok=True)
    rows = []

    array_cohorts = [
        ("GSE73661", "data/processed/GSE73661/expression.tsv",
         "data/processed/GSE73661/baseline_endpoint.tsv", "baseline_sample_id",
         [("mucosal_healing", 0.52115)]),
        ("GSE12251", "data/processed/GSE12251/expression_raw.tsv",
         "data/processed/GSE12251/baseline_endpoint.tsv", "sample_id",
         [("wk8_endoscopic_histologic_healing", 0.274677)]),
        ("GSE92415", "data/processed/GSE92415/expression_raw.tsv",
         "data/processed/GSE92415/baseline_endpoint.tsv", "sample_id",
         [("week6_clinical_response", 0.749952)]),
        ("GSE206285", "data/processed/GSE206285/expression_raw.tsv",
         "data/processed/GSE206285/baseline_endpoint.tsv", "sample_id",
         [("week8_mucosal_healing", 0.562673), ("week8_clinical_remission", 0.665193)]),
        ("GSE23597", "data/processed/GSE23597/expression_raw.tsv",
         "data/processed/GSE23597/baseline_endpoint.tsv", "sample_id",
         [("week8_response", 0.554221), ("week30_response", 0.604487)]),
        ("GSE16879", "data/processed/GSE16879/expression_raw.tsv",
         "data/processed/GSE16879/baseline_endpoint.tsv", "sample_id",
         [("infliximab_response", 0.193463)]),
    ]
    for dataset_id, expr_path, ep_path, sample_col, endpoints in array_cohorts:
        print(f"[{dataset_id}] loading expression ...", flush=True)
        expr = pd.read_csv(ROOT / expr_path, sep="\t")
        expr = expr.set_index(expr.columns[0])
        ep = read_tab(ROOT / ep_path)
        for endpoint, observed_or in endpoints:
            ep_e = ep[[sample_col, endpoint]].copy()
            ep_e[endpoint] = pd.to_numeric(ep_e[endpoint], errors="coerce")
            ep_e = ep_e.dropna(subset=[endpoint])
            ep_e = ep_e[ep_e[sample_col].isin(expr.columns)]
            samples = ep_e[sample_col].tolist()
            x = expr[samples].to_numpy(dtype=float)
            y = ep_e[endpoint].to_numpy(dtype=float).astype(int)
            zmat = zscore_matrix(x)
            n_features = zmat.shape[0]
            row = run_benchmark(dataset_id, endpoint, observed_or, zmat, y,
                                "probe", n_features, int(len(y)))
            rows.append(row)
            print(f"  {endpoint}: observed OR {observed_or:.3f} -> p={row['empirical_p_twosided']:.4f} "
                  f"pct={row['percentile_extreme_direction']:.3f} (random median {row['random_or_median']:.3f})", flush=True)

    # GSE193677 gene-level benchmark (IBD vs control, one biopsy per participant).
    print("[GSE193677] loading gene-level matrix ...", flush=True)
    mat, y, n_genes = load_gse193677_one_per_participant()
    zmat = zscore_matrix(mat)
    observed_or = observed_gse193677_or()
    row = run_benchmark("GSE193677", "ibd_vs_control_one_per_participant", observed_or,
                        zmat, y, "gene", n_genes, int(len(y)))
    rows.append(row)
    print(f"  ibd_vs_control: observed OR {observed_or:.3f} -> p={row['empirical_p_twosided']:.4f} "
          f"pct={row['percentile_extreme_direction']:.3f} (random median {row['random_or_median']:.3f})", flush=True)

    pd.DataFrame(rows).to_csv(OUT, sep="\t", index=False)
    print(f"wrote={OUT} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
