#!/usr/bin/env python3
"""Expression- and variance-matched random-gene null benchmark.

The current defensible matched-null analysis is run in GSE193677, where a
gene-level RNA-seq matrix is available. Older adult endpoint arrays remain in
the same-size random benchmark because their processed public matrices are
probe-level and were not remapped genome-wide for broad null matching.
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
OUT = ROOT / "results/validation/matched_null_benchmark.tsv"
ITER_OUT = ROOT / "results/validation/matched_null_iterations.tsv"
N_ITER = 1000
SEED = 20260808
NEAREST_K = 200


MODULE_GENES = {
    "upstream": [("IPMK", -1.0), ("IPPK", -1.0), ("HDAC3", -1.0), ("NCOR1", -1.0), ("NCOR2", -1.0)],
    "mmp": [("MMP1", 1.0), ("MMP3", 1.0), ("MMP10", 1.0), ("MMP12", 1.0), ("MMP13", 1.0)],
    "junction": [("TJP1", -1.0), ("OCLN", -1.0), ("CLDN2", 1.0)],
}


def import_gse193677_module():
    spec = importlib.util.spec_from_file_location("r39", ROOT / "scripts/39_gse193677_replication.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def zscore_rows(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x, axis=1, keepdims=True)
    sd = np.nanstd(x, axis=1, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def zscore_vector(x: np.ndarray) -> np.ndarray:
    sd = np.nanstd(x, ddof=1)
    return (x - np.nanmean(x)) / (sd if sd else 1.0)


def load_one_per_participant_matrix():
    r39 = import_gse193677_module()
    meta = r39.parse_series_matrix()
    meta["disease_group"] = meta["ibd_disease"].replace({"CD": "IBD", "UC": "IBD", "Control": "Control"})
    meta["ibd_vs_control"] = (meta["disease_group"] == "IBD").astype(int)
    with gzip.open(r39.ADJCOUNTS, "rt", errors="replace") as handle:
        header = [t.strip('"') for t in handle.readline().split()]
    order = {sid: i for i, sid in enumerate(header)}
    meta["_order"] = meta["sample_id"].map(order)
    one = meta.sort_values("_order").groupby("participant_id", as_index=False).first()
    samples = one["sample_id"].tolist()
    keep_idx = np.asarray([order[s] for s in samples], dtype=int)
    genes = []
    rows = []
    with gzip.open(r39.ADJCOUNTS, "rt", errors="replace") as handle:
        header = [t.strip('"') for t in handle.readline().split()]
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            genes.append(parts[0].strip('"'))
            values = np.asarray([float(t.strip('"')) for t in parts[1:]], dtype=np.float32)
            rows.append(values)
    matrix = np.vstack(rows)
    y = one.set_index("sample_id").loc[samples, "ibd_vs_control"].to_numpy(dtype=int)
    return genes, matrix, keep_idx, y


def observed_ids():
    r39 = import_gse193677_module()
    return {symbol: ens for symbol, ens in r39.GENE_ENSEMBL.items()}


def odds_ratio(score: np.ndarray, y: np.ndarray) -> float:
    score_z = zscore_vector(score)
    model = sm.Logit(y, sm.add_constant(score_z)).fit(disp=False)
    return float(math.exp(model.params[1]))


def observed_or_from_replication() -> float:
    models = pd.read_csv(ROOT / "results/replication/GSE193677_replication_models.tsv", sep="\t")
    row = models[models["analysis"].eq("one_per_participant_ibd_vs_control_unadjusted")]
    if row.empty:
        raise RuntimeError("Missing GSE193677 one-per-participant replication model")
    return float(row["or_per_1sd"].iloc[0])


def score_from_gene_indices(zmat: np.ndarray, module_indices: dict[str, list[int]]) -> np.ndarray:
    module_scores = []
    for module, genes in MODULE_GENES.items():
        idx = module_indices[module]
        signs = np.asarray([sign for _symbol, sign in genes], dtype=float)
        module_scores.append((zmat[idx, :] * signs[:, None]).mean(axis=0))
    score = np.vstack(module_scores).mean(axis=0)
    return score


def nearest_candidates(metric: np.ndarray, observed_index: int, excluded: set[int]) -> np.ndarray:
    distance = np.abs(metric - metric[observed_index])
    order = np.argsort(distance)
    candidates = [i for i in order if i not in excluded and np.isfinite(metric[i])]
    return np.asarray(candidates[:NEAREST_K], dtype=int)


def matched_iteration(
    rng: np.random.Generator,
    observed_idx_by_symbol: dict[str, int],
    candidate_pools: dict[str, np.ndarray],
) -> dict[str, list[int]]:
    chosen: set[int] = set(observed_idx_by_symbol.values())
    module_indices: dict[str, list[int]] = {}
    for module, genes in MODULE_GENES.items():
        module_indices[module] = []
        for symbol, _sign in genes:
            candidates = np.asarray([i for i in candidate_pools[symbol] if i not in chosen], dtype=int)
            if len(candidates) == 0:
                raise RuntimeError(f"No matched candidates available for {symbol}")
            pick = int(rng.choice(candidates))
            chosen.add(pick)
            module_indices[module].append(pick)
    return module_indices


def main() -> int:
    os.makedirs(OUT.parent, exist_ok=True)
    genes, matrix, keep_idx, y = load_one_per_participant_matrix()
    observed_map = observed_ids()
    gene_to_index = {g: i for i, g in enumerate(genes)}
    observed_idx_by_symbol = {s: gene_to_index[e] for s, e in observed_map.items() if e in gene_to_index}
    missing = sorted(set(observed_map) - set(observed_idx_by_symbol))
    if missing:
        raise RuntimeError(f"Observed genes missing from GSE193677 matrix: {missing}")

    zmat = zscore_rows(matrix)
    observed_indices = {
        module: [observed_idx_by_symbol[symbol] for symbol, _sign in genes_with_sign]
        for module, genes_with_sign in MODULE_GENES.items()
    }
    observed_score = score_from_gene_indices(zmat, observed_indices)[keep_idx]
    observed_or = observed_or_from_replication()
    observed_or_recomputed = odds_ratio(observed_score, y)

    mean_expr = np.nanmean(matrix, axis=1)
    var_expr = np.nanvar(matrix, axis=1, ddof=1)
    mean_rank = pd.Series(mean_expr).rank(method="average").to_numpy()
    var_rank = pd.Series(var_expr).rank(method="average").to_numpy()
    composite = mean_rank + var_rank
    observed_indices_set = set(observed_idx_by_symbol.values())
    candidate_pools = {
        symbol: nearest_candidates(composite, obs, observed_indices_set)
        for symbol, obs in observed_idx_by_symbol.items()
    }

    rng = np.random.default_rng(SEED)
    random_ors = []
    valid = 0
    for _ in range(N_ITER):
        try:
            idx = matched_iteration(rng, observed_idx_by_symbol, candidate_pools)
            score = score_from_gene_indices(zmat, idx)[keep_idx]
            random_ors.append(odds_ratio(score, y))
            valid += 1
        except Exception:
            continue
    random_ors = np.asarray(random_ors)
    observed_abs = abs(math.log(observed_or))
    null_abs = np.abs(np.log(random_ors))
    empirical_p = (1 + np.sum(null_abs >= observed_abs)) / (1 + len(random_ors))
    percentile = float(np.mean(random_ors >= observed_or))

    rows = [
        {
            "dataset_id": "GSE193677",
            "endpoint": "IBD versus control, one biopsy per participant",
            "null_design": "expression_and_variance_matched_module_structure",
            "feature_level": "gene",
            "n_samples": int(len(y)),
            "n_features_universe": int(len(genes)),
            "n_observed_genes": 13,
            "n_iter_requested": N_ITER,
            "n_iter_valid": valid,
            "seed": SEED,
            "nearest_k": NEAREST_K,
            "observed_or": observed_or,
            "observed_or_recomputed": observed_or_recomputed,
            "observed_abs_log_or": observed_abs,
            "null_or_median": float(np.median(random_ors)),
            "null_or_p2_5": float(np.percentile(random_ors, 2.5)),
            "null_or_p97_5": float(np.percentile(random_ors, 97.5)),
            "empirical_p_twosided": float(empirical_p),
            "percentile_extreme_direction": percentile,
            "status": "modeled",
        },
        {
            "dataset_id": "GSE193677",
            "endpoint": "IBD versus control, one biopsy per participant",
            "null_design": "compartment_matched_random_gene_set",
            "feature_level": "gene",
            "n_samples": int(len(y)),
            "n_features_universe": int(len(genes)),
            "n_observed_genes": 13,
            "n_iter_requested": N_ITER,
            "n_iter_valid": 0,
            "seed": SEED,
            "nearest_k": "",
            "observed_or": observed_or,
            "observed_or_recomputed": observed_or_recomputed,
            "observed_abs_log_or": observed_abs,
            "null_or_median": "",
            "null_or_p2_5": "",
            "null_or_p97_5": "",
            "empirical_p_twosided": "",
            "percentile_extreme_direction": "",
            "status": "not_performed_no_full_gene_compartment_map_for_random_universe",
        },
    ]
    pd.DataFrame(rows).to_csv(OUT, sep="\t", index=False)
    pd.DataFrame(
        {
            "dataset_id": "GSE193677",
            "endpoint": "IBD versus control, one biopsy per participant",
            "null_design": "expression_and_variance_matched_module_structure",
            "iteration": np.arange(1, len(random_ors) + 1),
            "random_or": random_ors,
            "random_abs_log_or": np.abs(np.log(random_ors)),
        }
    ).to_csv(ITER_OUT, sep="\t", index=False)
    print(f"wrote={OUT} rows={len(rows)} observed_or={observed_or:.4f} empirical_p={empirical_p:.4f}")
    print(f"wrote={ITER_OUT} rows={len(random_ors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
