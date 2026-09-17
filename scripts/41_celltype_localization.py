#!/usr/bin/env python3
"""Cell-type localization of the 13 barrier-injury score genes in IBD single-cell data.

Uses GSE214695 (18 colonic scRNA-seq samples: 6 HC / 6 UC / 6 CD) with the GEO
cell annotation file. Reports per-cell-type detection fraction and mean
log1p(CPM) expression, aggregated by broad compartment and disease group.
Descriptive localization evidence only; no mechanism claim.
"""

from __future__ import annotations

import csv
import gzip
import io
import math
import os
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io as sio


ROOT = Path(__file__).resolve().parent.parent
TAR_PATH = ROOT / "data/raw/GSE214695/GSE214695_RAW.tar"
ANNOT_PATH = ROOT / "data/raw/GSE214695/GSE214695_cell_annotation.csv.gz"
OUT_CELLTYPE = ROOT / "results/validation/celltype_localization.tsv"
OUT_FINE = ROOT / "results/validation/celltype_localization_fine.tsv"

SCORE_GENES = [
    "IPMK", "IPPK", "HDAC3", "NCOR1", "NCOR2",
    "MMP1", "MMP3", "MMP10", "MMP12", "MMP13",
    "TJP1", "OCLN", "CLDN2",
]

COMPARTMENT_RULES = [
    ("epithelial", re.compile(r"Colonocyt|Epithelium|Goblet|TA$|Cycling TA|Secretory progenitor|Tuft|Paneth|Enteroendocrine|BEST4", re.I)),
    ("immune", re.compile(r"CD4|CD8|PC |Plasma|B cell|T cell|Treg|IEL|Mast|M[0-9]|macrophage|Neutrophil|N[0-9]|monocyte|DC|NK|ILC|MAIT|Eosinophil|Cycling T|Cycling myeloid|DN|Plasmablast|Memory B|GC B|Na[iï]ve B", re.I)),
    ("stromal", re.compile(r"Fibroblast|Endothel|Pericyt|Myofibroblast|Glia|FRC|^S[123]$|^S[123][ab]$", re.I)),
]


def open_member(tar: tarfile.TarFile, suffix: str) -> io.IOBase:
    for member in tar.getmembers():
        if member.name.endswith(suffix):
            return tar.extractfile(member)
    raise FileNotFoundError(suffix)


def parse_features(fh) -> dict[str, int]:
    """Map gene symbol -> row index from a 10X features.tsv."""
    mapping: dict[str, int] = {}
    with gzip.open(fh, "rt", errors="replace") as handle:
        for idx, line in enumerate(handle):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                mapping[parts[1].upper()] = idx
    return mapping


def read_annotations() -> pd.DataFrame:
    frame = pd.read_csv(ANNOT_PATH, compression="gzip")
    frame["sample"] = frame["sample"].astype(str)
    frame["disease"] = frame["sample"].str[:2]
    frame["barcode"] = frame["cell_id"].astype(str)
    frame["compartment"] = "other"
    for name, pattern in COMPARTMENT_RULES:
        frame.loc[frame["nanostring_reference"].str.contains(pattern), "compartment"] = name
    return frame


def main() -> int:
    os.makedirs(OUT_CELLTYPE.parent, exist_ok=True)
    annot = read_annotations()
    print(f"annotated cells={len(annot)} samples={annot['sample'].nunique()}")
    print("compartment counts:", annot["compartment"].value_counts().to_dict())

    features = None
    cell_matrix: dict[str, dict[str, float]] = {}  # cell_key -> {gene: log1p_cpm}
    per_sample_counts: dict[str, int] = {}

    with tarfile.open(TAR_PATH, "r") as tar:
        tar_members = {m.name: m for m in tar.getmembers()}
        # Map each tar sample file set (e.g., GSM6614348_HC-1) to annotation sample (HC1).
        gsm_to_sample: dict[str, str] = {}
        for mname in tar_members:
            base = mname.rsplit("/", 1)[-1]
            if base.endswith("_barcodes.tsv.gz"):
                token = base[: -len("_barcodes.tsv.gz")]
                gsm, _, sample_token = token.partition("_")
                gsm_to_sample[gsm] = sample_token.replace("-", "")
        print("tar samples:", sorted(set(gsm_to_sample.values())))
        for sample in sorted(set(annot["sample"])):
            gsm = next((g for g, s in gsm_to_sample.items() if s == sample), None)
            if gsm is None:
                print(f"  !! no tar files for {sample}")
                continue
            def find(suffix: str):
                name = f"{gsm}_{suffix}"
                if name in tar_members:
                    return tar.extractfile(tar_members[name])
                # fall back to any member containing the gsm and suffix
                for mname, m in tar_members.items():
                    if gsm in mname and mname.endswith(suffix):
                        return tar.extractfile(m)
                return None
            feats_fh = find("features.tsv.gz")
            if feats_fh is None:
                feats_fh = find("genes.tsv.gz")
            barcodes_fh = find("barcodes.tsv.gz")
            mtx_fh = find("matrix.mtx.gz")
            if feats_fh is None or barcodes_fh is None or mtx_fh is None:
                print(f"  !! missing files for {sample}: feat={feats_fh is not None} bar={barcodes_fh is not None} mtx={mtx_fh is not None}")
                continue
            if features is None:
                features = parse_features(feats_fh)
            else:
                feats_fh.close()
            with gzip.open(barcodes_fh, "rt", errors="replace") as bh:
                barcodes = [line.strip() for line in bh]
            per_sample_counts[sample] = len(barcodes)
            sample_annot = annot[annot["sample"] == sample]
            bc_to_idx = {bc: i for i, bc in enumerate(barcodes)}
            pairs = [(bc, bc_to_idx[bc]) for bc in sample_annot["barcode"] if bc in bc_to_idx]
            keep_idx = np.array([c for _, c in pairs], dtype=int)
            mtx = sio.mmread(gzip.open(mtx_fh, "rb")).tocsc()
            totals = np.asarray(mtx[:, keep_idx].sum(axis=0)).ravel()
            totals[totals == 0] = 1.0
            present_genes = [g for g in SCORE_GENES if g in features]
            gene_rows = [features[g] for g in present_genes]
            sub = mtx[gene_rows][:, keep_idx].toarray()  # n_genes x n_kept
            expr = np.log1p(sub * (1e4 / totals))
            kept_barcodes = [b for b, _ in pairs]
            for j, bc in enumerate(kept_barcodes):
                key = f"{sample}_{bc}"
                cell_matrix[key] = {g: float(expr[i, j]) for i, g in enumerate(present_genes)}
            print(f"  {sample}: {len(barcodes)} barcodes, annot={len(sample_annot)}, "
                  f"matched={len(keep_idx)}", flush=True)

    # Sanity: annotation counts vs matrix counts per sample.
    annot_counts = annot.groupby("sample").size()
    print("matrix barcodes per sample (raw/unfiltered):", dict(sorted(per_sample_counts.items())))
    print("annotated cells per sample:", annot_counts.to_dict())

    cells = pd.DataFrame.from_dict(cell_matrix, orient="index")
    cells.index.name = "cell_key"
    cells = cells.reset_index()
    cells["sample"] = cells["cell_key"].str.rsplit("_", n=1).str[0]
    cells["barcode"] = cells["cell_key"].str.rsplit("_", n=1).str[1]
    meta = annot.set_index(["sample", "barcode"])
    cells = cells.join(meta, on=["sample", "barcode"], how="inner")
    print(f"cells with expression + annotation: {len(cells)}")

    long = cells.melt(
        id_vars=["cell_key", "sample", "disease", "compartment", "nanostring_reference"],
        value_vars=SCORE_GENES, var_name="gene", value_name="log1p_cpm",
    )
    long["detected"] = long["log1p_cpm"].notna() & (long["log1p_cpm"] > 0)

    summary = (
        long.groupby(["gene", "compartment", "disease"])
        .agg(
            n_cells=("cell_key", "nunique"),
            pct_detected=("detected", "mean"),
            mean_log1p_cpm=("log1p_cpm", "mean"),
            median_log1p_cpm=("log1p_cpm", "median"),
        )
        .reset_index()
    )
    summary["pct_detected"] = (summary["pct_detected"] * 100).round(3)
    summary["mean_log1p_cpm"] = summary["mean_log1p_cpm"].round(5)
    summary["median_log1p_cpm"] = summary["median_log1p_cpm"].round(5)
    summary.to_csv(OUT_CELLTYPE, sep="\t", index=False)

    fine = (
        long.groupby(["gene", "nanostring_reference", "compartment", "disease"])
        .agg(
            n_cells=("cell_key", "nunique"),
            pct_detected=("detected", "mean"),
            mean_log1p_cpm=("log1p_cpm", "mean"),
        )
        .reset_index()
    )
    fine["pct_detected"] = (fine["pct_detected"] * 100).round(3)
    fine["mean_log1p_cpm"] = fine["mean_log1p_cpm"].round(5)
    fine.to_csv(OUT_FINE, sep="\t", index=False)

    print("wrote:", OUT_CELLTYPE, "rows", len(summary))
    print("wrote:", OUT_FINE, "rows", len(fine))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
