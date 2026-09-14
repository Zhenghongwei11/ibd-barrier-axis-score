#!/usr/bin/env python3
"""Compute the prespecified IPMK-HDAC3-MMP barrier-axis score."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
from collections import defaultdict


MODULES = {
    "upstream": {
        "IPMK": -1,
        "IPPK": -1,
        "HDAC3": -1,
        "NCOR1": -1,
        "NCOR2": -1,
    },
    "mmp": {
        "MMP1": 1,
        "MMP3": 1,
        "MMP10": 1,
        "MMP12": 1,
        "MMP13": 1,
    },
    "junction": {
        "TJP1": -1,
        "OCLN": -1,
        "CLDN2": 1,
    },
}

ALL_GENES = {gene for module in MODULES.values() for gene in module}


def open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return open(path, "rt", errors="replace")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def zscores(values: list[float]) -> list[float]:
    mu = mean(values)
    variance = sum((x - mu) ** 2 for x in values) / max(len(values) - 1, 1)
    sd = math.sqrt(variance)
    if sd == 0 or math.isnan(sd):
        return [0.0 for _ in values]
    return [(x - mu) / sd for x in values]


def read_expression(path: str, gene_col: str, log2_transform: bool):
    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        gene_idx = header.index(gene_col)
        sample_indices = [i for i, name in enumerate(header) if i != gene_idx and name not in {"Gene ID", "ID_REF"}]
        sample_ids = [header[i] for i in sample_indices]
        gene_values: dict[str, list[list[float]]] = defaultdict(list)

        for row in reader:
            if len(row) <= max(sample_indices):
                continue
            gene = row[gene_idx].strip().upper()
            if gene not in ALL_GENES:
                continue
            values = []
            for i in sample_indices:
                try:
                    value = float(row[i])
                except ValueError:
                    value = float("nan")
                if log2_transform and not math.isnan(value):
                    value = math.log2(value + 1.0)
                values.append(value)
            gene_values[gene].append(values)

    aggregated = {}
    for gene, rows in gene_values.items():
        cols = list(zip(*rows))
        aggregated[gene] = [mean([x for x in col if not math.isnan(x)]) for col in cols]
    return sample_ids, aggregated


def component_scores(aggregated: dict[str, list[float]], sample_count: int):
    signed_z: dict[str, list[float]] = {}
    for module in MODULES.values():
        for gene, direction in module.items():
            if gene in aggregated:
                signed_z[gene] = [direction * z for z in zscores(aggregated[gene])]

    rows = []
    for idx in range(sample_count):
        module_values = {}
        valid_modules = 0
        for module_name, genes in MODULES.items():
            values = [signed_z[g][idx] for g in genes if g in signed_z]
            required = math.ceil(len(genes) * 0.5)
            if len(values) >= required:
                module_values[module_name] = mean(values)
                valid_modules += 1
            else:
                module_values[module_name] = float("nan")
        all_values = [v for v in module_values.values() if not math.isnan(v)]
        axis_score = mean(all_values) if valid_modules >= 2 and "mmp" in module_values and not math.isnan(module_values["mmp"]) else float("nan")
        rows.append((axis_score, module_values))
    return rows


def write_scores(dataset_id: str, sample_ids: list[str], rows, output: str, append: bool) -> None:
    os.makedirs(os.path.dirname(output), exist_ok=True)
    write_header = not append or not os.path.exists(output)
    mode = "a" if append else "w"
    with open(output, mode, newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        if write_header:
            writer.writerow(["sample_id", "dataset_id", "axis_score", "upstream_score", "mmp_score", "junction_score", "scoring_method", "qc_flags"])
        for sample_id, (axis_score, modules) in zip(sample_ids, rows):
            qc = "ok" if not math.isnan(axis_score) else "insufficient_axis_genes"
            writer.writerow(
                [
                    sample_id,
                    dataset_id,
                    f"{axis_score:.6g}" if not math.isnan(axis_score) else "NA",
                    f"{modules['upstream']:.6g}" if not math.isnan(modules["upstream"]) else "NA",
                    f"{modules['mmp']:.6g}" if not math.isnan(modules["mmp"]) else "NA",
                    f"{modules['junction']:.6g}" if not math.isnan(modules["junction"]) else "NA",
                    "signed_within_dataset_zscore_mean",
                    qc,
                ]
            )


def write_availability(dataset_id: str, aggregated: dict[str, list[float]], output: str, append: bool) -> None:
    os.makedirs(os.path.dirname(output), exist_ok=True)
    write_header = not append or not os.path.exists(output)
    mode = "a" if append else "w"
    with open(output, mode, newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        if write_header:
            writer.writerow(["dataset_id", "module", "gene", "available"])
        for module_name, genes in MODULES.items():
            for gene in genes:
                writer.writerow([dataset_id, module_name, gene, "yes" if gene in aggregated else "no"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--gene-col", required=True)
    parser.add_argument("--output", default="results/axis/barrier_axis_scores.tsv")
    parser.add_argument("--availability-output", default="results/axis/gene_availability.tsv")
    parser.add_argument("--log2", action="store_true")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    sample_ids, aggregated = read_expression(args.input, args.gene_col, args.log2)
    rows = component_scores(aggregated, len(sample_ids))
    write_scores(args.dataset_id, sample_ids, rows, args.output, args.append)
    write_availability(args.dataset_id, aggregated, args.availability_output, args.append)
    print(f"dataset={args.dataset_id} samples={len(sample_ids)} genes_available={len(aggregated)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
