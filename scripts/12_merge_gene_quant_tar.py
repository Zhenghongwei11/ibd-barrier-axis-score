#!/usr/bin/env python3
"""Merge GEO per-sample gene quantification files stored inside a RAW tar."""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import tarfile
from collections import defaultdict


AXIS_AND_COMPARATOR_GENES = {
    "IPMK",
    "IPPK",
    "HDAC3",
    "NCOR1",
    "NCOR2",
    "MMP1",
    "MMP3",
    "MMP10",
    "MMP12",
    "MMP13",
    "TJP1",
    "OCLN",
    "CLDN2",
    "OSM",
    "TREM1",
    "TNF",
    "IL13RA2",
    "TNFRSF1B",
}


def sample_id_from_name(name: str) -> str:
    base = os.path.basename(name)
    if base.endswith(".txt.gz"):
        base = base[: -len(".txt.gz")]
    elif base.endswith(".gz"):
        base = base[: -len(".gz")]
    return base.split("_", 1)[0]


def parse_member(handle, target_genes: set[str]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    with gzip.open(handle, "rt", errors="replace") as text:
        reader = csv.reader(text, delimiter="\t")
        header = next(reader, None)
        if not header or len(header) < 2:
            return {}
        for row in reader:
            if len(row) < 2:
                continue
            gene = row[0].strip().upper()
            if target_genes and gene not in target_genes:
                continue
            try:
                value = float(row[1])
            except ValueError:
                continue
            values[gene].append(value)
    return {gene: sum(v) / len(v) for gene, v in values.items() if v}


def merge_tar(tar_path: str, target_genes: set[str]) -> tuple[list[str], dict[str, dict[str, float]]]:
    sample_values: dict[str, dict[str, float]] = {}
    sample_order: list[str] = []
    with tarfile.open(tar_path) as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".txt.gz"):
                continue
            sample_id = sample_id_from_name(member.name)
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            values = parse_member(extracted, target_genes)
            sample_order.append(sample_id)
            sample_values[sample_id] = values
    return sample_order, sample_values


def write_matrix(sample_order: list[str], sample_values: dict[str, dict[str, float]], output: str) -> None:
    os.makedirs(os.path.dirname(output), exist_ok=True)
    genes = sorted({gene for values in sample_values.values() for gene in values})
    with open(output, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["Gene Symbol", *sample_order])
        for gene in genes:
            writer.writerow([gene, *[f"{sample_values[sample].get(gene, float('nan')):.8g}" for sample in sample_order]])


def write_sample_map(sample_order: list[str], tar_path: str, output: str) -> None:
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with tarfile.open(tar_path) as archive, open(output, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample_id", "tar_member", "source_label"])
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".txt.gz"):
                continue
            sample_id = sample_id_from_name(member.name)
            base = os.path.basename(member.name)
            label = base[: -len(".txt.gz")] if base.endswith(".txt.gz") else base
            writer.writerow([sample_id, member.name, label])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--tar", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-map-out", required=True)
    parser.add_argument("--all-genes", action="store_true", help="Export every gene instead of the axis/comparator gene subset.")
    args = parser.parse_args()

    target_genes = set() if args.all_genes else AXIS_AND_COMPARATOR_GENES
    sample_order, sample_values = merge_tar(args.tar, target_genes)
    write_matrix(sample_order, sample_values, args.output)
    write_sample_map(sample_order, args.tar, args.sample_map_out)
    available_genes = len({gene for values in sample_values.values() for gene in values})
    print(f"dataset={args.dataset_id} samples={len(sample_order)} genes_exported={available_genes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
