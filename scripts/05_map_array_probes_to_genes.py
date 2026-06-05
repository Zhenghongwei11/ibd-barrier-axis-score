#!/usr/bin/env python3
"""Map GEO array probe expression to gene symbols using a GPL annotation file."""

from __future__ import annotations

import argparse
import csv
import gzip
import os
from collections import defaultdict


def open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return open(path, "rt", errors="replace")


def read_annotation(path: str, targets: set[str] | None = None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    in_table = False
    id_idx = None
    symbol_idx = None
    with open_text(path) as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line == "!platform_table_begin":
                in_table = True
                continue
            if line == "!platform_table_end":
                break
            if not in_table:
                continue
            parts = line.split("\t")
            if id_idx is None:
                id_idx = parts.index("ID")
                symbol_idx = parts.index("Gene symbol")
                continue
            symbol = parts[symbol_idx].strip().upper() if symbol_idx < len(parts) else ""
            if not symbol or "///" in symbol:
                continue
            if targets and symbol not in targets:
                continue
            mapping[parts[id_idx].strip()] = symbol
    return mapping


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def map_expression(expression: str, annotation: str, output: str, target_genes: set[str] | None) -> None:
    mapping = read_annotation(annotation, target_genes)
    gene_rows: dict[str, list[list[float]]] = defaultdict(list)
    with open(expression, newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        sample_ids = header[1:]
        for row in reader:
            probe_id = row[0]
            gene = mapping.get(probe_id)
            if not gene:
                continue
            try:
                values = [float(x) for x in row[1:]]
            except ValueError:
                continue
            gene_rows[gene].append(values)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["Gene Symbol", *sample_ids])
        for gene in sorted(gene_rows):
            columns = list(zip(*gene_rows[gene]))
            writer.writerow([gene, *[f"{mean(list(col)):.8g}" for col in columns]])
    print(f"mapped_genes={len(gene_rows)} probes_mapped={len(mapping)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expression", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-genes", default="")
    args = parser.parse_args()
    targets = {x.strip().upper() for x in args.target_genes.split(",") if x.strip()} or None
    map_expression(args.expression, args.annotation, args.output, targets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
