#!/usr/bin/env python3
"""Extract sample metadata and expression values from a GEO series matrix file."""

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


def clean(value: str) -> str:
    return value.strip().strip('"')


def parse_matrix(path: str):
    sample_ids: list[str] = []
    titles: list[str] = []
    sample_metadata: list[dict[str, str]] = []
    expression_rows: list[list[str]] = []
    in_table = False
    header: list[str] | None = None

    with open_text(path) as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [clean(x) for x in line.split("\t")[1:]]
                sample_metadata = [{"sample_id": sample_id} for sample_id in sample_ids]
            elif line.startswith("!Sample_title"):
                titles = [clean(x) for x in line.split("\t")[1:]]
                if sample_metadata and len(titles) == len(sample_metadata):
                    for idx, title in enumerate(titles):
                        sample_metadata[idx]["title"] = title
            elif line.startswith("!Sample_characteristics_ch1"):
                values = [clean(x) for x in line.split("\t")[1:]]
                for idx, value in enumerate(values):
                    if idx >= len(sample_metadata) or ":" not in value:
                        continue
                    key, parsed = value.split(":", 1)
                    key = key.strip().lower().replace(" ", "_")
                    sample_metadata[idx][key] = parsed.strip()
            elif line == "!series_matrix_table_begin":
                in_table = True
            elif line == "!series_matrix_table_end":
                in_table = False
            elif in_table:
                parts = [clean(x) for x in line.split("\t")]
                if header is None:
                    header = parts
                else:
                    expression_rows.append(parts)
    if titles and sample_metadata and "title" not in sample_metadata[0] and len(titles) == len(sample_metadata):
        for idx, title in enumerate(titles):
            sample_metadata[idx]["title"] = title
    return sample_ids, sample_metadata, header, expression_rows


def write_metadata(sample_ids: list[str], sample_metadata: list[dict[str, str]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = ["title", "sample_id"]
    for row in sample_metadata:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(keys)
        for idx in range(len(sample_ids)):
            row = sample_metadata[idx] if idx < len(sample_metadata) else {"sample_id": sample_ids[idx]}
            writer.writerow([row.get(k, "") for k in keys])


def write_expression(header: list[str] | None, rows: list[list[str]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        if header:
            writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--metadata-out", required=True)
    parser.add_argument("--expression-out", required=True)
    args = parser.parse_args()

    sample_ids, metadata, header, expression_rows = parse_matrix(args.input)
    write_metadata(sample_ids, metadata, args.metadata_out)
    write_expression(header, expression_rows, args.expression_out)
    print(f"samples={len(sample_ids)} expression_rows={len(expression_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
