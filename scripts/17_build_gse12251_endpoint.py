#!/usr/bin/env python3
"""Build GSE12251 baseline endpoint table for week-8 endoscopic/histologic healing."""

from __future__ import annotations

import csv
import os
import re


METADATA = "data/processed/GSE12251/metadata.tsv"
AXIS = "results/axis/barrier_axis_scores.tsv"
OUTPUT = "data/processed/GSE12251/baseline_endpoint.tsv"
ALL_OUTPUT = "data/processed/GSE12251/baseline_endpoint_all_arrays.tsv"


def read_axis() -> dict[str, dict[str, str]]:
    with open(AXIS, newline="") as handle:
        return {
            row["sample_id"]: row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["dataset_id"] == "GSE12251" and row["qc_flags"] == "ok"
        }


def patient_id(title: str) -> str:
    match = re.match(r"(P\d+)", title)
    return match.group(1) if match else title.split("/")[0]


def dose(title: str) -> str:
    parts = title.split("/")
    return parts[1] if len(parts) > 1 else ""


def response(value: str) -> str:
    value = value.strip().lower()
    if value == "yes":
        return "1"
    if value == "no":
        return "0"
    return ""


def build_rows() -> list[dict[str, str]]:
    axis = read_axis()
    with open(METADATA, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    output_rows = []
    for row in rows:
        sample_id = row["sample_id"]
        axis_row = axis.get(sample_id)
        if not axis_row:
            continue
        out = {
            "patient_id": patient_id(row["title"]),
            "sample_id": sample_id,
            "title": row["title"],
            "dose": dose(row["title"]),
            "timepoint": "W0",
            "tissue_site": "colonic biopsy",
            "treatment_context": "pre-infliximab ACT1 biopsy",
            "endpoint_name": "week8_endoscopic_histologic_healing",
            "wk8_endoscopic_histologic_healing": response(row.get("wk8rsphm", "")),
            "axis_score": axis_row["axis_score"],
            "upstream_score": axis_row["upstream_score"],
            "mmp_score": axis_row["mmp_score"],
            "junction_score": axis_row["junction_score"],
            "endpoint_rule": "WK8RSPHM Yes=1 No=0; response defined by GEO as endoscopic and histologic healing at week 8",
        }
        output_rows.append(out)
    return output_rows


def write_rows(rows: list[dict[str, str]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "patient_id",
        "sample_id",
        "title",
        "dose",
        "timepoint",
        "tissue_site",
        "treatment_context",
        "endpoint_name",
        "wk8_endoscopic_histologic_healing",
        "axis_score",
        "upstream_score",
        "mmp_score",
        "junction_score",
        "endpoint_rule",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    all_rows = build_rows()
    write_rows(all_rows, ALL_OUTPUT)
    unique_rows = []
    seen: set[str] = set()
    dropped = 0
    for row in all_rows:
        if row["patient_id"] in seen:
            dropped += 1
            continue
        seen.add(row["patient_id"])
        unique_rows.append(row)
    write_rows(unique_rows, OUTPUT)
    responders = sum(int(r["wk8_endoscopic_histologic_healing"]) for r in unique_rows)
    print(f"wrote={OUTPUT} n={len(unique_rows)} responders={responders} nonresponders={len(unique_rows)-responders} dropped_duplicate_arrays={dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
