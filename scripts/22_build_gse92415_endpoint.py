#!/usr/bin/env python3
"""Build GSE92415 baseline week-6 response endpoint table."""

from __future__ import annotations

import csv
import os


METADATA = "data/processed/GSE92415/metadata.tsv"
AXIS = "results/axis/barrier_axis_scores.tsv"
OUTPUT = "data/processed/GSE92415/baseline_endpoint.tsv"
ALL_OUTPUT = "data/processed/GSE92415/endpoint_all_uc.tsv"


def read_axis() -> dict[str, dict[str, str]]:
    with open(AXIS, newline="") as handle:
        return {
            row["sample_id"]: row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["dataset_id"] == "GSE92415" and row["qc_flags"] == "ok"
        }


def response(value: str) -> str:
    value = value.strip().lower()
    if value == "yes":
        return "1"
    if value == "no":
        return "0"
    return ""


def build_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    axis = read_axis()
    all_rows: list[dict[str, str]] = []
    baseline_rows: list[dict[str, str]] = []
    with open(METADATA, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if "Ulcerative Colitis" not in row.get("disease", ""):
                continue
            axis_row = axis.get(row["sample_id"])
            if not axis_row:
                continue
            out = {
                "subject_id": row.get("subject", ""),
                "sample_id": row["sample_id"],
                "title": row.get("title", ""),
                "age": row.get("age", ""),
                "tissue_site": row.get("tissue", "colon mucosa"),
                "treatment": row.get("treatment", ""),
                "visit": row.get("visit", ""),
                "baseline_mayo_score": row.get("mayo_score", ""),
                "endpoint_name": "week6_clinical_response",
                "week6_clinical_response": response(row.get("wk6response", "")),
                "axis_score": axis_row["axis_score"],
                "upstream_score": axis_row["upstream_score"],
                "mmp_score": axis_row["mmp_score"],
                "junction_score": axis_row["junction_score"],
                "endpoint_rule": "WK6response Yes=1 No=0 from GEO sample metadata; primary table uses Week 0 UC baseline biopsies only",
            }
            all_rows.append(out)
            if out["visit"] == "Week 0" and out["week6_clinical_response"] in {"0", "1"}:
                baseline_rows.append(out)
    return all_rows, baseline_rows


def write_rows(rows: list[dict[str, str]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "subject_id",
        "sample_id",
        "title",
        "age",
        "tissue_site",
        "treatment",
        "visit",
        "baseline_mayo_score",
        "endpoint_name",
        "week6_clinical_response",
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
    all_rows, baseline_rows = build_rows()
    write_rows(all_rows, ALL_OUTPUT)
    write_rows(baseline_rows, OUTPUT)
    responders = sum(int(r["week6_clinical_response"]) for r in baseline_rows)
    treatments = sorted({r["treatment"] for r in baseline_rows})
    print(f"wrote={OUTPUT} n={len(baseline_rows)} responders={responders} nonresponders={len(baseline_rows)-responders} treatments={','.join(treatments)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
