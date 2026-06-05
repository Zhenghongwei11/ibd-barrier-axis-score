#!/usr/bin/env python3
"""Build endpoint tables for adult expansion cohorts."""

from __future__ import annotations

import csv
import os


AXIS = "results/axis/barrier_axis_scores.tsv"


def read_axis(dataset_id: str) -> dict[str, dict[str, str]]:
    with open(AXIS, newline="") as handle:
        return {
            row["sample_id"]: row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["dataset_id"] == dataset_id and row["qc_flags"] == "ok"
        }


def yn(value: str) -> str:
    value = value.strip().lower()
    if value in {"y", "yes"}:
        return "1"
    if value in {"n", "no"}:
        return "0"
    return ""


def write_rows(rows: list[dict[str, str]], path: str, fields: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def axis_fields(out: dict[str, str], axis_row: dict[str, str]) -> None:
    out["axis_score"] = axis_row["axis_score"]
    out["upstream_score"] = axis_row["upstream_score"]
    out["mmp_score"] = axis_row["mmp_score"]
    out["junction_score"] = axis_row["junction_score"]


def build_gse206285() -> None:
    dataset = "GSE206285"
    axis = read_axis(dataset)
    rows = []
    with open(f"data/processed/{dataset}/metadata.tsv", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("diagnosis") != "ulcerative colitis":
                continue
            if row.get("visit") != "WEEK I-0":
                continue
            axis_row = axis.get(row["sample_id"])
            if not axis_row:
                continue
            out = {
                "donor_id": row.get("donor_id", ""),
                "sample_id": row["sample_id"],
                "title": row.get("title", ""),
                "tissue_site": row.get("tissue", "colon"),
                "visit": row.get("visit", ""),
                "treatment": row.get("treatment", ""),
                "endpoint_name": "week8_mucosal_healing",
                "week8_mucosal_healing": yn(row.get("mucosal_healing_at_week_8", "")),
                "week8_clinical_remission": yn(row.get("clinical_remission_at_week_8", "")),
                "endpoint_rule": "Week-8 mucosal healing and clinical remission from GEO sample metadata; baseline UC biopsies only",
            }
            axis_fields(out, axis_row)
            rows.append(out)
    fields = [
        "donor_id",
        "sample_id",
        "title",
        "tissue_site",
        "visit",
        "treatment",
        "endpoint_name",
        "week8_mucosal_healing",
        "week8_clinical_remission",
        "axis_score",
        "upstream_score",
        "mmp_score",
        "junction_score",
        "endpoint_rule",
    ]
    write_rows(rows, f"data/processed/{dataset}/baseline_endpoint.tsv", fields)
    mh = [r for r in rows if r["week8_mucosal_healing"] in {"0", "1"}]
    cr = [r for r in rows if r["week8_clinical_remission"] in {"0", "1"}]
    print(f"{dataset} n={len(rows)} mucosal_healing_n={len(mh)} remission_n={len(cr)}")


def build_gse23597() -> None:
    dataset = "GSE23597"
    axis = read_axis(dataset)
    rows = []
    with open(f"data/processed/{dataset}/metadata.tsv", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("time") != "W0":
                continue
            axis_row = axis.get(row["sample_id"])
            if not axis_row:
                continue
            out = {
                "subject_id": row.get("subject", ""),
                "sample_id": row["sample_id"],
                "title": row.get("title", ""),
                "dose": row.get("dose", ""),
                "time": row.get("time", ""),
                "endpoint_name": "week8_response",
                "week8_response": yn(row.get("wk8_response", "")),
                "week30_response": yn(row.get("wk30_response", "")),
                "endpoint_rule": "WK8/WK30 response from GEO sample metadata; W0 baseline biopsies only",
            }
            axis_fields(out, axis_row)
            rows.append(out)
    fields = [
        "subject_id",
        "sample_id",
        "title",
        "dose",
        "time",
        "endpoint_name",
        "week8_response",
        "week30_response",
        "axis_score",
        "upstream_score",
        "mmp_score",
        "junction_score",
        "endpoint_rule",
    ]
    write_rows(rows, f"data/processed/{dataset}/baseline_endpoint.tsv", fields)
    print(f"{dataset} n={len(rows)} wk8_responders={sum(int(r['week8_response']) for r in rows if r['week8_response'])}")


def build_gse16879() -> None:
    dataset = "GSE16879"
    axis = read_axis(dataset)
    rows = []
    with open(f"data/processed/{dataset}/metadata.tsv", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("before_or_after_first_infliximab_treatment") != "Before first infliximab treatment":
                continue
            if row.get("response_to_infliximab") not in {"Yes", "No"}:
                continue
            axis_row = axis.get(row["sample_id"])
            if not axis_row:
                continue
            out = {
                "sample_id": row["sample_id"],
                "title": row.get("title", ""),
                "tissue_site": row.get("tissue", ""),
                "disease": row.get("disease", ""),
                "timepoint": row.get("before_or_after_first_infliximab_treatment", ""),
                "endpoint_name": "infliximab_response",
                "infliximab_response": yn(row.get("response_to_infliximab", "")),
                "endpoint_rule": "Response to first infliximab from GEO sample metadata; pretreatment IBD biopsies only",
            }
            axis_fields(out, axis_row)
            rows.append(out)
    fields = [
        "sample_id",
        "title",
        "tissue_site",
        "disease",
        "timepoint",
        "endpoint_name",
        "infliximab_response",
        "axis_score",
        "upstream_score",
        "mmp_score",
        "junction_score",
        "endpoint_rule",
    ]
    write_rows(rows, f"data/processed/{dataset}/baseline_endpoint.tsv", fields)
    print(f"{dataset} n={len(rows)} responders={sum(int(r['infliximab_response']) for r in rows if r['infliximab_response'])}")


def main() -> int:
    build_gse206285()
    build_gse23597()
    build_gse16879()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
