#!/usr/bin/env python3
"""Build baseline-to-follow-up mucosal-healing endpoint table for GSE73661."""

from __future__ import annotations

import csv
import os
from collections import defaultdict


METADATA = "data/processed/GSE73661/metadata.tsv"
AXIS = "results/axis/barrier_axis_scores.tsv"
OUTPUT = "data/processed/GSE73661/baseline_endpoint.tsv"

WEEK_ORDER = {
    "W4_W6": 1,
    "W6": 2,
    "W12": 3,
    "W52": 4,
}


def response_from_title(title: str):
    normalized = title.replace("_", " ")
    if "UC R " in normalized:
        return 1
    if "UC NR " in normalized:
        return 0
    return None


def treatment_family(value: str) -> str:
    if value == "IFX":
        return "IFX"
    if value.startswith("vdz"):
        return "VDZ"
    return "OTHER"


def read_axis() -> dict[str, dict[str, str]]:
    with open(AXIS, newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return {row["sample_id"]: row for row in rows if row["dataset_id"] == "GSE73661"}


def main() -> int:
    with open(METADATA, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    axis = read_axis()
    by_patient: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_patient[row["study_individual_number"]].append(row)

    output_rows = []
    for patient, patient_rows in by_patient.items():
        baselines = [
            r
            for r in patient_rows
            if r["week_(w)"] == "W0"
            and r["induction_therapy_maintenance_therapy"] != "CO"
            and not r["induction_therapy_maintenance_therapy"].startswith("plac")
        ]
        if not baselines:
            continue
        baseline = baselines[0]
        followups = []
        for row in patient_rows:
            if row["week_(w)"] in {"W0", "CO"}:
                continue
            resp = response_from_title(row["title"])
            if resp is None:
                continue
            followups.append((WEEK_ORDER.get(row["week_(w)"], 99), row, resp))
        if not followups:
            continue
        _, followup, response = sorted(followups, key=lambda x: x[0])[0]
        baseline_axis = axis.get(baseline["sample_id"])
        if not baseline_axis:
            continue
        output_rows.append(
            {
                "patient_id": patient,
                "baseline_sample_id": baseline["sample_id"],
                "followup_sample_id": followup["sample_id"],
                "treatment_family": treatment_family(baseline["induction_therapy_maintenance_therapy"]),
                "baseline_treatment": baseline["induction_therapy_maintenance_therapy"],
                "followup_week": followup["week_(w)"],
                "baseline_mayo": baseline["mayo_endoscopic_subscore"],
                "followup_mayo": followup["mayo_endoscopic_subscore"],
                "mucosal_healing": str(response),
                "axis_score": baseline_axis["axis_score"],
                "upstream_score": baseline_axis["upstream_score"],
                "mmp_score": baseline_axis["mmp_score"],
                "junction_score": baseline_axis["junction_score"],
                "endpoint_rule": "earliest_followup_UC_R_or_UC_NR_title; R=1 NR=0; Mayo<=1 expected for R",
            }
        )

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", newline="") as handle:
        fields = [
            "patient_id",
            "baseline_sample_id",
            "followup_sample_id",
            "treatment_family",
            "baseline_treatment",
            "followup_week",
            "baseline_mayo",
            "followup_mayo",
            "mucosal_healing",
            "axis_score",
            "upstream_score",
            "mmp_score",
            "junction_score",
            "endpoint_rule",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)
    responders = sum(int(r["mucosal_healing"]) for r in output_rows)
    print(f"wrote={OUTPUT} n={len(output_rows)} responders={responders} nonresponders={len(output_rows)-responders}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
