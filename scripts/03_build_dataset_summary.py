#!/usr/bin/env python3
"""Build results/dataset_summary.tsv from the feasibility table."""

from __future__ import annotations

import csv
import os


INPUT = "results/cohorts/cohort_feasibility.tsv"
OUTPUT = "results/dataset_summary.tsv"
QC_OUTPUT = "results/cohorts/cohort_qc_summary.tsv"


def main() -> int:
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(INPUT, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    fields = [
        "dataset_id",
        "sample_size",
        "disease_scope",
        "tissue_site",
        "platform",
        "endpoint_family",
        "intended_role",
        "usability_decision",
        "key_qc_status",
    ]

    with open(OUTPUT, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset_id": row["dataset_id"],
                    "sample_size": row["n_total"],
                    "disease_scope": row["disease_scope"],
                    "tissue_site": row["tissue_site"],
                    "platform": row["platform"],
                    "endpoint_family": row["endpoint_family"],
                    "intended_role": row["primary_role"],
                    "usability_decision": row["usability_decision"],
                    "key_qc_status": "pre_download_feasibility_complete",
                }
            )
    with open(QC_OUTPUT, "w", newline="") as handle:
        fields = [
            "dataset_id",
            "qc_stage",
            "metadata_status",
            "endpoint_status",
            "expression_status",
            "blocking_issue",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset_id": row["dataset_id"],
                    "qc_stage": "pre_download",
                    "metadata_status": "series_level_confirmed",
                    "endpoint_status": row["n_endpoint_available"],
                    "expression_status": row["platform"],
                    "blocking_issue": row["risk_note"],
                }
            )
    print(f"Wrote {OUTPUT} and {QC_OUTPUT} with {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
