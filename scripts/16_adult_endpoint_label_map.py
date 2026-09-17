#!/usr/bin/env python3
"""Inspect adult candidate GEO records for endpoint-label feasibility.

This is a label-map gate only. It reads GEO text records and does not download
or model expression matrices.
"""

from __future__ import annotations

import csv
import re
import urllib.request


CANDIDATES = {
    "GSE12251": {
        "text": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE12251&targ=self&form=text&view=quick",
        "matrix": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE12nnn/GSE12251/matrix/GSE12251_series_matrix.txt.gz",
        "endpoint_family": "endoscopic_histologic_healing",
        "expected": "22 UC baseline samples",
        "priority": "1",
    },
    "GSE92415": {
        "text": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92415&targ=self&form=text&view=quick",
        "matrix": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92415/matrix/GSE92415_series_matrix.txt.gz",
        "endpoint_family": "golimumab_response_or_remission",
        "expected": "183 total samples; 162 UC; 21 healthy",
        "priority": "2",
    },
    "GSE206285": {
        "text": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE206285&targ=self&form=text&view=quick",
        "matrix": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE206nnn/GSE206285/matrix/GSE206285_series_matrix.txt.gz",
        "endpoint_family": "UNIFI_response_or_remission",
        "expected": "568 total samples; 550 UC baseline; 18 healthy",
        "priority": "3",
    },
    "GSE23597": {
        "text": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE23597&targ=self&form=text&view=quick",
        "matrix": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE23nnn/GSE23597/matrix/GSE23597_series_matrix.txt.gz",
        "endpoint_family": "ACT1_response_timecourse",
        "expected": "113 biopsies from 48 UC patients",
        "priority": "4",
    },
    "GSE16879": {
        "text": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE16879&targ=self&form=text&view=quick",
        "matrix": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE16nnn/GSE16879/matrix/GSE16879_series_matrix.txt.gz",
        "endpoint_family": "infliximab_response",
        "expected": "61 IBD patients plus 12 controls",
        "priority": "5",
    },
    "GSE193677": {
        "text": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193677&targ=self&form=text&view=quick",
        "matrix": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE193nnn/GSE193677/matrix/GSE193677_series_matrix.txt.gz",
        "endpoint_family": "adult_activity_or_phenotype",
        "expected": "large adult IBD registry-scale biopsy cohort",
        "priority": "6",
    },
}

FIELD_PATTERNS = {
    "response": r"respond|response|non-respond|nonrespond",
    "remission": r"remission",
    "healing": r"healing|mucosal",
    "endoscopy_histology": r"endoscopic|histologic|histology|mayo",
    "activity": r"activity|severity",
    "treatment": r"infliximab|golimumab|ustekinumab|vedolizumab|anti-tnf|treatment|therapy|placebo",
    "baseline": r"before|baseline|pre-|pre |week 0|w0",
    "timepoint": r"week|timepoint|post",
}


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 endpoint-label-map"})
    with urllib.request.urlopen(req, timeout=45) as response:
        return response.read().decode("utf-8", errors="replace")


def series_values(text: str, field: str) -> list[str]:
    prefix = f"!Series_{field} = "
    return [line[len(prefix) :].strip() for line in text.splitlines() if line.startswith(prefix)]


def joined_series_text(text: str) -> str:
    fields = ["title", "summary", "overall_design", "type"]
    values: list[str] = []
    for field in fields:
        values.extend(series_values(text, field))
    return " ".join(values)


def detected_fields(body: str) -> str:
    body_l = body.lower()
    hits = [name for name, pattern in FIELD_PATTERNS.items() if re.search(pattern, body_l)]
    return ";".join(hits) if hits else "not_obvious_in_geo_record"


def sample_count(text: str) -> int:
    return len([line for line in text.splitlines() if line.startswith("!Series_sample_id = ")])


def supplementary_files(text: str) -> str:
    values = series_values(text, "supplementary_file")
    return ";".join(values) if values else "none_listed"


def decision(dataset_id: str, fields: str, body: str) -> tuple[str, str]:
    body_l = body.lower()
    if dataset_id == "GSE12251" and "endoscopic and histologic healing" in body_l:
        return "proceed_to_import", "Response definition is explicit in GEO overall design; sample-level parser must map patient responder IDs."
    if dataset_id in {"GSE92415", "GSE206285"} and any(x in fields for x in ["response", "remission", "treatment"]):
        return "inspect_supplement_first", "Trial response context is clear but sample-level response/remission labels need supplement or metadata mapping."
    if dataset_id == "GSE23597" and "week" in body_l and "infliximab" in body_l:
        return "inspect_supplement_first", "ACT1 time-course context is clear; baseline/timepoint and response labels need parser review."
    if dataset_id == "GSE16879" and any(x in fields for x in ["response", "endoscopy_histology"]):
        return "inspect_supplement_first", "Response context is visible; age scope and sample-level response mapping need confirmation."
    if dataset_id == "GSE193677" and "adult" in body_l:
        return "supportive_activity_candidate", "Adult registry-scale context is visible; use as activity/phenotype support unless direct endpoint labels are found."
    return "defer", "No sufficiently clean direct endpoint label map from GEO series text."


def main() -> int:
    rows = []
    for dataset_id, cfg in CANDIDATES.items():
        try:
            text = fetch_text(cfg["text"])
            body = joined_series_text(text)
            fields = detected_fields(body)
            usability, blocker = decision(dataset_id, fields, body)
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "priority": cfg["priority"],
                    "source_files": cfg["matrix"],
                    "label_fields": fields,
                    "endpoint_family": cfg["endpoint_family"],
                    "baseline_available": "yes_or_likely" if "baseline" in fields else "unclear",
                    "treatment_context": "detected" if "treatment" in fields else "requires_manual_review",
                    "tissue_site": "mucosal_biopsy_or_endoscopy_context" if "biopsy" in body.lower() or "mucosal" in body.lower() else "requires_manual_review",
                    "age_scope_evidence": "adult_or_trial_derived_from_prior_landscape; confirm if not explicit in series text",
                    "sample_count_expected": cfg["expected"],
                    "sample_count_series_record": str(sample_count(text)),
                    "usability_decision": usability,
                    "blocker": blocker,
                    "supplementary_files": supplementary_files(text),
                    "geo_text_record": cfg["text"],
                    "evidence_snippet": body[:600].replace("\t", " ").replace("\n", " "),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "priority": cfg["priority"],
                    "source_files": cfg["matrix"],
                    "label_fields": "fetch_or_parse_failed",
                    "endpoint_family": cfg["endpoint_family"],
                    "baseline_available": "unknown",
                    "treatment_context": "unknown",
                    "tissue_site": "unknown",
                    "age_scope_evidence": "unknown",
                    "sample_count_expected": cfg["expected"],
                    "sample_count_series_record": "unknown",
                    "usability_decision": "defer",
                    "blocker": f"{type(exc).__name__}: {exc}",
                    "supplementary_files": "unknown",
                    "geo_text_record": cfg["text"],
                    "evidence_snippet": "",
                }
            )

    output = "docs/ADULT_ENDPOINT_LABEL_MAP.tsv"
    fields = [
        "dataset_id",
        "priority",
        "source_files",
        "label_fields",
        "endpoint_family",
        "baseline_available",
        "treatment_context",
        "tissue_site",
        "age_scope_evidence",
        "sample_count_expected",
        "sample_count_series_record",
        "usability_decision",
        "blocker",
        "supplementary_files",
        "geo_text_record",
        "evidence_snippet",
    ]
    with open(output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={output} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
