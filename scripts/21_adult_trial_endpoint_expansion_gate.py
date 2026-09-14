#!/usr/bin/env python3
"""Strict adult trial endpoint expansion gate.

This script downloads official GEO series matrices and small supplementary
metadata files, then checks whether sample-level adult endpoint labels can be
joined to expression sample identifiers. It does not fit models.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


RAW = Path("data/raw")
OUT_TABLE = Path("docs/ADULT_TRIAL_ENDPOINT_EXPANSION_LABEL_MAP.tsv")
OUT_LOG = Path("docs/ADULT_TRIAL_ENDPOINT_EXPANSION_FEASIBILITY.md")
OUT_DOWNLOADS = Path("data/references/adult_expansion_downloads.tsv")


@dataclass(frozen=True)
class Candidate:
    dataset_id: str
    priority: int
    text_url: str
    matrix_url: str
    endpoint_family: str
    expected: str
    role_hint: str


CANDIDATES = [
    Candidate(
        "GSE206285",
        1,
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE206285&targ=self&form=text&view=quick",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE206nnn/GSE206285/matrix/GSE206285_series_matrix.txt.gz",
        "UNIFI_response_or_remission",
        "568 total; 550 UC baseline; 18 healthy",
        "highest_priority_adult_trial_endpoint_candidate",
    ),
    Candidate(
        "GSE92415",
        2,
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92415&targ=self&form=text&view=quick",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92415/matrix/GSE92415_series_matrix.txt.gz",
        "golimumab_response_or_remission",
        "183 total; 162 UC; 21 healthy",
        "adult_trial_endpoint_candidate",
    ),
    Candidate(
        "GSE23597",
        3,
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE23597&targ=self&form=text&view=quick",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE23nnn/GSE23597/matrix/GSE23597_series_matrix.txt.gz",
        "ACT1_response_timecourse",
        "113 biopsies from 48 UC patients",
        "adult_trial_endpoint_candidate",
    ),
    Candidate(
        "GSE16879",
        4,
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE16879&targ=self&form=text&view=quick",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE16nnn/GSE16879/matrix/GSE16879_series_matrix.txt.gz",
        "infliximab_response",
        "133 series samples; 61 IBD patients plus 12 controls in GEO text",
        "adult_or_mixed_endpoint_candidate",
    ),
    Candidate(
        "GSE193677",
        5,
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193677&targ=self&form=text&view=quick",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE193nnn/GSE193677/matrix/GSE193677_series_matrix.txt.gz",
        "adult_activity_or_phenotype",
        "2490 adult IBD/control biopsy samples",
        "adult_supportive_activity_candidate",
    ),
]

ENDPOINT_RE = re.compile(
    r"respond|response|remission|healing|endoscop|histolog|mayo|activity|severity|inflam|disease|clinical",
    re.I,
)
DIRECT_RE = re.compile(r"respond|response|remission|healing|endoscop|histolog|mayo", re.I)
BASELINE_RE = re.compile(r"baseline|week 0|wk0|w0|pre[- ]?treatment|before", re.I)
TREATMENT_RE = re.compile(r"ustekinumab|golimumab|infliximab|placebo|anti-tnf|treatment|dose", re.I)
SMALL_SUPP_RE = re.compile(r"metadata|sample|clinical|phenotype|annotation|tsv|csv|txt|xlsx", re.I)


def safe_name(url: str) -> str:
    return os.path.basename(urlparse(url).path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> tuple[str, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == 0:
        dest.unlink()
    if not dest.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "adult-endpoint-expansion-gate"})
        tmp = dest.with_suffix(dest.suffix + ".part")
        if tmp.exists():
            tmp.unlink()
        with urllib.request.urlopen(req, timeout=90) as response, tmp.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        tmp.replace(dest)
    return str(dest), sha256(dest)


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return path.open("rt", errors="replace")


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "adult-endpoint-expansion-gate"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def series_values(text: str, field: str) -> list[str]:
    prefix = f"!Series_{field} = "
    return [line[len(prefix) :].strip() for line in text.splitlines() if line.startswith(prefix)]


def sample_count(text: str) -> int:
    return len([line for line in text.splitlines() if line.startswith("!Series_sample_id = ")])


def parse_matrix_metadata(path: Path) -> tuple[list[dict[str, str]], list[str], int]:
    sample_ids: list[str] = []
    rows: list[dict[str, str]] = []
    expression_rows = 0
    in_table = False

    with open_text(path) as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line == "!series_matrix_table_begin":
                in_table = True
                continue
            if line == "!series_matrix_table_end":
                in_table = False
                continue
            if in_table:
                expression_rows += 1
                continue
            if not line.startswith("!Sample_"):
                continue
            parts = [x.strip().strip('"') for x in line.split("\t")]
            key = parts[0].replace("!Sample_", "").lower()
            values = parts[1:]
            if key == "geo_accession":
                sample_ids = values
                rows = [{"sample_id": sample_id} for sample_id in sample_ids]
            if not rows or len(values) != len(rows):
                continue
            if key == "characteristics_ch1":
                for idx, value in enumerate(values):
                    parsed_key = "characteristics"
                    parsed_value = value
                    if ":" in value:
                        parsed_key, parsed_value = value.split(":", 1)
                    parsed_key = parsed_key.strip().lower().replace(" ", "_").replace("-", "_")
                    rows[idx][parsed_key] = parsed_value.strip()
            else:
                key = key.strip().lower().replace(" ", "_").replace("-", "_")
                for idx, value in enumerate(values):
                    rows[idx][key] = value
    return rows, sample_ids, max(expression_rows - 1, 0)


def supplementary_urls(text: str) -> list[str]:
    urls = []
    for url in series_values(text, "supplementary_file"):
        if url and url not in urls:
            urls.append(url)
    return urls


def should_download_supp(url: str) -> bool:
    name = safe_name(url)
    if name.endswith(".tar"):
        return False
    if re.search(r"counts|adjcounts|raw", name, re.I):
        return False
    return bool(SMALL_SUPP_RE.search(name))


def sniff_delimiter(sample: str) -> str:
    if "\t" in sample:
        return "\t"
    if "," in sample:
        return ","
    return "\t"


def inspect_supp_table(path: Path) -> dict[str, str]:
    if path.suffix.lower() not in {".gz", ".txt", ".tsv", ".csv"} and not str(path).endswith(".txt.gz") and not str(path).endswith(".tsv.gz"):
        return {"rows": "not_parsed", "columns": "", "endpoint_columns": "", "join_columns": ""}
    with open_text(path) as handle:
        sample = handle.read(4096)
    if not sample.strip():
        return {"rows": "0", "columns": "", "endpoint_columns": "", "join_columns": ""}
    delim = sniff_delimiter(sample)
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter=delim)
        if reader.fieldnames is None:
            return {"rows": "not_parsed", "columns": "", "endpoint_columns": "", "join_columns": ""}
        rows = 0
        endpoint_cols = []
        join_cols = []
        for col in reader.fieldnames:
            cl = col.lower()
            if ENDPOINT_RE.search(cl):
                endpoint_cols.append(col)
            if any(token in cl for token in ["gsm", "geo", "sample", "title", "subject", "patient", "participant", "id"]):
                join_cols.append(col)
        for _ in reader:
            rows += 1
    return {
        "rows": str(rows),
        "columns": ";".join(reader.fieldnames or []),
        "endpoint_columns": ";".join(endpoint_cols),
        "join_columns": ";".join(join_cols),
    }


def detect_columns(rows: list[dict[str, str]], pattern: re.Pattern[str]) -> list[str]:
    hits = []
    keys = sorted({key for row in rows for key in row})
    for key in keys:
        values = " ".join(row.get(key, "") for row in rows[: min(len(rows), 80)])
        if pattern.search(key) or pattern.search(values):
            hits.append(key)
    return hits


def nonempty_count(rows: list[dict[str, str]], columns: list[str]) -> int:
    if not columns:
        return 0
    return sum(1 for row in rows if any(str(row.get(col, "")).strip() for col in columns))


def classify(dataset_id: str, matrix_direct_cols: list[str], supp_direct_cols: list[str], supp_join_cols: list[str], matrix_endpoint_cols: list[str]) -> tuple[str, str]:
    if matrix_direct_cols:
        return "direct_endpoint_import", "Direct endpoint-like sample metadata are present in the series matrix."
    if supp_direct_cols and supp_join_cols:
        return "direct_endpoint_import", "Supplementary metadata contain direct endpoint-like columns and join-key candidates."
    if dataset_id == "GSE193677" and matrix_endpoint_cols:
        return "supportive_activity_only", "Series matrix metadata support adult tissue/activity or phenotype contrasts but not treatment response."
    if dataset_id == "GSE92415":
        return "exclude_untraceable", "GEO text describes week-6 golimumab response, but no non-RAW supplementary sample-level endpoint metadata were found."
    if dataset_id in {"GSE23597", "GSE16879"}:
        return "exclude_untraceable", "Endpoint context is visible, but direct response labels are not traceable to sample identifiers from downloaded metadata."
    return "exclude_untraceable", "No traceable direct endpoint labels were found in matrix or small supplementary metadata."


def make_row(candidate: Candidate, downloads: list[dict[str, str]]) -> dict[str, str]:
    text = fetch_text(candidate.text_url)
    local_matrix, matrix_sha = download(candidate.matrix_url, RAW / candidate.dataset_id / safe_name(candidate.matrix_url))
    matrix_rows, sample_ids, expression_rows = parse_matrix_metadata(Path(local_matrix))

    supp_paths = []
    supp_endpoint_cols: list[str] = []
    supp_join_cols: list[str] = []
    supp_parse_notes = []
    skipped_supp = []
    for url in supplementary_urls(text):
        name = safe_name(url)
        if should_download_supp(url):
            local, digest = download(url, RAW / candidate.dataset_id / name)
            downloads.append({"dataset_id": candidate.dataset_id, "url": url, "local_path": local, "sha256": digest, "status": "downloaded"})
            info = inspect_supp_table(Path(local))
            supp_paths.append(local)
            if info["endpoint_columns"]:
                supp_endpoint_cols.extend(info["endpoint_columns"].split(";"))
            if info["join_columns"]:
                supp_join_cols.extend(info["join_columns"].split(";"))
            supp_parse_notes.append(f"{name}: rows={info['rows']} endpoint_cols={info['endpoint_columns'] or 'none'} join_cols={info['join_columns'] or 'none'}")
        else:
            skipped_supp.append(name)

    downloads.append({"dataset_id": candidate.dataset_id, "url": candidate.matrix_url, "local_path": local_matrix, "sha256": matrix_sha, "status": "downloaded"})

    matrix_endpoint_cols = detect_columns(matrix_rows, ENDPOINT_RE)
    matrix_direct_cols = detect_columns(matrix_rows, DIRECT_RE)
    baseline_cols = detect_columns(matrix_rows, BASELINE_RE)
    treatment_cols = detect_columns(matrix_rows, TREATMENT_RE)
    status, blocker = classify(candidate.dataset_id, matrix_direct_cols, sorted(set(supp_endpoint_cols)), sorted(set(supp_join_cols)), matrix_endpoint_cols)
    body = " ".join(series_values(text, "title") + series_values(text, "summary") + series_values(text, "overall_design"))

    return {
        "dataset_id": candidate.dataset_id,
        "priority": str(candidate.priority),
        "role_hint": candidate.role_hint,
        "endpoint_family": candidate.endpoint_family,
        "expected_sample_count": candidate.expected,
        "geo_series_sample_count": str(sample_count(text)),
        "matrix_sample_count": str(len(sample_ids)),
        "matrix_expression_rows": str(expression_rows),
        "matrix_endpoint_columns": ";".join(matrix_endpoint_cols),
        "matrix_direct_endpoint_columns": ";".join(matrix_direct_cols),
        "matrix_direct_nonempty_n": str(nonempty_count(matrix_rows, matrix_direct_cols)),
        "baseline_columns": ";".join(baseline_cols),
        "treatment_columns": ";".join(treatment_cols),
        "supplementary_downloaded": ";".join(supp_paths) if supp_paths else "none",
        "supplementary_skipped": ";".join(skipped_supp) if skipped_supp else "none",
        "supplementary_endpoint_columns": ";".join(sorted(set(supp_endpoint_cols))),
        "supplementary_join_columns": ";".join(sorted(set(supp_join_cols))),
        "supplementary_parse_notes": " | ".join(supp_parse_notes) if supp_parse_notes else "none",
        "feasibility_status": status,
        "blocker_or_rationale": blocker,
        "source_matrix": candidate.matrix_url,
        "source_geo": candidate.text_url,
        "evidence_snippet": body[:550].replace("\t", " ").replace("\n", " "),
    }


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_downloads(rows: list[dict[str, str]]) -> None:
    OUT_DOWNLOADS.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset_id", "url", "local_path", "sha256", "status"]
    with OUT_DOWNLOADS.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_log(rows: list[dict[str, str]]) -> None:
    lines = [
        "# Adult Trial Endpoint Expansion Feasibility",
        "",
        "This gate inspected official GEO series matrices and small non-RAW supplementary metadata. It did not fit endpoint models.",
        "",
        "## Decisions",
        "",
    ]
    for row in rows:
        lines.append(f"### {row['dataset_id']}: {row['feasibility_status']}")
        lines.append("")
        lines.append(f"- Endpoint family: {row['endpoint_family']}")
        lines.append(f"- Matrix samples: {row['matrix_sample_count']}; GEO series samples: {row['geo_series_sample_count']}")
        lines.append(f"- Matrix direct endpoint columns: {row['matrix_direct_endpoint_columns'] or 'none'}")
        lines.append(f"- Supplementary endpoint columns: {row['supplementary_endpoint_columns'] or 'none'}")
        lines.append(f"- Rationale/blocker: {row['blocker_or_rationale']}")
        lines.append("")
    OUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows: list[dict[str, str]] = []
    downloads: list[dict[str, str]] = []
    for candidate in CANDIDATES:
        rows.append(make_row(candidate, downloads))
    write_tsv(OUT_TABLE, rows)
    write_downloads(downloads)
    write_log(rows)
    print(f"wrote={OUT_TABLE} rows={len(rows)}")
    print(f"wrote={OUT_LOG}")
    print(f"wrote={OUT_DOWNLOADS} downloads={len(downloads)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
