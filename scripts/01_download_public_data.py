#!/usr/bin/env python3
"""Download files listed in data/manifest.tsv without raw FASTQ processing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import urllib.request


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def should_download(row: dict[str, str], dataset: str | None, include_conditional: bool) -> bool:
    if dataset and row["dataset_id"] != dataset:
        return False
    if row["data_type"] == "landing_page":
        return False
    if row["required_for_default"] == "yes":
        return True
    return include_conditional and row["required_for_default"] == "conditional"


def download(row: dict[str, str], dry_run: bool) -> None:
    destination = row["destination"]
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if dry_run:
        print(f"DRY-RUN\t{row['dataset_id']}\t{row['file_id']}\t{row['url']}\t{destination}")
        return
    if not os.path.exists(destination):
        print(f"Downloading {row['dataset_id']}:{row['file_id']} -> {destination}")
        urllib.request.urlretrieve(row["url"], destination)
    else:
        print(f"Exists {destination}")
    checksum = sha256_file(destination)
    with open(destination + ".sha256", "w") as handle:
        handle.write(f"{checksum}  {os.path.basename(destination)}\n")
    print(f"SHA256\t{row['dataset_id']}\t{checksum}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest.tsv")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--include-conditional", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    selected = [row for row in rows if should_download(row, args.dataset, args.include_conditional)]
    if not selected:
        print("No files selected.", file=sys.stderr)
        return 1
    for row in selected:
        download(row, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
