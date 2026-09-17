# Data Manifest

## Policy

All external inputs are listed in `data/manifest.tsv`. The default pipeline uses processed GEO matrices, processed expression/count files, or curated supplementary metadata. The project does not require raw FASTQ alignment.

## Current Modeled Dataset Roles

- `GSE73661`: adult trial-derived direct endpoint anchor with endoscopic mucosal healing labels after vedolizumab or infliximab.
- `GSE109142`: pediatric UC week-4 remission stratum.
- `GSE57945`: pediatric ileal mucosal injury and phenotype support.
- `GSE117993`: pediatric rectal IBD phenotype support.
- `GSE101794`: pediatric ileal Crohn phenotype support.
- `IBDMDB_HMP2`: deferred longitudinal extension after confirming usable host transcriptome and activity metadata intersections.

## Adult Endpoint Candidates

The current change reframes the project as age-stratified clinical biomarker validation. The following adult candidates are registered as conditional processed-data sources. They must pass endpoint-label mapping before download/modeling:

- `GSE12251`: adult ACT1 infliximab UC response candidate with week-8 endoscopic/histologic healing context.
- `GPL570`: Affymetrix HG-U133 Plus 2.0 annotation required for GSE12251 probe-to-gene mapping.
- `GSE92415`: adult PURSUIT-SC golimumab UC response/remission candidate.
- `GSE206285`: adult UNIFI ustekinumab UC response/remission candidate.
- `GSE23597`: adult ACT1 infliximab/placebo UC biopsy time-course candidate.
- `GSE16879`: infliximab response IBD mucosal biopsy candidate requiring age-scope confirmation.
- `GSE193677`: adult IBD activity/phenotype support candidate.

## Checksum Status

Checksums are marked in sidecar files written by `scripts/01_download_public_data.py` under `data/raw/<dataset_id>/`. Previously downloaded or confirmed resources include GSE73661, GSE57945, GSE109142, GSE117993, and GSE101794.

## Raw Sequencing Boundary

The project may download GEO `RAW.tar` archives only when they contain processed quantification files or microarray files needed for normalization. Raw FASTQ alignment is not part of the default workflow.

## Access Notes

GEO and IBDMDB are public resources. If a dataset requires controlled access or manual download in later stages, the file must be marked `manual` in `data/manifest.tsv`, with exact access steps documented here before analysis.
