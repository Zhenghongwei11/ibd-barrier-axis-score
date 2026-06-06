# IBD Barrier-Axis Score

This repository contains the public reproducibility package for an age-stratified retrospective clinical assessment of an epithelial barrier-axis score in inflammatory bowel disease mucosal transcriptomes.

The project evaluates a prespecified IPMK-HDAC3-MMP-inspired epithelial barrier-injury score across adult direct clinical endpoint cohorts and pediatric or early-onset supportive cohorts. The public package is intended for computational reproducibility only. It does not include manuscript drafts, cover letters, journal submission files, private data, credentials, or internal planning material.

## Repository Contents

- `scripts/`: analysis and figure-generation scripts.
- `data/processed/`: compact processed source tables needed to rerun endpoint models and figures.
- `docs/DATA_MANIFEST.tsv`: public GEO data source manifest for rebuilding from public data.
- `docs/RUN_MANIFEST.tsv`: command provenance for the analysis chain.
- `docs/FIGURE_PROVENANCE.tsv`: figure-to-script-to-source-data mapping.
- `results/`: derived tables supporting the manuscript analyses.
- `results/figures/source_data/`: compact source-data tables for each figure.
- `plots/publication/submission_grade/`: final figure exports in PNG, PDF, and SVG.

## Quick Reproduction

Create a Python environment and regenerate the endpoint models, sensitivity analyses, source-data tables, and figures from the included compact processed source tables:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/reproduce_one_click.sh
```

Expected runtime on a laptop is less than 5 minutes for the table refresh and figure regeneration path. The default path does not overwrite `results/axis/barrier_axis_scores.tsv`, because the included axis-score table contains compact scores for all cohorts including datasets whose source expression is downloaded only in the optional public-GEO rebuild path.

## Rebuild From Public GEO Sources

The full public-data rebuild starts from `docs/DATA_MANIFEST.tsv` and downloads GEO series matrices, processed supplementary expression files, and platform annotations. This route is network-dependent and larger than the default quick reproduction path.

```bash
bash scripts/reproduce_one_click.sh --download-public-data
```

The default repository includes compact processed source tables so reviewers can rerun the statistical models and regenerate figures without downloading large GEO files.

## Main Analysis Boundary

The analysis supports retrospective molecular stratification of mucosal biopsy transcriptomes. Clinical use, treatment-selection use, dietary or supplementation efficacy, and causal mechanism require dedicated prospective studies.

## Key Outputs

- Primary endpoint models: `results/clinical/age_stratified_endpoint_models.tsv`
- Clinical tertile summaries: `results/clinical/clinical_score_strata_summary.tsv`
- Inflammatory-response comparator models: `results/clinical/inflammation_specificity_models.tsv`
- Figure provenance: `docs/FIGURE_PROVENANCE.tsv`
- Final figures: `plots/publication/submission_grade/`

## Citation

Please cite the Zenodo version DOI for the exact release used. Citation metadata are provided in `CITATION.cff`.
