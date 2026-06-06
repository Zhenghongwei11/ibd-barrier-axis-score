#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--download-public-data" ]]; then
  python3 scripts/01_download_public_data.py --manifest docs/DATA_MANIFEST.tsv --include-conditional
fi

python3 scripts/06_build_gse73661_endpoint.py
python3 scripts/07_fit_endpoint_models.py
python3 scripts/08_fit_comparator_models.py
python3 scripts/09_external_validation_gse57945.py
python3 scripts/10_sensitivity_analyses.py
python3 scripts/13_fit_gse109142_endpoint.py
python3 scripts/14_external_validation_additional_risk.py
python3 scripts/17_build_gse12251_endpoint.py
python3 scripts/18_fit_age_stratified_endpoint_models.py
python3 scripts/22_build_gse92415_endpoint.py
python3 scripts/23_build_adult_expansion_endpoints.py
python3 scripts/24_build_clinical_score_strata_table.py
python3 scripts/25_inflammation_specificity_models.py
python3 scripts/31_module_contribution_analysis.py
python3 scripts/32_predictive_performance.py
python3 scripts/33_adult_healing_random_effects.py
python3 scripts/26_export_figure_source_data.py
python3 scripts/24_make_submission_grade_figures.py

echo "Reproduction complete."
