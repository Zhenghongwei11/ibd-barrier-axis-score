# Harmonization Notes

## Current Status

This document records harmonization decisions after the expanded multicohort import pass.

## Dataset-Level Decisions

### GSE73661

- Treat baseline `W0` samples as the primary predictor timepoint.
- Treat endoscopic mucosal healing as the primary endpoint where response labels can be mapped to Mayo endoscopic subscore 0 or 1.
- Do not mix baseline and follow-up samples in the same independent-sample model.
- Account for repeated patient identifiers in any longitudinal secondary model.

### GSE109142

- Treat pretreatment rectal RNA-seq as the discovery candidate.
- Use `week_4_remission` Yes/No as the direct clinical endpoint in UC patients only.
- Keep controls for phenotype/context summaries, but exclude them from remission models.
- Adjusted models may include baseline total Mayo score and initial treatment; do not overstate the directionally consistent but nonsignificant adjusted result.
- Pediatric UC scope must be explicit in claims.

### GSE12251

- Treat all samples as baseline `W0` colonic mucosal biopsies collected before infliximab treatment.
- Use `WK8RSPHM` Yes/No as the adult direct endpoint, defined by GEO as endoscopic and histologic healing at week 8.
- Keep one row per patient for the primary model; the duplicate P13 array is excluded from `baseline_endpoint.tsv` and retained only in `baseline_endpoint_all_arrays.tsv`.
- Use the unadjusted score-only model as the primary model because the cohort is small.
- Add dose-adjusted analysis as a conservative sensitivity model because sample titles encode 5 mg/kg versus 10 mg/kg infliximab dosing.
- Do not call this prospective prediction; the model is a retrospective baseline-expression association.

### RISK Datasets

- `GSE57945`, `GSE101794`, and `GSE117993` are not primary treatment-response cohorts based on GEO metadata.
- Use them for disease subtype, tissue-site, and pediatric CD/UC generalization rather than endpoint pooling.
- `GSE101794` supports ileal CD versus non-IBD phenotype validation.
- `GSE117993` supports rectal IBD/CD/UC versus control phenotype validation.
- Do not combine ileal and rectal samples without tissue-site stratification.

### IBDMDB/HMP2

- Use only processed tables and sample intersections that fit local compute.
- Do not introduce broad multi-omics integration unless it directly tests longitudinal activity or mechanism-localization claims.

## Cross-Dataset Harmonization Rules

- Expression values are normalized within dataset before axis scoring.
- Effects are estimated within dataset before any cross-cohort pooling.
- Endpoint definitions are harmonized by endpoint family, not by name alone.
- Disease type and tissue site must be retained in all model-ready metadata.

## Remaining Future Work

- Adult endpoint expansion was completed after a stricter label-map gate. GSE92415, GSE206285, GSE23597, and GSE16879 contained traceable sample-level adult response, remission, or healing labels in GEO matrix metadata and were imported as additional adult endpoint cohorts.
- GSE206285 provides the largest adult direct endpoint anchor, with baseline UNIFI sigmoid biopsies and week 8 mucosal-healing and clinical-remission labels. It substantially reduces the adult endpoint sample-size limitation.
- GSE23597 and GSE12251 are ACT1-related resources and should not be interpreted as fully independent trial evidence. They are retained because their sample-level labels and endpoint definitions are traceable, but the manuscript must acknowledge possible study-context overlap.
- GSE193677 was inspected as an adult activity/phenotype support candidate. Processed counts were not imported during this expansion because direct adult endpoint cohorts were sufficient and the large count matrix is not needed to address the primary adult sample-size limitation.
- IBDMDB host transcriptome and activity metadata may support secondary longitudinal validation if sample intersections are sufficient, but they are not part of the current modeled evidence.
