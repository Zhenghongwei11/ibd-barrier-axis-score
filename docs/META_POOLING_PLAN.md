# Cohort-Level Synthesis and Pooling Plan

## Article-Type Boundary

This manuscript is an age-stratified multicohort clinical biomarker validation study. It is not a formal systematic review or meta-analysis.

Allowed wording:

- cohort-level synthesis
- age-stratified effect summary
- multicohort clinical validation
- secondary random-effects summary within compatible strata

Forbidden wording:

- systematic review and meta-analysis
- comprehensive meta-analysis of all IBD transcriptomic studies
- pooled proof of clinical utility
- single pan-IBD pooled effect across adult and pediatric endpoints

## Primary Reporting Order

1. Cohort-specific endpoint models with effect sizes and confidence intervals.
2. Age-stratified adult versus pediatric/early-onset summary.
3. Endpoint-family summary.
4. Secondary pooling only if compatibility criteria are met.

## Pooling Eligibility

Pooling requires at least two independent cohorts with:

- compatible endpoint family
- comparable expression timing, preferably baseline expression before endpoint assessment
- comparable tissue context or a documented tissue-stratified rationale
- consistent effect direction definition
- defensible age-stratum handling

## Non-Poolable Endpoint Families

Do not pool:

- adult mucosal healing with pediatric week-4 remission unless explicitly justified as a broader response family
- treatment response with diagnosis or disease-versus-control status
- ulceration or macroscopic inflammation with biologic response
- longitudinal activity slopes with cross-sectional endpoint effects
- adult and pediatric estimates as a single unqualified effect

## Output Tables

- `results/meta/age_endpoint_synthesis.tsv`
- `results/meta/non_poolable_endpoints.tsv`
- `results/meta/random_effects_pooling.tsv` only if endpoint-compatible pooling is actually performed

## Current Pooling Outlook

Current modeled endpoints are clinically informative but heterogeneous. GSE73661 is an adult trial-derived mucosal-healing anchor, GSE109142 is a pediatric remission context, and GSE57945/GSE101794/GSE117993 are pediatric phenotype or injury support cohorts. Adult candidates must be imported and modeled before any adult endpoint-compatible pooled summary can be considered.
