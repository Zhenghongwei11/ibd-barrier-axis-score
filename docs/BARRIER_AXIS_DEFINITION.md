# Barrier-Axis Definition

## Locked Prespecified Axis

The main exposure is a prespecified mucosal IPMK-HDAC3-MMP epithelial barrier-axis score. It is defined before fitting clinical endpoint models and must not be optimized against outcomes.

## Gene Modules

### Upstream Regulatory Module

- `IPMK`: inositol polyphosphate multikinase; mechanistic upstream regulator of InsP generation and HDAC3 activation.
- `IPPK`: converts InsP5 to InsP6; included to separate IPMK-wide HOIP depletion from InsP6-specific biology where data permit.
- `HDAC3`: effector deacetylase in the epigenetic barrier axis.
- `NCOR1`, `NCOR2`: HDAC3 corepressor-complex genes; included as complex-context markers rather than presumed activity markers.

### MMP Injury Module

- `MMP1`, `MMP3`, `MMP10`, `MMP12`, `MMP13`

These genes represent MMP-mediated extracellular matrix and junctional injury programs linked to barrier disruption. Their direction is expected to increase with barrier injury and poorer endpoint status.

### Junction Barrier Module

- `TJP1`, `OCLN`, `CLDN2`

`TJP1` and `OCLN` are expected to decrease with barrier impairment. `CLDN2` is treated as a leak-associated marker and is expected to increase with barrier impairment.

## Primary Score

Default scoring method:

1. Normalize expression within each dataset.
2. Z-score each available axis gene within dataset.
3. Reverse signs so higher values always indicate greater barrier-axis injury:
   - regulatory protective genes: `IPMK`, `IPPK`, `HDAC3`, `NCOR1`, `NCOR2` are multiplied by `-1`.
   - injury genes: MMP genes and `CLDN2` retain positive direction.
   - protective junction genes: `TJP1`, `OCLN` are multiplied by `-1`.
4. Compute module means and the overall axis score as the mean of available signed z-scores.

## Component Scores

- `upstream_score`: lower regulatory activity proxy, signed so higher means worse barrier-axis state.
- `mmp_score`: MMP injury score, higher means more injury.
- `junction_score`: epithelial junction impairment score, higher means worse barrier integrity.

## Missing-Gene Rule

A module is valid if at least 50% of its prespecified genes are available after platform annotation. The overall score is valid if at least two of three modules are valid and at least one MMP gene is measured.

## Exploratory Extensions

Any later genes, cell-type markers, or alternative weighting schemes must be labeled exploratory and cannot replace the primary score for the main claim.

## Biological Anchor

This axis is anchored by the Nature Communications report "Phytic acid (InsP6) activates HDAC3 epigenetic axis to maintain intestinal barrier function" (DOI: 10.1038/s41467-026-68994-0).
