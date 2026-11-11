# Transfer and SMITH-Agent

A new experiment can draw on several kinds of reference data. **Spatial atlases** contribute tissue organization, while related assays reveal modality-specific priorities. SMITH treats both as **reusable design information**.

:::{admonition} Key idea
:class: tip

Transfer lets a source atlas inform a target assay, but biological relevance and alignment quality still determine whether that information is trustworthy.
:::

```{figure} ../_static/figures/transfer_modes.png
:alt: Cross-assay, cross-study, and cross-modality transfer in SMITH
:width: 85%
:align: center

SMITH can transfer panel-design information across assays, studies, and molecular modalities. Adapted from Figure 1f of the SMITH manuscript.
```

## Reference modes

- **Single-cell reference (`SMITH_sc`).** Selects targets from a matched scRNA-seq atlas using profile reconstruction and cell-type information. It is useful when broad molecular coverage matters, but spatial organization cannot be learned directly from dissociated data.

- **Spatial reference (`SMITH_sp`).** Adds anatomical regions or continuous coordinates to the profile and cell-type objectives. A previous spatial assay can therefore inform the tissue organization expected in a new experiment.

- **Aligned single-cell and spatial reference (`SMITH_sc+sp`).** Combines broad single-cell coverage with spatial context. If $A$ maps single cells to spatial locations, a pseudo-spatial profile is

  $$
  \widehat{X}^{\mathrm{SP}}_{i,:}
  =\sum_j A_{ji}X^{\mathrm{SC}}_{j,:}.
  $$

  SMITH can then use profile, cell-type, and spatial objectives on the aligned representation. The alignment method is external to the selector, so alignment quality remains part of the design assessment.

- **Cross-study or cross-modality transfer.** When source and target assays share measurable target identities but cannot be aligned cell by cell, SMITH is trained on the source and restricted to the target's measurable universe. The manuscript uses this setting for transcriptomic-to-regulatory transfer and STARmap-to-RIBOMap transfer.

Transfer is only as reliable as the **biological match**. Tissue, condition, developmental stage, assay chemistry, and annotation differences can change which targets are useful.

## From SMITH to SMITH-Agent

The core **`smith` package** answers: given a prepared reference, objectives, and a panel budget, which targets should be prioritized? **`smith_agent`** organizes the surrounding design decisions:

1. structure the biological intent, assay, panel size, objectives, and required targets;
2. retrieve compatible references and validate their annotations and measurable candidate universe;
3. run SMITH within each reference, convert rankings to a common percentile scale, and aggregate multi-reference evidence;
4. evaluate candidate panels, apply assay-specific feasibility backends, and record the configuration, targets, metrics, and output files.

Each reference contributes an **inspectable ranking** before consensus is formed. Feasibility tools add assay-specific probe and platform checks to the biological ranking, producing a traceable panel-design report.

**SMITH-Agent organizes the workflow around the SMITH optimizer**. SMITH learns from biological references; the agent structures reference retrieval, ranking aggregation, evaluation and feasibility checks into a traceable process.
