# Transfer and SMITH-Agent

The best reference for a new experiment is not always a matched single-cell dataset. **Spatial atlases** contain tissue organization that dissociated data cannot provide, and related assays can reveal modality-specific priorities. SMITH treats these references as **reusable design information**.

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

The agent does not concatenate rankings blindly. Each reference contributes an **inspectable ranking** before consensus is formed. Feasibility is also separate from biological informativeness: a highly ranked target may fail probe specificity or platform constraints. External probe-design backends and experimental review are therefore still required before deployment.

**SMITH-Agent is a decision-support layer around SMITH**, not a replacement for the optimizer or expert judgment. The optimizer learns from biological references; the agent makes the surrounding process structured, traceable, and repeatable.
