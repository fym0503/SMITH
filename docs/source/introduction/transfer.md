# Transfer and SMITH-Agent

The most informative reference for a new experiment is not always a matched scRNA-seq dataset. Existing spatial atlases contain tissue organization that dissociated data cannot provide, and related assays can reveal modality-specific priorities. SMITH therefore treats molecular target selection as a reference-reuse problem.

```{figure} ../_static/figures/transfer_modes.png
:alt: Cross-assay, cross-study, and cross-modality transfer in SMITH
:width: 85%
:align: center

SMITH can transfer panel-design information across assays, studies, and molecular modalities. Adapted from Figure 1f of the SMITH manuscript.
```

## Single-cell reference

$\mathrm{SMITH}_{\mathrm{sc}}$ selects targets from a single-cell reference matched to the tissue and species of the intended experiment. This is the conventional design setting for targeted spatial transcriptomics. Because dissociated data do not contain native coordinates, the usual objectives are molecular profile reconstruction and cell type information.

This mode is appropriate when broad target coverage is important and no relevant spatial reference is available. Its boundary is equally important: spatial organization cannot be learned directly from a reference that does not contain spatial information.

## Spatial reference

$\mathrm{SMITH}_{\mathrm{sp}}$ learns directly from an existing spatial dataset. In addition to profile and cell type objectives, it can use anatomical regions or continuous coordinates. A previous assay can therefore contribute information about tissue organization to the design of a later experiment.

The source and target do not need to be the same physical sample, but transfer depends on biological relevance. Differences in tissue, condition, developmental stage, assay chemistry, or annotation quality can change which targets are useful.

## Aligned single-cell and spatial reference

$\mathrm{SMITH}_{\mathrm{sc+sp}}$ combines the broad molecular coverage of single-cell data with the spatial context of a reference assay. Let $X^{\mathrm{SC}}$ be the single-cell matrix, $X^{\mathrm{SP}}$ the spatial matrix, and $A$ a cell-to-location mapping produced by an established alignment method. A pseudo-spatial profile over the single-cell target universe is constructed as

$$
\widehat{X}^{\mathrm{SP}}_{i,:}
=
\sum_j A_{ji}X^{\mathrm{SC}}_{j,:}.
$$

SMITH then selects targets from this aligned representation using profile, cell type, and spatial objectives. The alignment tool is deliberately external to the selector: users can choose an appropriate mapping method, while SMITH consumes the resulting representation. Poor alignment can propagate bias into the selected panel, so alignment quality remains part of the experimental design assessment.

## Cross-study and cross-modality transfer

Direct transfer is used when a source and target assay share measurable target identities but do not support cell-level alignment. SMITH is trained on the source reference, candidates are restricted to the target assay's measurable universe, and the selected identities are evaluated or deployed in the target modality. The manuscript uses this setting to transfer transcriptomic information to transcription-factor activity profiling and STARmap information to RIBOMap profiling.

For references that can be aligned across modalities, SMITH can first construct a shared representation and then train with the annotations available on the aligned cell axis. The RIBOMap analysis, for example, combines shared-gene expression, cell type consistency, and spatial structure through a soft optimal-transport coupling before target selection. This mode is more expressive than direct transfer, but it also introduces additional assumptions from feature matching and alignment.

## From SMITH to SMITH-Agent

The core `smith` package answers a focused computational question: given a prepared reference, selected objectives, and a panel budget, which molecular targets should be prioritized? Real experimental design requires decisions before and after that optimization. `smith_agent` organizes those decisions as an explicit workflow.

Starting from a biological design intent, SMITH-Agent can:

1. structure the species, tissue, assay, panel size, objectives, and required prior targets;
2. retrieve or register compatible reference datasets;
3. validate annotations and assemble a measurable candidate universe;
4. run SMITH on the source and relevant references;
5. convert rankings to percentile priority scores and aggregate multi-reference evidence;
6. evaluate candidate panels on biological objectives;
7. apply assay-specific feasibility and probe-design backends;
8. record the configuration, selected targets, evaluations, and output files in a report.

When several references are available, the agent does not concatenate their rankings blindly. It trains SMITH within each compatible reference, converts ranks to a common percentile scale, and builds a consensus ranking from the evidence contributed by those references. This keeps the origin of each target priority inspectable.

Feasibility checking is also separate from biological informativeness. A highly ranked gene may yield too few assay-compatible probes or fail specificity filters. SMITH-Agent provides a modular bridge to probe-design tools so these constraints can be considered before a panel is finalized. Because chemistry and probe rules differ among platforms, external backends and experimental review remain necessary for deployment.

SMITH-Agent is therefore a decision-support layer around SMITH, not a replacement for the optimizer or for expert judgment. The optimizer learns a target ranking from biological references; the agent makes the surrounding design process structured, traceable, and repeatable.
