# The Panel-Design Problem

## Spatial information changes the experiment

Dissociated single-cell assays can describe molecular heterogeneity at high depth, but they remove cells from their native tissue. Imaging-based spatial profiling preserves location, allowing molecular states to be interpreted together with tissue architecture, cell-cell neighborhoods, and anatomical context. This distinction is central in systems where position shapes function, including developmental patterning, neural organization, immune niches, and disease-associated tissue remodeling.

```{figure} ../_static/figures/panel_design_problem.png
:alt: Target panel selection in imaging-based spatial molecular profiling
:width: 100%
:align: center

Imaging-based spatial molecular profiling requires a target panel to be selected before data generation. The selected targets determine which molecular signals can be measured and which downstream biological analyses the experiment can support. Adapted from Figure 1a of the SMITH manuscript.
```

The same experimental design principle now spans several molecular layers. Targeted spatial transcriptomics measures selected messenger RNAs. Regulatory imaging assays can measure transcription-factor or miRNA activities. Translatomic assays can spatially resolve actively translated RNA. SMITH uses the term **molecular target** because the selectable unit is not restricted to a gene-expression measurement.

## Target selection is an irreversible bottleneck

Most imaging assays cannot measure every candidate target. Probe designability, optical crowding, barcode capacity, imaging cycles, reporter construction, and decoding error impose a finite panel budget. The targets must be specified before data generation, so information excluded at this stage cannot be recovered by a downstream analysis.

This makes panel design different from ordinary feature selection. Feature selection is often judged by how well a subset predicts a label in an existing dataset. A spatial panel must instead support a future experiment whose biological outputs may include several distinct analyses:

- identifying cell types and rare populations;
- reconstructing unmeasured molecular profiles;
- resolving anatomical regions or continuous spatial organization;
- following developmental or temporal changes;
- retaining disease-associated variation;
- satisfying platform-specific probe constraints.

A target can be valuable for one objective and dispensable for another. Canonical markers may distinguish known cell classes but provide limited coverage of within-class state. Highly expressed genes may be easy to detect but spatially uninformative. Spatially patterned targets may define tissue domains without reconstructing the broader molecular profile. The useful panel is therefore determined by the biological purpose and by the trade-offs among objectives.

## Formal definition

Assume that a reference contains $K$ cells or observations and $N$ candidate molecular targets. SMITH represents it as

$$
X \in \mathbb{R}^{K \times N}.
$$

For a requested panel size $M$, the output is a subset $S$ of the candidates, where $|S|=M$, together with a ranking over all $N$ targets. The reduced matrix $X_S$ contains only the selected columns.

Let $P_i$ denote one biological property that should remain recoverable from the panel, such as cell identity, molecular profile, spatial position, or developmental time. The conceptual goal is to make the information recovered from $X_S$ resemble the information available in the full reference $X$ for every selected property:

$$
P_i(X_S) \approx P_i(X), \qquad i=1,\ldots,m.
$$

Because these properties are not measured on a common scale and can favor different targets, SMITH does not collapse them into a single predefined score. It learns a shared target ranking while retaining a separate prediction loss for each biological objective.

## What enters a SMITH design

A design begins with four pieces of information.

**Experimental purpose**
: The intended tissue, species, assay, panel size, and downstream biological analyses determine which objectives are relevant.

**Candidate universe**
: Candidate targets are the columns of the reference matrix, optionally restricted to targets measurable by the intended assay.

**Reference data**
: The reference may be single-cell, spatial, aligned single-cell and spatial data, or a related molecular modality. Its annotations determine which objectives can be trained.

**Prior targets**
: User-specified markers or required assay controls can be retained in the final panel. SMITH ranks the remaining candidates around these fixed choices rather than forcing users to discard established biological knowledge.

The result is a ranked target list and a panel of the requested size. Evaluation is performed on held-out observations or a held-out target dataset, so the selected panel is judged by the biology it preserves rather than by its training loss alone.
