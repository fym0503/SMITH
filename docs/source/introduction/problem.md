# The Panel-Design Problem

## Spatial information changes the experiment

Dissociated single-cell assays measure molecular heterogeneity deeply, but they remove cells from their native tissue. Imaging-based profiling preserves location, so molecular states can be interpreted with cell-cell neighborhoods, anatomical regions, and tissue architecture. This matters whenever position shapes function, including development, neural organization, immune niches, and disease-associated remodeling.

```{figure} ../_static/figures/panel_design_problem.png
:alt: Target panel selection in imaging-based spatial molecular profiling
:width: 100%
:align: center

Imaging-based spatial molecular profiling requires a target panel before data generation. The selected targets determine which molecular signals can be measured and which downstream analyses the experiment can support. Adapted from Figure 1a of the SMITH manuscript.
```

The same design constraint appears across targeted transcriptomics, transcription-factor and miRNA activity assays, and translatomic imaging. We use **molecular target** because the selectable unit is not limited to a gene-expression measurement.

:::{admonition} Key idea
:class: tip

Panel design is an irreversible bottleneck: information excluded before imaging cannot be recovered by downstream analysis.
:::

## Why panel choice is biological

Probe designability, optical crowding, barcode capacity, imaging cycles, reporter construction, and decoding error impose a finite panel budget. A useful panel may need to preserve several signals at once:

- cell identities and rare populations;
- unmeasured molecular profiles;
- anatomical regions and spatial organization;
- developmental, temporal, or disease-associated states.

A canonical marker may separate known cell classes but miss within-class state. A highly expressed target may be easy to detect but spatially uninformative. The right subset therefore depends on the intended biological analysis, not on abundance or generic marker status alone.

## SMITH formulation

Let a reference contain $K$ observations and $N$ candidate targets:

$$
X \in \mathbb{R}^{K \times N}.
$$

For a requested panel size $M$, SMITH returns a subset $S$ with $|S|=M$ and a ranking over all candidates. If $P_i$ is a biological property such as cell identity, molecular profile, spatial position, or developmental time, the design goal is

$$
P_i(X_S) \approx P_i(X), \qquad i=1,\ldots,m.
$$

These properties have different labels, scales, and preferred targets. SMITH therefore keeps a separate objective for each property and learns one shared ranking rather than collapsing everything into a fixed score.

## Inputs and output

A run specifies four inputs:

- **Experimental purpose:** tissue, species, assay, panel size, and analyses.
- **Candidate universe:** targets measurable by the intended platform.
- **Reference data:** single-cell, spatial, aligned, or related-modality data with the available annotations.
- **Prior targets:** required markers or controls that must remain in the panel.

The output is a ranked target list and one or more panel sizes. Evaluation on held-out observations or a held-out target dataset tests the biology preserved by the selected panel, rather than relying on training loss alone.
