# Introduction

Spatial biology asks where molecular states occur, not only which states are present. Imaging-based assays preserve tissue position while measuring selected molecular targets, including transcripts, transcription-factor activity, miRNA activity, and translation-related signals.

The targets usually have to be chosen before imaging. Probe capacity, optical crowding, barcode design, imaging time, and decoding accuracy make a finite panel unavoidable. The panel therefore determines which cell identities, tissue domains, developmental programs, and disease-associated states can be observed later.

:::{admonition} Key idea
:class: tip

A target panel is an experimental decision: it determines which biological signals can be measured in the completed assay.
:::

```{figure} ../_static/figures/smith_framework.png
:alt: SMITH framework for molecular target selection
:width: 100%
:align: center

SMITH combines the experimental purpose, candidate targets, optional prior targets, and existing reference studies. Stochastic gates produce one target ranking while task-specific networks evaluate the biological information retained by the candidate panel. Adapted from Figure 1 of the SMITH manuscript.
```

SMITH treats panel design as a multi-objective learning problem. A shared stochastic gate ranks candidates, objective-specific networks test whether the gated data preserve complementary biology, and Pareto optimization balances those objectives without a manually fixed loss weighting. Existing single-cell, spatial, cross-study, and cross-modality references can all provide design information. **SMITH-Agent** adds reference retrieval, feasibility checks, and reporting around the core selector.

The introduction is organized around four questions:

- **Problem:** What makes target selection a biological and experimental bottleneck?
- **Model:** How does a differentiable gate turn a reference into a ranked panel?
- **Optimization:** How are competing biological objectives balanced?
- **Transfer:** How can an existing atlas inform a new assay?

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} The Panel-Design Problem
:link: problem
:link-type: doc

Why target selection determines the biology that remains observable.
:::

:::{grid-item-card} Model Principles
:link: model
:link-type: doc

How stochastic gates and objective heads produce a target ranking.
:::

:::{grid-item-card} Multi-objective Optimization
:link: optimization
:link-type: doc

How Pareto gradient balancing handles competing objectives.
:::

:::{grid-item-card} Transfer and SMITH-Agent
:link: transfer
:link-type: doc

How references are reused and design decisions are recorded.
:::
::::

```{toctree}
:maxdepth: 2
:hidden:

problem
model
optimization
transfer
```
