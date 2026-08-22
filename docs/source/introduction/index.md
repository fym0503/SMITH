# Introduction

Spatial biology asks not only which molecular states are present in a tissue, but also where those states occur and how neighboring cells organize into functional niches. Imaging-based spatial profiling addresses this question by measuring molecular signals together with cellular or subcellular position. Mature assays such as MERFISH and seqFISH profile targeted transcripts, while newer imaging strategies extend the same principle to transcription-factor activity, miRNA activity, and the translatome.

These technologies share an important experimental constraint: the molecular targets must usually be chosen before imaging begins. Each target can require a dedicated probe set, barcode, reporter, or genetic tag, and the number of reliable measurements is limited by assay chemistry, imaging throughput, and decoding accuracy. A panel is therefore not a neutral subset of the molecular atlas. It determines which cell identities, tissue domains, developmental signals, and disease-associated programs remain observable in the completed experiment.

**SMITH is a framework for selecting compact molecular target panels from existing single-cell and spatial references.** It treats panel design as a multi-objective problem. A shared stochastic gate learns which targets to retain, while objective-specific networks ask whether the gated data preserve complementary forms of biological information. Pareto multi-task optimization balances these objectives without requiring a manually fixed loss weighting.

SMITH also treats existing atlases as design priors. A panel can be learned from a matched single-cell reference, an existing spatial study, an aligned single-cell and spatial representation, or a related molecular modality. This transfer view allows earlier experiments to inform new assays across studies, platforms, and molecular readouts. SMITH-Agent builds on the optimizer by organizing biological intent, reference retrieval, model execution, feasibility checks, and probe-level outputs into a reproducible workflow.

```{figure} ../_static/figures/smith_framework.png
:alt: SMITH framework for molecular target selection
:width: 100%
:align: center

SMITH combines the experimental purpose, candidate targets, optional prior targets, and existing reference studies. Stochastic gates produce one target ranking while task-specific networks evaluate the biological information retained by the candidate panel. Adapted from Figure 1 of the SMITH manuscript.
```

The framework is built around four ideas:

1. **The panel is part of the biological question.** Informativeness depends on the intended analysis, not only on expression abundance or generic marker status.
2. **Selection is learned jointly with biological objectives.** Targets receive high priority when they collectively preserve the signals needed for the experiment.
3. **Competing objectives are balanced explicitly.** The optimizer searches for a common Pareto descent direction rather than hiding the trade-off in fixed loss weights.
4. **Reference data are reusable design information.** Single-cell, spatial, cross-study, and cross-modality datasets can all contribute to a new panel.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} The Panel-Design Problem
:link: problem
:link-type: doc

Why target selection is an experimental bottleneck and how SMITH defines biological information preservation.
:::

:::{grid-item-card} Model Principles
:link: model
:link-type: doc

How the stochastic gate, shared representation, objective heads, and prior targets produce a ranked panel.
:::

:::{grid-item-card} Multi-objective Optimization
:link: optimization
:link-type: doc

How Pareto gradient balancing trains one selector against several biological objectives.
:::

:::{grid-item-card} Transfer and SMITH-Agent
:link: transfer
:link-type: doc

How existing atlases inform new assays and how the agent turns a design intent into deployable outputs.
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
