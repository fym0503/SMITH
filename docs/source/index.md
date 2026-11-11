```{image} _static/brand/SMITH_illu.png
:alt: SMITH logo showing target selection from a spatial reference
:width: 900px
:align: center
```

# SMITH

**Agentic and transferable panel design for imaging-based spatial molecular profiling.**

SMITH selects compact target panels from single-cell and spatial references. Its multi-task model combines cell identity, expression, spatial and temporal signals; SMITH-Agent organizes the surrounding data, selection and evaluation workflow.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Install and run
:link: installation
Set up the Python package and train a panel from AnnData.
:::

:::{grid-item-card} Understand the model
:link: introduction/index
Explore target selection, multi-task learning and reference transfer.
:::

:::{grid-item-card} Follow a tutorial
:link: tutorials/index
Run the TF, RIBOMap and liver MERFISH examples.
:::

:::{grid-item-card} Use the command line
:link: cli
Inspect tools and run reproducibility workflows.
:::
::::

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

installation
introduction/index
quickstart
```

```{toctree}
:maxdepth: 2
:caption: Core commands
:hidden:

cli
```

```{toctree}
:maxdepth: 2
:caption: Paper examples
:hidden:

tutorials/index
```
