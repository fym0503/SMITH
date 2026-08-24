# Model Principles

SMITH learns target selection and biological prediction in one model. A stochastic gate acts on candidate targets, a shared representation summarizes the gated measurements, and objective-specific heads test which biological signals remain recoverable.

:::{admonition} Key idea
:class: tip

One shared gate learns a target ranking; the objective heads tell the gate what information the experiment must preserve.
:::

```{figure} ../_static/figures/stochastic_gates.png
:alt: Stochastic gates used for differentiable molecular target selection
:width: 100%
:align: center

Each candidate target has a learnable gate weight. Noise during training makes selection stochastic and differentiable; the optimized weights define the final target ranking. Adapted from Figure 1d of the SMITH manuscript.
```

## 1. Differentiable target selection

Let $x_{ij}$ be target $j$ in observation $i$, and let $w_j$ be its learnable weight. SMITH samples

$$
\epsilon_{ij} \sim \mathcal{N}(0,\sigma^2)
$$

and applies a clipped gate:

$$
g_{ij}=\max\!\left(0,\min\!\left(1,w_j+\epsilon_{ij}\right)\right),
\qquad z_{ij}=x_{ij}g_{ij}.
$$

The gated matrix $Z$ enters the prediction heads. Low-weight targets are masked more often, while high-weight targets remain active. Stochastic removal encourages the model to use robust combinations rather than a brittle dependence on every input. A sparsity regularizer controls the expected number of active gates (`--lam`); `--sigma` controls gate noise.

SMITH does not force exactly $M$ gates to be active during training. After optimization, targets are ranked by $w_j$ and the highest-ranked $M$ form the requested panel. One ranking can therefore be inspected at several panel sizes.

## 2. Shared representation and objective heads

The gated measurements first pass through a shared representation. Each biological objective has its own head and loss:

$$
L_i(\theta_{\mathrm{shared}},\theta_i),
$$

where the shared parameters include the selector and representation, while $\theta_i$ is specific to objective $i$.

- **Profile reconstruction:** recover the full reference profile from the selected targets.
- **Cell type:** predict annotated cell types with cross-entropy.
- **Spatial organization:** predict regions with cross-entropy or coordinates with regression loss.
- **Temporal or pathology state:** predict developmental time or condition labels when those annotations exist.

The heads are modular. A single-cell reference may provide reconstruction and cell type information; a spatial or developmental reference can add region, coordinate, or time objectives.

## 3. Prior targets and panel output

Required markers and assay controls can be supplied as prior targets. They are kept in the final ranking while the gate learns which additional candidates best complement them. The primary output is the complete ranking; a panel is the leading $M$ targets, including fixed prior targets. Held-out biological evaluation and assay-feasibility checks are performed after this export.
