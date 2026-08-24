# Model Principles

**SMITH learns target selection and biological prediction in one model.** A stochastic gate acts on candidate targets, a shared representation summarizes the gated measurements, and objective-specific heads test which biological signals remain recoverable.

## Model overview

```{figure} ../_static/figures/model_overview_c.png
:alt: Overview of SMITH target panel selection and multi-task learning
:width: 100%
:align: center

SMITH starts from the experimental purpose, candidate targets, prior targets, and existing single-cell or spatial studies. Stochastic gates transform the candidate universe into a shared representation, while multiple biological objectives guide the final selected panel. Adapted from Figure 1c of the SMITH manuscript.
```

:::{admonition} Key idea
:class: tip

One shared gate learns a target ranking; the objective heads tell the gate what information the experiment must preserve.
:::

## Stochastic gates

```{figure} ../_static/figures/stochastic_gates_d.png
:alt: Stochastic gates used for differentiable molecular target selection
:width: 90%
:align: center

Each candidate target has a learnable gate weight. Noise makes the selector stochastic and differentiable during training; the learned weights define the final target ranking. Adapted from Figure 1d of the SMITH manuscript.
```

### Differentiable target selection

Let $x_{ij}$ be target $j$ in observation $i$, and let $w_j$ be its learnable weight. SMITH samples

$$
\epsilon_{ij} \sim \mathcal{N}(0,\sigma^2)
$$

and applies a clipped gate:

$$
g_{ij}=\max\!\left(0,\min\!\left(1,w_j+\epsilon_{ij}\right)\right),
\qquad z_{ij}=x_{ij}g_{ij}.
$$

The gated matrix $Z$ enters the prediction heads. **Low-weight targets** are masked more often, while **high-weight targets** remain active. Stochastic removal encourages the model to use robust combinations rather than a brittle dependence on every input. A **sparsity regularizer** controls the expected number of active gates (`--lam`); `--sigma` controls gate noise.

SMITH does not force exactly $M$ gates to be active during training. After optimization, targets are ranked by $w_j$ and the **highest-ranked $M$ form the requested panel**. One ranking can therefore be inspected at several panel sizes.

## Multi-task learning

```{figure} ../_static/figures/multitask_learning_e.png
:alt: Multi-task learning and Pareto gradient selection in SMITH
:width: 70%
:align: center

The shared selector receives gradients from several biological objectives. A Frank-Wolfe minimum-norm update finds a common direction when profile, cell-type, spatial, and temporal objectives disagree. Adapted from Figure 1e of the SMITH manuscript.
```

### Shared representation and objective heads

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

## Pareto optimization

A useful panel often has to preserve **several biological signals at once**. Cell-type prediction may favor different targets from molecular reconstruction, spatial organization, or developmental time. A fixed weighted sum can combine these losses, but its result depends on manually chosen coefficients and on the numerical scale of each loss.

```{figure} ../_static/figures/pareto_optimization.png
:alt: Pareto gradient balancing in SMITH
:width: 85%
:align: center

Objective gradients can point in competing directions. SMITH uses a Frank-Wolfe minimum-norm solver to obtain a common update direction for the shared selector. Adapted from Figure 1e of the SMITH manuscript.
```

### Gradient balancing

For objective $i$, SMITH computes a gradient $g_i$ from loss $L_i$. Objective-specific parameters use their own gradients; the **gate and shared representation receive information from all objectives**. Gradients are normalized within each batch so that an objective does not dominate only because its raw loss or gradient norm is larger.

The shared update uses non-negative coefficients on the probability simplex:

$$
\alpha_i \geq 0, \qquad \sum_{i=1}^{m}\alpha_i=1.
$$

SMITH chooses the **minimum-norm combination**

$$
\boldsymbol{\alpha}^{*}
=\arg\min_{\boldsymbol{\alpha}}
\left\|\sum_{i=1}^{m}\alpha_i g_i\right\|_2^2,
$$

then updates the shared parameters with

$$
g^{*}=\sum_{i=1}^{m}\alpha_i^{*}g_i.
$$

The package solves this constrained problem with a **Frank-Wolfe minimum-norm solver**. Agreeing objectives retain their common direction; conflicting objectives receive a compromise direction on their convex hull.

### Training and evaluation

Each batch follows three conceptual steps:

1. Apply the stochastic gate, compute the shared representation, and evaluate every active objective.
2. Normalize objective gradients and solve for the Pareto coefficients $\alpha_i$.
3. Update the model and repeat until the requested epoch, then export gate weights as the target ranking.

The result is a locally Pareto-compatible compromise, not a claim that every objective is maximized simultaneously. Manuscript analyses **evaluate the exported panel separately on held-out data**, testing cell type, profile reconstruction, spatial organization, developmental time, or cross-dataset transfer rather than relying on a training curve.

## Prior targets and panel output

Required markers and assay controls can be supplied as prior targets. They are kept in the final ranking while the gate learns which additional candidates best complement them. The **primary output is the complete ranking**; a panel is the leading $M$ targets, including fixed prior targets. Held-out biological evaluation and assay-feasibility checks are performed after this export.
