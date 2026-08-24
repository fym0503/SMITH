# Multi-objective Optimization

A useful panel often has to preserve several biological signals at once. Cell-type prediction may favor different targets from molecular reconstruction, spatial organization, or developmental time. A fixed weighted sum can combine these losses, but its result depends on manually chosen coefficients and on the numerical scale of each loss.

:::{admonition} Key idea
:class: tip

The shared selector is updated with a Pareto-compatible combination of task gradients, rather than a manually chosen weighted sum.
:::

```{figure} ../_static/figures/pareto_optimization.png
:alt: Pareto gradient balancing in SMITH
:width: 100%
:align: center

Objective gradients can point in competing directions. SMITH uses a Frank-Wolfe minimum-norm solver to obtain a common update direction for the shared selector. Adapted from Figure 1e of the SMITH manuscript.
```

## Pareto update

For objective $i$, SMITH computes a gradient $g_i$ from loss $L_i$. Objective-specific parameters use their own gradients; the gate and shared representation receive information from all objectives. Gradients are normalized within the batch so that an objective does not dominate only because its raw loss or gradient norm is larger.

The shared update uses non-negative coefficients on the probability simplex:

$$
\alpha_i \geq 0, \qquad \sum_{i=1}^{m}\alpha_i=1.
$$

SMITH chooses the minimum-norm combination

$$
\boldsymbol{\alpha}^{*}
=\arg\min_{\boldsymbol{\alpha}}
\left\|\sum_{i=1}^{m}\alpha_i g_i\right\|_2^2,
$$

then updates the shared parameters with

$$
g^{*}=\sum_{i=1}^{m}\alpha_i^{*}g_i.
$$

The package solves this constrained problem with a Frank-Wolfe minimum-norm solver. Agreeing objectives retain their common direction; conflicting objectives receive a compromise direction on their convex hull.

## Training and evaluation

Each batch follows three conceptual steps:

1. Apply the stochastic gate, compute the shared representation, and evaluate every active objective.
2. Normalize the objective gradients and solve for the Pareto coefficients $\alpha_i$.
3. Update the model and repeat until the requested epoch, then export gate weights as the target ranking.

The result is a locally Pareto-compatible compromise, not a claim that every objective is maximized simultaneously. Manuscript analyses evaluate the exported panel separately on held-out data: for example, cell type, profile reconstruction, spatial-region recovery, coordinate prediction, developmental time, or cross-dataset transfer. This separation tests what can be recovered from selected-target measurements rather than relying on a training curve.
