# Multi-objective Optimization

A useful panel often has to preserve several biological signals at once. Optimizing only cell type classification can discard genes needed for molecular reconstruction. Optimizing only reconstruction can favor broadly varying targets while weakening rare-cell or spatial resolution. A fixed weighted sum of losses can combine these objectives, but its behavior depends on manually chosen coefficients and on the numerical scale of each loss.

SMITH instead searches for a Pareto-compatible update during training.

```{figure} ../_static/figures/pareto_optimization.png
:alt: Pareto gradient balancing in SMITH
:width: 100%
:align: center

Objective gradients can point in competing directions. SMITH uses a Frank-Wolfe minimum-norm solver to obtain a common update direction for the shared selector. Adapted from Figure 1e of the SMITH manuscript.
```

## Separate objective gradients

For each objective $i$, SMITH computes a gradient $g_i$ from its loss $L_i$. Objective-specific network parameters are updated from their own losses. The stochastic gate and shared representation, however, receive information from every objective and therefore require one combined update.

Before combination, gradients are normalized so that an objective does not dominate only because its raw loss or gradient norm is numerically larger. The implementation uses loss-and-gradient normalization during each training batch.

## Minimum-norm combination

SMITH chooses non-negative coefficients $\alpha_1,\ldots,\alpha_m$ on the probability simplex:

$$
\alpha_i \geq 0,
\qquad
\sum_{i=1}^{m}\alpha_i=1.
$$

The coefficients minimize the norm of the combined gradient:

$$
\boldsymbol{\alpha}^{*}
=
\arg\min_{\boldsymbol{\alpha}}
\left\|
\sum_{i=1}^{m}\alpha_i g_i
\right\|_2^2.
$$

The package solves this constrained problem with a Frank-Wolfe minimum-norm solver. The shared parameters are then updated in the direction

$$
g^{*}=\sum_{i=1}^{m}\alpha_i^{*}g_i.
$$

When the task gradients agree, the solution follows their common direction. When they conflict, the solver finds a compromise direction on their convex hull. At a Pareto-stationary point, no local update can improve one objective without worsening at least one other objective.

## Training sequence

Each batch follows the same sequence:

1. Apply the stochastic gate and compute the shared representation.
2. Run every active objective head and calculate its task loss plus gate regularization.
3. Compute and normalize the objective gradients used by the minimum-norm solver.
4. Solve for the Pareto coefficients $\alpha_i$.
5. Recompute the weighted task losses, back-propagate their sum, and update the model.
6. Repeat until the requested training epoch, then export the learned gate weights as the target ranking.

This procedure does not claim that all biological goals are simultaneously maximized. Instead, it makes their competition part of the optimization problem. The resulting ranking reflects a locally Pareto-compatible compromise under the chosen reference data, annotations, model configuration, and candidate universe.

## From model loss to biological evaluation

The objective heads guide target selection during training, but manuscript analyses evaluate the exported panel independently. Depending on the experiment, held-out evaluation includes cell type classification, profile reconstruction, spatial-region recovery, coordinate prediction, developmental-time prediction, module coverage, or cross-dataset transfer. This separation is important: the panel is judged by what can be recovered from newly generated selected-target profiles, not by a training curve alone.
