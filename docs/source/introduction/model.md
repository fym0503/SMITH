# Model Principles

SMITH learns target selection and biological prediction in the same model. A stochastic gate acts on the candidate molecular targets, a shared representation summarizes the gated measurements, and objective-specific networks evaluate what biological information remains.

```{figure} ../_static/figures/stochastic_gates.png
:alt: Stochastic gates used for differentiable molecular target selection
:width: 100%
:align: center

Each candidate target has a learnable gate weight. Noise during training makes selection stochastic and differentiable; the optimized weights define the final target ranking. Adapted from Figure 1d of the SMITH manuscript.
```

## Differentiable target selection

Let $x_{ij}$ be the value of target $j$ in observation $i$, and let $w_j$ be the learnable weight assigned to that target. During training, SMITH samples Gaussian noise

$$
\epsilon_{ij} \sim \mathcal{N}(0,\sigma^2)
$$

and applies a clipped stochastic gate:

$$
g_{ij}=\max\!\left(0,\min\!\left(1,w_j+\epsilon_{ij}\right)\right),
\qquad
z_{ij}=x_{ij}g_{ij}.
$$

The gated matrix $Z$ is passed to the prediction networks. A low weight makes a target more likely to be masked; a high weight keeps it active more consistently. The noise encourages the downstream networks to rely on targets that remain informative under stochastic removal instead of depending on a brittle combination of all inputs.

A sparsity regularizer penalizes the expected number of active gates. Its strength is controlled by the `--lam` parameter, while `--sigma` controls gate noise. The optimizer therefore learns both which targets matter and how compact the usable representation should become.

SMITH does not force exactly $M$ gates to be active during training. After optimization, targets are ranked by their learned gate weights and the highest-ranked $M$ targets form the requested panel. This separates the learning problem from the experimental panel budget and allows one trained ranking to be inspected at several panel sizes.

## Shared representation and objective heads

The gated measurements first enter a shared representation network. Each biological objective then has its own prediction head. If the objectives are indexed by $i=1,\ldots,m$, the loss of head $i$ is written as

$$
L_i(\theta_{\mathrm{shared}},\theta_i),
$$

where $\theta_{\mathrm{shared}}$ contains the target selector and shared representation, and $\theta_i$ contains the parameters specific to objective $i$. The shared parameters must support every objective, while each head can model the form of its own output.

## Biological objectives

The available annotations determine which heads are included in a run.

**Molecular profile reconstruction**
: The selected targets are used to reconstruct the full reference profile. This is the base objective because it rewards panels that retain information beyond a small set of known labels. SMITH supports mean squared error and a hurdle loss that separately models zero-valued and positive measurements.

**Cell type information**
: A classification head predicts cell type from the gated representation using cross-entropy loss. This objective favors panels that preserve distinctions among annotated cell populations.

**Spatial information**
: Discrete anatomical regions or spatial domains are modeled by cross-entropy loss. When continuous coordinates are available, a regression head predicts spatial position with mean squared error. Both forms ask whether the panel retains tissue organization.

**Temporal information**
: A regression head predicts developmental or experimental time. This objective is useful for lineage-resolved atlases in which a panel must preserve dynamic molecular activity rather than only terminal identity.

**Pathology information**
: A classification head can preserve disease or pathology annotations when the study design requires condition-associated variation.

These objectives are modular. A single-cell reference may provide reconstruction and cell type labels but no spatial coordinates. A spatial reference can add region or coordinate objectives. A developmental activity atlas can add time. SMITH uses the information available in each reference rather than requiring every dataset to have the same annotation schema.

## Prior targets and panel output

An experiment may need to retain established markers, controls, or targets chosen for a separate hypothesis. SMITH accepts a prior-target list and assigns those targets a sufficiently large selection weight so that they remain in the final ranking. The stochastic gate then learns which additional candidates best complement the prior panel.

The primary model output is the complete target ranking. A requested panel is obtained from its leading entries, including fixed prior targets. Downstream workflows can then evaluate the panel on held-out biology, pass it through assay-feasibility filters, or compare several panel budgets without changing the meaning of the learned weights.
