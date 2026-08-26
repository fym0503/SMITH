# SMITH: Agentic and Transferable Panel Design

SMITH is a computational framework for target panel design in imaging-based spatial molecular profiling. It selects compact gene or molecular target panels from reference atlases so future targeted spatial assays can preserve cell identity, molecular profiles, spatial organization, temporal signals, or disease-associated variation under a fixed panel budget.

This directory is a clean, repository-style package assembled from the local SMITH workspace. It follows the lightweight organization used by DeconX: source code in `src/`, runnable entry points in `scripts/`, configuration in `configs/`, documentation in `docs/`, and placeholder `data/` and `outputs/` directories.

## Project Architecture

The package exposes two Python namespaces: the core optimizer and the agent workflow.

* **src/smith/**: Core SMITH optimization code.
  * `stgmodel.py`: stochastic-gate target selection and task heads.
  * `datasets.py`: AnnData loading, task tensor construction, balancing, and prior target lookup.
  * `losses.py`: reconstruction/classification/regression losses, including hurdle loss support.
  * `model_selector.py`: model and optimizer construction.
  * `eval.py`: panel export and downstream evaluation helpers.
  * `min_norm_solvers.py`: multi-task Pareto gradient solver.
* **src/smith_agent/**: Agent and CLI orchestration layer.
  * Parses panel-design intents into structured requests.
  * Registers tools for dataset inspection, reference retrieval, SMITH selection, feasibility filtering, evaluation, plotting, and reporting.
  * Provides the `smith-cli` command.
  * `feasibility/` normalizes ODT/SCRINSHOT, OligoMiner, PaintSHOP, and ProbeDealer outputs.
  * `probedealer/` contains the lightweight ProbeDealer-style implementation.
* **scripts/**: Main execution entry points.
  * `main.py`: original SMITH training and panel-selection runner.
  * `eval.py`: downstream panel evaluation runner.
  * `submit_eval.py`: original batch/evaluation helper.
  * `agent_examples/`: analysis and figure-generation scripts from `smith-agent`.
* **configs/agent/**: SMITH-Agent registries for datasets, tools, workflow skills, models, policies, and feasibility backends.
* **docs/source/**: Local documentation copied from the original SMITH, baseline, interface, and agent workspaces.
* **data/**: Placeholder for user-provided or downloaded datasets. Large H5AD files are intentionally not copied here.
* **outputs/**: Placeholder for new runs. Historical HPO and benchmark outputs are intentionally excluded.
* **reproducibility/**: Paper-section manifests and compact, checksum-pinned example inputs.

### Agent workflow skills

The files in `configs/agent/skills/` are YAML workflow definitions loaded by SMITH-Agent. They describe task intake, reference retrieval, candidate assembly, panel selection, feasibility filtering, probe generation, baseline comparison, and reporting. They are application configuration, not Codex `SKILL.md` packages. The wheel embeds a runtime copy under `smith_agent.resources/configs/agent/skills/`; edit the source files in `configs/agent/skills/`.

### Paper reproducibility

Each main Results section has one representative executable example. The examples do not imply that every manuscript panel is rebuilt by the default command; data and tool requirements are documented separately.

```bash
smith-repro list
smith-repro check
smith-repro run 02_regulatory_activity
smith-repro run 03_ribomap_transfer
smith-repro check 02_regulatory_activity --data-root data/tutorials
smith-repro run 02_regulatory_activity --data-root data/tutorials --output-dir outputs/tutorials/regulatory
smith-repro run 05_agent
```

The runnable biological examples are documented under `docs/source/tutorials/`; their workflows write input checksums, model rankings, panels, evaluations and run manifests to the selected output directory.

## Technical Workflow

1. **Define the panel-design task.** Specify the assay context, panel size, candidate targets, optional prior targets, and objectives such as cell type recovery, profile reconstruction, spatial preservation, time prediction, or disease-aware variation.
2. **Load reference data.** Use matched scRNA-seq, spatial references, modality-related references, or aligned single-cell/spatial representations.
3. **Run SMITH optimization.** A stochastic gate layer learns target weights while objective-specific heads evaluate complementary biological signals.
4. **Balance objectives.** Pareto multi-task optimization combines gradients without hand-tuned objective weights.
5. **Transfer and evaluate.** Panels can be selected from scRNA-seq, spatial references, aligned scRNA+spatial data, or cross-modality references and evaluated on held-out target assays.
6. **Optionally run SMITH-Agent.** The agent workflow can retrieve references, run feasibility filters, tune hyperparameters, evaluate panels, generate probe candidates, and build reports.

## Installation

Create an environment with Python 3.10+ and install the package in editable mode:

```bash
python -m pip install -e .
```

For the original SMITH optimizer, the important scientific dependencies are:

```bash
python -m pip install numpy pandas scipy scikit-learn scanpy anndata torch tqdm h5py
```

For SMITH-Agent, install:

```bash
python -m pip install pyyaml requests rich prompt-toolkit biopython
```

Some optional feasibility backends depend on external tools and reference indexes. See `docs/source/interface_readme.md` and `configs/agent/feasibility_backends/`.

## Packaging

Build source and wheel distributions:

```bash
python -m build --sdist --wheel
```

The source distribution keeps the DeconX-style project layout: `configs/`, `scripts/`, `docs/`, `manifests/`, `tests/`, `data/.gitkeep`, and `outputs/.gitkeep` are included. The wheel also embeds default agent configs, scripts, docs, and manifests under `smith_agent.resources`, so installed CLI commands can load the default registry without a local source checkout. Runtime outputs still go to the current working directory unless `SMITH_RUNTIME_ROOT` is set.

## Usage

Run the original SMITH optimizer on an AnnData object:

```bash
python scripts/main.py \
  --adata_file data/example.h5ad \
  --saving_dir outputs/example/saving \
  --log_dir outputs/example/log \
  --tasks recon,cls \
  --task_name celltype \
  --panel_size 64 \
  --epoch 100 \
  --record 10 \
  --device cpu
```

Inspect SMITH-Agent registries from the source tree:

```bash
smith-cli --config configs/agent/agent.yaml tools
smith-cli --config configs/agent/agent.yaml datasets
```

After installing the wheel, the default packaged registry is available without `--config`:

```bash
smith-cli tools
smith-cli datasets
```

Start the interactive agent shell:

```bash
smith-cli --config configs/agent/agent.yaml shell
```

## Notebook Tutorials

The paper-oriented examples are executable Jupyter notebooks grouped by manuscript Results section. Editable and pre-executed notebooks live under `docs/source/tutorials/notebooks/` and are published in the [online tutorials](https://smith-panel-design.readthedocs.io/en/latest/tutorials/index.html).

```bash
python -m pip install -e '.[notebooks]'
jupyter lab docs/source/tutorials/notebooks
```

Use `python scripts/build_tutorial_notebooks.py --execute` to regenerate every documented notebook and its figures.

## Run Without External Data

The package includes a synthetic AnnData generator so the optimizer can be tested without downloading manuscript-scale datasets:

```bash
python scripts/make_smoke_h5ad.py --output data/smoke/smoke_panel.h5ad
python scripts/main.py \
  --adata_file data/smoke/smoke_panel.h5ad \
  --saving_dir outputs/smoke/saving \
  --log_dir outputs/smoke/logs \
  --tasks recon,cls,region,pathology,coordination \
  --task_name celltype \
  --panel_size 8 \
  --epoch 2 \
  --record 1 \
  --batch_size 16 \
  --rep_dim 8 \
  --rep_hidden_dims 8 \
  --head_hidden_dims 8 \
  --dim 8 \
  --device cpu \
  --balance_mode off
```

The selected target ranking is written to `outputs/smoke/saving/epoch_*.csv`.

Run the test suite:

```bash
python -m pytest tests
```

## Input AnnData Schema

SMITH expects an `.h5ad` file with genes or molecular targets in `adata.var_names` and observations in `adata.obs_names`. The core optimizer reads `adata.X` by default. If `--layer <name>` is supplied, it reads `adata.layers[<name>]` instead.

Supported task fields:

* `recon`: uses the expression matrix itself and requires no extra annotation.
* `cls`: reads the first available column from `obs['celltype']`, `obs['cell_type']`, or `obs['subclass']`.
* `region`: reads `obs['region']`.
* `pathology`: reads `obs['pathology']`.
* `coordination` or `standard_coordination`: reads `obsm['spatial']` and optimizes coordinate regression as `coo`.

Prior targets can be supplied with `--prior_file`, a CSV containing a `names` column matching `adata.var_names`.

## Data Policy

This clean package intentionally excludes large local assets:

* raw and processed H5AD data bundles
* benchmark/HPO output directories
* conda environments
* PDF build artifacts
* Python caches

The packaged dataset registry uses relative paths under `data/` and stores the original local source path in `metadata.original_local_path`. The original local workspace still contains those assets under directories such as `SMITH_unified/`, `SMITH_htapp/`, `SMITH_data_ribomap/`, `SMITH_data_elegans/`, `SMITH_ribomap/`, and `smith-agent/outputs/`.

## Relationship to the Local Workspace

This package was assembled from:

* `/workspace/fanyimin/SMITH_tool-main`
* `/workspace/fanyimin/smith-agent`
* `/workspace/fanyimin/smith_interface`
* selected README files from baseline and data directories

Use this directory for code cleanup, documentation, packaging, and future repository publication. Use the original data/output directories only when reproducing the full benchmark experiments.
