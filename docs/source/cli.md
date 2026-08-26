# Command-line Reference

Use this page when you already know which workflow you want to run and need the exact command names. The biological rationale and worked examples are in [Introduction](introduction/index) and [Paper examples](tutorials/index).

## Core optimizer

Use `python scripts/main.py --help` for the complete training interface. The runner accepts an H5AD input, task list, panel size, output directories, model dimensions, optimization parameters and deterministic seed.

## SMITH-Agent

Install the package in the environment that will run the experiments. This
registers both `smith-agent` (the interactive shell) and `smith-cli` (the
command-oriented interface):

```bash
python -m pip install -e .
```

Configure an OpenAI-compatible planner before using free-form natural-language
requests. The agent still exposes deterministic tools when no planner is
configured, but it cannot interpret arbitrary questions without a model:

```bash
export OPENAI_BASE_URL=https://api.babelark.com/v1
export OPENAI_MODEL=<model-name>
export OPENAI_API_KEY=<api-key>
smith-agent --config configs/agent/agent.yaml shell
```

The shell accepts dataset paths in a message with an `@/absolute/path.h5ad`
mention, then can run registered SMITH, evaluation, feasibility, and reporting
tools. It is a terminal interface; the package does not provide a web server.

```bash
smith-cli tools
smith-cli datasets
smith-cli models
smith-cli skills
smith-cli run-tool <tool> --arguments-json '{}'
```

## Paper reproducibility

```bash
smith-repro list
smith-repro check [case|all]
smith-repro run <case> [--output-dir PATH]
```

Public case ids are `01_wmb`, `02_regulatory_activity`, `03_ribomap_transfer` and `05_agent`. `01_wmb` is a source-availability record. Check downloaded H5AD inputs with `smith-repro check <case> --data-root data/tutorials`. End-to-end `smith-repro run` commands require a SMITH GitHub checkout because the workflows and notebooks are repository artifacts.
