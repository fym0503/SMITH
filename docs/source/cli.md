# Command-line Reference

Use these commands to train panels, inspect Agent registries and run the tutorial workflows. The biological background is in [Introduction](introduction/index); worked analyses are in [Paper examples](tutorials/index).

## Core optimizer

Use `python scripts/main.py --help` for the complete training interface. The runner accepts an H5AD input, task list, panel size, output directories, model dimensions, optimization parameters and deterministic seed.

## SMITH-Agent

Install the package in the environment that will run the experiments. This
registers both `smith-agent` (the interactive shell) and `smith-cli` (the
command-oriented interface):

```bash
python -m pip install -e .
```

Set your OpenAI model and API key to route natural-language requests. Registry tools can also be invoked directly:

```bash
export OPENAI_MODEL="your-model-id"
read -r -s -p "API key: " OPENAI_API_KEY
export OPENAI_API_KEY
smith-agent --config configs/agent/agent.yaml shell
```

The shell accepts dataset paths in a message with an `@/absolute/path.h5ad`
mention, then can run registered SMITH, evaluation, feasibility, and reporting
tools. The command starts an interactive terminal session.

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
smith-repro run <case> --data-root PATH [--output-dir PATH]
```

Workflow ids are `02_regulatory_activity`, `03_ribomap_transfer` and `05_agent`. Check H5AD inputs with `smith-repro check <case> --data-root data/tutorials`, then run a workflow with `smith-repro run <case> --data-root data/tutorials`.
