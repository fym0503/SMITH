# Command-Line Reference

## Core optimizer

Use `python scripts/main.py --help` for the complete training interface. The runner accepts an H5AD input, task list, panel size, output directories, model dimensions, optimization parameters and deterministic seed.

## SMITH-Agent

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
