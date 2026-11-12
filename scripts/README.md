# Scripts

## Core commands

- `main.py`: train SMITH and export ranked targets.
- `eval.py`: evaluate a selected target panel.
- `download_tutorial_data.py`: download the complete tutorial bundle from Zenodo and verify its inputs.
- `build_tutorial_notebooks.py`: validate the editable tutorial notebooks and optionally execute them.
- `make_smoke_h5ad.py`: generate a small synthetic AnnData object for an installation check.

## Agent and feasibility utilities

The root utilities and `agent_examples/` support registered Agent tools, data preparation, probe screening and tutorial workflows. External screens use their own executable, index and reference sequence.

Run `python scripts/main.py --help` or `smith-cli --help` for command options.

Edit the `_source.ipynb` notebooks directly. Run `python scripts/build_tutorial_notebooks.py --execute` to refresh their `_executed.ipynb` companions; training settings come from the task profiles.

The training and evaluation commands delegate to `smith.train` and `smith.evaluate`. Standalone probe utilities live under `src/smith_agent/resources/scripts/` so they can also run in an external tool environment; the checkout scripts launch those same files.
