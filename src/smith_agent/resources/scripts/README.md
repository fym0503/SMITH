# Scripts

## Core commands

- `main.py`: train SMITH and export ranked targets.
- `eval.py`: evaluate a selected target panel.
- `download_tutorial_data.py`: download and verify a tutorial archive after its URL is published.
- `build_tutorial_notebooks.py`: build tutorial source notebooks and optionally execute them.
- `make_smoke_h5ad.py`: generate a small synthetic AnnData object for an installation check.

## Agent and feasibility utilities

The root utilities and `agent_examples/` support registered Agent tools, data preparation, probe screening and tutorial workflows. External screens use their own executable, index and reference sequence.

Run `python scripts/main.py --help` or `smith-cli --help` for command options.
