# Installation

SMITH requires Python 3.10 or newer. Create an isolated environment and install the package from a checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For notebook development and plotting, install the notebook extra:

```bash
python -m pip install -e '.[notebooks]'
```

Check the installation and available workflows:

```bash
smith-cli models
smith-repro list
python -m pip install -e ".[dev]"
python -m pytest -q tests
```

External probe-design workflows need their respective executables, reference indexes and sequence files. Configure these with the environment variables described in the Agent backend settings.

## Tutorial reference environment

For the tutorial reference settings, use Python 3.12 and install the pinned numerical dependencies before starting Jupyter:

```bash
python -m pip install -r requirements-reproducibility.txt -e ".[notebooks]"
export SMITH_TUTORIAL_DEVICE=cuda:0
```

The reference results use CUDA training. Notebooks select CUDA when available and otherwise use CPU; the selected device is printed at startup. CPU training can produce different target rankings with the same seed.
