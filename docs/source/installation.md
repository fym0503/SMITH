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
python -m pytest -q tests
```

External probe-design workflows need their respective executables, reference indexes and sequence files. Configure these with the environment variables described in the Agent backend settings.
