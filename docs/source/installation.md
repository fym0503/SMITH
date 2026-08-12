# Installation

SMITH requires Python 3.10 or newer. Install the source checkout in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For documentation and tests:

```bash
python -m pip install -e '.[dev,docs]'
```

Verify the optimizer, Agent registry and reproducibility layer:

```bash
smith-cli models
smith-repro list
smith-repro check
python -m pytest -q tests
```

ODT/SCRINSHOT, OligoMiner, PaintSHOP and ProbeDealer full screens require backend-specific executables, indexes and reference assets. Their absence does not prevent core SMITH training or the compact paper examples from running.
