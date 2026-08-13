# SMITH Reproducibility Layer

This directory maps each main Results section to one representative executable example. It does not claim that every panel of every manuscript figure is rebuilt by the default examples.

Use `smith-repro list`, `smith-repro check`, and `smith-repro run <case>` after installing the package. Full-paper sources and data-access constraints are recorded in `manifests/`.

The files under `fixtures/` are compact result tables copied from completed paper-workspace runs, not synthetic examples or raw biological datasets. Their SHA-256 values are pinned in the corresponding manifests. The original generation scripts for the real examples are archived under `workflows/`. The WMB source/data are unavailable and are documented as such rather than replaced with fake output.
