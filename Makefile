DATA_ROOT ?= data/tutorials

.PHONY: test repro-check repro-run docs wheel

test:
	python -m pytest -q tests

repro-check:
	smith-repro check --data-root "$(DATA_ROOT)"

repro-run:
	@set -e; for case in 02_regulatory_activity 03_ribomap_transfer 05_agent; do \
		smith-repro run $$case --data-root "$(DATA_ROOT)" --output-dir outputs/reproducibility/$$case; \
	done

docs:
	sphinx-build -W -b html docs/source docs/_build/html

wheel:
	python -m build --wheel
